"""Path C: prosodic features + a lightweight classifier for emotional tone.

Why this exists rather than a speech-emotion model:

* Audio LLMs read the words, not the voice. arXiv:2510.10444 tested Gemini 2.5
  directly and found emotion accuracy collapses when the cue is prosodic rather
  than lexical. Our own data reproduces it - a flatly delivered obscenity is
  labelled `neutral` in ground truth and Gemini returns `upset`.
* Every off-the-shelf SER model is 660 MB - 1.1 GB against a 512 MB instance,
  and the two strongest (audEERING MSP-dim, emotion2vec+) are non-commercial or
  commercially restricted by their training data.

The brief's section 4 explicitly allows "acoustic features plus a lightweight
classifier", which is what this is: ~30 numbers per clip and a logistic
regression whose coefficients ship as JSON. No weights, no licence, no download.

The feature set targets arousal and valence rather than emotion categories,
because the five target labels map onto those two axes far more cleanly than
onto any categorical emotion set - `frustrated` has no acted analogue as a
category but is simply low valence at moderate arousal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ingest import decode
from .path_b_acoustic import FRAME, HOP, SR, _frames, _vad

COEF_PATH = Path(__file__).with_name("tone_coefficients.json")

PROSODY_FEATURES: tuple[str, ...] = (
    # arousal: pitch height and movement
    "f0_mean", "f0_std", "f0_range", "f0_p90", "f0_slope_mean", "f0_slope_abs",
    "f0_voiced_frac",
    # arousal: energy dynamics
    "rms_mean_db", "rms_std_db", "rms_range_db", "rms_attack",
    "energy_p95_p50_db",
    # tempo and rhythm
    "speech_rate", "pause_rate", "pause_mean_s", "pause_max_s", "voiced_run_mean_s",
    # voice quality -> tension / valence
    "jitter", "shimmer", "hnr_db", "spectral_tilt_db", "hf500_ratio",
    "spectral_centroid_hz", "centroid_std_hz",
    # contour shape
    "f0_final_slope", "rms_final_slope",
)

TONES = ("neutral", "satisfied", "frustrated", "upset", "distressed")
INTENSITIES = ("low", "medium", "high")


@dataclass
class ProsodyResult:
    emotional_tone: str = "neutral"
    emotional_intensity: str = "low"
    tone_probs: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    tts_separated: bool = False
    notes: list[str] = field(default_factory=list)


def _f0_track(fr: np.ndarray, voiced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised autocorrelation F0 tracker.

    Returns (f0_hz per frame, peak autocorrelation per frame); unvoiced frames
    are zero. Computing the autocorrelation of every frame at once via FFT keeps
    this to a few milliseconds per clip - a per-frame Python loop is roughly 20x
    slower and this runs on every file in production.
    """
    n = fr.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0)
    x = fr - fr.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * FRAME)))
    spec = np.fft.rfft(x, n=nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=1)[:, :FRAME]
    zero = ac[:, :1].copy()
    zero[zero <= 0] = 1e-20
    ac = ac / zero

    lo, hi = int(SR / 320), int(SR / 70)
    band = ac[:, lo:hi]
    idx = np.argmax(band, axis=1)
    peak = band[np.arange(n), idx]
    f0 = SR / (lo + idx).astype(np.float64)

    ok = voiced & (peak > 0.30)
    return np.where(ok, f0, 0.0), np.where(ok, peak, 0.0)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def separate_tts_human(
    fr: np.ndarray, voiced: np.ndarray, f0: np.ndarray, peak: np.ndarray
) -> tuple[np.ndarray, bool, str]:
    """Try to isolate the human customer from the synthetic agent.

    The agent on these calls is AutoAce's own TTS voice ("Erica", confirmed in
    the T0 transcripts). Synthetic speech is markedly more regular than human
    speech: lower cycle-to-cycle pitch perturbation and a higher, more stable
    autocorrelation peak. Clustering voiced runs on those two statistics is a far
    easier problem than human-vs-human diarization and needs no model.

    Returns (mask of frames judged human, whether separation was accepted, note).
    Separation is only accepted when the two clusters are actually distinct - a
    clip with one speaker must not be split down the middle.
    """
    runs = [(s, e) for s, e in _runs(voiced) if (e - s) >= 6]
    if len(runs) < 4:
        return voiced, False, "too few voiced runs to attempt separation"

    stats = []
    for s, e in runs:
        seg_f0 = f0[s:e][f0[s:e] > 0]
        seg_pk = peak[s:e][peak[s:e] > 0]
        if seg_f0.size < 3:
            stats.append((0.0, 0.0))
            continue
        jit = float(np.mean(np.abs(np.diff(seg_f0))) / (np.mean(seg_f0) + 1e-9))
        stats.append((jit, float(np.mean(seg_pk))))
    S = np.array(stats)
    if S.shape[0] < 4 or S[:, 0].std() < 1e-6:
        return voiced, False, "no usable variation between runs"

    Z = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-9)
    # 2-means on (jitter, periodicity), seeded deterministically.
    c = np.array([Z[np.argmin(Z[:, 0])], Z[np.argmax(Z[:, 0])]])
    for _ in range(30):
        d = ((Z[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        if len(set(lab.tolist())) < 2:
            return voiced, False, "clustering collapsed to one group"
        c = np.array([Z[lab == k].mean(axis=0) for k in (0, 1)])

    sep = float(np.linalg.norm(c[0] - c[1]))
    within = float(np.mean([np.linalg.norm(Z[lab == k] - c[k], axis=1).mean() for k in (0, 1)]))
    if sep < 1.5 * max(within, 1e-6):
        return voiced, False, f"clusters not distinct (sep {sep:.2f} vs within {within:.2f})"

    # The human cluster is the one with HIGHER jitter (less regular).
    human_k = int(np.argmax([S[lab == k][:, 0].mean() for k in (0, 1)]))
    mask = np.zeros_like(voiced)
    for (s, e), k in zip(runs, lab):
        if k == human_k:
            mask[s:e] = True
    if mask.sum() < 0.15 * voiced.sum():
        return voiced, False, "human cluster too small to trust"
    return mask, True, f"separated (sep {sep:.2f}, {mask.sum()}/{voiced.sum()} frames kept)"


def prosodic_features(path: str | Path, *, try_separation: bool = True) -> dict[str, float]:
    x = decode(path, sr=SR, mono=True)
    fr = _frames(x)
    if fr.shape[0] < 4:
        return dict.fromkeys(PROSODY_FEATURES, 0.0)

    voiced, rms_db = _vad(fr)
    f0, peak = _f0_track(fr, voiced)

    target = voiced
    if try_separation:
        target, ok, _note = separate_tts_human(fr, voiced, f0, peak)
        if not ok:
            target = voiced

    sel = target & (f0 > 0)
    f0v = f0[sel]
    if f0v.size < 3:
        f0v = f0[f0 > 0]
    if f0v.size < 3:
        return dict.fromkeys(PROSODY_FEATURES, 0.0)

    rms_lin = np.sqrt((fr**2).mean(axis=1))
    rms_t = rms_db[target] if target.any() else rms_db
    df0 = np.diff(f0v)

    # jitter / shimmer: cycle-to-cycle irregularity, the classic tension cues
    jitter = float(np.mean(np.abs(df0)) / (np.mean(f0v) + 1e-9))
    amp = rms_lin[sel] if sel.any() else rms_lin
    shimmer = float(np.mean(np.abs(np.diff(amp))) / (np.mean(amp) + 1e-12)) if amp.size > 2 else 0.0
    pk = peak[sel]
    pk = np.clip(pk[pk > 0], 1e-6, 0.999)
    hnr = float(10 * np.log10(np.mean(pk / (1 - pk)))) if pk.size else 0.0

    psd = np.abs(np.fft.rfft(fr[target] * np.hanning(FRAME), axis=1)) ** 2 \
        if target.any() else np.abs(np.fft.rfft(fr * np.hanning(FRAME), axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)
    mean_psd = psd.mean(axis=0) + 1e-20
    low = mean_psd[freqs < 500].sum()
    high = mean_psd[freqs >= 500].sum()
    tilt = float(10 * np.log10(low / (high + 1e-20)))
    centroids = (freqs * psd).sum(axis=1) / (psd.sum(axis=1) + 1e-20)

    v_runs = _runs(target)
    sil_runs = _runs(~target)
    dur_s = x.size / SR
    pause_lens = [(e - s) * HOP / SR for s, e in sil_runs]
    pause_lens = [p for p in pause_lens if p > 0.15]
    v_lens = [(e - s) * HOP / SR for s, e in v_runs]

    tail = max(3, len(f0v) // 4)
    f0_final = float(np.polyfit(np.arange(tail), f0v[-tail:], 1)[0]) if len(f0v) >= 4 else 0.0
    rt = rms_t[-max(3, len(rms_t) // 4):]
    rms_final = float(np.polyfit(np.arange(len(rt)), rt, 1)[0]) if len(rt) >= 4 else 0.0

    return {
        "f0_mean": float(f0v.mean()),
        "f0_std": float(f0v.std()),
        "f0_range": float(np.percentile(f0v, 95) - np.percentile(f0v, 5)),
        "f0_p90": float(np.percentile(f0v, 90)),
        "f0_slope_mean": float(df0.mean()) if df0.size else 0.0,
        "f0_slope_abs": float(np.abs(df0).mean()) if df0.size else 0.0,
        "f0_voiced_frac": float(sel.mean()),
        "rms_mean_db": float(rms_t.mean()),
        "rms_std_db": float(rms_t.std()),
        "rms_range_db": float(np.percentile(rms_t, 95) - np.percentile(rms_t, 5)),
        "rms_attack": float(np.mean(np.abs(np.diff(rms_t)))) if rms_t.size > 1 else 0.0,
        "energy_p95_p50_db": float(np.percentile(rms_t, 95) - np.percentile(rms_t, 50)),
        "speech_rate": float(len(v_runs) / max(dur_s, 1e-6)),
        "pause_rate": float(len(pause_lens) / max(dur_s, 1e-6)),
        "pause_mean_s": float(np.mean(pause_lens)) if pause_lens else 0.0,
        "pause_max_s": float(np.max(pause_lens)) if pause_lens else 0.0,
        "voiced_run_mean_s": float(np.mean(v_lens)) if v_lens else 0.0,
        "jitter": jitter,
        "shimmer": shimmer,
        "hnr_db": hnr,
        "spectral_tilt_db": tilt,
        "hf500_ratio": float(high / (low + high + 1e-20)),
        "spectral_centroid_hz": float(centroids.mean()),
        "centroid_std_hz": float(centroids.std()),
        "f0_final_slope": f0_final,
        "rms_final_slope": rms_final,
    }


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def analyse_prosody(path: str | Path) -> ProsodyResult:
    """Predict tone and intensity from prosody using the fitted coefficients."""
    feats = prosodic_features(path)
    res = ProsodyResult(features=feats)
    if not COEF_PATH.exists():
        res.notes.append("tone_coefficients.json missing - run scripts/fit_tone.py")
        return res

    model = json.loads(COEF_PATH.read_text())
    names = model["feature_names"]
    v = np.array([feats.get(k, 0.0) for k in names], dtype=np.float64)
    v = (v - np.array(model["mean"])) / np.array(model["std"])

    for field_name, classes, attr in (
        ("emotional_tone", TONES, "emotional_tone"),
        ("emotional_intensity", INTENSITIES, "emotional_intensity"),
    ):
        spec = model["fields"].get(field_name)
        if not spec:
            continue
        z = np.array(spec["coef"]) @ v + np.array(spec["intercept"])
        p = _softmax(z)
        order = spec.get("classes", list(classes))
        setattr(res, attr, order[int(np.argmax(p))])
        if field_name == "emotional_tone":
            res.tone_probs = {c: round(float(pi), 4) for c, pi in zip(order, p)}
    return res
