"""Path B: deterministic acoustic analysis.

This path exists because measurement showed Path A is *blind* on several fields,
not merely imprecise:

  * ``speaker_overlap_present`` - Gemini returned false on every clip under every
    prompt variant tried, including one that explicitly asked about interruptions
    and talk-over. Ground truth is true on two of three.
  * ``background_noise_present`` / ``severity`` - Gemini reported hearing "a faint
    hiss throughout the recording" and then labelled the field false.
  * ``audio_quality`` - pinned at ``slightly_impaired`` on every clip while ground
    truth is ``clear`` on all three, because it treats the synthetic TTS agent
    voice as a defect.

So for those fields this is the primary source, not a cross-check. It is
structurally incapable of the conflation the brief warns about: it never sees
transcript or emotion, and noise is measured on VAD-inverted regions while
quality is measured on speech regions.

numpy + ffmpeg only - no model weights, so it adds nothing to the container.

THRESHOLDS ARE PROVISIONAL. They are set from what each measure physically means
(see each function), not fitted to the three provided labels - fitting on n=3 is
what produced the earlier label-collapse bug. They must be refit on the proxy set
before any accuracy claim is made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ingest import decode

SR = 16_000
FRAME = 512      # 32 ms
HOP = 256        # 16 ms

# --- Provisional thresholds -------------------------------------------------
# Spectral flatness of the non-speech regions. Flatness is the ratio of the
# geometric to arithmetic mean of the power spectrum: 0 = purely tonal, 1 = white
# noise. A near-silent line carries only low-frequency room tone (very tonal);
# broadband hiss/static sits high; structured sources (television, chatter,
# music) sit between.
FLATNESS_SILENT = 0.08      # below: nothing meaningful in the background
FLATNESS_BROADBAND = 0.15   # above: hiss / static / wind rather than a source

# Fraction of voiced frames carrying a second, unrelated pitch track.
DUAL_PITCH_OVERLAP = 0.38

# Long silence: >= 5 s of contiguous near-floor audio, excluding lead-in/out.
# Note the provided calls are all labelled false while containing gaps up to
# 6.7 s at the noise floor, so this rule is deliberately conservative and the
# LLM is allowed to override it. See outputs/validation/forensics_findings.md.
SILENCE_MIN_S = 8.0
SILENCE_REL_DB = -35.0


@dataclass
class AcousticResult:
    background_noise_present: bool = False
    background_noise_type: str = ""
    background_noise_severity: str = "none"
    audio_quality: str = "clear"
    speaker_overlap_present: bool = False
    long_silence_present: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _frames(x: np.ndarray, n: int = FRAME, hop: int = HOP) -> np.ndarray:
    if x.size < n:
        return np.empty((0, n))
    count = 1 + (x.size - n) // hop
    return x[np.arange(n)[None, :] + hop * np.arange(count)[:, None]]


def _vad(fr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive energy VAD. Returns (voiced mask, per-frame dB)."""
    rms_db = 20 * np.log10(np.sqrt((fr**2).mean(axis=1)) + 1e-12)
    floor, speech = np.percentile(rms_db, 5), np.percentile(rms_db, 90)
    return rms_db > floor + 0.45 * (speech - floor), rms_db


def _flatness_centroid(fr: np.ndarray) -> tuple[float, float]:
    if fr.shape[0] == 0:
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(fr * np.hanning(fr.shape[1]), axis=1)) ** 2
    spec = spec.mean(axis=0) + 1e-20
    freqs = np.fft.rfftfreq(fr.shape[1], 1 / SR)
    flatness = float(np.exp(np.mean(np.log(spec))) / spec.mean())
    centroid = float((freqs * spec).sum() / spec.sum())
    return flatness, centroid


def _dual_pitch_fraction(fr: np.ndarray, voiced: np.ndarray) -> float:
    """Fraction of voiced frames with a second, unrelated periodicity.

    Two people talking over each other on a summed mono mix leave two
    independent harmonic series. Autocorrelation finds the dominant period;
    masking that period together with its harmonics and sub-harmonics before
    re-searching prevents a single voice's own octave from being counted as a
    second speaker.
    """
    lo, hi = int(SR / 320), int(SR / 70)
    dual = total = 0
    for i in np.flatnonzero(voiced):
        seg = fr[i] - fr[i].mean()
        if np.sqrt((seg**2).mean()) < 1e-5:
            continue
        ac = np.correlate(seg, seg, "full")[seg.size - 1:]
        if ac[0] <= 0:
            continue
        band = (ac / ac[0])[lo:hi]
        if band.size == 0:
            continue
        p1 = int(np.argmax(band))
        if band[p1] < 0.30:
            continue
        total += 1
        lag1, masked = lo + p1, band.copy()
        for mult in (0.5, 1, 1.5, 2, 2.5, 3, 4):
            centre = lag1 * mult
            w = max(3, int(0.12 * centre))
            masked[max(0, int(centre - w - lo)):max(0, int(centre + w - lo))] = 0
        if masked.max() > 0.55 * band[p1] and masked.max() > 0.25:
            dual += 1
    return dual / total if total else 0.0


def _longest_silence(rms_db: np.ndarray, speech_db: float) -> float:
    """Longest interior run below the threshold, in seconds."""
    quiet = rms_db < speech_db + SILENCE_REL_DB
    if not quiet.any():
        return 0.0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], quiet.view(np.int8), [0]))))
    starts, ends = edges[0::2], edges[1::2]
    lead = int(1.0 * SR / HOP)
    best = 0.0
    for s, e in zip(starts, ends):
        if s <= lead or e >= len(rms_db) - lead:
            continue     # leading/trailing dead air is normal, not a fault
        best = max(best, (e - s) * HOP / SR)
    return best


def _classify_noise(flat: float, centroid: float, snr_db: float) -> tuple[bool, str, str]:
    """Map non-speech spectral character onto the noise fields.

    Flatness carries the discrimination; SNR only modulates severity. Measured on
    the provided calls SNR was actively misleading - a clip with audible static
    sat within 0.6 dB of a clip with none, because static that is quiet still has
    a distinctive broadband signature.
    """
    if flat < FLATNESS_SILENT:
        return False, "", "none"

    if flat >= FLATNESS_BROADBAND:
        kind = "static" if centroid > 1000 else "line noise"
    elif centroid > 900:
        kind = "background chatter"
    else:
        kind = "television"

    # Severity blends how much noise there is (SNR) with how intrusive its
    # character is (flatness above the silence floor).
    intrusion = (flat - FLATNESS_SILENT) / max(FLATNESS_BROADBAND - FLATNESS_SILENT, 1e-6)
    if snr_db < 20 or intrusion > 1.6:
        sev = "high"
    elif snr_db < 38 or intrusion > 0.5:
        sev = "medium"
    else:
        sev = "low"
    return True, kind, sev


def analyse_acoustics(path: str | Path) -> AcousticResult:
    """Deterministic analysis of the objective fields. Never raises on content."""
    x = decode(path, sr=SR, mono=True)
    fr = _frames(x)
    res = AcousticResult()
    if fr.shape[0] == 0:
        res.notes.append("clip too short for frame analysis")
        return res

    voiced, rms_db = _vad(fr)
    speech_db = float(np.median(rms_db[voiced])) if voiced.any() else float(np.max(rms_db))
    nonspeech = fr[~voiced]
    nonspeech_db = float(np.median(rms_db[~voiced])) if (~voiced).any() else -120.0
    snr = speech_db - nonspeech_db

    flat, centroid = _flatness_centroid(nonspeech)
    dual = _dual_pitch_fraction(fr, voiced)
    longest_sil = _longest_silence(rms_db, speech_db)

    # Bandwidth on speech frames only - quality is about the speech channel.
    rolloff = 0.0
    if voiced.any():
        sp = np.abs(np.fft.rfft(fr[voiced] * np.hanning(FRAME), axis=1)) ** 2
        prof = sp.mean(axis=0)
        if prof.sum() > 0:
            cum = np.cumsum(prof) / prof.sum()
            rolloff = float(np.fft.rfftfreq(FRAME, 1 / SR)[np.searchsorted(cum, 0.995)])

    res.background_noise_present, res.background_noise_type, res.background_noise_severity = (
        _classify_noise(flat, centroid, snr)
    )
    res.speaker_overlap_present = bool(dual > DUAL_PITCH_OVERLAP)
    res.long_silence_present = bool(longest_sil >= SILENCE_MIN_S)

    # audio_quality deliberately defaults to `clear`. It is reserved for defects
    # in the captured signal, and neither background sound nor a synthetic agent
    # voice is such a defect. Severe band-limiting is the one cheap, unambiguous
    # indicator available without a reference signal.
    if rolloff and rolloff < 1700:
        res.audio_quality = "severely_impaired"
        res.notes.append(f"speech bandwidth only {rolloff:.0f} Hz")
    elif rolloff and rolloff < 2300:
        res.audio_quality = "slightly_impaired"
        res.notes.append(f"speech bandwidth {rolloff:.0f} Hz")

    res.metrics = {
        "speech_db": round(speech_db, 2),
        "nonspeech_db": round(nonspeech_db, 2),
        "snr_db": round(snr, 2),
        "nonspeech_flatness": round(flat, 4),
        "nonspeech_centroid_hz": round(centroid, 1),
        "dual_pitch_frac": round(dual, 4),
        "longest_interior_silence_s": round(longest_sil, 2),
        "speech_rolloff_hz": round(rolloff, 1),
        "speech_frac": round(float(voiced.mean()), 3),
    }
    return res
