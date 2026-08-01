"""Measure the acoustic signals behind the four fields Gemini cannot produce.

Before building a deterministic detector it is worth establishing that the target
labels are recoverable from the signal at all. This script measures, per clip:

  * speech / non-speech split and the level difference between them  -> noise presence
  * spectral flatness and centroid of the non-speech regions          -> noise character
  * frames containing two simultaneous pitch tracks                   -> speaker overlap
  * bandwidth and clipping                                            -> audio quality

Everything is numpy + ffmpeg. No model weights, so it runs anywhere the
container runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoace.config import REPO_ROOT  # noqa: E402
from autoace.ingest import decode  # noqa: E402

SR = 16_000
FRAME = 512          # 32 ms
HOP = 256            # 16 ms


def frames(x: np.ndarray, n: int = FRAME, hop: int = HOP) -> np.ndarray:
    if x.size < n:
        return np.empty((0, n))
    count = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]


def vad(fr: np.ndarray) -> np.ndarray:
    """Energy VAD with an adaptive floor.

    The threshold sits between the noise floor (5th percentile) and the speech
    level (90th percentile) rather than at a fixed dBFS, so it survives the
    loudness differences between clips.
    """
    rms_db = 20 * np.log10(np.sqrt((fr**2).mean(axis=1)) + 1e-12)
    floor, speech = np.percentile(rms_db, 5), np.percentile(rms_db, 90)
    thr = floor + 0.45 * (speech - floor)
    return rms_db > thr


def spectral_stats(fr: np.ndarray) -> tuple[float, float]:
    """Return (spectral flatness, spectral centroid Hz).

    Flatness near 1 means broadband/noise-like (hiss, static); near 0 means
    tonal/structured (speech, music, television). This is what separates
    'sharp static' from 'TV' without a classifier.
    """
    if fr.shape[0] == 0:
        return float("nan"), float("nan")
    spec = np.abs(np.fft.rfft(fr * np.hanning(fr.shape[1]), axis=1)) ** 2
    spec = spec.mean(axis=0) + 1e-20
    freqs = np.fft.rfftfreq(fr.shape[1], 1 / SR)
    gmean = np.exp(np.mean(np.log(spec)))
    flatness = float(gmean / spec.mean())
    centroid = float((freqs * spec).sum() / spec.sum())
    return flatness, centroid


def dual_pitch_fraction(fr: np.ndarray, voiced: np.ndarray) -> tuple[float, int]:
    """Fraction of voiced frames containing a second, unrelated pitch track.

    Overlapping talkers on a summed mono mix leave two independent harmonic
    series. Autocorrelation finds the dominant period; suppressing that period
    with its harmonics and sub-harmonics and re-searching reveals whether a
    second, unrelated periodicity is present. Suppression is what stops a single
    voice's own octave from counting as a second speaker.
    """
    lo, hi = int(SR / 320), int(SR / 70)     # 70-320 Hz plausible F0 range
    n_dual = n_voiced = 0
    for i in np.flatnonzero(voiced):
        seg = fr[i] - fr[i].mean()
        if np.sqrt((seg**2).mean()) < 1e-5:
            continue
        ac = np.correlate(seg, seg, "full")[seg.size - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        band = ac[lo:hi]
        if band.size == 0:
            continue
        p1 = int(np.argmax(band))
        if band[p1] < 0.30:                  # unvoiced after all
            continue
        n_voiced += 1
        lag1 = lo + p1
        masked = band.copy()
        # Suppress the primary lag, its harmonics and its sub-harmonics.
        for mult in (0.5, 1, 1.5, 2, 2.5, 3, 4):
            centre = lag1 * mult
            w = max(3, int(0.12 * centre))
            a, b = int(centre - w - lo), int(centre + w - lo)
            masked[max(0, a):max(0, b)] = 0
        if masked.max() > 0.55 * band[p1] and masked.max() > 0.25:
            n_dual += 1
    return (n_dual / n_voiced if n_voiced else 0.0), n_voiced


def analyse(path: Path) -> dict:
    x = decode(path, sr=SR, mono=True)
    fr = frames(x)
    voiced = vad(fr)
    rms_db = 20 * np.log10(np.sqrt((fr**2).mean(axis=1)) + 1e-12)

    speech_db = float(np.median(rms_db[voiced])) if voiced.any() else float("nan")
    nonspeech = fr[~voiced]
    nonspeech_db = float(np.median(rms_db[~voiced])) if (~voiced).any() else float("nan")
    flat, centroid = spectral_stats(nonspeech)
    dual, n_voiced = dual_pitch_fraction(fr, voiced)

    # Bandwidth: highest frequency holding meaningful energy in speech frames.
    sp = np.abs(np.fft.rfft(fr[voiced] * np.hanning(FRAME), axis=1)) ** 2
    prof = sp.mean(axis=0) if sp.shape[0] else np.zeros(FRAME // 2 + 1)
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)
    cum = np.cumsum(prof) / (prof.sum() + 1e-20)
    rolloff = float(freqs[np.searchsorted(cum, 0.995)]) if prof.sum() > 0 else 0.0

    return {
        "duration_s": round(x.size / SR, 1),
        "speech_frac": round(float(voiced.mean()), 3),
        "speech_db": round(speech_db, 1),
        "nonspeech_db": round(nonspeech_db, 1),
        "snr_db": round(speech_db - nonspeech_db, 1),
        "nonspeech_flatness": round(flat, 4),
        "nonspeech_centroid_hz": round(centroid, 0),
        "dual_pitch_frac": round(dual, 4),
        "n_voiced_frames": n_voiced,
        "rolloff_995_hz": round(rolloff, 0),
    }


def main() -> int:
    d = REPO_ROOT / "data" / "provided_calls"
    truth = {}
    import csv
    for row in csv.DictReader(open(d / "labels.csv", newline="", encoding="utf-8-sig")):
        truth[row["name"]] = json.loads(row["result_json"])

    results = {}
    for p in sorted(d.glob("*.ogg")):
        results[p.name] = analyse(p)

    print("=" * 104)
    print("ACOUSTIC MEASUREMENTS vs GROUND TRUTH")
    print("=" * 104)
    keys = list(next(iter(results.values())).keys())
    print(f"  {'metric':<24}" + "".join(f"{n.replace('.ogg',''):>18}" for n in results))
    print("  " + "-" * 100)
    for k in keys:
        print(f"  {k:<24}" + "".join(f"{results[n][k]:>18}" for n in results))

    print("\n  " + "-" * 100)
    print(f"  {'GROUND TRUTH':<24}" + "".join(f"{n.replace('.ogg',''):>18}" for n in results))
    print("  " + "-" * 100)
    for field in ("background_noise_present", "background_noise_type",
                  "background_noise_severity", "speaker_overlap_present", "audio_quality"):
        print(f"  {field:<24}" + "".join(f"{str(truth[n][field]):>18}" for n in results))

    print("\n" + "=" * 104)
    print("IS EACH LABEL RECOVERABLE FROM THE SIGNAL?")
    print("=" * 104)
    names = list(results)
    snr = [results[n]["snr_db"] for n in names]
    noise_truth = [truth[n]["background_noise_present"] for n in names]
    print(f"  noise present : SNR {snr}  vs truth {noise_truth}")
    print(f"                  -> {'SEPARABLE' if max(s for s,t in zip(snr,noise_truth) if t) < min(s for s,t in zip(snr,noise_truth) if not t) else 'NOT separable by SNR alone'}")
    dp = [results[n]["dual_pitch_frac"] for n in names]
    ov_truth = [truth[n]["speaker_overlap_present"] for n in names]
    print(f"  overlap       : dual-pitch {dp}  vs truth {ov_truth}")
    pos = [d_ for d_, t in zip(dp, ov_truth) if t]
    neg = [d_ for d_, t in zip(dp, ov_truth) if not t]
    print(f"                  -> {'SEPARABLE' if pos and neg and min(pos) > max(neg) else 'NOT cleanly separable'}")
    flats = [results[n]["nonspeech_flatness"] for n in names]
    print(f"  noise type    : flatness {flats}  vs truth "
          f"{[truth[n]['background_noise_type'] for n in names]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
