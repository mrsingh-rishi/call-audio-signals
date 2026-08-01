"""Reproducible audio forensics - the measurements behind plan section 2.

Run as ``python -m autoace.forensics <files-or-dir>``. Every number quoted in
the memo comes from here, so the reviewer can re-derive them on their own
clips rather than taking the write-up on trust.

The clipping test deserves a note. Peak dBFS is *not* evidence of clipping:
lossy transform codecs routinely decode to intersample peaks above full scale
without the source ever having clipped. On the provided calls, call_001 peaks
at +0.035 dBFS but has only 3 samples at full scale out of 1.49 M and zero
flat-topped runs - decoder overshoot, not clipping. Real clipping shows up as
sustained runs of consecutive samples pinned at full scale, so that is what
:func:`clipping_report` measures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .ingest import (
    ANALYSIS_SR,
    AudioIngestError,
    analyse_channels,
    decode,
    probe,
)

BAND_EDGES: tuple[int, ...] = (
    0, 125, 250, 500, 1000, 2000, 3000, 3500, 4000,
    5000, 6000, 8000, 10000, 12000, 14000, 16000, 20000, 24000,
)


def clipping_report(x: np.ndarray) -> dict[str, Any]:
    """Distinguish true clipping from codec intersample overshoot.

    ``x`` must be the RAW (un-normalised) signal - normalising first would
    rescale the very peaks being tested.
    """
    n = int(x.size)
    peak = float(np.abs(x).max()) if n else 0.0
    out: dict[str, Any] = {
        "n_samples": n,
        "peak_abs": peak,
        "peak_dbfs": float(20 * np.log10(peak)) if peak > 0 else float("-inf"),
        "samples_at_or_above_full_scale": int(np.sum(np.abs(x) >= 1.0)),
        "thresholds": {},
    }
    for thr in (0.999, 0.99, 0.95):
        at = np.abs(x) >= thr
        # Run-length encode the boolean mask by finding transition indices.
        edges = np.flatnonzero(np.diff(np.concatenate(([0], at.view(np.int8), [0]))))
        starts, ends = edges[0::2], edges[1::2]
        runs = ends - starts
        out["thresholds"][str(thr)] = {
            "n_samples": int(at.sum()),
            "fraction": float(at.mean()) if n else 0.0,
            "runs_ge_3": int(np.sum(runs >= 3)),
            "longest_run": int(runs.max()) if runs.size else 0,
        }
    # Flat-topping is the signature of genuine hard clipping: high-amplitude
    # samples with near-zero slope between them.
    if n > 1:
        hi = np.abs(x[:-1]) > 0.95
        flat = hi & (np.abs(np.diff(x)) < 1e-5)
        out["flat_top_samples"] = int(flat.sum())
    else:
        out["flat_top_samples"] = 0

    t = out["thresholds"]["0.999"]
    out["true_clipping_detected"] = bool(
        out["flat_top_samples"] > 0 and t["runs_ge_3"] > 0 and t["fraction"] > 1e-4
    )
    return out


def spectral_bands(
    x: np.ndarray, sr: int, edges: tuple[int, ...] = BAND_EDGES, nfft: int = 8192
) -> dict[str, Any]:
    """Mean PSD per band over speech-active frames, in dB relative to peak.

    Restricting to active frames matters: averaging over silence would bury the
    band structure under the noise floor. The output distinguishes wideband
    VoIP (gradual rolloff, content past 8 kHz) from narrowband telephony (a
    30 dB+ wall at 3.4-4 kHz).
    """
    hop = nfft // 2
    if x.size < nfft:
        return {"error": "clip shorter than FFT window", "bands": {}}
    nfr = 1 + (x.size - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(nfr)[:, None]
    frames = x[idx] * np.hanning(nfft)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1 / sr)

    total = power.sum(axis=1)
    active = total > np.percentile(total, 70)
    if not active.any():
        active = np.ones_like(total, dtype=bool)
    prof = power[active].mean(axis=0)
    ref, tot = prof.max(), prof.sum()

    bands: dict[str, dict[str, float]] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            continue
        bands[f"{lo}-{hi}"] = {
            "mean_psd_db_rel_peak": float(10 * np.log10(prof[m].mean() / ref + 1e-30)),
            "share_pct": float(100 * prof[m].sum() / tot),
        }
    return {"n_frames": int(nfr), "n_active_frames": int(active.sum()), "bands": bands}


def silence_sweep(
    x: np.ndarray, sr: int, min_durations: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)
) -> dict[str, Any]:
    """Contiguous low-energy gaps at several thresholds relative to speech level.

    Reported as a sweep rather than a single verdict because the operative
    threshold is fitted on the proxy set, not guessed here. The provided calls
    are all labelled ``long_silence_present: false`` while containing gaps up
    to 3.54 s, which is why the fitted rule sits at >= 5.0 s.
    """
    n, hop = int(0.02 * sr), int(0.01 * sr)
    if x.size < n:
        return {"error": "clip too short", "sweep": {}}
    nfr = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(nfr)[:, None]
    fr_rms = np.sqrt((x[idx] ** 2).mean(axis=1))
    fr_db = 20 * np.log10(fr_rms + 1e-300)

    speech_db = float(np.percentile(fr_db, 90))
    out: dict[str, Any] = {"speech_level_dbfs": speech_db, "sweep": {}}

    for rel in (-35.0, -45.0, -55.0):
        thr = speech_db + rel
        quiet = fr_db < thr
        edges = np.flatnonzero(np.diff(np.concatenate(([0], quiet.view(np.int8), [0]))))
        starts, ends = edges[0::2], edges[1::2]
        durations = (ends - starts) * hop / sr
        # Exclude leading/trailing dead air - a call that starts before the
        # first word is normal, not a call-flow fault.
        interior = [
            float(d)
            for s, d in zip(starts, durations)
            if (s * hop / sr) > 1.0 and ((s + (d * sr / hop)) * hop / sr) < (x.size / sr - 1.0)
        ]
        out["sweep"][f"{rel:+.0f}dB_rel_speech"] = {
            "threshold_dbfs": float(thr),
            "longest_interior_gap_s": max(interior) if interior else 0.0,
            "n_gaps_over": {
                f"{md}s": sum(1 for d in interior if d >= md) for md in min_durations
            },
        }
    return out


def forensics_for_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    pr = probe(p)
    report: dict[str, Any] = {
        "file": p.name,
        "probe": {
            "codec": pr.codec,
            "sample_rate": pr.sample_rate,
            "channels": pr.channels,
            "channel_layout": pr.channel_layout,
            "duration_s": pr.duration_s,
            "bit_rate": pr.bit_rate,
            "format_name": pr.format_name,
            "size_bytes": pr.size_bytes,
            "encoder": pr.encoder,
        },
    }
    if pr.channels >= 2:
        ch = analyse_channels(p)
        report["channels"] = {
            "rms_l_dbfs": ch.rms_l_dbfs,
            "rms_r_dbfs": ch.rms_r_dbfs,
            "rms_l_minus_r_dbfs": ch.rms_diff_dbfs,
            "max_abs_diff": ch.max_abs_diff,
            "identical_fraction": ch.identical_fraction,
            "pearson_r": ch.pearson_r,
            "is_duplicated_mono": ch.is_duplicated_mono,
        }

    # Clipping is measured per channel and reported for the worst one. A mono
    # downmix - even a level-preserving one - can average away clipping that
    # exists in only one channel of a true-stereo recording.
    if pr.channels > 1:
        multi = decode(p, sr=ANALYSIS_SR, mono=False)
        per_channel = [clipping_report(multi[:, i]) for i in range(multi.shape[1])]
        worst = max(per_channel, key=lambda c: c["thresholds"]["0.999"]["n_samples"])
        worst["measured_on"] = f"worst of {multi.shape[1]} channels"
        report["clipping"] = worst
    else:
        report["clipping"] = clipping_report(decode(p, sr=ANALYSIS_SR, mono=True))
        report["clipping"]["measured_on"] = "mono"

    mono = decode(p, sr=ANALYSIS_SR, mono=True)
    report["spectrum"] = spectral_bands(mono, ANALYSIS_SR)
    report["silence"] = silence_sweep(mono, ANALYSIS_SR)
    return report


def _print_human(r: dict[str, Any]) -> None:
    pr = r["probe"]
    print(f"\n{'=' * 72}\n{r['file']}\n{'=' * 72}")
    print(f"  {pr['codec']}  {pr['sample_rate']} Hz  {pr['channels']}ch "
          f"({pr['channel_layout']})  {pr['duration_s']:.3f}s  "
          f"{(pr['bit_rate'] or 0) // 1000} kbps  [{pr['format_name']}]")
    if pr["encoder"]:
        print(f"  encoder: {pr['encoder']}")

    if "channels" in r:
        c = r["channels"]
        print("\n  -- channel layout --")
        print(f"    RMS L / R           : {c['rms_l_dbfs']:.4f} / {c['rms_r_dbfs']:.4f} dBFS")
        print(f"    RMS(L-R)            : {c['rms_l_minus_r_dbfs']:.4f} dBFS")
        print(f"    max |L-R|           : {c['max_abs_diff']:.6e}")
        print(f"    identical samples   : {100 * c['identical_fraction']:.4f}%")
        print(f"    pearson r           : {c['pearson_r']:.12f}")
        verdict = "DUPLICATED MONO" if c["is_duplicated_mono"] else "TRUE DUAL-CHANNEL"
        print(f"    verdict             : {verdict}")

    cl = r["clipping"]
    print("\n  -- clipping (runs + flat-top, not peak dBFS) --")
    print(f"    peak                : {cl['peak_dbfs']:+.4f} dBFS")
    print(f"    samples >= FS       : {cl['samples_at_or_above_full_scale']} / {cl['n_samples']}")
    t = cl["thresholds"]["0.999"]
    print(f"    @0.999: n={t['n_samples']} ({100 * t['fraction']:.5f}%)  "
          f"runs>=3={t['runs_ge_3']}  longest={t['longest_run']}")
    print(f"    flat-top samples    : {cl['flat_top_samples']}")
    print(f"    TRUE CLIPPING       : {cl['true_clipping_detected']}")

    sp = r["spectrum"]
    if sp.get("bands"):
        print(f"\n  -- spectrum ({sp['n_active_frames']}/{sp['n_frames']} active frames) --")
        for name, b in sp["bands"].items():
            if b["mean_psd_db_rel_peak"] > -100:
                print(f"    {name:>12} Hz : {b['mean_psd_db_rel_peak']:8.2f} dB  "
                      f"({b['share_pct']:7.4f}%)")

    si = r["silence"]
    if si.get("sweep"):
        print(f"\n  -- silence (speech level {si['speech_level_dbfs']:.1f} dBFS) --")
        for name, s in si["sweep"].items():
            gaps = ", ".join(f"{k}:{v}" for k, v in s["n_gaps_over"].items())
            print(f"    {name:>18} (={s['threshold_dbfs']:7.1f} dBFS) "
                  f"longest={s['longest_interior_gap_s']:6.2f}s  gaps[{gaps}]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audio forensics for call clips.")
    ap.add_argument("paths", nargs="+", help="audio files or a directory")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args(argv)

    targets: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            targets.extend(sorted(q for q in p.iterdir() if q.is_file()))
        else:
            targets.append(p)

    results, failures = [], 0
    for t in targets:
        try:
            results.append(forensics_for_file(t))
        except AudioIngestError as exc:
            failures += 1
            results.append({"file": t.name, "error": exc.reason})
            if not args.json:
                print(f"\n{'=' * 72}\n{t.name}\n{'=' * 72}\n  SKIPPED: {exc.reason}")

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for r in results:
            if "error" not in r:
                _print_human(r)
    # Non-zero only if nothing could be read at all; a partial failure is
    # reported per file and must not fail the whole run.
    return 1 if targets and failures == len(targets) else 0


if __name__ == "__main__":
    raise SystemExit(main())
