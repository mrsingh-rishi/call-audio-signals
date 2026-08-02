"""Fit the overlap decision threshold for Path D on the proxy set.

The model itself needs no fitting - reading the powerset argmax is exact. The one
free parameter is how much detected overlap counts as "enough to affect
understanding or analysis", which is the brief's wording and a judgement the
model cannot make for us.

Discipline, same as `fit_detectors.py`:

* The threshold is chosen on the **fit** split only (271 clips, disjoint
  speakers), then reported on the **eval** split it has never seen.
* The three provided calls are NOT used to choose it. They are printed at the
  end as a smoke test and explicitly labelled n=3.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoace.config import REPO_ROOT  # noqa: E402
from autoace.path_d_overlap import COEF_PATH, analyse_overlap  # noqa: E402

PROXY = REPO_ROOT / "data" / "proxy"
CACHE = PROXY / "overlap_features.npz"


def _manifest_key(rows: list[dict]) -> str:
    """Identify this build of the proxy set - see `fit_detectors._manifest_key`.

    Clip names carry a spec fingerprint, so a regenerated set invalidates the
    cache. Matching on row count alone would silently reuse measurements taken
    from the previous build's audio.
    """
    import hashlib

    return hashlib.sha256("|".join(r["name"] for r in rows).encode()).hexdigest()[:16]


def measure(force: bool = False) -> tuple[np.ndarray, list[dict]]:
    """Overlap seconds + longest-run seconds per proxy clip, cached."""
    rows = list(csv.DictReader((PROXY / "labels.csv").open(newline="")))
    key = _manifest_key(rows)
    if CACHE.exists() and not force:
        d = np.load(CACHE, allow_pickle=True)
        if len(d["M"]) == len(rows) and str(d.get("key", "")) == key:
            return d["M"], rows
        print("  proxy set changed since the cache was written; re-measuring")
    M = np.zeros((len(rows), 2), dtype=np.float64)
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        res = analyse_overlap(PROXY / "audio" / r["name"])
        M[i] = (res.overlap_seconds, res.longest_overlap_s)
        if (i + 1) % 50 == 0:
            rate = (time.perf_counter() - t0) / (i + 1)
            print(f"  {i + 1}/{len(rows)}  ({rate:.2f}s/clip)")
    np.savez_compressed(CACHE, M=M, key=key)
    return M, rows


def _balanced_acc(y: np.ndarray, pred: np.ndarray) -> float:
    out = []
    for c in (False, True):
        m = y == c
        if m.any():
            out.append(float((pred[m] == c).mean()))
    return float(np.mean(out)) if out else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="recompute the cache")
    ap.add_argument("--write", action="store_true", help="write the coefficients file")
    args = ap.parse_args()

    M, rows = measure(args.force)
    y = np.array([json.loads(r["result_json"])["speaker_overlap_present"] for r in rows])
    split = np.array([r["split"] for r in rows])
    fit_m, ev_m = split == "fit", split == "eval"

    print(f"\nproxy: {fit_m.sum()} fit / {ev_m.sum()} eval, speakers disjoint")
    print(f"overlap positives: fit {y[fit_m].mean():.3f}  eval {y[ev_m].mean():.3f}\n")

    # Sweep the threshold on the FIT split only. The grid starts near one frame
    # (~17 ms) rather than at 0.1 s: the first sweep bottomed out at its own
    # lower bound, which meant the grid, not the data, was choosing.
    grid = np.round(np.concatenate([
        np.arange(0.01, 0.30, 0.01), np.arange(0.30, 4.01, 0.05)
    ]), 3)
    scores = [_balanced_acc(y[fit_m], M[fit_m, 0] >= t) for t in grid]
    best_i = int(np.argmax(scores))
    best_t = float(grid[best_i])
    if best_i == len(grid) - 1:
        print("  WARNING: optimum is at the TOP of the grid - widen it.")
    elif best_i == 0:
        # Not a truncated grid: the model's frame is ~16.9 ms, so the smallest
        # non-zero measurement is one frame and every threshold below that is
        # the identical rule. The grid has saturated, it has not been cut off.
        print("  note: optimum is 'any detected overlap frame' (one frame ~17 ms); "
              "thresholds below that are the same rule, so the grid has saturated.")
    print(f"threshold chosen on FIT split: {best_t:.3f}s "
          f"(fit balanced acc {scores[best_i]:.3f})")
    print("   fit-split curve:", ", ".join(
        f"{t}s={s:.3f}" for t, s in zip(grid, scores)
        if t in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)))

    # Report on the EVAL split, which played no part in choosing it.
    pred_ev = M[ev_m, 0] >= best_t
    bal = _balanced_acc(y[ev_m], pred_ev)
    tp = int((pred_ev & y[ev_m]).sum()); fp = int((pred_ev & ~y[ev_m]).sum())
    fn = int((~pred_ev & y[ev_m]).sum()); tn = int((~pred_ev & ~y[ev_m]).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    baseline = _balanced_acc(y[ev_m], np.ones_like(y[ev_m], dtype=bool))

    print("\n=== EVAL SPLIT (unseen speakers) ===")
    print(f"  balanced accuracy  {bal:.3f}   (Path B dual-pitch scores 0.544 on these same "
          f"labels, baseline {baseline:.3f})")
    print(f"  precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}")
    print(f"  confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    # Does detected duration track the overlap that is actually THERE? Correlate
    # against `actual_overlap_s` (measured two-voice time), not `overlap_s` (what
    # was requested) - the two diverge badly, which is the whole point of the
    # generator fix. The dual-pitch cue scored 0.069 against the requested
    # duration, which is what "unsolved" meant.
    act = np.array([float(r.get("actual_overlap_s") or 0) for r in rows])
    req = np.array([float(r["overlap_s"] or 0) for r in rows])
    corr = float(np.corrcoef(act[ev_m], M[ev_m, 0])[0, 1])
    corr_req = float(np.corrcoef(req[ev_m], M[ev_m, 0])[0, 1])
    print(f"  corr(ACTUAL two-voice s, detected s) = {corr:.3f}")
    print(f"  corr(requested s,        detected s) = {corr_req:.3f} "
          f"(dual-pitch cue scored 0.069 here)")

    if args.write:
        COEF_PATH.write_text(json.dumps({
            "min_overlap_s": best_t,
            "fitted_on": "proxy fit split (271 clips, speakers disjoint from eval)",
            "eval_balanced_accuracy": round(bal, 4),
            "eval_f1": round(f1, 4),
            "eval_correlation_injected_vs_detected": round(corr, 4),
            "model": "pyannote/segmentation-3.0 (MIT), ONNX export via k2-fsa",
        }, indent=2) + "\n")
        print(f"\nwrote {COEF_PATH}")
    else:
        print("\n(dry run - pass --write to save the threshold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
