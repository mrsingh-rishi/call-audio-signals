"""Fit the prosodic tone/intensity classifier on the proxy set.

Same discipline as fit_detectors.py: fitted on the fit split only, standardised
with fit-split statistics only, evaluated on speakers never seen during fitting.
scikit-learn trains; only JSON coefficients ship.

The honest caveat, recorded here and repeated in the memo: the emotion labels
come from *acted* corpora (RAVDESS, CREMA-D). The SER literature is explicit
that acted prosody exaggerates cues relative to spontaneous speech, so the
numbers this produces are an optimistic bound on real dealership calls. The
derived fields (noise, overlap, silence, quality) do not have this problem -
their ground truth comes from our own degradation chain.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoace.config import REPO_ROOT  # noqa: E402
from autoace.path_c_prosody import (  # noqa: E402
    INTENSITIES,
    PROSODY_FEATURES,
    TONES,
    prosodic_features,
)

PROXY = REPO_ROOT / "data" / "proxy"
CACHE = PROXY / "prosody_features.npz"
OUT = REPO_ROOT / "src" / "autoace" / "tone_coefficients.json"


def build(force: bool = False) -> tuple[np.ndarray, list[dict]]:
    rows = list(csv.DictReader((PROXY / "labels.csv").open(newline="")))
    if CACHE.exists() and not force:
        d = np.load(CACHE, allow_pickle=True)
        if len(d["X"]) == len(rows):
            return d["X"], rows
    X = []
    for i, r in enumerate(rows):
        f = prosodic_features(PROXY / "audio" / r["name"])
        X.append([f[k] for k in PROSODY_FEATURES])
        if (i + 1) % 50 == 0:
            print(f"  prosody {i + 1}/{len(rows)}", flush=True)
    X = np.asarray(X, dtype=np.float64)
    np.savez_compressed(CACHE, X=X)
    return X, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score

    X, rows = build(args.force)
    truth = [json.loads(r["result_json"]) for r in rows]
    split = np.array([r["split"] for r in rows])
    fit_m, ev_m = split == "fit", split == "eval"

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X[fit_m].mean(axis=0), X[fit_m].std(axis=0) + 1e-9
    Xs = (X - mu) / sd

    model = {"feature_names": list(PROSODY_FEATURES),
             "mean": mu.tolist(), "std": sd.tolist(), "fields": {}}
    lines: list[str] = []

    for field, classes in (("emotional_tone", TONES),
                           ("emotional_intensity", INTENSITIES)):
        y = np.array([classes.index(str(t[field])) for t in truth])
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.3)
        clf.fit(Xs[fit_m], y[fit_m])
        pred = clf.predict(Xs[ev_m])
        macro = f1_score(y[ev_m], pred, average="macro", zero_division=0)
        acc = float((pred == y[ev_m]).mean())
        # Majority-class floor on the SAME eval split - this is baseline B0.
        maj = np.bincount(y[fit_m]).argmax()
        b0_acc = float((y[ev_m] == maj).mean())
        b0_f1 = f1_score(y[ev_m], np.full_like(y[ev_m], maj),
                         average="macro", zero_division=0)

        model["fields"][field] = {
            "classes": list(classes),
            "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
        }
        lines.append(f"\n  {field}: accuracy={acc:.3f}  macroF1={macro:.3f}"
                     f"   [B0 majority: acc={b0_acc:.3f} macroF1={b0_f1:.3f}]")
        per = f1_score(y[ev_m], pred, average=None, zero_division=0,
                       labels=list(range(len(classes))))
        for c, f1c in zip(classes, per):
            support = int((y[ev_m] == classes.index(c)).sum())
            lines.append(f"      {c:<12} F1={f1c:.3f}  (n={support})")
        cm = confusion_matrix(y[ev_m], pred, labels=list(range(len(classes))))
        lines.append("      confusion (rows=truth): "
                     + " | ".join(f"{c[:4]}:{list(r)}" for c, r in zip(classes, cm)))

    OUT.write_text(json.dumps(model, indent=2))
    print("EVAL SPLIT (unseen speakers)" + "".join(lines))
    print(f"\nwrote {OUT.name} ({OUT.stat().st_size / 1024:.1f} KB)")
    print("\nNOTE: emotion labels come from ACTED corpora; the SER literature is "
          "explicit that acted prosody exaggerates cues. Treat these as an "
          "optimistic bound for spontaneous dealership calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
