"""Fit the objective-field detectors on the proxy set.

Replaces hand-set thresholds with fitted ones. This exists because the
hand-tuned versions - which scored 3/3 on the three provided calls - collapsed
to 3/16 on the proxy set. Three clips cannot distinguish a rule that works from
a rule that memorises.

Discipline enforced here:

* Coefficients are fitted on the **fit** split only. The **eval** split holds
  entirely different speakers and different noise files.
* Features are standardised using fit-split statistics only, so no evaluation
  data leaks through the scaler.
* scikit-learn is used for fitting and never ships. Coefficients are exported to
  JSON and applied at inference with a numpy dot product, so the container adds
  no dependency.
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
from autoace.path_b_acoustic import FEATURE_NAMES, extract_features  # noqa: E402

PROXY = REPO_ROOT / "data" / "proxy"
CACHE = PROXY / "features.npz"
OUT_MODEL = REPO_ROOT / "src" / "autoace" / "detector_coefficients.json"

BINARY_FIELDS = ("background_noise_present", "speaker_overlap_present",
                 "long_silence_present")
ORDINAL_FIELDS = {
    "background_noise_severity": ["none", "low", "medium", "high"],
    "audio_quality": ["clear", "slightly_impaired", "severely_impaired"],
}


def build_features(force: bool = False) -> tuple[np.ndarray, list[dict]]:
    rows = list(csv.DictReader((PROXY / "labels.csv").open(newline="")))
    if CACHE.exists() and not force:
        d = np.load(CACHE, allow_pickle=True)
        if len(d["X"]) == len(rows):
            return d["X"], rows
    X = []
    for i, r in enumerate(rows):
        X.append([extract_features(PROXY / "audio" / r["name"])[k] for k in FEATURE_NAMES])
        if (i + 1) % 50 == 0:
            print(f"  features {i + 1}/{len(rows)}")
    X = np.asarray(X, dtype=np.float64)
    np.savez_compressed(CACHE, X=X)
    return X, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="recompute cached features")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score

    X, rows = build_features(args.force)
    truth = [json.loads(r["result_json"]) for r in rows]
    split = np.array([r["split"] for r in rows])
    fit_m, ev_m = split == "fit", split == "eval"
    print(f"\nfit {fit_m.sum()} clips / eval {ev_m.sum()} clips "
          f"({len(set(r['speaker_id'] for r in rows))} speakers, disjoint by split)")

    # Standardise on fit-split statistics only.
    mu, sd = X[fit_m].mean(axis=0), X[fit_m].std(axis=0) + 1e-9
    Xs = (X - mu) / sd

    model: dict[str, object] = {
        "feature_names": list(FEATURE_NAMES),
        "mean": mu.tolist(), "std": sd.tolist(), "fields": {},
    }
    report: list[str] = []

    for field in BINARY_FIELDS:
        y = np.array([bool(t[field]) for t in truth])
        if len(set(y[fit_m])) < 2:
            print(f"  {field}: only one class in fit split, skipped")
            continue
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)
        clf.fit(Xs[fit_m], y[fit_m])
        pred = clf.predict(Xs[ev_m])
        bal = balanced_accuracy_score(y[ev_m], pred)
        f1 = f1_score(y[ev_m], pred, zero_division=0)
        base = max((y[ev_m] == v).mean() for v in (True, False))
        model["fields"][field] = {  # type: ignore[index]
            "type": "binary",
            "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
        }
        report.append(f"  {field:<28} balanced_acc={bal:.3f}  F1={f1:.3f}  "
                      f"(majority baseline {base:.3f})")

    for field, classes in ORDINAL_FIELDS.items():
        y = np.array([classes.index(str(t[field])) for t in truth])
        if len(set(y[fit_m])) < 2:
            continue
        clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                                 C=0.5, multi_class="multinomial")
        clf.fit(Xs[fit_m], y[fit_m])
        pred = clf.predict(Xs[ev_m])
        exact = float((pred == y[ev_m]).mean())
        within1 = float((np.abs(pred - y[ev_m]) <= 1).mean())
        macro = f1_score(y[ev_m], pred, average="macro", zero_division=0)
        model["fields"][field] = {  # type: ignore[index]
            "type": "ordinal", "classes": classes,
            "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
        }
        report.append(f"  {field:<28} exact={exact:.3f}  within1={within1:.3f}  "
                      f"macroF1={macro:.3f}")

    OUT_MODEL.write_text(json.dumps(model, indent=2))
    print("\nEVAL SPLIT (unseen speakers, unseen noise files)")
    print("\n".join(report))
    print(f"\nwrote {OUT_MODEL.relative_to(REPO_ROOT)} "
          f"({OUT_MODEL.stat().st_size / 1024:.1f} KB - ships in the container)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
