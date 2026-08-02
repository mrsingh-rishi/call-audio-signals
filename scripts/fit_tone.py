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


def _manifest_key(rows: list[dict]) -> str:
    """Identify this build of the proxy set - see `fit_detectors._manifest_key`.

    Matching on row count alone let a regenerated set reuse the previous build's
    features, so coefficients were scored against audio they had never seen.
    Clip names carry a spec fingerprint, so hashing them catches it.
    """
    import hashlib

    return hashlib.sha256("|".join(r["name"] for r in rows).encode()).hexdigest()[:16]


def build(force: bool = False) -> tuple[np.ndarray, list[dict]]:
    rows = list(csv.DictReader((PROXY / "labels.csv").open(newline="")))
    key = _manifest_key(rows)
    if CACHE.exists() and not force:
        d = np.load(CACHE, allow_pickle=True)
        if len(d["X"]) == len(rows) and str(d.get("key", "")) == key:
            return d["X"], rows
        print("  proxy set changed since the cache was written; recomputing prosody")
    X = []
    for i, r in enumerate(rows):
        f = prosodic_features(PROXY / "audio" / r["name"])
        X.append([f[k] for k in PROSODY_FEATURES])
        if (i + 1) % 50 == 0:
            print(f"  prosody {i + 1}/{len(rows)}", flush=True)
    X = np.asarray(X, dtype=np.float64)
    np.savez_compressed(CACHE, X=X, key=key)
    return X, rows


# --- Prior correction ------------------------------------------------------
# The classifier is fitted on RAVDESS + CREMA-D, which are acted and roughly
# balanced across emotions. Spontaneous call-centre speech is not: it is
# overwhelmingly neutral, and the SER literature is explicit that acted prosody
# exaggerates arousal cues. That mismatch is a *prior shift*, and the standard
# correction is logit adjustment - add `tau * log(pi_target)` to each class
# logit, which is a per-class offset on the intercept and costs nothing at
# inference (Menon et al., "Long-tail learning via logit adjustment").
#
# `class_weight="balanced"` already makes the effective training prior uniform,
# so the offset below shifts FROM uniform TO the deployment prior rather than
# undoing the training distribution twice.
#
# These are order-of-magnitude estimates of a real dealership call mix, NOT
# measured from AutoAce data. tau scales how far to trust them, and tau is
# chosen on held-out speakers - if the data says 0, the correction is off.
DEPLOY_TONE_PRIOR: dict[str, float] = {
    "neutral": 0.55, "satisfied": 0.20, "frustrated": 0.15,
    "upset": 0.07, "distressed": 0.03,
}
DEPLOY_INTENSITY_PRIOR: dict[str, float] = {"low": 0.45, "medium": 0.40, "high": 0.15}


def _prior_offset(classes: tuple[str, ...]) -> np.ndarray:
    """log(pi_target) per class, centred so tau only tilts, never rescales."""
    table = DEPLOY_TONE_PRIOR if len(classes) == len(TONES) else DEPLOY_INTENSITY_PRIOR
    p = np.array([table.get(c, 1.0 / len(classes)) for c in classes], dtype=float)
    p = p / p.sum()
    lg = np.log(p)
    return lg - lg.mean()


def _choose_tau(Xs, y, fit_m, speakers, classes, LogisticRegression, f1_score):
    """Pick tau on held-out SPEAKERS inside the fit split.

    Never on the eval split - that would make the reported number a fitted one -
    and never on the three provided calls. The inner split is grouped by speaker
    for the same reason the outer one is: a threshold tuned to a voice it is
    later scored on tells you nothing.
    """
    fit_idx = np.flatnonzero(fit_m)
    uniq = sorted(set(speakers[fit_idx]))
    if len(uniq) < 6:
        return 0.0, "too few fit speakers for an inner split; correction disabled"
    hold = set(uniq[::3])                       # ~1/3 of speakers held out
    inner_tr = np.array([i for i in fit_idx if speakers[i] not in hold])
    inner_va = np.array([i for i in fit_idx if speakers[i] in hold])
    if inner_tr.size < 30 or inner_va.size < 20 or len(set(y[inner_tr])) < 2:
        return 0.0, "inner split too small; correction disabled"

    clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.3)
    clf.fit(Xs[inner_tr], y[inner_tr])
    off = _prior_offset(classes)
    best_tau, best_f1 = 0.0, -1.0
    curve = []
    for tau in np.arange(0.0, 1.01, 0.1):
        pred = np.argmax(Xs[inner_va] @ clf.coef_.T + clf.intercept_ + off * tau, axis=1)
        f1 = f1_score(y[inner_va], pred, average="macro", zero_division=0)
        curve.append((round(float(tau), 1), round(float(f1), 3)))
        if f1 > best_f1 + 1e-6:
            best_tau, best_f1 = float(tau), float(f1)
    note = (f"inner macroF1 {curve[0][1]:.3f}@0 -> {best_f1:.3f}@{best_tau:.1f}; "
            f"curve {curve}")
    return best_tau, note


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

    speakers = np.array([r["speaker_id"] for r in rows])

    for field, classes in (("emotional_tone", TONES),
                           ("emotional_intensity", INTENSITIES)):
        y = np.array([classes.index(str(t[field])) for t in truth])
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.3)
        clf.fit(Xs[fit_m], y[fit_m])

        tau, tau_note = _choose_tau(
            Xs, y, fit_m, speakers, classes, LogisticRegression, f1_score
        )
        adj = _prior_offset(classes) * tau
        model_intercept = clf.intercept_ + adj
        pred = np.argmax(Xs[ev_m] @ clf.coef_.T + model_intercept, axis=1)

        # Report the eval-split effect of the correction ALONE, so its
        # contribution is not conflated with any other change. tau was chosen on
        # held-out speakers inside the fit split; these two numbers are both
        # read-outs, neither is a selection.
        pred_tau0 = np.argmax(Xs[ev_m] @ clf.coef_.T + clf.intercept_, axis=1)
        f1_tau0 = f1_score(y[ev_m], pred_tau0, average="macro", zero_division=0)
        f1_tau = f1_score(y[ev_m], pred, average="macro", zero_division=0)
        lines.append(f"\n  [{field}] logit adjustment tau={tau:.2f} - {tau_note}")
        lines.append(f"      eval macroF1 {f1_tau0:.3f} at tau=0  ->  {f1_tau:.3f} at "
                     f"tau={tau:.2f}   (delta {f1_tau - f1_tau0:+.3f})")
        macro = f1_score(y[ev_m], pred, average="macro", zero_division=0)
        acc = float((pred == y[ev_m]).mean())
        # Majority-class floor on the SAME eval split - this is baseline B0.
        maj = np.bincount(y[fit_m]).argmax()
        b0_acc = float((y[ev_m] == maj).mean())
        b0_f1 = f1_score(y[ev_m], np.full_like(y[ev_m], maj),
                         average="macro", zero_division=0)

        model["fields"][field] = {
            "classes": list(classes),
            "coef": clf.coef_.tolist(),
            "intercept": model_intercept.tolist(),
            # Recorded so the adjustment is auditable and reversible: the raw
            # intercept plus the offset that was folded into it.
            "intercept_raw": clf.intercept_.tolist(),
            "logit_adjustment_tau": round(float(tau), 3),
            "logit_adjustment_offset": adj.tolist(),
        }
        lines.append(f"\n  {field}: accuracy={acc:.3f}  macroF1={macro:.3f}"
                     f"   [B0 majority: acc={b0_acc:.3f} macroF1={b0_f1:.3f}]")
        per = f1_score(y[ev_m], pred, average=None, zero_division=0,
                       labels=list(range(len(classes))))
        for c, f1c in zip(classes, per):
            support = int((y[ev_m] == classes.index(c)).sum())
            lines.append(f"      {c:<12} F1={f1c:.3f}  (n={support})")
        cm = confusion_matrix(y[ev_m], pred, labels=list(range(len(classes))))
        # As an aligned grid rather than one long line - this is copied straight
        # into the validation report, and `list(np.int64 row)` renders as
        # "np.int64(31)" per cell, which is unreadable.
        lines.append("      confusion (rows=truth, cols=pred):")
        lines.append("          " + "".join(f"{c[:5]:>7}" for c in classes))
        for c, row in zip(classes, cm):
            lines.append(f"      {c:<10}" + "".join(f"{int(v):>7}" for v in row))

    # --- Is the tone posterior worth putting into `confidence`? -------------
    # fusion.py folds Path C's top posterior into the reported confidence, so it
    # has to be shown that the posterior actually tracks correctness. If a
    # barely-committed prediction were as accurate as a confident one, the signal
    # would be decoration and should not be shipped.
    y_t = np.array([TONES.index(str(t["emotional_tone"])) for t in truth])
    clf_t = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.3)
    clf_t.fit(Xs[fit_m], y_t[fit_m])
    z = Xs[ev_m] @ clf_t.coef_.T + clf_t.intercept_
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    top, hit = p.max(axis=1), (p.argmax(axis=1) == y_t[ev_m])
    order = np.argsort(top)
    lines.append("\n  [confidence] does the tone posterior track correctness?")
    for name, sel in (("lowest third ", order[: len(order) // 3]),
                      ("middle third ", order[len(order) // 3: 2 * len(order) // 3]),
                      ("highest third", order[2 * len(order) // 3:])):
        lines.append(f"      {name}  mean posterior {top[sel].mean():.3f}  "
                     f"accuracy {hit[sel].mean():.3f}  (n={len(sel)})")
    lines.append(f"      correlation(posterior, correct) = "
                 f"{float(np.corrcoef(top, hit.astype(float))[0, 1]):+.3f}")

    OUT.write_text(json.dumps(model, indent=2))
    # "\n".join, not "".join: only some entries carry their own leading newline,
    # so the per-class rows used to run together into one unreadable line.
    # Entries that do start with "\n" become a blank separator line, which is
    # what we want between field blocks.
    print("EVAL SPLIT (unseen speakers)\n" + "\n".join(lines))
    print(f"\nwrote {OUT.name} ({OUT.stat().st_size / 1024:.1f} KB)")
    print("\nNOTE: emotion labels come from ACTED corpora; the SER literature is "
          "explicit that acted prosody exaggerates cues. Treat these as an "
          "optimistic bound for spontaneous dealership calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
