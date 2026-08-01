"""Scoring helpers shared by the validation scripts and the live dashboard.

Ordinal fields are scored two ways deliberately. Predicting `medium` when the
truth is `high` is not the same error as predicting `none`, so exact-match alone
would hide how close a wrong answer was. Both numbers are always reported.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .schema import FIELD_ORDER, ORDINAL_RANKS

BOOLEAN_FIELDS = (
    "background_noise_present",
    "speaker_overlap_present",
    "long_silence_present",
)
CATEGORICAL_FIELDS = ("emotional_tone",)
ORDINAL_FIELDS = tuple(ORDINAL_RANKS)
# Open text: not exact-matchable, reported separately and never folded into an
# accuracy average.
FREE_TEXT_FIELDS = ("background_noise_type",)
SCORED_FIELDS = CATEGORICAL_FIELDS + ORDINAL_FIELDS + BOOLEAN_FIELDS


@dataclass
class FieldScore:
    field_name: str
    n: int = 0
    exact: int = 0
    off_by_one: int = 0
    is_ordinal: bool = False
    confusion: Counter = field(default_factory=Counter)

    @property
    def accuracy(self) -> float:
        return self.exact / self.n if self.n else 0.0

    @property
    def within_one(self) -> float:
        if not self.n or not self.is_ordinal:
            return self.accuracy
        return (self.exact + self.off_by_one) / self.n


def macro_f1(pairs: Iterable[tuple[str, str]]) -> tuple[float, dict[str, dict[str, float]]]:
    """Macro-averaged F1 plus per-class precision/recall/F1.

    Macro rather than micro because the brief names it and because the tone
    classes are imbalanced - a micro average would let a dominant class hide
    total failure on a rare one.
    """
    pairs = list(pairs)
    classes = sorted({c for pair in pairs for c in pair})
    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for c in classes:
        tp = sum(1 for t, p in pairs if t == c and p == c)
        fp = sum(1 for t, p in pairs if t != c and p == c)
        fn = sum(1 for t, p in pairs if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {
            "precision": prec, "recall": rec, "f1": f1, "support": float(tp + fn)
        }
        # Classes absent from truth contribute no recall signal; excluding them
        # stops a never-predicted class dragging the macro average.
        if (tp + fn) > 0:
            f1s.append(f1)
    return (sum(f1s) / len(f1s) if f1s else 0.0), per_class


def score_batch(
    truths: list[dict[str, Any]], preds: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare predictions against ground truth field by field."""
    if len(truths) != len(preds):
        raise ValueError("truths and preds must be the same length")

    scores: dict[str, FieldScore] = {
        f: FieldScore(f, is_ordinal=f in ORDINAL_RANKS) for f in SCORED_FIELDS
    }
    for t, p in zip(truths, preds):
        for f in SCORED_FIELDS:
            if f not in t or f not in p:
                continue
            tv, pv = t[f], p[f]
            sc = scores[f]
            sc.n += 1
            sc.confusion[(str(tv), str(pv))] += 1
            if tv == pv:
                sc.exact += 1
            elif f in ORDINAL_RANKS:
                rank = ORDINAL_RANKS[f]
                if abs(rank.get(str(tv), 0) - rank.get(str(pv), 0)) == 1:
                    sc.off_by_one += 1

    tone_pairs = [
        (str(t["emotional_tone"]), str(p["emotional_tone"]))
        for t, p in zip(truths, preds)
        if "emotional_tone" in t and "emotional_tone" in p
    ]
    tone_macro_f1, tone_per_class = macro_f1(tone_pairs)

    text_matches = []
    for t, p in zip(truths, preds):
        tv = str(t.get("background_noise_type", "")).strip().lower()
        pv = str(p.get("background_noise_type", "")).strip().lower()
        if not tv and not pv:
            text_matches.append(("both empty", True))
        else:
            # Token overlap, since exact string match on open text is not a
            # plausible scoring mechanism.
            tt, pt = set(tv.split()), set(pv.split())
            text_matches.append((f"{tv!r} vs {pv!r}", bool(tt & pt)))

    return {
        "n": len(truths),
        "fields": {
            f: {
                "n": s.n,
                "exact_accuracy": s.accuracy,
                "within_one": s.within_one if s.is_ordinal else None,
                "confusion": {f"{a}->{b}": c for (a, b), c in sorted(s.confusion.items())},
            }
            for f, s in scores.items()
        },
        "emotional_tone_macro_f1": tone_macro_f1,
        "emotional_tone_per_class": tone_per_class,
        "background_noise_type": {
            "token_overlap_rate": (
                sum(1 for _, ok in text_matches if ok) / len(text_matches)
                if text_matches else 0.0
            ),
            "detail": [d for d, _ in text_matches],
        },
        "mean_scored_field_accuracy": (
            sum(s.accuracy for s in scores.values() if s.n) / max(
                1, sum(1 for s in scores.values() if s.n)
            )
        ),
    }


def format_report(result: dict[str, Any], title: str = "") -> str:
    lines: list[str] = []
    if title:
        lines += [title, "=" * len(title)]
    n = result["n"]
    lines.append(f"n = {n}")
    if n < 30:
        lines.append(
            f"WARNING: n={n} supports no statistical claim. Reported as raw "
            f"agreement only, not as a metric."
        )
    lines.append("")
    lines.append(f"{'field':<28} {'exact':>8} {'within-1':>10}   confusion (truth->pred)")
    lines.append("-" * 100)
    for f in FIELD_ORDER:
        if f not in result["fields"]:
            continue
        s = result["fields"][f]
        if not s["n"]:
            continue
        w1 = f"{s['within_one']:.3f}" if s["within_one"] is not None else "-"
        errs = {k: v for k, v in s["confusion"].items() if k.split("->")[0] != k.split("->")[1]}
        lines.append(
            f"{f:<28} {s['exact_accuracy']:>8.3f} {w1:>10}   "
            f"{', '.join(f'{k} x{v}' for k, v in errs.items()) or 'all correct'}"
        )
    lines.append("")
    lines.append(f"emotional_tone macro-F1: {result['emotional_tone_macro_f1']:.3f}")
    for cls, m in result["emotional_tone_per_class"].items():
        if m["support"]:
            lines.append(
                f"    {cls:<12} P={m['precision']:.2f} R={m['recall']:.2f} "
                f"F1={m['f1']:.2f} (n={int(m['support'])})"
            )
    bnt = result["background_noise_type"]
    lines.append(f"\nbackground_noise_type token overlap: {bnt['token_overlap_rate']:.3f}")
    for d in bnt["detail"]:
        lines.append(f"    {d}")
    return "\n".join(lines)
