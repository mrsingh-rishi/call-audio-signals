"""Role-labelled transcript handling.

Two jobs: repairing model-emitted timestamps, and extracting customer-only
regions so emotional tone can be attributed to the right speaker.

The timestamp repair is not defensive programming for its own sake. Measured on
the provided calls, Gemini emits ``2.50`` to mean "2 minutes 50 seconds" even
when the schema declares a NUMBER of seconds and the instruction says total
elapsed seconds. Interpreted literally, a 171.9 s call appears to end at 2.5 s.
That silently breaks windowing and customer-turn extraction, so every transcript
is validated against the known clip duration and repaired when the
minutes.seconds pattern is detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Turn:
    start_s: float
    end_s: float
    role: str
    text: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _mmss_to_seconds(v: float) -> float:
    """Interpret ``M.SS`` as total seconds: 2.50 -> 170.0."""
    minutes = int(v)
    seconds = round((v - minutes) * 100)
    return minutes * 60 + seconds


def repair_timestamps(
    turns: list[dict[str, Any]], duration_s: float
) -> tuple[list[dict[str, Any]], str | None]:
    """Detect and repair minutes.seconds timestamps.

    Returns the (possibly repaired) turns plus a note describing what was done,
    or ``None`` if the timestamps were already sane. The note is surfaced in
    results so a silent repair never goes unnoticed.
    """
    if not turns or duration_s <= 0:
        return turns, None

    def _f(t: dict[str, Any], k: str) -> float:
        try:
            return float(t.get(k, 0.0))
        except (TypeError, ValueError):
            return 0.0

    raw_max = max(_f(t, "end_s") for t in turns)
    if raw_max <= 0:
        return turns, None

    # Already plausible: covers a reasonable share of the clip and stays inside it.
    if 0.4 * duration_s <= raw_max <= 1.10 * duration_s:
        return turns, None

    mmss_max = max(_mmss_to_seconds(_f(t, "end_s")) for t in turns)
    if 0.4 * duration_s <= mmss_max <= 1.10 * duration_s:
        repaired = []
        for t in turns:
            u = dict(t)
            u["start_s"] = _mmss_to_seconds(_f(t, "start_s"))
            u["end_s"] = _mmss_to_seconds(_f(t, "end_s"))
            repaired.append(u)
        return repaired, (
            f"timestamps were minutes.seconds (max {raw_max:.2f} vs duration "
            f"{duration_s:.1f}s); converted to seconds (max {mmss_max:.0f}s)"
        )

    # Neither reading fits. Clamp rather than propagate nonsense downstream.
    clamped = []
    for t in turns:
        u = dict(t)
        u["start_s"] = min(max(0.0, _f(t, "start_s")), duration_s)
        u["end_s"] = min(max(0.0, _f(t, "end_s")), duration_s)
        clamped.append(u)
    return clamped, (
        f"timestamps implausible (max end {raw_max:.2f} vs duration {duration_s:.1f}s "
        f"and M.SS reading gives {mmss_max:.0f}s); clamped to clip bounds"
    )


def parse_turns(result: dict[str, Any], duration_s: float) -> tuple[list[Turn], str | None]:
    raw, note = repair_timestamps(result.get("turns") or [], duration_s)
    turns: list[Turn] = []
    for t in raw:
        try:
            start, end = float(t.get("start_s", 0.0)), float(t.get("end_s", 0.0))
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        turns.append(
            Turn(
                start_s=start,
                end_s=end,
                role=str(t.get("role", "unknown")).strip().lower(),
                text=str(t.get("text", "")),
            )
        )
    return turns, note


def role_seconds(turns: list[Turn]) -> dict[str, float]:
    out: dict[str, float] = {}
    for t in turns:
        out[t.role] = out.get(t.role, 0.0) + t.duration_s
    return out


def customer_regions(
    turns: list[Turn], pad_s: float = 0.25, merge_gap_s: float = 0.75
) -> list[tuple[float, float]]:
    """Merged, padded time ranges where the customer is speaking.

    Used to score emotional tone on customer audio only. Padding preserves turn
    onsets and offsets, which carry prosody; merging avoids fragmenting a single
    utterance into unusably short slices.
    """
    spans = sorted(
        (max(0.0, t.start_s - pad_s), t.end_s + pad_s)
        for t in turns
        if t.role == "customer" and t.duration_s > 0
    )
    if not spans:
        return []
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def transcript_text(turns: list[Turn], roles: tuple[str, ...] = ("agent", "customer")) -> str:
    """Flatten to text for the B1 transcript-only baseline."""
    return "\n".join(f"{t.role}: {t.text}" for t in turns if t.role in roles and t.text.strip())
