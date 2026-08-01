"""Resolution rules between the Gemini path and the deterministic path.

Authority is assigned per field from measured capability, not from a general
preference for one path:

| field                     | authority | why                                          |
|---------------------------|-----------|----------------------------------------------|
| emotional_tone/intensity  | Path A    | Path B has no emotion model, by design       |
| background_noise_*        | Path B    | Path A reported hearing hiss then labelled it absent |
| speaker_overlap_present   | Path B    | Path A returned false on every clip, every prompt |
| audio_quality             | Path B    | Path A pins `slightly_impaired`, mistaking the synthetic agent voice for a defect |
| long_silence_present      | Path B, Path A may veto | the brief scopes this to silence "that may indicate a call-flow or audio problem" - a semantic judgement Path B cannot make |

Disagreement between the paths is the input to the confidence score, which is
the one honest use available for that field given the labels ship a constant
0.82 placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .path_b_acoustic import AcousticResult
from .schema import CallAnalysis, coerce_to_schema


@dataclass
class FusionOutcome:
    analysis: CallAnalysis
    disagreements: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    repairs: list[str] = field(default_factory=list)


def fuse(
    gemini: CallAnalysis | None,
    acoustic: AcousticResult | None,
    *,
    trust_llm_silence_veto: bool = True,
) -> FusionOutcome:
    """Combine both paths into one result."""
    if gemini is None and acoustic is None:
        raise ValueError("fuse() needs at least one path")

    disagreements: list[str] = []
    sources: dict[str, str] = {}
    out: dict[str, Any] = {}

    # --- emotion: Path A only -------------------------------------------
    if gemini is not None:
        out["emotional_tone"] = gemini.emotional_tone
        out["emotional_intensity"] = gemini.emotional_intensity
        sources["emotional_tone"] = sources["emotional_intensity"] = "gemini"
    else:
        # No emotion signal at all. Emitting a confident guess would be worse
        # than admitting it, so this is flagged through confidence below.
        out["emotional_tone"] = "neutral"
        out["emotional_intensity"] = "low"
        sources["emotional_tone"] = sources["emotional_intensity"] = "default(no-llm)"

    # --- noise, overlap, quality: Path B authoritative -------------------
    if acoustic is not None:
        out["background_noise_present"] = acoustic.background_noise_present
        out["background_noise_type"] = acoustic.background_noise_type
        out["background_noise_severity"] = acoustic.background_noise_severity
        out["speaker_overlap_present"] = acoustic.speaker_overlap_present
        out["audio_quality"] = acoustic.audio_quality
        for f in ("background_noise_present", "background_noise_severity",
                  "speaker_overlap_present", "audio_quality"):
            sources[f] = "acoustic"
        sources["background_noise_type"] = "acoustic"

        if gemini is not None:
            # Path A keeps a say on noise *type*: it names sources ("television",
            # "office chatter") that spectral shape can only approximate. Its
            # label is adopted when both paths agree something is audible.
            if acoustic.background_noise_present and gemini.background_noise_present \
                    and gemini.background_noise_type:
                out["background_noise_type"] = gemini.background_noise_type
                sources["background_noise_type"] = "gemini(type)+acoustic(presence)"

            for f in ("background_noise_present", "speaker_overlap_present", "audio_quality"):
                if getattr(gemini, f) != out[f]:
                    disagreements.append(
                        f"{f}: gemini={getattr(gemini, f)!r} acoustic={out[f]!r}"
                    )

            # Path A may escalate quality when it can name a concrete defect that
            # bandwidth alone cannot see (clipping, dropouts, echo).
            rank = {"clear": 0, "slightly_impaired": 1, "severely_impaired": 2}
            if rank[str(gemini.audio_quality)] > rank[str(out["audio_quality"])] + 1:
                out["audio_quality"] = "slightly_impaired"
                sources["audio_quality"] = "acoustic+gemini(escalated)"
    elif gemini is not None:
        for f in ("background_noise_present", "background_noise_type",
                  "background_noise_severity", "speaker_overlap_present", "audio_quality"):
            out[f] = getattr(gemini, f)
            sources[f] = "gemini(no-acoustic)"

    # --- long silence ----------------------------------------------------
    if acoustic is not None:
        out["long_silence_present"] = acoustic.long_silence_present
        sources["long_silence_present"] = "acoustic"
        if (gemini is not None and trust_llm_silence_veto
                and acoustic.long_silence_present and not gemini.long_silence_present):
            # The field is scoped to silence indicating a *problem*. A long pause
            # while an advisor looks something up is normal, and only the
            # content-aware path can tell those apart.
            out["long_silence_present"] = False
            sources["long_silence_present"] = "acoustic+gemini(veto)"
            disagreements.append("long_silence_present: acoustic=True vetoed by gemini")
    elif gemini is not None:
        out["long_silence_present"] = gemini.long_silence_present
        sources["long_silence_present"] = "gemini(no-acoustic)"

    # --- confidence ------------------------------------------------------
    base = float(gemini.confidence) if gemini is not None else 0.4
    conf = base - 0.08 * len(disagreements)
    if gemini is None or acoustic is None:
        conf -= 0.15                      # only one path ran
    out["confidence"] = round(max(0.05, min(0.99, conf)), 2)
    sources["confidence"] = "fused(agreement-weighted)"

    analysis, repairs = coerce_to_schema(out)
    return FusionOutcome(
        analysis=analysis, disagreements=disagreements, sources=sources, repairs=repairs
    )
