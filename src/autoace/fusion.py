"""Resolution rules between the Gemini path and the deterministic path.

Authority is assigned per field from measured capability, not from a general
preference for one path:

| field                     | authority | why                                          |
|---------------------------|-----------|----------------------------------------------|
| emotional_tone/intensity  | Path C    | prosody beats a lexical reader; Path A used when C is unsure |
| background_noise_*        | Path B    | Path A reported hearing hiss then labelled it absent |
| speaker_overlap_present   | **Path D** | pyannote segmentation-3.0: 0.792 balanced acc vs 0.544 for Path B's dual-pitch cue. Path A returned false on every clip, every prompt |
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


_UNINFORMATIVE_TYPES = {
    "", "none", "nothing", "nothing audible", "n/a", "unknown", "unspecified",
    "background noise", "noise", "silence",
}
"""Names that carry no information and would score zero against any real label.

If the only name available is one of these, the spectral character from Path B is
kept instead - "static" or "television" is a guess, but it is a guess with content.
"""


@dataclass
class FusionOutcome:
    analysis: CallAnalysis
    disagreements: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    repairs: list[str] = field(default_factory=list)


def fuse(
    gemini: CallAnalysis | None,
    acoustic: AcousticResult | None,
    prosody: Any = None,
    overlap: Any = None,
    *,
    gemini_noise_source: str = "",
    trust_llm_silence_veto: bool = True,
) -> FusionOutcome:
    """Combine the available paths into one result.

    ``gemini_noise_source`` is Path A's one-to-three-word naming of the dominant
    background sound. It lives in the evidence block rather than the label
    schema, deliberately: it is descriptive and is emitted independently of Path
    A's own presence boolean, so it stays usable even when that boolean is wrong.
    """
    if gemini is None and acoustic is None:
        raise ValueError("fuse() needs at least one path")

    disagreements: list[str] = []
    sources: dict[str, str] = {}
    out: dict[str, Any] = {}

    # --- emotion --------------------------------------------------------
    # Path C (prosody) is authoritative when it has fitted coefficients. On the
    # proxy eval split it reaches macro-F1 0.421 against 0.082 for the majority
    # baseline, whereas Gemini is documented to read lexical content rather than
    # delivery (arXiv:2510.10444) and returned `upset` for a flatly-delivered
    # obscenity whose ground truth is `neutral`.
    if prosody is not None and getattr(prosody, "tone_probs", None):
        out["emotional_tone"] = prosody.emotional_tone
        out["emotional_intensity"] = prosody.emotional_intensity
        sources["emotional_tone"] = sources["emotional_intensity"] = "prosody"
        if gemini is not None and gemini.emotional_tone != prosody.emotional_tone:
            disagreements.append(
                f"emotional_tone: gemini={gemini.emotional_tone!r} "
                f"prosody={prosody.emotional_tone!r}"
            )
    elif gemini is not None:
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
            # "office chatter") that spectral shape can only approximate.
            #
            # Presence stays with Path B, naming comes from Path A - and crucially
            # the naming is taken even when Path A's OWN boolean is false. Path A
            # is unreliable on the boolean (it once wrote "a faint hiss throughout"
            # and then labelled the field absent) but is good at saying what the
            # sound is. Requiring both to agree meant the name was only available
            # when it was least needed, and the output fell back to the useless
            # literal string "background noise".
            if acoustic.background_noise_present:
                named = (gemini_noise_source or "").strip() or (
                    gemini.background_noise_type or ""
                ).strip()
                if named and named.lower() not in _UNINFORMATIVE_TYPES:
                    out["background_noise_type"] = named
                    sources["background_noise_type"] = "gemini(type)+acoustic(presence)"

            # `speaker_overlap_present` is deliberately absent here. Path A
            # returns false on it universally, so a mismatch is a property of the
            # model rather than evidence of uncertainty - counting it would
            # penalise confidence on every clip that genuinely has overlap.
            for f in ("background_noise_present", "audio_quality"):
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

    # --- speaker overlap: Path D authoritative when the model is present ---
    # Path D reaches 0.792 balanced accuracy on the proxy eval split against
    # 0.544 for Path B's dual-pitch cue, so it takes the field outright rather
    # than voting with it. When the ONNX model is missing, Path B's value (set
    # above) stands.
    if overlap is not None and getattr(overlap, "available", False):
        out["speaker_overlap_present"] = overlap.speaker_overlap_present
        sources["speaker_overlap_present"] = "segmentation"
        if acoustic is not None and \
                acoustic.speaker_overlap_present != overlap.speaker_overlap_present:
            disagreements.append(
                f"speaker_overlap_present: acoustic={acoustic.speaker_overlap_present!r} "
                f"segmentation={overlap.speaker_overlap_present!r}"
            )
        # Path A returned false on every clip under every prompt variant tried,
        # so a disagreement with it carries no information and is not recorded.

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
    # Three inputs, in decreasing order of how much they are trusted:
    #
    # 1. Path C's own posterior on `emotional_tone`. This is the only genuinely
    #    probabilistic signal in the system, and tone is the hardest field, so a
    #    barely-committed 0.38 top class should not be reported at 0.82.
    # 2. Disagreement between paths, which is evidence of a hard clip.
    # 3. Whether every path actually ran.
    #
    # It is deliberately NOT calibrated against the provided labels: those carry
    # a constant 0.82 on all three calls, and 0.82 is also the value in the
    # brief's own example output, so it is a copied placeholder. Fitting a
    # calibrator to a constant would produce a constant. Calibration against the
    # proxy eval split is reported in the validation report.
    base = float(gemini.confidence) if gemini is not None else 0.4
    probs = getattr(prosody, "tone_probs", None) if prosody is not None else None
    if probs:
        top = max(float(v) for v in probs.values())
        # Blend rather than replace: the posterior of a 5-class model rarely
        # exceeds 0.6 even when right, so using it raw would understate every
        # result. Anchored so top=0.20 (chance) pulls down hard and top=0.60
        # leaves the base roughly intact.
        base = 0.5 * base + 0.5 * min(0.95, 0.30 + 1.15 * (top - 0.20))
        sources["confidence"] = "fused(prosody-posterior + agreement)"
    else:
        sources["confidence"] = "fused(agreement-weighted)"
    conf = base - 0.08 * len(disagreements)
    if gemini is None or acoustic is None:
        conf -= 0.15                      # only one path ran
    out["confidence"] = round(max(0.05, min(0.99, conf)), 2)

    analysis, repairs = coerce_to_schema(out)
    return FusionOutcome(
        analysis=analysis, disagreements=disagreements, sources=sources, repairs=repairs
    )
