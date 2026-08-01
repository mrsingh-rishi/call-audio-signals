"""Single source of truth for the nine-field output contract.

Every prompt, every validator, the Gemini ``response_schema`` and the memo all
derive from this module. Nothing else defines an enum value or a label
definition, so they cannot drift apart.

The label definitions in ``LABEL_DEFINITIONS`` are quoted verbatim from the
trial brief. They are injected into prompts unmodified.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class EmotionalTone(StrEnum):
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    UPSET = "upset"
    DISTRESSED = "distressed"


class EmotionalIntensity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NoiseSeverity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AudioQuality(StrEnum):
    CLEAR = "clear"
    SLIGHTLY_IMPAIRED = "slightly_impaired"
    SEVERELY_IMPAIRED = "severely_impaired"


# --- Ordering -------------------------------------------------------------
# Three of the enums are ordinal, so predicting `medium` when the truth is
# `high` is not the same error as predicting `none`. Validation reports both
# exact match and off-by-one using these ranks.

INTENSITY_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}
SEVERITY_RANK: dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3}
QUALITY_RANK: dict[str, int] = {"clear": 0, "slightly_impaired": 1, "severely_impaired": 2}

# `emotional_tone` is NOT ordinal, so "max" needs an explicit definition for
# long-call aggregation. `satisfied` is positive and sits off the negative
# escalation axis, sharing rank 0 with `neutral`; ties at rank 0 are broken by
# duration-weighted mode. See plan section 7.2.
ESCALATION_RANK: dict[str, int] = {
    "neutral": 0,
    "satisfied": 0,
    "frustrated": 1,
    "upset": 2,
    "distressed": 3,
}

ORDINAL_RANKS: dict[str, dict[str, int]] = {
    "emotional_intensity": INTENSITY_RANK,
    "background_noise_severity": SEVERITY_RANK,
    "audio_quality": QUALITY_RANK,
}


# --- Verbatim label definitions from the brief ----------------------------

LABEL_DEFINITIONS: dict[str, str] = {
    "emotional_tone": (
        "The primary emotional tone expressed by the customer. "
        "neutral: no clear positive or negative emotion. "
        "satisfied: pleased, relieved, appreciative, or clearly positive. "
        "frustrated: annoyed, impatient, or dissatisfied without strong anger or distress. "
        "upset: clearly angry, agitated, or strongly dissatisfied. "
        "distressed: highly emotional, overwhelmed, panicked, crying, or otherwise "
        "emotionally escalated."
    ),
    "emotional_intensity": (
        "The strength of the detected emotional tone. low is subtle or mild. "
        "medium is clear and sustained. high is strong, escalated, or likely to "
        "require attention."
    ),
    "background_noise_present": (
        "Whether meaningful non-speech sound is audible in the background. "
        "Barely perceptible artifacts should not automatically count as background noise."
    ),
    "background_noise_type": (
        "A concise description of the dominant background noise, such as office chatter, "
        "music, road noise, television, keyboard typing, wind, or mechanical noise. "
        "Empty string when no noise is present."
    ),
    "background_noise_severity": (
        "How much the noise affects the call. none means no meaningful noise. "
        "low is audible but does not interfere. medium occasionally interferes with "
        "understanding. high materially impairs the conversation or analysis."
    ),
    "audio_quality": (
        "The overall technical quality of the audio, independent of emotional tone. "
        "Consider distortion, clipping, echo, static, low volume, muffled speech, "
        "robotic audio, and packet loss."
    ),
    "speaker_overlap_present": (
        "Whether two or more speakers talk at the same time enough to affect "
        "understanding or analysis."
    ),
    "long_silence_present": (
        "Whether the clip contains an unusually long period of silence or dead air that "
        "may indicate a call-flow or audio problem."
    ),
    "confidence": (
        "The model's confidence in the overall result. Values near 1.0 indicate high "
        "confidence; values near 0.0 indicate substantial uncertainty."
    ),
}

# The brief's explicit anti-shortcut warning. Injected verbatim into every
# prompt that emits labels.
SCORING_WARNING = (
    "Emotional tone, background-noise detection, and technical audio quality are scored "
    "separately. Do not infer frustration or distress solely from loudness, and do not "
    "infer background noise solely from poor audio quality."
)

# Observed from the one real labelled example (call_003): "sharp static" was
# labelled as background_noise_type while audio_quality stayed "clear". The
# labeller treats static/hiss as NOISE, not as a quality defect. This is the
# inverse of the naive mapping, so it is stated explicitly to the model.
ONTOLOGY_NOTES = (
    "Static, hiss and line noise count as background noise (background_noise_type), not "
    "as an audio_quality defect. audio_quality describes the technical integrity of the "
    "captured signal - distortion, clipping, echo, muffling, robotic artifacts, packet "
    "loss - not what is audible in the room."
)


# --- Output contract ------------------------------------------------------

# Declaration order below IS the emitted JSON key order. It matches the brief's
# example output exactly. Do not reorder.
FIELD_ORDER: tuple[str, ...] = (
    "emotional_tone",
    "emotional_intensity",
    "background_noise_present",
    "background_noise_type",
    "background_noise_severity",
    "audio_quality",
    "speaker_overlap_present",
    "long_silence_present",
    "confidence",
)


class CallAnalysis(BaseModel):
    """The exact output contract. Strict: invariants raise rather than repair.

    Use :func:`coerce_to_schema` for untrusted input (model output, CSV
    manifests) - it repairs and reports what it changed instead of raising.
    """

    model_config = {"extra": "forbid", "use_enum_values": True}

    emotional_tone: EmotionalTone
    emotional_intensity: EmotionalIntensity
    background_noise_present: bool
    background_noise_type: str = Field(default="")
    background_noise_severity: NoiseSeverity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("background_noise_type", mode="before")
    @classmethod
    def _clean_type(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @model_validator(mode="after")
    def _check_noise_coherence(self) -> CallAnalysis:
        """background_noise_present must agree with type and severity.

        Without this the model can emit present=false alongside a non-empty
        type, which would be scored as two contradictory answers.
        """
        if self.background_noise_present:
            if not self.background_noise_type:
                raise ValueError("background_noise_present=true requires a non-empty type")
            if self.background_noise_severity == NoiseSeverity.NONE:
                raise ValueError("background_noise_present=true requires severity != none")
        else:
            if self.background_noise_type:
                raise ValueError("background_noise_present=false requires an empty type")
            if self.background_noise_severity != NoiseSeverity.NONE:
                raise ValueError("background_noise_present=false requires severity=none")
        return self

    def to_output_dict(self) -> dict[str, Any]:
        """Emit in the brief's exact key order."""
        d = self.model_dump()
        return {k: d[k] for k in FIELD_ORDER}


def coerce_to_schema(raw: dict[str, Any]) -> tuple[CallAnalysis, list[str]]:
    """Repair untrusted input into a valid :class:`CallAnalysis`.

    Returns the model plus a list of human-readable repairs applied. Never
    raises for recoverable problems - a malformed prediction must degrade to a
    valid-but-flagged row rather than failing the batch.
    """
    repairs: list[str] = []
    d = dict(raw)

    def _enum_or(field: str, enum_cls: type[StrEnum], default: str) -> None:
        val = d.get(field)
        valid = {e.value for e in enum_cls}
        if isinstance(val, str) and val.strip().lower() in valid:
            d[field] = val.strip().lower()
            return
        repairs.append(f"{field}={val!r} invalid, defaulted to {default!r}")
        d[field] = default

    _enum_or("emotional_tone", EmotionalTone, "neutral")
    _enum_or("emotional_intensity", EmotionalIntensity, "low")
    _enum_or("background_noise_severity", NoiseSeverity, "none")
    _enum_or("audio_quality", AudioQuality, "clear")

    for field, default in (
        ("background_noise_present", False),
        ("speaker_overlap_present", False),
        ("long_silence_present", False),
    ):
        val = d.get(field)
        if isinstance(val, bool):
            continue
        if isinstance(val, str) and val.strip().lower() in {"true", "false"}:
            d[field] = val.strip().lower() == "true"
            repairs.append(f"{field} coerced from string")
        else:
            repairs.append(f"{field}={val!r} invalid, defaulted to {default}")
            d[field] = default

    ntype = d.get("background_noise_type")
    d["background_noise_type"] = "" if ntype is None else str(ntype).strip()

    conf = d.get("confidence")
    try:
        c = float(conf)  # type: ignore[arg-type]
        if not (0.0 <= c <= 1.0):
            raise ValueError
        d["confidence"] = c
    except (TypeError, ValueError):
        repairs.append(f"confidence={conf!r} invalid, defaulted to 0.5")
        d["confidence"] = 0.5

    # Resolve the noise coherence invariant. `present` is treated as the
    # authoritative field because it is the one the brief scores as a boolean;
    # type and severity are brought into line with it.
    if d["background_noise_present"]:
        if not d["background_noise_type"]:
            d["background_noise_type"] = "unspecified"
            repairs.append("noise present but type empty, set to 'unspecified'")
        if d["background_noise_severity"] == "none":
            d["background_noise_severity"] = "low"
            repairs.append("noise present but severity=none, raised to 'low'")
    else:
        if d["background_noise_type"]:
            repairs.append(
                f"noise absent but type={d['background_noise_type']!r}, cleared"
            )
            d["background_noise_type"] = ""
        if d["background_noise_severity"] != "none":
            repairs.append(
                f"noise absent but severity={d['background_noise_severity']!r}, set to 'none'"
            )
            d["background_noise_severity"] = "none"

    for extra in set(d) - set(FIELD_ORDER):
        d.pop(extra)
        repairs.append(f"dropped unexpected field {extra!r}")

    return CallAnalysis(**d), repairs


def gemini_response_schema(include_evidence: bool = True) -> dict[str, Any]:
    """Build the Gemini ``response_schema``.

    When ``include_evidence`` is true the three evidence strings are declared
    BEFORE any label. Generation is autoregressive, so ordering the schema this
    way forces the model to commit to separate observations for noise, quality
    and emotion before it picks labels - which is the anti-conflation mechanism
    described in plan section 5. Set false for the compact per-window schema on
    long calls, where evidence strings would erode the cost headroom (plan 7.1).
    """
    props: dict[str, Any] = {}
    order: list[str] = []

    if include_evidence:
        for name, desc in (
            (
                "noise_evidence",
                "Non-speech sounds you can hear in the background. Describe ONLY sounds, "
                "not signal degradation and not emotion. Empty string if none.",
            ),
            (
                "quality_evidence",
                "Technical defects in the captured signal: distortion, clipping, echo, "
                "muffling, robotic artifacts, packet loss, low volume. Describe ONLY "
                "signal integrity, not background sounds and not emotion.",
            ),
            (
                "customer_identification",
                "Which speaker is the customer (not the agent), and how you can tell.",
            ),
            (
                "emotion_evidence",
                "What the CUSTOMER says and how they say it that indicates their emotional "
                "state. Quote or paraphrase. Do not cite loudness alone.",
            ),
        ):
            props[name] = {"type": "STRING", "description": desc}
            order.append(name)

    props.update(
        {
            "emotional_tone": {
                "type": "STRING",
                "enum": [e.value for e in EmotionalTone],
                "description": LABEL_DEFINITIONS["emotional_tone"],
            },
            "emotional_intensity": {
                "type": "STRING",
                "enum": [e.value for e in EmotionalIntensity],
                "description": LABEL_DEFINITIONS["emotional_intensity"],
            },
            "background_noise_present": {
                "type": "BOOLEAN",
                "description": LABEL_DEFINITIONS["background_noise_present"],
            },
            "background_noise_type": {
                "type": "STRING",
                "description": LABEL_DEFINITIONS["background_noise_type"],
            },
            "background_noise_severity": {
                "type": "STRING",
                "enum": [e.value for e in NoiseSeverity],
                "description": LABEL_DEFINITIONS["background_noise_severity"],
            },
            "audio_quality": {
                "type": "STRING",
                "enum": [e.value for e in AudioQuality],
                "description": LABEL_DEFINITIONS["audio_quality"],
            },
            "speaker_overlap_present": {
                "type": "BOOLEAN",
                "description": LABEL_DEFINITIONS["speaker_overlap_present"],
            },
            "long_silence_present": {
                "type": "BOOLEAN",
                "description": LABEL_DEFINITIONS["long_silence_present"],
            },
            "confidence": {
                "type": "NUMBER",
                "description": LABEL_DEFINITIONS["confidence"],
            },
        }
    )
    order.extend(FIELD_ORDER)

    return {
        "type": "OBJECT",
        "properties": props,
        "required": order,
        "propertyOrdering": order,
    }
