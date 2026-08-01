"""Schema contract tests.

The output contract is what the hidden set is scored against, so these tests
guard it tightly: exact key order, exact enum vocabularies, and the
noise-coherence invariant that stops the system emitting two contradictory
answers about the same thing.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from autoace.schema import (
    ESCALATION_RANK,
    FIELD_ORDER,
    AudioQuality,
    CallAnalysis,
    EmotionalIntensity,
    EmotionalTone,
    NoiseSeverity,
    coerce_to_schema,
    gemini_response_schema,
)

VALID = {
    "emotional_tone": "frustrated",
    "emotional_intensity": "medium",
    "background_noise_present": True,
    "background_noise_type": "office chatter",
    "background_noise_severity": "low",
    "audio_quality": "clear",
    "speaker_overlap_present": False,
    "long_silence_present": False,
    "confidence": 0.82,
}


def test_enum_vocabularies_match_the_brief():
    assert [e.value for e in EmotionalTone] == [
        "neutral", "satisfied", "frustrated", "upset", "distressed",
    ]
    assert [e.value for e in EmotionalIntensity] == ["low", "medium", "high"]
    assert [e.value for e in NoiseSeverity] == ["none", "low", "medium", "high"]
    assert [e.value for e in AudioQuality] == [
        "clear", "slightly_impaired", "severely_impaired",
    ]


def test_output_key_order_matches_brief_exactly():
    got = list(CallAnalysis(**VALID).to_output_dict().keys())
    assert got == list(FIELD_ORDER)
    assert got == [
        "emotional_tone", "emotional_intensity", "background_noise_present",
        "background_noise_type", "background_noise_severity", "audio_quality",
        "speaker_overlap_present", "long_silence_present", "confidence",
    ]


def test_output_is_json_serialisable_with_plain_types():
    d = CallAnalysis(**VALID).to_output_dict()
    round_tripped = json.loads(json.dumps(d))
    assert round_tripped["emotional_tone"] == "frustrated"
    assert isinstance(round_tripped["background_noise_present"], bool)
    assert isinstance(round_tripped["confidence"], float)


def test_escalation_rank_treats_satisfied_as_off_axis():
    """`emotional_tone` is not ordinal; satisfied is positive, so it shares
    rank 0 with neutral rather than sitting between tones on a severity line."""
    assert ESCALATION_RANK["satisfied"] == ESCALATION_RANK["neutral"] == 0
    assert ESCALATION_RANK["frustrated"] < ESCALATION_RANK["upset"]
    assert ESCALATION_RANK["upset"] < ESCALATION_RANK["distressed"]


# --- noise coherence invariant -------------------------------------------

def test_noise_present_requires_type_and_severity():
    with pytest.raises(ValidationError):
        CallAnalysis(**{**VALID, "background_noise_type": ""})
    with pytest.raises(ValidationError):
        CallAnalysis(**{**VALID, "background_noise_severity": "none"})


def test_noise_absent_requires_empty_type_and_none_severity():
    absent = {
        **VALID,
        "background_noise_present": False,
        "background_noise_type": "",
        "background_noise_severity": "none",
    }
    assert CallAnalysis(**absent).background_noise_type == ""
    with pytest.raises(ValidationError):
        CallAnalysis(**{**absent, "background_noise_type": "music"})
    with pytest.raises(ValidationError):
        CallAnalysis(**{**absent, "background_noise_severity": "low"})


def test_confidence_bounds_enforced():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValidationError):
            CallAnalysis(**{**VALID, "confidence": bad})


# --- coercion of untrusted output ----------------------------------------

def test_coerce_repairs_contradictory_noise_fields():
    model, repairs = coerce_to_schema(
        {**VALID, "background_noise_present": False, "background_noise_type": "music"}
    )
    assert model.background_noise_present is False
    assert model.background_noise_type == ""
    assert model.background_noise_severity == "none"
    assert any("cleared" in r for r in repairs)


def test_coerce_fills_missing_type_when_noise_present():
    model, repairs = coerce_to_schema({**VALID, "background_noise_type": ""})
    assert model.background_noise_present is True
    assert model.background_noise_type == "unspecified"
    assert any("unspecified" in r for r in repairs)


def test_coerce_handles_garbage_without_raising():
    """A malformed prediction must degrade to a valid flagged row, never crash."""
    model, repairs = coerce_to_schema(
        {
            "emotional_tone": "furious",          # not in the enum
            "emotional_intensity": None,
            "background_noise_present": "true",   # stringly-typed bool
            "background_noise_type": "  static ",
            "background_noise_severity": "MEDIUM",
            "audio_quality": "clear",
            "speaker_overlap_present": 1,
            "long_silence_present": False,
            "confidence": "not a number",
            "extra_field": "should be dropped",
        }
    )
    assert model.emotional_tone == "neutral"
    assert model.background_noise_present is True
    assert model.background_noise_type == "static"
    assert model.background_noise_severity == "medium"
    assert model.confidence == 0.5
    assert any("extra_field" in r for r in repairs)
    assert list(model.to_output_dict().keys()) == list(FIELD_ORDER)


def test_coerce_accepts_already_valid_payload_without_repairs():
    model, repairs = coerce_to_schema(dict(VALID))
    assert repairs == []
    assert model.to_output_dict() == VALID


# --- Gemini response schema ----------------------------------------------

def test_response_schema_orders_evidence_before_labels():
    """Generation is autoregressive, so evidence fields must be declared first
    for the anti-conflation mechanism to work."""
    s = gemini_response_schema(include_evidence=True)
    order = s["propertyOrdering"]
    assert order.index("noise_evidence") < order.index("background_noise_present")
    assert order.index("quality_evidence") < order.index("audio_quality")
    assert order.index("emotion_evidence") < order.index("emotional_tone")
    assert order.index("customer_identification") < order.index("emotion_evidence")


def test_compact_schema_drops_evidence_for_windowed_scoring():
    """Long-call windows use a compact schema; evidence strings would erode the
    cost headroom (plan 7.1)."""
    s = gemini_response_schema(include_evidence=False)
    assert s["propertyOrdering"] == list(FIELD_ORDER)
    assert "noise_evidence" not in s["properties"]


def test_response_schema_enums_match_the_python_enums():
    props = gemini_response_schema()["properties"]
    assert props["emotional_tone"]["enum"] == [e.value for e in EmotionalTone]
    assert props["background_noise_severity"]["enum"] == [e.value for e in NoiseSeverity]
    assert props["audio_quality"]["enum"] == [e.value for e in AudioQuality]
