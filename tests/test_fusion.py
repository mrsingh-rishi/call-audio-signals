"""Fusion authority tests.

Fusion decides which path owns which field, so these tests pin the decisions
that were made from measurement rather than preference - and pin the failure
modes that motivated them. A regression here silently changes what the hidden
set is scored on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from autoace.fusion import fuse
from autoace.path_b_acoustic import AcousticResult
from autoace.schema import CallAnalysis


def _gemini(**over) -> CallAnalysis:
    base = {
        "emotional_tone": "neutral",
        "emotional_intensity": "low",
        "background_noise_present": False,
        "background_noise_type": "",
        "background_noise_severity": "none",
        "audio_quality": "clear",
        "speaker_overlap_present": False,
        "long_silence_present": False,
        "confidence": 0.8,
    }
    base.update(over)
    return CallAnalysis(**base)


@dataclass
class _Overlap:
    speaker_overlap_present: bool = False
    available: bool = True
    overlap_seconds: float = 0.0
    overlap_fraction: float = 0.0
    longest_overlap_s: float = 0.0
    notes: list = field(default_factory=list)


@dataclass
class _Prosody:
    emotional_tone: str = "upset"
    emotional_intensity: str = "high"
    tone_probs: dict = field(default_factory=lambda: {"upset": 0.9})


# --- speaker overlap ------------------------------------------------------

def test_segmentation_path_overrides_the_dual_pitch_cue():
    """Path D is authoritative: 0.792 balanced acc vs 0.544 on the proxy set."""
    ac = AcousticResult(speaker_overlap_present=False)
    out = fuse(_gemini(), ac, None, _Overlap(speaker_overlap_present=True))
    assert out.analysis.speaker_overlap_present is True
    assert out.sources["speaker_overlap_present"] == "segmentation"


def test_dual_pitch_cue_is_kept_when_the_model_is_unavailable():
    """A missing ONNX model must degrade, not fail."""
    ac = AcousticResult(speaker_overlap_present=True)
    out = fuse(_gemini(), ac, None, _Overlap(available=False))
    assert out.analysis.speaker_overlap_present is True
    assert out.sources["speaker_overlap_present"] == "acoustic"


def test_gemini_overlap_disagreement_does_not_dent_confidence():
    """Path A returns false universally, so that mismatch carries no signal.

    Counting it would penalise confidence on every clip that genuinely has
    overlap - punishing the system for being right.
    """
    ac = AcousticResult(speaker_overlap_present=True)
    with_overlap = fuse(_gemini(), ac, None, _Overlap(speaker_overlap_present=True))
    assert not any("speaker_overlap" in d for d in with_overlap.disagreements)


# --- background noise type ------------------------------------------------

def test_noise_named_by_gemini_even_when_its_own_boolean_is_false():
    """The naming is decoupled from Path A's unreliable presence boolean.

    Path A once wrote "a faint hiss throughout the recording" and then set
    background_noise_present=false. Presence comes from Path B; the name should
    still be used.
    """
    ac = AcousticResult(background_noise_present=True,
                        background_noise_type="static",
                        background_noise_severity="medium")
    out = fuse(_gemini(background_noise_present=False), ac, None, None,
               gemini_noise_source="television")
    assert out.analysis.background_noise_type == "television"


def test_uninformative_noise_names_are_rejected():
    """'background noise' scores zero against any real label; keep the spectral guess."""
    ac = AcousticResult(background_noise_present=True,
                        background_noise_type="static",
                        background_noise_severity="low")
    out = fuse(_gemini(), ac, None, None, gemini_noise_source="background noise")
    assert out.analysis.background_noise_type == "static"


def test_noise_type_cleared_when_path_b_says_absent():
    """Coherence: no name may survive a false presence boolean."""
    ac = AcousticResult(background_noise_present=False)
    out = fuse(_gemini(), ac, None, None, gemini_noise_source="television")
    assert out.analysis.background_noise_present is False
    assert out.analysis.background_noise_type == ""
    assert out.analysis.background_noise_severity == "none"


# --- emotion --------------------------------------------------------------

def test_prosody_owns_tone_when_it_has_coefficients():
    out = fuse(_gemini(emotional_tone="neutral"), AcousticResult(), _Prosody(), None)
    assert out.analysis.emotional_tone == "upset"
    assert out.sources["emotional_tone"] == "prosody"


def test_gemini_owns_tone_when_prosody_has_no_coefficients():
    class _Bare:
        emotional_tone = "neutral"
        emotional_intensity = "low"
        tone_probs: dict = {}

    out = fuse(_gemini(emotional_tone="frustrated"), AcousticResult(), _Bare(), None)
    assert out.analysis.emotional_tone == "frustrated"
    assert out.sources["emotional_tone"] == "gemini"


def test_fuse_requires_at_least_one_path():
    import pytest

    with pytest.raises(ValueError):
        fuse(None, None, None, None)
