"""Baselines. Required by the brief: "Define a simple baseline before attempting
a complex system."

**B0 - majority class.** Predicts the fit-split mode for every field. This is the
floor any real system must clear, and it is not a hypothetical: an earlier
version of this project scored 1/3 on tone against the three provided calls
purely by always answering `neutral`. A constant predictor and a working
classifier are indistinguishable at n=3, which is the entire argument for the
proxy set.

**B1 - transcript only.** Transcribe, then classify tone from the text alone with
no acoustic features. This measures how much the *voice* actually contributes.
The SER literature predicts audio LLMs lean on lexical content
(arXiv:2510.10444), and our ground truth contains two cases that inverta text
reading - a flatly delivered obscenity labelled `neutral`, and a customer
refused throughout labelled `satisfied`. If the audio paths beat B1 clearly,
the architecture is justified by measurement rather than assertion.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import FIELD_ORDER, CallAnalysis, coerce_to_schema


@dataclass
class MajorityBaseline:
    """B0: constant prediction of the training-split mode."""

    modes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(cls, truths: list[dict[str, Any]]) -> MajorityBaseline:
        modes: dict[str, Any] = {}
        for f in FIELD_ORDER:
            if f == "confidence":
                modes[f] = 0.82
                continue
            vals = [t[f] for t in truths if f in t]
            modes[f] = Counter(vals).most_common(1)[0][0] if vals else None
        return cls(modes=modes)

    def predict(self) -> CallAnalysis:
        analysis, _ = coerce_to_schema(dict(self.modes))
        return analysis

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.modes, indent=2))

    @classmethod
    def load(cls, path: Path) -> MajorityBaseline:
        return cls(modes=json.loads(path.read_text()))


TRANSCRIPT_SYSTEM = """You classify the emotional tone of a customer on a car dealership
service call, using ONLY a written transcript. You cannot hear the audio.

Report the tone of the CUSTOMER, not the agent.

emotional_tone: neutral | satisfied | frustrated | upset | distressed
emotional_intensity: low | medium | high

Return JSON with exactly those two keys."""

_B1_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "emotional_tone": {"type": "STRING",
                           "enum": ["neutral", "satisfied", "frustrated", "upset", "distressed"]},
        "emotional_intensity": {"type": "STRING", "enum": ["low", "medium", "high"]},
    },
    "required": ["emotional_tone", "emotional_intensity"],
    "propertyOrdering": ["emotional_tone", "emotional_intensity"],
}


def classify_transcript(text: str, *, client: Any = None, settings: Any = None
                        ) -> tuple[str, str, dict[str, int]]:
    """B1: tone from text alone. Returns (tone, intensity, token counts)."""
    from google import genai
    from google.genai import types

    from .config import get_settings

    s = settings or get_settings()
    client = client or genai.Client(api_key=s.gemini_api_key)
    resp = client.models.generate_content(
        model=s.gemini_model,
        contents=[types.Part.from_text(text=f"Transcript:\n\n{text}")],
        config=types.GenerateContentConfig(
            system_instruction=TRANSCRIPT_SYSTEM,
            response_mime_type="application/json",
            response_schema=_B1_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.0,
        ),
    )
    d = json.loads(resp.text or "{}")
    u = resp.usage_metadata
    return (
        str(d.get("emotional_tone", "neutral")),
        str(d.get("emotional_intensity", "low")),
        {"prompt": int(getattr(u, "prompt_token_count", 0) or 0),
         "output": int(getattr(u, "candidates_token_count", 0) or 0)},
    )
