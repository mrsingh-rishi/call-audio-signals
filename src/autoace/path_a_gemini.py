"""Path A: Gemini audio analysis.

Two variants behind one interface:

* ``A-single`` - one call. The response schema declares ``noise_evidence``,
  ``quality_evidence``, ``customer_identification`` and ``emotion_evidence``
  BEFORE any label. Generation is autoregressive, so schema ordering forces the
  model to commit to separate observations per field group before it picks
  labels. That is the anti-conflation mechanism.
* ``A-multi`` - three calls with restricted evidence per head.

Which one ships is decided by the ablation's adversarial-cell column, not by
assertion. Cost is measured from live ``usageMetadata`` rather than estimated.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .config import PRICING, Settings, get_settings
from .schema import (
    LABEL_DEFINITIONS,
    ONTOLOGY_NOTES,
    SCORING_WARNING,
    CallAnalysis,
    coerce_to_schema,
    gemini_response_schema,
)

MIME = {
    ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg",
    ".wav": "audio/wav", ".mp3": "audio/mp3", ".m4a": "audio/mp4",
    ".flac": "audio/flac", ".aac": "audio/aac", ".webm": "audio/webm",
}

# Weighting delivery over word choice is defensible: the target labels describe
# how a customer sounds, not what they ask for.
#
# An earlier version of this block also gave worked examples ("a customer who
# swears in a flat voice is neutral", "told no repeatedly but stays pleasant is
# not upset"). Those examples described the three provided calls almost exactly,
# and measurement showed they collapsed ALL SEVEN label fields to a single
# constant answer across acoustically different clips - the model stopped
# listening and recited the prompt. They are removed deliberately: no worked
# examples, and no label name is ever paired with a scenario.
_DELIVERY_RULE = """Weigh vocal delivery - pitch movement, pace, volume dynamics, tension,
tremor, sighing, interruption - at least as heavily as word choice when judging the
customer's emotional tone. Word choice alone can mislead in either direction: strong
language may be delivered flatly, and a serious complaint may be delivered calmly.
Describe what you actually hear in the voice, then choose the label that fits it."""

_CUSTOMER_RULE = """This recording is a single mixed mono channel from a car dealership service
department: the dealership agent and the customer are summed together. The agent
is often an automated voice assistant and usually speaks first with a scripted
greeting. The CUSTOMER is the person calling about their own vehicle.

Report the emotional tone OF THE CUSTOMER ONLY. Ignore the agent's tone entirely."""


def build_system_instruction() -> str:
    defs = "\n".join(f"- {k}: {v}" for k, v in LABEL_DEFINITIONS.items())
    return f"""You analyse recorded phone calls for a car dealership service department.

{_CUSTOMER_RULE}

{_DELIVERY_RULE}

FIELD DEFINITIONS
{defs}

{SCORING_WARNING}

{ONTOLOGY_NOTES}

The agent's voice may be synthetic text-to-speech. Synthetic or robotic timbre in
the AGENT's voice is NOT an audio_quality defect - audio_quality describes
transmission and capture integrity, not whether a speaker sounds artificial.

LISTEN ACTIVELY FOR NON-SPEECH SOUND. During pauses and behind the speech, note any
television, music, office chatter, traffic, typing, wind, machinery, hiss, static or
line noise. Hiss and static DO count as background noise.

LISTEN FOR OVERLAPPING SPEECH: any point where both parties are audible at once -
interruptions, talk-over, one party starting before the other has finished.

Before choosing any label, write your evidence fields. Describe background sounds in
noise_evidence WITHOUT reference to signal degradation. Describe signal defects in
quality_evidence WITHOUT reference to background sounds. These are scored separately
and must not be conflated.

Your labels must agree with your own evidence. If noise_evidence describes an audible
sound, background_noise_present must be true and the severity must not be none. If it
describes nothing, background_noise_present must be false."""


@dataclass
class CallUsage:
    """Measured token usage and cost for one or more API calls."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    cached_tokens: int = 0
    n_calls: int = 0
    latency_s: float = 0.0
    model: str = ""
    notes: list[str] = field(default_factory=list)

    def add(self, usage: Any, elapsed: float, model: str) -> None:
        self.prompt_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
        self.output_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
        self.thought_tokens += int(getattr(usage, "thoughts_token_count", 0) or 0)
        self.cached_tokens += int(getattr(usage, "cached_content_token_count", 0) or 0)
        self.latency_s += elapsed
        self.n_calls += 1
        self.model = model

    def cost_usd(self, model: str | None = None) -> float:
        """Dollar cost from measured tokens.

        Thinking tokens bill at OUTPUT rates, which is how an unbounded thinking
        budget silently breaches a per-minute ceiling. They are charged here
        explicitly rather than folded into output.
        """
        p = PRICING.get(model or self.model) or PRICING["gemini-2.5-flash-lite"]
        billed_in = max(0, self.prompt_tokens - self.cached_tokens)
        cost = billed_in * p.audio_in_per_m / 1e6
        if self.cached_tokens and p.cached_in_per_m is not None:
            cost += self.cached_tokens * p.cached_in_per_m / 1e6
        cost += (self.output_tokens + self.thought_tokens) * p.text_out_per_m / 1e6
        return cost

    def cost_uncached_usd(self, model: str | None = None) -> float:
        """Cost as if nothing had been cached - the honest worst case.

        Implicit caching is opportunistic: re-running the same clip caches the
        audio itself, which makes a repeated benchmark look far cheaper than a
        production run over distinct files. In production only the stable system
        prefix realistically caches, so this figure is the one to quote for
        ceiling compliance.
        """
        p = PRICING.get(model or self.model) or PRICING["gemini-2.5-flash-lite"]
        return (self.prompt_tokens * p.audio_in_per_m
                + (self.output_tokens + self.thought_tokens) * p.text_out_per_m) / 1e6

    def cost_per_audio_min(self, duration_s: float, model: str | None = None) -> float:
        minutes = max(duration_s / 60.0, 1e-9)
        return self.cost_usd(model) / minutes

    def cost_uncached_per_audio_min(self, duration_s: float, model: str | None = None) -> float:
        minutes = max(duration_s / 60.0, 1e-9)
        return self.cost_uncached_usd(model) / minutes


def _mime_for(path: Path) -> str:
    return MIME.get(path.suffix.lower(), "audio/ogg")


def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: list[Any],
    config: types.GenerateContentConfig,
    max_attempts: int = 4,
) -> tuple[Any, float]:
    """Call the API with exponential backoff on transient failures.

    A 429 or 5xx must degrade to a clean per-file error, never take down a
    batch, so retries are bounded and the final exception is allowed to
    propagate to the per-file handler.
    """
    last: Exception | None = None
    for attempt in range(max_attempts):
        t0 = time.perf_counter()
        try:
            resp = client.models.generate_content(
                model=model, contents=contents, config=config
            )
            return resp, time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc)
            transient = any(
                s in msg for s in ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED",
                                   "UNAVAILABLE", "DEADLINE_EXCEEDED")
            )
            if not transient or attempt == max_attempts - 1:
                raise
            sleep = (2**attempt) + random.uniform(0, 0.5)
            time.sleep(sleep)
    raise last  # type: ignore[misc]


def analyse_single(
    path: str | Path,
    *,
    client: genai.Client | None = None,
    settings: Settings | None = None,
    model: str | None = None,
    include_evidence: bool = True,
) -> tuple[CallAnalysis, dict[str, Any]]:
    """A-single: one call producing all nine fields plus evidence."""
    s = settings or get_settings()
    if not s.gemini_enabled:
        raise RuntimeError("Gemini is disabled (no API key, or LOCAL_ONLY=true)")
    client = client or genai.Client(api_key=s.gemini_api_key)
    model = model or s.gemini_model
    p = Path(path)

    config = types.GenerateContentConfig(
        system_instruction=build_system_instruction(),
        response_mime_type="application/json",
        response_schema=gemini_response_schema(include_evidence=include_evidence),
        thinking_config=types.ThinkingConfig(thinking_budget=s.gemini_thinking_budget),
        temperature=0.0,
    )
    contents = [
        types.Part.from_bytes(data=p.read_bytes(), mime_type=_mime_for(p)),
        types.Part.from_text(
            text="Analyse this call and return the required fields."
        ),
    ]

    resp, elapsed = _generate_with_retry(client, model, contents, config)
    usage = CallUsage()
    usage.add(resp.usage_metadata, elapsed, model)

    try:
        raw = json.loads(resp.text or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"model returned unparseable JSON: {exc}") from exc

    # Evidence fields are part of the reasoning contract, not the output
    # contract. Strip them before coercion so they are not reported as schema
    # repairs - otherwise every successful call looks like it had four errors.
    evidence_keys = (
        "noise_evidence", "quality_evidence",
        "customer_identification", "emotion_evidence",
    )
    evidence = {k: raw[k] for k in evidence_keys if k in raw}
    labels_only = {k: v for k, v in raw.items() if k not in evidence_keys}
    analysis, repairs = coerce_to_schema(labels_only)

    return analysis, {
        "variant": "A-single",
        "model": model,
        "evidence": evidence,
        "repairs": repairs,
        "usage": usage,
    }
