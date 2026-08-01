"""Diagnostic: why do all clips return identical labels?

Runs the same three clips under several prompt variants and reports how much
the labels actually vary. If a variant produces identical output for acoustically
different calls, that variant is broken regardless of how sensible it reads.

Also dumps the model's own evidence strings, which settle whether the audio is
reaching the model at all: distinct evidence + identical labels means the model
hears the difference and the prompt is overriding it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoace.config import REPO_ROOT, get_settings  # noqa: E402
from autoace.ingest import probe  # noqa: E402
from autoace.path_a_gemini import MIME, build_system_instruction  # noqa: E402
from autoace.schema import (  # noqa: E402
    LABEL_DEFINITIONS,
    SCORING_WARNING,
    gemini_response_schema,
)

LABELS = ("emotional_tone", "emotional_intensity", "background_noise_present",
          "background_noise_severity", "audio_quality", "speaker_overlap_present",
          "long_silence_present")

DEFS = "\n".join(f"- {k}: {v}" for k, v in LABEL_DEFINITIONS.items())

VARIANTS: dict[str, str | None] = {
    # What ships today.
    "A_current": build_system_instruction(),

    # No steering at all - schema descriptions only.
    "B_none": None,

    # The brief's definitions verbatim, nothing else.
    "C_defs_only": f"""You analyse recorded phone calls for a car dealership service department.
Report the emotional tone OF THE CUSTOMER (not the agent).

FIELD DEFINITIONS
{DEFS}

{SCORING_WARNING}""",

    # Definitions + a neutral-worded delivery cue that does not name any label.
    "D_defs_plus_prosody": f"""You analyse recorded phone calls for a car dealership service department.

This recording is a single mixed mono channel: the dealership agent (often an automated
voice assistant, usually speaking first with a scripted greeting) and the customer are
summed together. Report the emotional tone OF THE CUSTOMER ONLY.

Weigh vocal delivery - pitch movement, pace, volume dynamics, tension, sighing,
interruption - at least as heavily as word choice. Word choice alone can mislead in
both directions.

The agent's voice may be synthetic text-to-speech. Synthetic timbre is NOT an
audio_quality defect: audio_quality describes transmission and capture integrity
(distortion, clipping, echo, static, dropouts, muffling), not whether a speaker
sounds artificial.

FIELD DEFINITIONS
{DEFS}

{SCORING_WARNING}""",
}


def run(client, model, path, system, thinking, temperature):
    data = path.read_bytes()
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=gemini_response_schema(include_evidence=True),
        thinking_config=types.ThinkingConfig(thinking_budget=thinking),
        temperature=temperature,
    )
    if system:
        cfg.system_instruction = system
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=data, mime_type=MIME.get(path.suffix.lower(), "audio/ogg")),
            types.Part.from_text(text="Analyse this call and return the required fields."),
        ],
        config=cfg,
    )
    return json.loads(resp.text or "{}"), resp.usage_metadata


def main() -> int:
    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)
    model = s.gemini_model
    files = sorted((REPO_ROOT / "data" / "provided_calls").glob("*.ogg"))

    print("=" * 90)
    print("STEP 1 - is the audio actually distinct and actually reaching the model?")
    print("=" * 90)
    for p in files:
        pr = probe(p)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        print(f"  {p.name:<16} {pr.duration_s:7.1f}s  sha256={digest}  bytes={p.stat().st_size}")

    for name, system in VARIANTS.items():
        print()
        print("=" * 90)
        print(f"VARIANT {name}  (system instruction: "
              f"{'none' if system is None else str(len(system)) + ' chars'})")
        print("=" * 90)
        rows = []
        for p in files:
            try:
                raw, usage = run(client, model, p, system, 0, 0.0)
            except Exception as exc:  # noqa: BLE001
                print(f"  {p.name}: FAILED {type(exc).__name__}: {str(exc)[:140]}")
                continue
            rows.append({k: raw.get(k) for k in LABELS})
            print(f"  {p.name:<16} in={usage.prompt_token_count:<6} "
                  + "  ".join(f"{k.split('_')[-1][:4]}={str(raw.get(k))[:12]}" for k in LABELS))
            ev = raw.get("emotion_evidence", "")
            if ev:
                print(f"                   evidence: {ev[:150]}")
        if len(rows) > 1:
            varying = [k for k in LABELS if len({json.dumps(r[k]) for r in rows}) > 1]
            identical = [k for k in LABELS if k not in varying]
            print(f"\n  --> FIELDS THAT VARY across the 3 clips: {varying or 'NONE (collapsed)'}")
            print(f"  --> identical across all clips        : {identical}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
