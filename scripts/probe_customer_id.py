"""T0 gate: can Gemini identify the customer on a summed mono mix?

`emotional_tone` is defined as the tone *of the customer*, and it is the
headline of the 45% hidden-set bucket. The provided calls are duplicated mono
(agent and customer summed into one channel), so the customer cannot be
isolated by channel, and F0 clustering separates the two speakers by only
~2.5 sigma - not enough to attribute emotion reliably.

The plan's bet is that semantic identification beats acoustic diarization: the
model reads who is the customer from what is being said. This script tests that
bet before anything is built on top of it.

Decision rule (plan section 12):
  3/3 correct -> proceed; tone is attributed to customer turns only
  2/3         -> add a scripted-opener heuristic (agent speaks first), re-probe
  <=1/3       -> redesign the emotion path now, not on day 5

Transcripts are retained because the B1 transcript-only baseline reuses them.
They contain customer speech, so they are written to a gitignored directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoace.config import REPO_ROOT, get_settings  # noqa: E402
from autoace.ingest import probe  # noqa: E402
from autoace.transcript import customer_regions, parse_turns, role_seconds  # noqa: E402

TRANSCRIPT_DIR = REPO_ROOT / "data" / "transcripts"

MIME = {
    ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg",
    ".wav": "audio/wav", ".mp3": "audio/mp3", ".m4a": "audio/mp4",
    ".flac": "audio/flac", ".aac": "audio/aac", ".webm": "audio/webm",
}

TRANSCRIPT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "speakers_detected": {
            "type": "INTEGER",
            "description": "How many distinct human speakers are audible.",
        },
        "customer_identification": {
            "type": "STRING",
            "description": (
                "Which speaker is the customer and how you can tell. Cite what they "
                "say - who is calling about their own vehicle, who represents the "
                "dealership. Do not guess from voice pitch."
            ),
        },
        "turns": {
            "type": "ARRAY",
            "description": "Every speech turn in chronological order.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start_s": {
                        "type": "NUMBER",
                        "description": (
                            "Total elapsed seconds from the start of the recording. "
                            "NOT minutes.seconds. 95.5 means ninety-five and a half "
                            "seconds, i.e. one minute 35.5 seconds."
                        ),
                    },
                    "end_s": {
                        "type": "NUMBER",
                        "description": "Total elapsed seconds from the start of the recording.",
                    },
                    "role": {"type": "STRING", "enum": ["agent", "customer", "unknown"]},
                    "text": {"type": "STRING"},
                },
                "required": ["start_s", "end_s", "role", "text"],
                "propertyOrdering": ["start_s", "end_s", "role", "text"],
            },
        },
        "call_summary": {
            "type": "STRING",
            "description": "One sentence: what the customer is calling about.",
        },
    },
    "required": ["speakers_detected", "customer_identification", "turns", "call_summary"],
    "propertyOrdering": [
        "speakers_detected", "customer_identification", "turns", "call_summary",
    ],
}

SYSTEM = """You are transcribing a recorded phone call from a car dealership service department.

The recording is a single mixed mono channel: the dealership agent and the customer
are summed together, so you must separate them by what they say, not by which
channel they are on.

Identify the CUSTOMER: the person calling about their own vehicle, appointment, repair
or bill. The AGENT represents the dealership - they greet the caller, look things up,
and offer to help. The agent usually speaks first with a scripted greeting.

Transcribe every turn with accurate start and end times, expressed as TOTAL ELAPSED
SECONDS from the beginning of the recording. Do not use minutes.seconds notation:
a turn beginning at one minute thirty-five seconds is 95.0, not 1.35.

Attribute each turn to 'agent' or 'customer'. Use 'unknown' only if genuinely
undecidable.

Do not summarise or paraphrase the speech - transcribe what is actually said."""


def transcribe(
    client: genai.Client, model: str, path: Path, duration_s: float
) -> tuple[dict[str, Any], Any, float]:
    data = path.read_bytes()
    mime = MIME.get(path.suffix.lower(), "audio/ogg")
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=data, mime_type=mime),
            types.Part.from_text(
                text=(
                    f"Transcribe this call with speaker roles. The recording is "
                    f"exactly {duration_s:.1f} seconds long, so every end_s must fall "
                    f"between 0 and {duration_s:.1f}."
                )
            ),
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=TRANSCRIPT_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.0,
        ),
    )
    elapsed = time.perf_counter() - t0
    return json.loads(resp.text or "{}"), resp.usage_metadata, elapsed


def analyse(result: dict[str, Any], duration_s: float) -> dict[str, Any]:
    turns, note = parse_turns(result, duration_s)
    by_role = role_seconds(turns)
    total = sum(by_role.values()) or 1.0
    roles_seq = [t.role for t in turns]
    flips = sum(1 for a, b in zip(roles_seq, roles_seq[1:]) if a != b)
    regions = customer_regions(turns)
    return {
        "n_turns": len(turns),
        "speakers_detected": result.get("speakers_detected"),
        "first_speaker": roles_seq[0] if roles_seq else None,
        "role_seconds": {k: round(v, 2) for k, v in by_role.items()},
        "customer_speech_fraction": round(by_role.get("customer", 0.0) / total, 3),
        "role_changes": flips,
        "unknown_turns": sum(1 for r in roles_seq if r == "unknown"),
        "timestamp_repair": note,
        "customer_regions": [(round(a, 2), round(b, 2)) for a, b in regions],
        "customer_audio_seconds": round(sum(b - a for a, b in regions), 2),
        "turns_parsed": turns,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(REPO_ROOT / "data" / "provided_calls"))
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)

    s = get_settings()
    if not s.gemini_enabled:
        print("ERROR: Gemini is not enabled (missing key, or LOCAL_ONLY=true).",
              file=sys.stderr)
        return 2

    model = args.model or s.gemini_model
    client = genai.Client(api_key=s.gemini_api_key)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in Path(args.dir).iterdir()
        if p.is_file() and p.suffix.lower() in MIME
    )
    if not files:
        print(f"ERROR: no audio found in {args.dir}", file=sys.stderr)
        return 2

    print(f"T0 customer-identification probe | model={model} | {len(files)} clips\n")
    total_cost = 0.0
    pricing = s.pricing

    for path in files:
        pr = probe(path)
        try:
            result, usage, elapsed = transcribe(client, model, path, pr.duration_s)
        except Exception as exc:  # noqa: BLE001 - probe must report, not crash
            print(f"{path.name}: FAILED {type(exc).__name__}: {str(exc)[:200]}\n")
            continue

        stats = analyse(result, pr.duration_s)
        audio_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        cost = (audio_tok * pricing.audio_in_per_m
                + out_tok * pricing.text_out_per_m) / 1e6
        total_cost += cost

        # Persist the repaired form - downstream consumers must never see the
        # unrepaired timestamps.
        persisted = dict(result)
        persisted["turns"] = [
            {"start_s": t.start_s, "end_s": t.end_s, "role": t.role, "text": t.text}
            for t in stats["turns_parsed"]
        ]
        persisted["_duration_s"] = pr.duration_s
        persisted["_timestamp_repair"] = stats["timestamp_repair"]
        (TRANSCRIPT_DIR / f"{path.stem}.json").write_text(json.dumps(persisted, indent=2))

        print(f"{'=' * 74}\n{path.name}  ({pr.duration_s:.1f}s)\n{'=' * 74}")
        print(f"  speakers detected     : {stats['speakers_detected']}")
        print(f"  turns                 : {stats['n_turns']}  "
              f"(role changes {stats['role_changes']}, unknown {stats['unknown_turns']})")
        print(f"  first speaker         : {stats['first_speaker']}")
        print(f"  speech seconds by role: {stats['role_seconds']}")
        print(f"  customer fraction     : {stats['customer_speech_fraction']}")
        print(f"  customer audio        : {stats['customer_audio_seconds']}s across "
              f"{len(stats['customer_regions'])} region(s)")
        if stats["timestamp_repair"]:
            print(f"  TIMESTAMP REPAIR      : {stats['timestamp_repair']}")
        print(f"  latency               : {elapsed:.2f}s  "
              f"({elapsed / max(pr.duration_s / 60, 1e-9):.2f}s per audio-min)")
        print(f"  tokens                : in={audio_tok} out={out_tok}   cost=${cost:.6f}")
        print(f"  audio tok/s (measured): "
              f"{(audio_tok - 180) / max(pr.duration_s, 1e-9):.1f}  (expected ~32)")
        print(f"\n  customer identification:\n    {result.get('customer_identification', '')}")
        print(f"\n  call summary:\n    {result.get('call_summary', '')}")
        print("\n  transcript:")
        for t in stats["turns_parsed"][:40]:
            print(f"    [{t.start_s:7.2f}-{t.end_s:7.2f}] {t.role:<9} {t.text}")
        print()

    print(f"{'=' * 74}\ntotal probe cost: ${total_cost:.6f}")
    print(f"transcripts written to {TRANSCRIPT_DIR} (gitignored; reused by the B1 baseline)")
    print("\nVERIFY MANUALLY: is the customer correctly identified in each call?")
    print("  3/3 -> proceed | 2/3 -> add scripted-opener heuristic | <=1/3 -> redesign")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
