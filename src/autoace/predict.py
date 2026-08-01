"""Prediction orchestrator.

One rule governs this module: **a single bad file must never fail a batch.**
Every failure mode - unreadable audio, API error, unparseable model output - is
caught here and converted into a result row with ``status="error"`` and a
human-readable reason. Callers can assume :func:`analyse_file` does not raise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai

from .config import COST_CEILING_PER_AUDIO_MIN, Settings, get_settings
from .ingest import AudioIngestError, probe
from .path_a_gemini import analyse_single
from .schema import CallAnalysis


@dataclass
class PredictionResult:
    name: str
    status: str = "ok"  # "ok" | "error"
    analysis: dict[str, Any] | None = None
    reason: str | None = None
    duration_s: float = 0.0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    cost_per_audio_min: float = 0.0
    model: str = ""
    tokens: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    repairs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        """Flat dict for the results table / CSV export."""
        row: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "reason": self.reason or "",
            "duration_s": round(self.duration_s, 3),
            "latency_s": round(self.latency_s, 3),
            "cost_usd": round(self.cost_usd, 8),
            "cost_per_audio_min": round(self.cost_per_audio_min, 8),
            "model": self.model,
        }
        if self.analysis:
            row.update(self.analysis)
        return row


def analyse_file(
    path: str | Path,
    *,
    settings: Settings | None = None,
    client: genai.Client | None = None,
    name: str | None = None,
) -> PredictionResult:
    """Analyse one clip. Never raises."""
    s = settings or get_settings()
    p = Path(path)
    result = PredictionResult(name=name or p.name)
    t0 = time.perf_counter()

    try:
        pr = probe(p)
        result.duration_s = pr.duration_s
    except AudioIngestError as exc:
        result.status = "error"
        result.reason = exc.reason
        result.latency_s = time.perf_counter() - t0
        return result
    except Exception as exc:  # noqa: BLE001
        result.status = "error"
        result.reason = f"could not read audio: {type(exc).__name__}"
        result.latency_s = time.perf_counter() - t0
        return result

    if not s.gemini_enabled:
        result.status = "error"
        result.reason = (
            "no analysis path available: Gemini is disabled "
            "(LOCAL_ONLY=true or no API key) and the local path is not installed"
        )
        result.latency_s = time.perf_counter() - t0
        return result

    try:
        analysis, meta = analyse_single(p, settings=s, client=client)
    except Exception as exc:  # noqa: BLE001 - deliberate: isolate per file
        result.status = "error"
        result.reason = f"{type(exc).__name__}: {str(exc)[:280]}"
        result.latency_s = time.perf_counter() - t0
        return result

    usage = meta["usage"]
    result.analysis = analysis.to_output_dict()
    result.model = meta["model"]
    result.latency_s = time.perf_counter() - t0
    result.cost_usd = usage.cost_usd(result.model)
    result.cost_per_audio_min = usage.cost_per_audio_min(pr.duration_s, result.model)
    result.tokens = {
        "prompt": usage.prompt_tokens,
        "output": usage.output_tokens,
        "thoughts": usage.thought_tokens,
        "cached": usage.cached_tokens,
    }
    result.evidence = {k: str(v) for k, v in meta.get("evidence", {}).items()}
    result.repairs = list(meta.get("repairs", []))

    if result.cost_per_audio_min > COST_CEILING_PER_AUDIO_MIN:
        result.warnings.append(
            f"cost ${result.cost_per_audio_min:.6f}/audio-min exceeds the "
            f"${COST_CEILING_PER_AUDIO_MIN} ceiling"
        )
    if usage.thought_tokens:
        result.warnings.append(
            f"{usage.thought_tokens} thinking tokens billed at output rates"
        )
    if pr.duration_s < 1.0:
        result.warnings.append("clip under 1s; result is unreliable")
    return result


def validate_analysis(d: dict[str, Any]) -> CallAnalysis:
    """Re-validate a stored row against the strict contract."""
    return CallAnalysis(**{k: v for k, v in d.items() if k in CallAnalysis.model_fields})
