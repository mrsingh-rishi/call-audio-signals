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
from .fusion import fuse
from .ingest import AudioIngestError, probe
from .path_a_gemini import analyse_single
from .path_b_acoustic import analyse_acoustics
from .path_c_prosody import analyse_prosody
from .schema import CallAnalysis


class _EmptyUsage:
    """Zero-cost stand-in when only the deterministic path ran."""

    prompt_tokens = output_tokens = thought_tokens = cached_tokens = 0
    latency_s = 0.0

    def cost_usd(self, model: str | None = None) -> float:
        return 0.0

    def cost_per_audio_min(self, duration_s: float, model: str | None = None) -> float:
        return 0.0


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
    cost_uncached_usd: float = 0.0
    cost_uncached_per_audio_min: float = 0.0
    model: str = ""
    tokens: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    repairs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    disagreements: list[str] = field(default_factory=list)
    acoustic_metrics: dict[str, float] = field(default_factory=dict)

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

    # Path B is deterministic, local and free, so it always runs - including in
    # LOCAL_ONLY mode, where it is the only analysis available.
    acoustic = None
    try:
        acoustic = analyse_acoustics(p)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"acoustic path failed: {type(exc).__name__}")

    prosody = None
    try:
        prosody = analyse_prosody(p)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"prosody path failed: {type(exc).__name__}")

    gemini_analysis = None
    meta: dict[str, Any] = {}
    if s.gemini_enabled:
        try:
            gemini_analysis, meta = analyse_single(p, settings=s, client=client)
        except Exception as exc:  # noqa: BLE001 - deliberate: isolate per file
            result.warnings.append(f"gemini path failed: {type(exc).__name__}: {str(exc)[:120]}")

    if gemini_analysis is None and acoustic is None:
        result.status = "error"
        result.reason = "both analysis paths failed for this file"
        result.latency_s = time.perf_counter() - t0
        return result

    outcome = fuse(gemini_analysis, acoustic, prosody)
    usage = meta.get("usage") or _EmptyUsage()
    result.analysis = outcome.analysis.to_output_dict()
    result.model = meta.get("model", "acoustic-only")
    result.sources = outcome.sources
    result.disagreements = outcome.disagreements
    if acoustic is not None:
        result.acoustic_metrics = acoustic.metrics
    result.latency_s = time.perf_counter() - t0
    result.cost_usd = usage.cost_usd(result.model)
    result.cost_per_audio_min = usage.cost_per_audio_min(pr.duration_s, result.model)
    if hasattr(usage, 'cost_uncached_usd'):
        result.cost_uncached_usd = usage.cost_uncached_usd(result.model)
        result.cost_uncached_per_audio_min = usage.cost_uncached_per_audio_min(
            pr.duration_s, result.model)
    result.tokens = {
        "prompt": usage.prompt_tokens,
        "output": usage.output_tokens,
        "thoughts": usage.thought_tokens,
        "cached": usage.cached_tokens,
    }
    result.evidence = {k: str(v) for k, v in meta.get("evidence", {}).items()}
    result.repairs = list(meta.get("repairs", [])) + list(outcome.repairs)

    if max(result.cost_per_audio_min, result.cost_uncached_per_audio_min) > COST_CEILING_PER_AUDIO_MIN:
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
