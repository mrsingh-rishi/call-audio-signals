"""Centralised configuration.

Every environment lookup goes through here so that:

- model IDs and pricing live in config, not code (they change, and a retirement
  should be a config edit rather than a code change);
- secrets are never logged - :func:`redacted_summary` is the only thing allowed
  to describe the configuration, and it reports presence, never values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env_once() -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


# --- Pricing ---------------------------------------------------------------
# USD per 1M tokens, verified 2026-08-01. Audio input is billed at a different
# rate from text input on every model, which is the number that matters here.
# The cost model recomputes from live usageMetadata; this table is only used
# to convert measured token counts into dollars.

@dataclass(frozen=True)
class ModelPricing:
    audio_in_per_m: float
    text_in_per_m: float
    text_out_per_m: float
    cached_in_per_m: float | None = None
    notes: str = ""


PRICING: dict[str, ModelPricing] = {
    "gemini-2.5-flash-lite": ModelPricing(
        audio_in_per_m=0.30, text_in_per_m=0.10, text_out_per_m=0.40,
        cached_in_per_m=0.03,
        notes="Retires 2026-10-16. Only model where 3 heads fit uncached.",
    ),
    "gemini-3.1-flash-lite": ModelPricing(
        audio_in_per_m=0.50, text_in_per_m=0.25, text_out_per_m=1.50,
        cached_in_per_m=0.05,
        notes="Forward path. 3 uncached heads breach the ceiling; 1 pass fits.",
    ),
    "gemini-3.5-flash-lite": ModelPricing(
        audio_in_per_m=0.30, text_in_per_m=0.30, text_out_per_m=2.50,
        notes="Audio rate to be confirmed empirically.",
    ),
    "gemini-3.5-flash": ModelPricing(
        audio_in_per_m=1.50, text_in_per_m=1.50, text_out_per_m=9.00,
    ),
}

AUDIO_TOKENS_PER_SECOND = 32
"""Gemini bills audio at 32 tokens/second = 1,920 tokens per audio minute."""

COST_CEILING_PER_AUDIO_MIN = 0.003
"""Hard constraint from the brief. Asserted against measured usage, not estimates."""

BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_model_fallback: str = "gemini-3.1-flash-lite"
    gemini_thinking_budget: int = 0

    local_only: bool = False
    upload_retention_hours: float = 24.0

    app_user: str = "autoace"
    app_password: str = ""
    session_secret: str = ""

    max_concurrency: int = 4
    database_url: str = "sqlite:///./autoace.db"

    # Long-call handling. Clips at or below the threshold are scored whole;
    # longer clips are windowed with a COMPACT per-window schema, because
    # evidence strings per window erode the cost headroom (plan 7.1).
    window_threshold_s: float = 90.0
    window_length_s: float = 60.0
    window_overlap_s: float = 5.0

    extra: dict[str, str] = field(default_factory=dict)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key) and not self.local_only

    @property
    def pricing(self) -> ModelPricing:
        return PRICING.get(self.gemini_model, PRICING["gemini-2.5-flash-lite"])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env_once()
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip(),
        gemini_model_fallback=os.getenv(
            "GEMINI_MODEL_FALLBACK", "gemini-3.1-flash-lite"
        ).strip(),
        gemini_thinking_budget=int(_num("GEMINI_THINKING_BUDGET", 0)),
        local_only=_flag("LOCAL_ONLY", False),
        upload_retention_hours=_num("UPLOAD_RETENTION_HOURS", 24.0),
        app_user=os.getenv("APP_USER", "autoace").strip(),
        app_password=os.getenv("APP_PASSWORD", ""),
        session_secret=os.getenv("SESSION_SECRET", ""),
        max_concurrency=int(_num("MAX_CONCURRENCY", 4)),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./autoace.db"),
    )


def redacted_summary() -> dict[str, object]:
    """Describe the active configuration without exposing any secret.

    This is the only function permitted to report on configuration in logs, the
    dashboard, or the compliance record.
    """
    s = get_settings()
    return {
        "gemini_api_key_present": bool(s.gemini_api_key),
        "gemini_enabled": s.gemini_enabled,
        "gemini_model": s.gemini_model,
        "gemini_model_fallback": s.gemini_model_fallback,
        "gemini_thinking_budget": s.gemini_thinking_budget,
        "local_only": s.local_only,
        "upload_retention_hours": s.upload_retention_hours,
        "app_password_set": bool(s.app_password),
        "session_secret_set": bool(s.session_secret),
        "max_concurrency": s.max_concurrency,
        "window_threshold_s": s.window_threshold_s,
        "window_length_s": s.window_length_s,
    }
