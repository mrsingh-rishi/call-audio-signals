"""Audio ingest: probe, decode, channel analysis, canonical normalisation.

Deliberately depends only on numpy plus the ffmpeg/ffprobe binaries, so the
forensics CLI and the batch pre-flight check stay light enough to run anywhere.

Two design points worth stating because they are easy to get wrong:

1. :class:`LoadedAudio` carries BOTH ``raw`` (original gain) and ``norm``
   (loudness-normalised). Level-dependent measurements - clipping above all -
   must run on ``raw``; normalising first would erase the very thing being
   measured. Models and spectral comparisons use ``norm``.

2. Canonical normalisation exists so the deterministic thresholds see one
   distribution regardless of input container. The provided calls are 48 kHz
   Opus, but the brief only promises "the same general format", so the proxy
   set varies container/bandwidth as a factor and this function is what makes
   the thresholds portable. See plan section 6.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WORK_SR = 16_000
"""Canonical working sample rate. 16 kHz is what silero-VAD expects and is
sufficient for every Path B measurement except wideband spectral analysis,
which requests a higher rate explicitly."""

ANALYSIS_SR = 48_000
"""Rate used for spectral forensics, matching the provided files natively."""

TARGET_DBFS = -23.0
"""Loudness normalisation target, approximating EBU R128 -23 LUFS with a
deterministic RMS estimate over speech-active frames."""

SUPPORTED_SUFFIXES = {".ogg", ".oga", ".opus", ".wav", ".mp3", ".m4a", ".flac", ".aac", ".webm"}


class AudioIngestError(Exception):
    """Raised when a file cannot be ingested.

    ``reason`` is written straight into the batch results table, so it must be
    a complete, human-readable sentence - the evaluator reads it, not a stack
    trace.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class AudioProbe:
    codec: str
    sample_rate: int
    channels: int
    channel_layout: str
    duration_s: float
    bit_rate: int | None
    format_name: str
    size_bytes: int
    encoder: str | None = None


@dataclass(frozen=True)
class ChannelReport:
    """Result of the L-R duplication test.

    On the three provided calls this returns ``is_duplicated_mono=True`` with
    ~91-98% bit-identical samples: the source is mono, encoded as stereo. That
    kills any per-channel speaker separation strategy, so the result is
    recorded rather than assumed either way.
    """

    channels: int
    rms_l_dbfs: float
    rms_r_dbfs: float
    rms_diff_dbfs: float
    max_abs_diff: float
    identical_fraction: float
    pearson_r: float
    is_duplicated_mono: bool


@dataclass
class LoadedAudio:
    raw: np.ndarray
    norm: np.ndarray
    sr: int
    duration_s: float
    probe: AudioProbe
    gain_db: float
    channels: ChannelReport | None = field(default=None)


def _clean_ffmpeg_error(stderr: str | bytes | None, path: Path) -> str:
    """Reduce ffmpeg/ffprobe stderr to one safe, readable line.

    Raw stderr embeds the absolute server path of the staged upload. These
    strings are surfaced in the dashboard and in downloadable results, so the
    directory layout must not travel with them.
    """
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "replace")
    lines = [ln.strip() for ln in (stderr or "").strip().splitlines() if ln.strip()]
    if not lines:
        return "unreadable by ffmpeg"
    hint = lines[-1]
    hint = hint.replace(str(path), path.name).replace(str(path.parent) + "/", "")
    hint = hint.replace(str(path.parent), "")
    return hint[:160]


def _require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise AudioIngestError(
                f"{tool} is not installed or not on PATH; audio decoding is unavailable"
            )


def probe(path: str | Path) -> AudioProbe:
    """Read container/stream metadata. Raises :class:`AudioIngestError`."""
    _require_ffmpeg()
    p = Path(path)
    if not p.exists():
        raise AudioIngestError(f"file not found: {p.name}")
    if p.stat().st_size == 0:
        raise AudioIngestError(f"file is zero bytes: {p.name}")

    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(p),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AudioIngestError(f"not decodable as audio ({_clean_ffmpeg_error(proc.stderr, p)})")

    try:
        meta = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AudioIngestError(f"ffprobe returned unparseable metadata: {exc}") from exc

    streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise AudioIngestError("file contains no audio stream")
    s, fmt = streams[0], meta.get("format", {})

    duration = _as_float(s.get("duration")) or _as_float(fmt.get("duration")) or 0.0
    if duration <= 0.0:
        raise AudioIngestError("audio stream has zero or unknown duration")

    tags = {**fmt.get("tags", {}), **s.get("tags", {})}
    encoder = tags.get("encoder") or tags.get("ENCODER")

    return AudioProbe(
        codec=s.get("codec_name", "unknown"),
        sample_rate=int(_as_float(s.get("sample_rate")) or 0),
        channels=int(s.get("channels") or 0),
        channel_layout=s.get("channel_layout", "unknown"),
        duration_s=duration,
        bit_rate=int(_as_float(fmt.get("bit_rate")) or 0) or None,
        format_name=fmt.get("format_name", "unknown"),
        size_bytes=int(_as_float(fmt.get("size")) or p.stat().st_size),
        encoder=encoder,
    )


def _as_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _downmix_filter(channels: int) -> str:
    """Level-preserving mono downmix expression.

    ffmpeg's ``-ac 1`` is NOT level-preserving: for stereo it applies a 1/sqrt(2)
    weight per channel, so identical L/R sum to +3.01 dB. That silently breaks
    every level-dependent measurement - measured on the provided calls it turned
    3 full-scale samples into 2,498 and produced a false clipping verdict. An
    equal-weight average (weights summing to 1) leaves duplicated-mono content
    at its original level.
    """
    if channels <= 1:
        return "pan=mono|c0=c0"
    w = 1.0 / channels
    return "pan=mono|c0=" + "+".join(f"{w:.10f}*c{i}" for i in range(channels))


def decode(path: str | Path, sr: int = WORK_SR, mono: bool = True) -> np.ndarray:
    """Decode to float64 PCM.

    Returns shape ``(n,)`` when ``mono`` else ``(n, channels)``. Decoding to
    float preserves intersample overshoot above +/-1.0, which matters: peak
    values above full scale are usually codec overshoot rather than clipping,
    and distinguishing them requires the un-clamped samples.
    """
    _require_ffmpeg()
    pr = probe(path)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0"]
    if mono:
        cmd += ["-af", _downmix_filter(pr.channels)]
    else:
        cmd += ["-ac", str(pr.channels)]
    cmd += ["-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise AudioIngestError(
            f"audio decode failed ({_clean_ffmpeg_error(proc.stderr, Path(path))})"
        )

    x = np.frombuffer(proc.stdout, dtype=np.float32).astype(np.float64)
    if x.size == 0:
        raise AudioIngestError("decoded to zero samples (truncated or empty stream)")
    if not mono and pr.channels > 1:
        usable = (x.size // pr.channels) * pr.channels
        x = x[:usable].reshape(-1, pr.channels)
    return x


def analyse_channels(path: str | Path, sr: int = ANALYSIS_SR) -> ChannelReport:
    """Test whether a stereo file is genuinely dual-channel or duplicated mono.

    Duplicated mono means agent and customer are summed together, so speaker
    overlap cannot be derived from cross-channel energy and the customer cannot
    be isolated by channel. Detection thresholds are deliberately loose because
    lossy stereo coding perturbs the channels slightly even when the source was
    mono - on the provided calls the residual reaches 4e-3 while 91%+ of
    samples remain bit-identical.
    """
    pr = probe(path)
    if pr.channels < 2:
        return ChannelReport(pr.channels, 0.0, 0.0, -np.inf, 0.0, 1.0, 1.0, True)

    st = decode(path, sr=sr, mono=False)
    left, right = st[:, 0], st[:, 1]
    diff = left - right

    def _db(v: np.ndarray) -> float:
        return float(20 * np.log10(np.sqrt(np.mean(v**2)) + 1e-300))

    identical = float(np.mean(left == right))
    r = 1.0
    if left.std() > 0 and right.std() > 0:
        r = float(np.corrcoef(left, right)[0, 1])

    rms_l, rms_diff = _db(left), _db(diff)
    # Duplicated when the difference sits far below the signal AND the channels
    # correlate almost perfectly. Requiring both avoids calling a quiet true-
    # stereo recording "duplicated".
    is_dup = bool((rms_diff < rms_l - 40.0) and (r > 0.999))

    return ChannelReport(
        channels=pr.channels,
        rms_l_dbfs=rms_l,
        rms_r_dbfs=_db(right),
        rms_diff_dbfs=rms_diff,
        max_abs_diff=float(np.abs(diff).max()),
        identical_fraction=identical,
        pearson_r=r,
        is_duplicated_mono=is_dup,
    )


def _speech_active_rms(x: np.ndarray, sr: int) -> float:
    """RMS over the louder 40% of frames, as a speech-level proxy.

    Plain whole-signal RMS would be dragged down by silence, so two clips with
    identical speech loudness but different pause ratios would normalise to
    different levels - exactly the instability normalisation is meant to remove.
    """
    n, hop = int(0.032 * sr), int(0.016 * sr)
    if x.size < n:
        return float(np.sqrt(np.mean(x**2)) + 1e-300)
    nfr = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(nfr)[:, None]
    frames = x[idx]
    fr_rms = np.sqrt((frames**2).mean(axis=1))
    thr = np.percentile(fr_rms, 60)
    active = fr_rms[fr_rms >= thr]
    return float(active.mean() + 1e-300) if active.size else float(fr_rms.mean() + 1e-300)


def load(
    path: str | Path,
    sr: int = WORK_SR,
    with_channels: bool = True,
) -> LoadedAudio:
    """Full ingest: probe, decode, channel report, canonical normalisation."""
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise AudioIngestError(
            f"unsupported extension {p.suffix!r} "
            f"(supported: {', '.join(sorted(SUPPORTED_SUFFIXES))})"
        )

    pr = probe(p)
    raw = decode(p, sr=sr, mono=True)

    level = _speech_active_rms(raw, sr)
    target = 10 ** (TARGET_DBFS / 20)
    gain = target / level
    # Clamp so a near-silent clip is not amplified into pure noise.
    gain = float(np.clip(gain, 10 ** (-20 / 20), 10 ** (30 / 20)))
    norm = raw * gain

    channels = None
    if with_channels and pr.channels >= 2:
        try:
            channels = analyse_channels(p)
        except AudioIngestError:
            channels = None  # non-fatal; the report is diagnostic only

    return LoadedAudio(
        raw=raw,
        norm=norm,
        sr=sr,
        duration_s=raw.size / sr,
        probe=pr,
        gain_db=float(20 * np.log10(gain)),
        channels=channels,
    )
