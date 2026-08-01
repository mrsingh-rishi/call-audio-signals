"""Seeded degradation chain that manufactures exact ground truth.

The trial provides three labelled calls, which supports no statistical claim.
The way out is to build clips whose labels are *known by construction*: if we mix
a noise file in at a chosen SNR, we know `background_noise_present` and its
severity without anyone listening. The same holds for injected overlap, injected
silence, and applied quality defects.

Two rules keep this honest:

1. **Factors are applied independently.** Noise level and quality damage are
   drawn separately, so the set contains clean-but-noisy and
   distorted-but-quiet-background clips. If they were coupled, a model could
   score well by conflating them - the exact shortcut the brief warns about.
2. **Every draw is seeded and recorded**, so a clip's manifest row fully
   determines its audio.

Ground-truth thresholds here are the *definitions* the proxy set is built to;
the detector thresholds in ``path_b_acoustic`` are then fitted against them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SR = 16_000

# --- Ground-truth definitions ---------------------------------------------
# SNR ladder -> (noise_present, severity). 30 dB is deliberately labelled
# absent: the brief says "barely perceptible artifacts should not automatically
# count as background noise", and 30 dB SNR is exactly that case.
SNR_LADDER: dict[float, tuple[bool, str]] = {
    float("inf"): (False, "none"),
    30.0: (False, "none"),
    20.0: (True, "low"),
    15.0: (True, "medium"),
    10.0: (True, "medium"),
    5.0: (True, "high"),
    0.0: (True, "high"),
}

# Injected silence -> long_silence_present. The detector threshold is fitted
# against this, not assumed; see path_b_acoustic.SILENCE_MIN_S.
SILENCE_TRUE_AT_S = 5.0

# Cumulative overlap above this counts as present.
OVERLAP_TRUE_AT_S = 0.5

QUALITY_OPS = {
    "clip":        ("severely_impaired", "hard clipping with flat-topped runs"),
    "packet_loss": ("severely_impaired", "20-200 ms dropouts"),
    "echo":        ("slightly_impaired", "room reverberation"),
    "low_volume":  ("slightly_impaired", "-20 dB level"),
    "bandlimit":   ("slightly_impaired", "narrowband filtering"),
    "opus_low":    ("severely_impaired", "very low bitrate codec artifacts"),
}
_QUALITY_RANK = {"clear": 0, "slightly_impaired": 1, "severely_impaired": 2}
_RANK_QUALITY = {v: k for k, v in _QUALITY_RANK.items()}

FORMAT_CHAINS = ("opus48_dupstereo", "wav16", "g711_mp3")


@dataclass
class DegradationSpec:
    """Everything drawn for one clip. Fully determines the output audio."""

    seed: int
    snr_db: float
    noise_class: str = ""
    noise_file: str = ""
    quality_ops: list[str] = field(default_factory=list)
    overlap_s: float = 0.0
    overlap_file: str = ""
    silence_s: float = 0.0
    format_chain: str = "opus48_dupstereo"
    adversarial_cell: str = ""

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def ground_truth(spec: DegradationSpec, tone: str, intensity: str) -> dict:
    """Derive the nine-field label from the spec. No listening required."""
    present, severity = SNR_LADDER[spec.snr_db]

    quality = "clear"
    for op in spec.quality_ops:
        lvl = QUALITY_OPS[op][0]
        if _QUALITY_RANK[lvl] > _QUALITY_RANK[quality]:
            quality = lvl

    return {
        "emotional_tone": tone,
        "emotional_intensity": intensity,
        "background_noise_present": present,
        "background_noise_type": spec.noise_class if present else "",
        "background_noise_severity": severity,
        "audio_quality": quality,
        "speaker_overlap_present": spec.overlap_s > OVERLAP_TRUE_AT_S,
        "long_silence_present": spec.silence_s >= SILENCE_TRUE_AT_S,
        "confidence": 0.82,
    }


# --- Signal operations -----------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)) + 1e-12)


def _fit_length(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Tile or crop a noise clip to the target length, starting at random."""
    if noise.size == 0:
        return np.zeros(n)
    if noise.size < n:
        reps = int(np.ceil(n / noise.size))
        noise = np.tile(noise, reps)
    start = int(rng.integers(0, max(1, noise.size - n)))
    return noise[start:start + n]


def mix_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float,
              rng: np.random.Generator) -> np.ndarray:
    """Mix at an exact SNR measured on speech-active frames.

    Using whole-signal RMS would make the achieved SNR depend on how much
    silence a clip happens to contain, so two clips at nominally the same SNR
    would sound different. Speech-active RMS keeps the ladder meaningful.
    """
    if not np.isfinite(snr_db):
        return speech
    n = int(0.032 * SR)
    if speech.size >= n:
        fr = speech[: (speech.size // n) * n].reshape(-1, n)
        fr_rms = np.sqrt((fr**2).mean(axis=1))
        # Anchor to the loudest content, not to a fixed percentile. These clips
        # are concatenated turns with gaps plus injected silence, so more than
        # half the frames can be silent - a percentile-60 cut then lands inside
        # the silence and the "speech level" collapses, which scaled the noise
        # far too quietly and made the injected SNR (and therefore the ground
        # truth) wrong by 20-50 dB.
        db = 20 * np.log10(fr_rms + 1e-12)
        ref = float(np.percentile(db, 95))
        active = fr_rms[db > ref - 25.0]
        speech_level = float(active.mean()) if active.size else _rms(speech)
    else:
        speech_level = _rms(speech)

    noise = _fit_length(noise, speech.size, rng)
    target_noise_level = speech_level / (10 ** (snr_db / 20))
    return speech + noise * (target_noise_level / _rms(noise))


def inject_overlap(speech: np.ndarray, other: np.ndarray, overlap_s: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Sum a second talker over part of the clip, as a mono mix would."""
    if overlap_s <= 0 or other.size == 0:
        return speech
    n_overlap = min(int(overlap_s * SR), speech.size)
    seg = _fit_length(other, n_overlap, rng)
    start = int(rng.integers(0, max(1, speech.size - n_overlap)))
    out = speech.copy()
    # Slightly quieter, as an interrupting party on a phone mix usually is.
    out[start:start + n_overlap] += seg * (_rms(speech) / _rms(seg)) * 0.7
    return out


def inject_silence(speech: np.ndarray, silence_s: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Splice digital silence into the middle, away from the edges.

    Leading and trailing dead air is normal on a recording; only an interior gap
    is evidence of a call-flow problem, so that is what gets injected.
    """
    if silence_s <= 0:
        return speech
    gap = np.zeros(int(silence_s * SR))
    lo, hi = int(0.25 * speech.size), int(0.75 * speech.size)
    cut = int(rng.integers(lo, max(lo + 1, hi)))
    return np.concatenate([speech[:cut], gap, speech[cut:]])


def apply_quality(x: np.ndarray, ops: list[str], rng: np.random.Generator) -> np.ndarray:
    """Apply signal-integrity defects. Independent of the noise axis."""
    y = x.copy()
    for op in ops:
        if op == "clip":
            # Hard clip at a fraction of peak so it produces genuine flat-topped
            # runs - the thing forensics.clipping_report actually looks for.
            thr = float(np.abs(y).max()) * 0.35
            y = np.clip(y, -thr, thr)
        elif op == "packet_loss":
            n_drops = int(rng.integers(4, 14))
            for _ in range(n_drops):
                d = int(rng.uniform(0.02, 0.2) * SR)
                s = int(rng.integers(0, max(1, y.size - d)))
                y[s:s + d] = 0.0
        elif op == "echo":
            delay = int(rng.uniform(0.06, 0.16) * SR)
            decay = float(rng.uniform(0.35, 0.6))
            e = np.zeros_like(y)
            e[delay:] = y[:-delay] * decay
            y = y + e
        elif op == "low_volume":
            y = y * 0.1
        elif op == "bandlimit":
            # Crude one-pole low-pass around 1.6 kHz - muffled speech.
            a = 0.82
            out = np.zeros_like(y)
            acc = 0.0
            for i in range(y.size):
                acc = a * acc + (1 - a) * y[i]
                out[i] = acc
            y = out
        # opus_low is handled in the format stage, where the codec runs.
    return y


# --- Format / container stage ---------------------------------------------

def write_format(x: np.ndarray, out_path: Path, chain: str, low_bitrate: bool = False) -> Path:
    """Render to one of the container chains.

    Container is a *factor*, not a fixed terminal state. The provided calls are
    48 kHz Opus duplicated-stereo, but the brief only promises "the same general
    format", so thresholds are checked for portability across all three.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.abs(x).max())
    if peak > 0.999:
        x = x / peak * 0.999          # avoid container-level clipping artefacts
    raw = (x * 32767.0).astype(np.int16).tobytes()

    base = ["ffmpeg", "-v", "error", "-y",
            "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0"]

    if chain == "opus48_dupstereo":
        br = "24k" if low_bitrate else "128k"
        cmd = base + ["-af", "aresample=48000,pan=stereo|c0=c0|c1=c0",
                      "-c:a", "libopus", "-b:a", br, str(out_path.with_suffix(".ogg"))]
        final = out_path.with_suffix(".ogg")
    elif chain == "wav16":
        cmd = base + ["-ar", "16000", "-ac", "1",
                      "-c:a", "pcm_s16le", str(out_path.with_suffix(".wav"))]
        final = out_path.with_suffix(".wav")
    elif chain == "g711_mp3":
        # 8 kHz mu-law encode/decode, then MP3 - the narrowband telephony path.
        cmd = base + ["-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw",
                      "-f", "wav", "pipe:1"]
        mu = subprocess.run(cmd, input=raw, capture_output=True)
        final = out_path.with_suffix(".mp3")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "pipe:0",
                        "-c:a", "libmp3lame", "-b:a", "64k", str(final)],
                       input=mu.stdout, capture_output=True, check=True)
        return final
    else:
        raise ValueError(f"unknown format chain {chain!r}")

    subprocess.run(cmd, input=raw, capture_output=True, check=True)
    return final


def degrade(
    speech: np.ndarray,
    spec: DegradationSpec,
    out_path: Path,
    *,
    noise: np.ndarray | None = None,
    overlap_speech: np.ndarray | None = None,
) -> Path:
    """Run the full chain for one clip and write it in the target format."""
    rng = np.random.default_rng(spec.seed)
    y = speech.astype(np.float64)

    y = inject_silence(y, spec.silence_s, rng)
    if overlap_speech is not None and spec.overlap_s > 0:
        y = inject_overlap(y, overlap_speech.astype(np.float64), spec.overlap_s, rng)
    if noise is not None and np.isfinite(spec.snr_db):
        y = mix_noise(y, noise.astype(np.float64), spec.snr_db, rng)
    y = apply_quality(y, spec.quality_ops, rng)

    return write_format(y, out_path, spec.format_chain,
                        low_bitrate="opus_low" in spec.quality_ops)


def load_wav(path: str | Path, sr: int = SR) -> np.ndarray:
    """Decode any audio file to mono float at the working rate."""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-af", "pan=mono|c0=c0", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True,
    )
    return np.frombuffer(p.stdout, dtype=np.float32).astype(np.float64)


def make_babble(sources: list[Path], n_samples: int, n_voices: int,
                rng: np.random.Generator) -> np.ndarray:
    """Synthesise office chatter by overlaying several speakers.

    Babble is the standard way to make 'office chatter' noise, and building it
    from HELD-OUT speakers keeps the noise axis disjoint from the speech axis -
    otherwise a clip could contain the same voice as both signal and noise,
    which is a leakage path.
    """
    out = np.zeros(n_samples)
    for _ in range(n_voices):
        src = load_wav(sources[int(rng.integers(0, len(sources)))])
        if src.size == 0:
            continue
        out += _fit_length(src, n_samples, rng)
    return out / max(n_voices, 1)
