"""Ingest tests, including regressions for two bugs found during development."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from autoace.ingest import (
    AudioIngestError,
    _downmix_filter,
    analyse_channels,
    decode,
    load,
    probe,
)

pytestmark = pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg not available",
)


def _make(path: Path, *, channels: int = 2, seconds: float = 2.0, amp: float = 0.9,
          rate: int = 48000, codec: str = "libopus", ext: str = "ogg") -> Path:
    """Synthesise a test clip. Never uses the confidential production audio."""
    out = path.with_suffix(f".{ext}")
    layout = "stereo" if channels == 2 else "mono"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}:sample_rate={rate}",
         "-af", f"volume={amp},aformat=channel_layouts={layout}",
         "-c:a", codec, str(out)],
        check=True, capture_output=True,
    )
    return out


def test_downmix_filter_weights_sum_to_one():
    """Regression: ffmpeg's `-ac 1` is not level-preserving.

    For stereo it weights each channel by 1/sqrt(2), so identical L/R sum to
    +3.01 dB. Measured on real duplicated-mono input that turned 3 full-scale
    samples into 2,498 and produced a false clipping verdict. Our filter must
    use weights that sum to exactly 1.
    """
    for nch in (1, 2, 4):
        expr = _downmix_filter(nch)
        weights = [float(t.split("*")[0]) for t in expr.split("c0=")[1].split("+")] \
            if nch > 1 else [1.0]
        assert sum(weights) == pytest.approx(1.0), f"{nch}ch downmix must preserve level"


def test_downmix_preserves_level_on_duplicated_mono(tmp_path):
    """The mono downmix of duplicated-mono audio must equal a single channel."""
    f = _make(tmp_path / "dup", channels=2, amp=0.9)
    mono = decode(f, sr=48000, mono=True)
    multi = decode(f, sr=48000, mono=False)
    left_peak = float(np.abs(multi[:, 0]).max())
    mono_peak = float(np.abs(mono).max())
    # Well inside the +3.01 dB error the old code produced.
    assert mono_peak == pytest.approx(left_peak, rel=0.02), (
        f"downmix changed level: mono {mono_peak:.4f} vs left {left_peak:.4f}"
    )


def test_duplicated_mono_is_detected(tmp_path):
    f = _make(tmp_path / "dup", channels=2)
    rep = analyse_channels(f)
    assert rep.channels == 2
    assert rep.is_duplicated_mono is True
    assert rep.pearson_r > 0.999


def test_true_stereo_is_not_flagged_as_duplicated(tmp_path):
    """Uncorrelated channels must not be called duplicated mono."""
    out = tmp_path / "stereo.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000",
         "-f", "lavfi", "-i", "sine=frequency=997:duration=2:sample_rate=48000",
         "-filter_complex", "[0:a][1:a]amerge=inputs=2[a]", "-map", "[a]",
         str(out)],
        check=True, capture_output=True,
    )
    rep = analyse_channels(out)
    assert rep.is_duplicated_mono is False


def test_probe_reads_container_metadata(tmp_path):
    f = _make(tmp_path / "probe", seconds=1.5)
    pr = probe(f)
    assert pr.codec == "opus"
    assert pr.channels == 2
    assert pr.duration_s == pytest.approx(1.5, abs=0.2)


# --- Failure isolation: every one of these must raise a clean, readable
# --- reason rather than a traceback, so the batch can continue.

def test_zero_byte_file_rejected(tmp_path):
    f = tmp_path / "empty.wav"
    f.write_bytes(b"")
    with pytest.raises(AudioIngestError, match="zero bytes"):
        probe(f)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(AudioIngestError, match="not found"):
        probe(tmp_path / "nope.wav")


def test_mp3_that_is_actually_a_pdf_rejected(tmp_path):
    f = tmp_path / "fake.mp3"
    f.write_bytes(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<</Type/Catalog>>\nendobj\n")
    with pytest.raises(AudioIngestError, match="not decodable as audio"):
        probe(f)


def test_truncated_audio_rejected(tmp_path):
    f = _make(tmp_path / "trunc", seconds=2.0)
    data = f.read_bytes()
    f.write_bytes(data[: len(data) // 6])
    with pytest.raises(AudioIngestError):
        load(f)


def test_unsupported_extension_rejected(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(AudioIngestError, match="unsupported extension"):
        load(f)


def test_load_normalises_without_touching_raw(tmp_path):
    """`raw` must keep original gain so clipping stays measurable; `norm` is scaled."""
    quiet = _make(tmp_path / "quiet", amp=0.02)
    la = load(quiet, sr=16000)
    raw_rms = float(np.sqrt(np.mean(la.raw**2)))
    norm_rms = float(np.sqrt(np.mean(la.norm**2)))
    assert norm_rms > raw_rms * 4, "quiet input should be normalised upward"
    assert la.gain_db > 0
    # raw must be untouched by normalisation
    assert float(np.abs(la.raw).max()) < 0.1
