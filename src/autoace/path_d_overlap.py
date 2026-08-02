"""Path D: overlapped-speech detection with pyannote segmentation-3.0 (ONNX).

Why this exists
---------------
``speaker_overlap_present`` was the one field with no working solution. The
hand-built dual-pitch cue in Path B sits at **0.544 balanced accuracy against a
0.520 baseline** on the proxy eval split - chance - and the correlation between
overlap duration and the feature was 0.069.

Two things were wrong, and only one of them was the detector. The proxy
generator was injecting the interrupting talker into silent gaps, so most clips
labelled as containing overlap contained almost none; that is fixed in
``proxy/degrade.inject_overlap``. On the corrected labels, this path reaches
**0.792 balanced accuracy (F1 0.789)** against the same 0.544 for the dual-pitch
cue - same split, same labels.

Why this model
--------------
``pyannote/segmentation-3.0`` is **MIT licensed** (Copyright (c) 2022 CNRS),
which is the filter that disqualified every speech-emotion candidate surveyed in
RESEARCH.md - audEERING MSP-dim is CC-BY-NC, emotion2vec+ is restricted by its
training data. AutoAce is a commercial product, so licence is a hard constraint.

The ONNX export ships as a **5.7 MB** file and runs on CPU through onnxruntime,
so the container gains no torch, no CUDA and nothing to download at boot. That
matters: the 512 MiB instance was already OOM-killed once, and the architectural
decision to avoid 660 MB-1.1 GB speech models is what leaves any headroom at all.

How the output is read
----------------------
The network emits a **powerset** over 3 speakers with at most 2 concurrent, so
7 classes per frame::

    0            -> non-speech
    1, 2, 3      -> exactly one speaker active
    4, 5, 6      -> exactly two speakers active  <- overlap

Overlap is therefore ``argmax(frame) >= 4``. No threshold tuning is involved in
reading the model; the only fitted parameter is how much total overlap counts as
"enough to affect understanding or analysis", which is the brief's wording and is
fitted on the proxy set by ``scripts/fit_overlap.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ingest import decode

SR = 16_000
WINDOW_SAMPLES = 160_000        # 10 s, fixed by the model's training regime
WINDOW_HOP_SAMPLES = 80_000     # 50% overlap so a talk-over at a window edge is
                                # never split across two windows and missed
ONNX_BATCH = 8
"""Windows fed to the model at once.

A 60 s chunk is 11 windows of 160 000 float32 samples - 7 MB of input, but the
activations for all of them exist simultaneously inside the session. Batching in
eights bounds that regardless of clip length."""

DECODE_CHUNK_S = 60.0
"""Decode in 60 s chunks. A 45-minute call at 16 kHz float32 is 172 MB in one
array; chunking bounds peak memory by chunk rather than by clip length, matching
what Path B already does for the same reason."""

FIRST_PAIR_CLASS = 4
"""Classes 4..6 are the two-speaker combinations. Derived from the model's own
metadata (num_speakers=3, powerset_max_classes=2), asserted at load time."""

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "sherpa-onnx-pyannote-segmentation-3-0"
MODEL_PATH = MODEL_DIR / "model.onnx"
COEF_PATH = Path(__file__).with_name("overlap_coefficients.json")

# Fallback used only when no fitted coefficients are present. Deliberately a
# duration, not a fraction: the brief scopes the field to overlap "enough to
# affect understanding", and 0.6 s is roughly one interrupted word.
DEFAULT_MIN_OVERLAP_S = 0.6

_session = None
_session_failed = False


@dataclass
class OverlapResult:
    speaker_overlap_present: bool = False
    overlap_seconds: float = 0.0
    overlap_fraction: float = 0.0
    longest_overlap_s: float = 0.0
    speech_seconds: float = 0.0
    available: bool = False
    notes: list[str] = field(default_factory=list)


def _load_session():
    """Lazily build one shared InferenceSession.

    One session, module-level: ``InferenceSession.run()`` is thread-safe, and the
    batch worker runs files concurrently. A session per call would multiply
    memory by ``MAX_CONCURRENCY`` for no benefit.
    """
    global _session, _session_failed
    if _session is not None or _session_failed:
        return _session
    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # Single-threaded: the batch layer already runs files concurrently, so
        # intra-op threads would oversubscribe the 0.1-0.5 CPU the instance has.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        # The CPU arena caches freed blocks instead of returning them, which on a
        # 512 MiB instance matters more than the allocation churn it saves. With
        # the arena off and inference batched (ONNX_BATCH), whole-process peak on
        # the 172 s clip measured 447 MB -> 332 MB, i.e. this path's own footprint
        # dropped from ~144 MB to ~31 MB. Identical outputs either way.
        opts.enable_cpu_mem_arena = False
        sess = ort.InferenceSession(
            str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        meta = sess.get_modelmeta().custom_metadata_map
        # Guard the powerset assumption rather than trusting the checkpoint blindly.
        n_classes = int(meta.get("num_classes", 7))
        n_speakers = int(meta.get("num_speakers", 3))
        max_concurrent = int(meta.get("powerset_max_classes", 2))
        if (n_classes, n_speakers, max_concurrent) != (7, 3, 2):
            _session_failed = True
            return None
        _session = sess
    except Exception:  # noqa: BLE001 - absent model or runtime must not break a batch
        _session_failed = True
        return None
    return _session


def _min_overlap_s() -> float:
    if COEF_PATH.exists():
        try:
            return float(json.loads(COEF_PATH.read_text())["min_overlap_s"])
        except Exception:  # noqa: BLE001
            pass
    return DEFAULT_MIN_OVERLAP_S


def _window_starts(n_samples: int) -> list[int]:
    """Start offsets of the 10 s analysis windows, hopping by 5 s."""
    if n_samples <= WINDOW_SAMPLES:
        return [0]
    starts = list(range(0, n_samples - WINDOW_SAMPLES + 1, WINDOW_HOP_SAMPLES))
    # Snap the final window to the hop grid rather than to the exact end, so
    # every frame maps cleanly onto the shared timeline in `_timeline`.
    last = starts[-1]
    while last + WINDOW_SAMPLES < n_samples:
        last += WINDOW_HOP_SAMPLES
        starts.append(last)
    return starts


def _windows(x: np.ndarray) -> np.ndarray:
    """Slice into fixed 10 s windows, zero-padding the tail."""
    out = np.zeros((len(_window_starts(x.size)), WINDOW_SAMPLES), dtype=np.float32)
    for i, s in enumerate(_window_starts(x.size)):
        seg = x[s : s + WINDOW_SAMPLES]
        out[i, : seg.size] = seg
    return out


def _timeline(sess, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the model over a signal and project every frame onto absolute time.

    Windows advance by half their length, so interior frames are scored twice.
    Rather than counting both and halving - which turns a disagreement between
    the two passes into a phantom half-frame - each frame is written to its true
    position on a shared timeline and combined with a logical OR. A talk-over
    seen in either pass counts once, at the right place.

    Returns (overlap mask, speech mask, seconds per frame) over the timeline.
    """
    win = _windows(x)
    starts = _window_starts(x.size)
    # Batched so peak activation memory is bounded by ONNX_BATCH rather than by
    # how many windows the chunk happens to contain.
    parts = []
    for i in range(0, win.shape[0], ONNX_BATCH):
        blk = win[i : i + ONNX_BATCH, None, :]
        parts.append(np.argmax(sess.run(None, {"x": blk})[0], axis=-1))
    cls = np.concatenate(parts, axis=0)             # (n_windows, n_frames)
    n_frames = cls.shape[1]
    frame_s = WINDOW_SAMPLES / SR / n_frames
    frames_per_hop = WINDOW_HOP_SAMPLES / SR / frame_s

    total = int(np.ceil(max(x.size / SR, 1e-9) / frame_s)) + n_frames
    ov = np.zeros(total, dtype=bool)
    sp = np.zeros(total, dtype=bool)
    for i, s0 in enumerate(starts):
        off = int(round(s0 / WINDOW_HOP_SAMPLES * frames_per_hop))
        sl = slice(off, off + n_frames)
        ov[sl] |= cls[i] >= FIRST_PAIR_CLASS
        sp[sl] |= cls[i] > 0
    keep = int(np.ceil(max(x.size / SR, 1e-9) / frame_s))
    return ov[:keep], sp[:keep], frame_s


def analyse_overlap(path: str | Path) -> OverlapResult:
    """Detect overlapped speech. Never raises; degrades to ``available=False``.

    When the model is missing the caller keeps Path B's dual-pitch heuristic,
    which is weak but is better than emitting a hard failure for one field.
    """
    res = OverlapResult()
    sess = _load_session()
    if sess is None:
        res.notes.append("segmentation model unavailable; falling back to Path B cue")
        return res

    try:
        from .ingest import probe

        try:
            duration = probe(path).duration_s
        except Exception:  # noqa: BLE001
            duration = 0.0

        ov_parts: list[np.ndarray] = []
        sp_parts: list[np.ndarray] = []
        frame_s = 0.0

        starts = [0.0] if duration <= DECODE_CHUNK_S else [
            i * DECODE_CHUNK_S for i in range(int(np.ceil(duration / DECODE_CHUNK_S)))
        ]
        for start in starts:
            dur = None if duration <= DECODE_CHUNK_S else min(
                DECODE_CHUNK_S, duration - start
            )
            if dur is not None and dur < 0.5:
                continue
            x = decode(path, sr=SR, mono=True, start_s=start, dur_s=dur)
            if x.size == 0:
                continue
            ov, sp, frame_s = _timeline(sess, x)
            ov_parts.append(ov)
            sp_parts.append(sp)

        if frame_s <= 0 or not ov_parts:
            res.notes.append("no frames produced")
            return res

        # Concatenating the chunk timelines reconstructs the whole clip, so a run
        # is measured across its true extent rather than per chunk.
        ov_all = np.concatenate(ov_parts)
        sp_all = np.concatenate(sp_parts)
        edges = np.flatnonzero(
            np.diff(np.concatenate(([0], ov_all.view(np.int8), [0])))
        )
        longest_run = int((edges[1::2] - edges[0::2]).max()) if edges.size else 0

        res.overlap_seconds = round(float(ov_all.sum()) * frame_s, 3)
        res.speech_seconds = round(float(sp_all.sum()) * frame_s, 3)
        res.longest_overlap_s = round(longest_run * frame_s, 3)
        res.overlap_fraction = round(
            res.overlap_seconds / res.speech_seconds, 4
        ) if res.speech_seconds > 0 else 0.0
        res.speaker_overlap_present = bool(res.overlap_seconds >= _min_overlap_s())
        res.available = True
        res.notes.append(
            f"pyannote-segmentation-3.0: {res.overlap_seconds:.2f}s overlap "
            f"over {res.speech_seconds:.2f}s speech "
            f"(longest run {res.longest_overlap_s:.2f}s)"
        )
    except Exception as exc:  # noqa: BLE001
        res.available = False
        res.notes.append(f"overlap analysis failed: {type(exc).__name__}")
    return res
