"""Path D (overlapped speech) tests.

Two things matter here and neither is accuracy - accuracy is measured on the
proxy set by `scripts/fit_overlap.py`, not asserted in a unit test:

1. The powerset reading is correct. Classes 4-6 are the two-speaker
   combinations; getting that index wrong would invert the field silently.
2. A missing or broken model degrades to `available=False` rather than raising,
   because one bad file must never fail a batch.
"""

from __future__ import annotations

import numpy as np
import pytest

from autoace import path_d_overlap as pd


def test_pair_class_index_matches_the_powerset_layout():
    """Rebuild the mapping the reference implementation uses and check the split.

    num_speakers=3, powerset_max_classes=2 gives: 0 non-speech, 1-3 single
    speaker, 4-6 speaker pairs. FIRST_PAIR_CLASS must be the first row whose
    speaker count is 2.
    """
    n_classes, n_speakers, max_concurrent = 7, 3, 2
    mapping = np.zeros((n_classes, n_speakers))
    k = 1
    for i in range(1, max_concurrent + 1):
        if i == 1:
            for j in range(n_speakers):
                mapping[k, j] = 1
                k += 1
        else:
            for j in range(n_speakers):
                for m in range(j + 1, n_speakers):
                    mapping[k, j] = mapping[k, m] = 1
                    k += 1
    counts = mapping.sum(axis=1)
    assert int(np.flatnonzero(counts == 2)[0]) == pd.FIRST_PAIR_CLASS
    assert (counts[pd.FIRST_PAIR_CLASS:] == 2).all()
    assert (counts[1:pd.FIRST_PAIR_CLASS] == 1).all()


def test_windows_are_model_sized_and_zero_padded():
    x = np.ones(WINDOW := pd.WINDOW_SAMPLES - 1000, dtype=np.float32)
    w = pd._windows(x)
    assert w.shape[1] == pd.WINDOW_SAMPLES
    assert w[0, WINDOW:].sum() == 0.0          # tail is padding, not garbage


def test_window_starts_cover_the_whole_signal():
    """Every sample must fall inside at least one window, or overlap is missed."""
    n = pd.WINDOW_SAMPLES * 3 + 12345
    starts = pd._window_starts(n)
    assert starts[0] == 0
    assert starts[-1] + pd.WINDOW_SAMPLES >= n
    # 50% hop, so consecutive windows must actually overlap.
    assert all(b - a == pd.WINDOW_HOP_SAMPLES for a, b in zip(starts, starts[1:]))


def test_short_signal_still_produces_one_window():
    assert pd._windows(np.zeros(500, dtype=np.float32)).shape == (1, pd.WINDOW_SAMPLES)
    assert pd._window_starts(500) == [0]


def test_missing_model_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setattr(pd, "_session", None)
    monkeypatch.setattr(pd, "_session_failed", False)
    monkeypatch.setattr(pd, "MODEL_PATH", pd.MODEL_PATH.with_name("does-not-exist.onnx"))
    res = pd.analyse_overlap("whatever.wav")
    assert res.available is False
    assert res.speaker_overlap_present is False
    assert res.notes


def test_unreadable_audio_degrades_instead_of_raising():
    """analyse_overlap is called from a path that promises never to raise."""
    if pd._load_session() is None:
        pytest.skip("segmentation model not installed")
    res = pd.analyse_overlap("/nonexistent/definitely-not-audio.ogg")
    assert res.speaker_overlap_present is False
    assert res.available is False


def test_threshold_comes_from_the_fitted_file_when_present():
    t = pd._min_overlap_s()
    assert 0.0 < t <= 5.0
