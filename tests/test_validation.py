"""Tests for the temporal validation guarantees.

These are the tests that matter most. A bug in the model code produces bad
numbers; a bug here produces *good* numbers that are wrong, which is worse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lagos_flood.validation import (
    assert_no_temporal_leakage,
    build_forecast_target,
    expanding_window_folds,
    temporal_holdout,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    frames = [
        pd.DataFrame({"lga": lga, "date": dates, "value": np.arange(len(dates), dtype=float)})
        for lga in ("Ikeja", "Kosofe", "Eti-Osa")
    ]
    return pd.concat(frames, ignore_index=True)


def test_holdout_separates_on_date(panel):
    train_idx, test_idx = temporal_holdout(panel, "2023-01-01")
    train_max = panel.iloc[train_idx]["date"].max()
    test_min = panel.iloc[test_idx]["date"].min()
    assert train_max < test_min
    assert test_min == pd.Timestamp("2023-01-01")


def test_holdout_embargo_shrinks_training_not_test(panel):
    _, test_plain = temporal_holdout(panel, "2023-01-01", embargo_days=0)
    train_emb, test_emb = temporal_holdout(panel, "2023-01-01", embargo_days=30)

    # The test period is untouched by the embargo...
    assert len(test_plain) == len(test_emb)
    # ...and training now stops 30 days earlier.
    assert panel.iloc[train_emb]["date"].max() <= pd.Timestamp("2022-12-01")


def test_holdout_rejects_empty_side(panel):
    with pytest.raises(ValueError, match="no test rows"):
        temporal_holdout(panel, "2030-01-01")
    with pytest.raises(ValueError, match="no training rows"):
        temporal_holdout(panel, "2019-01-01")


def test_folds_are_time_ordered_and_expanding(panel):
    folds = list(expanding_window_folds(panel, n_folds=4, embargo_days=7))
    assert len(folds) >= 2

    for fold in folds:
        assert fold.train_end < fold.valid_start
        assert fold.valid_start <= fold.valid_end

    # strict=False is deliberate: pairwise iteration over consecutive folds.
    for earlier, later in zip(folds, folds[1:], strict=False):
        # Training grows, and each validation block sits strictly after the last.
        assert len(later.train_idx) > len(earlier.train_idx)
        assert later.valid_start > earlier.valid_start


def test_folds_respect_embargo(panel):
    embargo = 21
    for fold in expanding_window_folds(panel, n_folds=3, embargo_days=embargo):
        gap = (fold.valid_start - fold.train_end).days
        assert gap > embargo, f"gap {gap} does not clear the {embargo}-day embargo"


def test_folds_reject_impossible_embargo(panel):
    with pytest.raises(ValueError, match="embargo"):
        list(expanding_window_folds(panel, n_folds=200, embargo_days=60))


def test_leakage_detector_catches_overlap(panel):
    dates = pd.to_datetime(panel["date"]).to_numpy()
    train_idx = np.flatnonzero(dates <= np.datetime64("2022-06-30"))
    overlapping = np.flatnonzero(dates >= np.datetime64("2022-06-01"))

    with pytest.raises(AssertionError, match="temporal leakage"):
        assert_no_temporal_leakage(panel, train_idx, overlapping)


def test_leakage_detector_catches_short_embargo(panel):
    dates = pd.to_datetime(panel["date"]).to_numpy()
    train_idx = np.flatnonzero(dates <= np.datetime64("2022-06-30"))
    valid_idx = np.flatnonzero(
        (dates >= np.datetime64("2022-07-03")) & (dates <= np.datetime64("2022-08-01"))
    )
    # Three days of separation, but features look back thirty.
    with pytest.raises(AssertionError, match="embargo too short"):
        assert_no_temporal_leakage(panel, train_idx, valid_idx, embargo_days=30)

    assert_no_temporal_leakage(panel, train_idx, valid_idx, embargo_days=2)


def test_forecast_target_looks_forward(panel):
    labelled = panel.assign(label=(panel["value"] % 4).astype(int))
    shifted = build_forecast_target(labelled, 3, label_col="label")

    row = shifted.iloc[0]
    future = labelled[
        (labelled["lga"] == row["lga"])
        & (labelled["date"] == row["date"] + pd.Timedelta(days=3))
    ]
    assert row["target"] == future["label"].iloc[0]


def test_forecast_target_does_not_bleed_across_lgas(panel):
    labelled = panel.assign(label=(panel["value"] % 4).astype(int))
    shifted = build_forecast_target(labelled, 2, label_col="label")

    # Each LGA loses exactly `lead` rows off the end of its own series.
    per_lga = shifted.groupby("lga").size()
    original = labelled.groupby("lga").size()
    assert (original - per_lga == 2).all()


def test_forecast_target_rejects_nowcast(panel):
    with pytest.raises(ValueError, match="lead_time_days must be >= 1"):
        build_forecast_target(panel.assign(label=0), 0, label_col="label")
