"""Time-ordered validation for a spatiotemporal panel.

The panel is one row per (LGA, date). Two properties make naive cross-validation
dishonest here:

*Temporal autocorrelation.* Yesterday's flooded fraction predicts today's almost
regardless of the model. A random fold puts 14 June in training and 15 June in
validation, and the model scores well by memorising the neighbourhood rather
than by forecasting (Roberts et al., 2017; Bergmeir & Benítez, 2012).

*Shared feature windows.* A 30-day rainfall accumulation computed for 1 July
overlaps the one computed for 20 June. Even a clean date cutoff leaks unless
validation starts far enough after training ends that no validation sample's
lookback window reaches back into training. That distance is the embargo.

Every splitter below is time-ordered and embargoed. There is deliberately no
shuffled splitter in this module.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATE_COL = "date"


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold, carrying positional indices into the panel."""

    index: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Fold({self.index}: train≤{self.train_end.date()} "
            f"| valid {self.valid_start.date()}→{self.valid_end.date()} "
            f"| n_train={len(self.train_idx)} n_valid={len(self.valid_idx)})"
        )


def temporal_holdout(
    df: pd.DataFrame,
    test_start: str | pd.Timestamp,
    *,
    date_col: str = DATE_COL,
    embargo_days: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split into development and final-test positional indices.

    The test set is touched **once**, after model selection is finished. The
    embargo is removed from the *end of training*, not from the test set, so the
    test period stays intact and reported metrics cover it completely.
    """
    dates = pd.to_datetime(df[date_col])
    cutoff = pd.Timestamp(test_start)
    train_end = cutoff - pd.Timedelta(days=embargo_days)

    train_mask = (dates < train_end).to_numpy()
    test_mask = (dates >= cutoff).to_numpy()

    if not train_mask.any():
        raise ValueError(f"no training rows before {train_end.date()}")
    if not test_mask.any():
        raise ValueError(f"no test rows on or after {cutoff.date()}")

    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def expanding_window_folds(
    df: pd.DataFrame,
    n_folds: int,
    *,
    date_col: str = DATE_COL,
    embargo_days: int = 0,
    min_train_days: int | None = None,
) -> Iterator[Fold]:
    """Yield ``n_folds`` expanding-origin folds over the development period.

    Training always starts at the beginning of the record and grows; validation
    is the next contiguous block after the embargo. Fold *k*'s validation period
    is strictly later than fold *k-1*'s, so the sequence mimics repeatedly
    retraining an operational model as time passes.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    dates = pd.to_datetime(df[date_col])
    unique_days = np.array(sorted(dates.unique()))
    n_days = len(unique_days)

    span = n_days // (n_folds + 1)
    if min_train_days is None:
        min_train_days = span
    if span <= embargo_days:
        raise ValueError(
            f"each fold spans {span} days but the embargo is {embargo_days} days, "
            f"leaving no validation window. Use fewer folds, a longer record, or "
            f"shorter feature lookbacks."
        )

    dates_np = dates.to_numpy()
    for k in range(n_folds):
        train_end_pos = min_train_days + k * span
        valid_start_pos = train_end_pos + embargo_days
        valid_end_pos = min(valid_start_pos + span, n_days)
        if valid_start_pos >= n_days or valid_start_pos >= valid_end_pos:
            break

        train_end = pd.Timestamp(unique_days[train_end_pos - 1])
        valid_start = pd.Timestamp(unique_days[valid_start_pos])
        valid_end = pd.Timestamp(unique_days[valid_end_pos - 1])

        train_idx = np.flatnonzero(dates_np <= np.datetime64(train_end))
        valid_idx = np.flatnonzero(
            (dates_np >= np.datetime64(valid_start)) & (dates_np <= np.datetime64(valid_end))
        )
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue

        yield Fold(
            index=k,
            train_idx=train_idx,
            valid_idx=valid_idx,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
        )


def assert_no_temporal_leakage(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    *,
    date_col: str = DATE_COL,
    embargo_days: int = 0,
) -> None:
    """Raise if any training date is within ``embargo_days`` of a validation date.

    Cheap enough to call on every fold. Worth calling on every fold — a silent
    leak inflates the headline number and is invisible in the metrics.
    """
    dates = pd.to_datetime(df[date_col]).to_numpy()
    train_max = dates[train_idx].max()
    valid_min = dates[valid_idx].min()

    if train_max >= valid_min:
        raise AssertionError(
            f"temporal leakage: training extends to {pd.Timestamp(train_max).date()} "
            f"but validation starts {pd.Timestamp(valid_min).date()}"
        )

    gap = (pd.Timestamp(valid_min) - pd.Timestamp(train_max)).days
    if gap <= embargo_days:
        raise AssertionError(
            f"embargo too short: {gap} day(s) between training end and validation "
            f"start, but features look back {embargo_days} day(s), so validation "
            f"feature windows overlap the training period"
        )


def build_forecast_target(
    df: pd.DataFrame,
    lead_time_days: int,
    *,
    label_col: str,
    lga_col: str = "lga",
    date_col: str = DATE_COL,
    target_col: str = "target",
) -> pd.DataFrame:
    """Shift the label backwards by ``lead_time_days`` to make it a forecast.

    After this, row (LGA, t) carries features observed up to t and the class that
    actually occurred at t + lead. Rows at the end of each LGA's series have no
    future label and are dropped. Without this step the "model" is a nowcast that
    reads the answer off contemporaneous SFED.
    """
    if lead_time_days < 1:
        raise ValueError("lead_time_days must be >= 1 for a forecasting task")

    out = df.sort_values([lga_col, date_col]).copy()
    out[target_col] = out.groupby(lga_col, sort=False)[label_col].shift(-lead_time_days)
    return out.dropna(subset=[target_col]).reset_index(drop=True)


__all__ = [
    "DATE_COL",
    "Fold",
    "assert_no_temporal_leakage",
    "build_forecast_target",
    "expanding_window_folds",
    "temporal_holdout",
]
