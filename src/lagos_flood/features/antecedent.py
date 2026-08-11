"""Antecedent wetness features.

Whether a given rainfall event floods depends heavily on how saturated the ground
already was. Kim et al. (2019) put the size of that effect starkly: in the Napa
River basin a 7-year rainfall falling on saturated soil produced a 100-year
flood, while a 200-year rainfall on dry soil produced only a ~15-year one. Storm
totals alone therefore cannot separate the classes, and the antecedent state has
to enter the feature set explicitly.

Two complementary encodings are provided: the Antecedent Precipitation Index, a
decaying memory of past rainfall (Heggen, 2001; Schoener & Stone, 2020), and
plain lagged/rolling accumulations for the model to combine itself.

Every function here is causal. Each is computed within an LGA, on date-sorted
rows, using only values at or before the row's own date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureConfig


def antecedent_precipitation_index(
    rainfall: pd.Series,
    k: float = 0.90,
    window: int | None = None,
) -> pd.Series:
    """Recursive API: ``API_t = k * API_{t-1} + P_t``.

    ``k`` sets how fast the soil is treated as forgetting past rainfall; values
    near 1 hold moisture longer. Passing ``window`` truncates the memory to a
    finite sum, which is what makes the result reproducible from a fixed lookback
    rather than from the whole history — the truncated form is what the temporal
    embargo is sized against.
    """
    if not 0.0 < k < 1.0:
        raise ValueError(f"decay constant k must be in (0, 1), got {k}")

    values = rainfall.to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)

    if window is None:
        out = np.empty_like(values)
        running = 0.0
        for i, p in enumerate(values):
            running = k * running + p
            out[i] = running
        return pd.Series(out, index=rainfall.index, name="api")

    # Truncated form: a weighted sum over the last `window` days, weight k**lag.
    weights = k ** np.arange(window)
    padded = np.concatenate([np.zeros(window - 1), values])
    strided = np.lib.stride_tricks.sliding_window_view(padded, window)
    out = (strided * weights[::-1]).sum(axis=1)
    return pd.Series(out, index=rainfall.index, name="api")


def add_lag_features(
    df: pd.DataFrame,
    columns: dict[str, tuple[int, ...]],
    *,
    lga_col: str = "lga",
    date_col: str = "date",
) -> pd.DataFrame:
    """Add ``{col}_lag{n}`` for each requested lag, grouped within LGA."""
    out = df.sort_values([lga_col, date_col]).copy()
    grouped = out.groupby(lga_col, sort=False)
    new: dict[str, pd.Series] = {}
    for col, lags in columns.items():
        _require(out, col)
        for lag in lags:
            new[f"{col}_lag{lag}"] = grouped[col].shift(lag)
    return out.assign(**new)


def add_rolling_accumulations(
    df: pd.DataFrame,
    column: str,
    windows: tuple[int, ...],
    *,
    lga_col: str = "lga",
    date_col: str = "date",
    stat: str = "sum",
) -> pd.DataFrame:
    """Add ``{column}_{stat}{w}d`` rolling statistics ending at the current row.

    The window is inclusive of the current day, which is legitimate: the target
    sits at ``t + lead_time_days``, so today's rainfall is known at forecast time.
    """
    out = df.sort_values([lga_col, date_col]).copy()
    _require(out, column)
    grouped = out.groupby(lga_col, sort=False)[column]
    new: dict[str, pd.Series] = {}
    for w in windows:
        rolled = grouped.rolling(window=w, min_periods=w).agg(stat)
        new[f"{column}_{stat}{w}d"] = rolled.reset_index(level=0, drop=True)
    return out.assign(**new)


def add_api(
    df: pd.DataFrame,
    rainfall_col: str,
    config: FeatureConfig,
    *,
    lga_col: str = "lga",
    date_col: str = "date",
    out_col: str = "api",
) -> pd.DataFrame:
    """Add a truncated API column computed independently per LGA."""
    out = df.sort_values([lga_col, date_col]).copy()
    _require(out, rainfall_col)
    out[out_col] = (
        out.groupby(lga_col, sort=False)[rainfall_col]
        .transform(
            lambda s: antecedent_precipitation_index(
                s, k=config.api_decay_k, window=config.api_window_days
            )
        )
    )
    return out


def add_season_flags(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    """Flag the two Lagos rainfall peaks and encode day-of-year cyclically.

    Aniramu and Orimoogunje (2025) tie high inundation in coastal Lagos to the
    JJA long rains and the SON peak. Raw day-of-year is a poor feature for a tree
    because 31 December and 1 January land at opposite ends of the split range,
    so it is encoded as a sine/cosine pair as well.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    month = dates.dt.month
    doy = dates.dt.dayofyear

    out["is_long_rains_jja"] = month.isin([6, 7, 8]).astype(np.int8)
    out["is_son_peak"] = month.isin([9, 10, 11]).astype(np.int8)
    out["is_wet_season"] = month.isin([4, 5, 6, 7, 8, 9, 10]).astype(np.int8)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def _require(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise KeyError(f"column {column!r} not in panel; have {sorted(df.columns)[:12]}…")


__all__ = [
    "add_api",
    "add_lag_features",
    "add_rolling_accumulations",
    "add_season_flags",
    "antecedent_precipitation_index",
]
