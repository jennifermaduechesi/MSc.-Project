"""Assemble the modelling panel from per-LGA daily inputs.

Input is one row per (LGA, date) carrying raw dynamic variables — rainfall, soil
moisture, SFED flooded fraction — plus a static table of per-LGA terrain and
exposure attributes. Output is the same panel with engineered features attached
and a documented feature list.

The one rule this module enforces is that no feature may encode information from
after its own row's date. :func:`assert_causal_features` checks the property
directly rather than trusting the construction, because a leaked feature is
almost impossible to spot in aggregate metrics — it just makes everything look
good.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from .antecedent import (
    add_api,
    add_lag_features,
    add_rolling_accumulations,
    add_season_flags,
)

RAINFALL_COL = "rain_mm"
SOIL_MOISTURE_COL = "soil_moisture"

#: Per-LGA time-invariant attributes. Terrain drives where water collects;
#: building density and impervious fraction stand in for exposure and for the
#: drainage failures repeatedly identified as a Lagos-specific driver.
STATIC_FEATURES: tuple[str, ...] = (
    "elevation_mean",
    "slope_mean",
    "twi_mean",          # topographic wetness index
    "hand_mean",         # height above nearest drainage
    "dist_to_river_km",
    "building_density",  # GRID3 building footprints per km²
    "impervious_frac",
    "coastal_flag",
)


def build_panel(
    dynamic: pd.DataFrame,
    static: pd.DataFrame | None,
    config: FeatureConfig,
    *,
    lga_col: str = "lga",
    date_col: str = "date",
    rainfall_col: str = RAINFALL_COL,
    soil_moisture_col: str = SOIL_MOISTURE_COL,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(panel, feature_names)`` ready for modelling.

    Rows whose lookback window extends before the start of the record contain
    NaNs and are dropped, costing ``config.max_lookback_days`` at the head of
    each LGA's series.
    """
    panel = dynamic.sort_values([lga_col, date_col]).reset_index(drop=True)

    panel = add_rolling_accumulations(
        panel, rainfall_col, config.rain_accum_windows, lga_col=lga_col, date_col=date_col
    )
    panel = add_api(panel, rainfall_col, config, lga_col=lga_col, date_col=date_col)

    lag_spec: dict[str, tuple[int, ...]] = {rainfall_col: config.rain_lag_days}
    if soil_moisture_col in panel.columns:
        lag_spec[soil_moisture_col] = config.soil_moisture_lag_days
    panel = add_lag_features(panel, lag_spec, lga_col=lga_col, date_col=date_col)

    if config.add_season_flags:
        panel = add_season_flags(panel, date_col=date_col)

    if static is not None:
        available = [c for c in STATIC_FEATURES if c in static.columns]
        panel = panel.merge(static[[lga_col, *available]], on=lga_col, how="left")

    feature_names = _collect_feature_names(panel, config, rainfall_col, soil_moisture_col)
    panel = panel.dropna(subset=feature_names).reset_index(drop=True)
    return panel, feature_names


def _collect_feature_names(
    panel: pd.DataFrame,
    config: FeatureConfig,
    rainfall_col: str,
    soil_moisture_col: str,
) -> list[str]:
    names: list[str] = [rainfall_col, "api"]
    names += [f"{rainfall_col}_sum{w}d" for w in config.rain_accum_windows]
    names += [f"{rainfall_col}_lag{n}" for n in config.rain_lag_days]
    if soil_moisture_col in panel.columns:
        names.append(soil_moisture_col)
        names += [f"{soil_moisture_col}_lag{n}" for n in config.soil_moisture_lag_days]
    if config.add_season_flags:
        names += ["is_long_rains_jja", "is_son_peak", "is_wet_season", "doy_sin", "doy_cos"]
    names += [c for c in STATIC_FEATURES if c in panel.columns]
    return [n for n in names if n in panel.columns]


def assert_causal_features(
    panel: pd.DataFrame,
    feature_names: list[str],
    *,
    lga_col: str = "lga",
    date_col: str = "date",
    n_probes: int = 24,
    seed: int = 0,
) -> None:
    """Empirically check that no feature depends on the future.

    For a sample of rows, truncate the LGA's series at that row's date, rebuild
    the causal features from the truncated history, and confirm the values match
    those built from the full series. A feature that peeks ahead changes when the
    future is removed; a causal one does not.

    Only the recomputable dynamic features are probed — static and calendar
    columns cannot leak by construction.
    """
    from ..config import DEFAULT

    rng = np.random.default_rng(seed)
    dynamic_cols = [
        f for f in feature_names
        if f not in STATIC_FEATURES
        and not f.startswith(("is_", "doy_"))
    ]
    if not dynamic_cols:
        return

    panel = panel.sort_values([lga_col, date_col]).reset_index(drop=True)
    eligible = panel.index.to_numpy()
    probes = rng.choice(eligible, size=min(n_probes, len(eligible)), replace=False)

    for pos in probes:
        row = panel.loc[pos]
        history = panel[(panel[lga_col] == row[lga_col]) & (panel[date_col] <= row[date_col])]
        if len(history) <= DEFAULT.features.max_lookback_days:
            continue

        rebuilt, rebuilt_names = build_panel(
            history[[lga_col, date_col, RAINFALL_COL]
                    + ([SOIL_MOISTURE_COL] if SOIL_MOISTURE_COL in history else [])].copy(),
            None,
            DEFAULT.features,
            lga_col=lga_col,
            date_col=date_col,
        )
        if rebuilt.empty:
            continue
        last = rebuilt.iloc[-1]

        for col in rebuilt_names:
            if col not in dynamic_cols:
                continue
            full_value = float(row[col])
            trunc_value = float(last[col])
            if not np.isclose(full_value, trunc_value, rtol=1e-6, atol=1e-8):
                raise AssertionError(
                    f"feature {col!r} is not causal: rebuilt from history alone it is "
                    f"{trunc_value:.6g}, but the full-panel value is {full_value:.6g} "
                    f"({row[lga_col]}, {pd.Timestamp(row[date_col]).date()})"
                )


__all__ = [
    "RAINFALL_COL",
    "SOIL_MOISTURE_COL",
    "STATIC_FEATURES",
    "assert_causal_features",
    "build_panel",
]
