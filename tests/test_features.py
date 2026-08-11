"""Tests for feature engineering, focused on causality."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lagos_flood.config import FeatureConfig
from lagos_flood.features.antecedent import (
    add_lag_features,
    add_rolling_accumulations,
    add_season_flags,
    antecedent_precipitation_index,
)
from lagos_flood.features.build import assert_causal_features, build_panel


@pytest.fixture
def dynamic() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2019-01-01", periods=900, freq="D")
    frames = []
    for lga in ("Ikeja", "Kosofe"):
        rain = rng.gamma(1.1, 4.0, len(dates)) * (rng.random(len(dates)) < 0.35)
        frames.append(
            pd.DataFrame(
                {
                    "lga": lga,
                    "date": dates,
                    "rain_mm": rain,
                    "soil_moisture": np.clip(np.cumsum(rain) % 1.0, 0, 1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_api_matches_the_recursive_definition():
    rain = pd.Series([10.0, 0.0, 5.0, 0.0])
    api = antecedent_precipitation_index(rain, k=0.5)
    # 10 → 5 → 5+2.5=7.5 → 3.75
    assert list(api.round(6)) == [10.0, 5.0, 7.5, 3.75]


def test_truncated_api_converges_on_the_untruncated_one():
    rng = np.random.default_rng(1)
    mean_rain = 3.0
    rain = pd.Series(rng.gamma(1.0, mean_rain, 400))
    k, window = 0.9, 120

    full = antecedent_precipitation_index(rain, k=k, window=None)
    truncated = antecedent_precipitation_index(rain, k=k, window=window)

    # Truncating at `window` drops the geometric tail, so the two forms differ by
    # about mean_rain * k**window / (1 - k) — roughly 1e-4 here, not zero. Assert
    # against that bound rather than an arbitrary tolerance.
    bound = mean_rain * k**window / (1 - k)
    difference = (full.iloc[200:] - truncated.iloc[200:]).abs().max()
    assert difference < 10 * bound
    assert np.allclose(full.iloc[200:], truncated.iloc[200:], rtol=1e-4)


def test_api_rejects_out_of_range_decay():
    with pytest.raises(ValueError, match="k must be in"):
        antecedent_precipitation_index(pd.Series([1.0]), k=1.4)


def test_api_is_computed_per_lga(dynamic):
    out = add_lag_features(dynamic, {"rain_mm": (1,)})
    first_rows = out.groupby("lga").head(1)
    # Each LGA's series starts fresh: no lag value carried over from the previous.
    assert first_rows["rain_mm_lag1"].isna().all()


def test_rolling_accumulation_includes_today_only(dynamic):
    out = add_rolling_accumulations(dynamic, "rain_mm", (3,))
    ikeja = out[out["lga"] == "Ikeja"].reset_index(drop=True)
    expected = ikeja["rain_mm"].iloc[7:10].sum()
    assert ikeja["rain_mm_sum3d"].iloc[9] == pytest.approx(expected)


def test_lag_features_do_not_cross_lga_boundaries(dynamic):
    out = add_lag_features(dynamic, {"rain_mm": (1, 2)}).sort_values(["lga", "date"])
    for _, grp in out.groupby("lga"):
        assert grp["rain_mm_lag1"].iloc[1:].to_numpy() == pytest.approx(
            grp["rain_mm"].iloc[:-1].to_numpy()
        )


def test_season_flags_match_the_lagos_regime():
    df = pd.DataFrame({"date": pd.to_datetime(["2022-07-15", "2022-10-15", "2022-01-15"])})
    out = add_season_flags(df)
    assert list(out["is_long_rains_jja"]) == [1, 0, 0]
    assert list(out["is_son_peak"]) == [0, 1, 0]
    assert list(out["is_wet_season"]) == [1, 1, 0]


def test_doy_encoding_wraps_around_new_year():
    df = pd.DataFrame({"date": pd.to_datetime(["2022-12-31", "2023-01-01"])})
    out = add_season_flags(df)
    # Adjacent days must be adjacent in the encoding, despite doy 365 → 1.
    delta = np.hypot(
        out["doy_sin"].iloc[0] - out["doy_sin"].iloc[1],
        out["doy_cos"].iloc[0] - out["doy_cos"].iloc[1],
    )
    assert delta < 0.05


def test_build_panel_drops_the_incomplete_lookback_head(dynamic):
    config = FeatureConfig()
    panel, features = build_panel(dynamic, None, config)
    assert len(panel) < len(dynamic)
    assert panel[features].notna().all().all()

    for _, grp in panel.groupby("lga"):
        first = grp["date"].min()
        assert (first - dynamic["date"].min()).days >= config.max_lookback_days - 1


def test_engineered_features_are_causal(dynamic):
    """The check that would catch a look-ahead feature slipping into the panel."""
    panel, features = build_panel(dynamic, None, FeatureConfig())
    assert_causal_features(panel, features, n_probes=6)


def test_causality_check_catches_a_leaked_feature(dynamic):
    panel, features = build_panel(dynamic, None, FeatureConfig())

    # A centred rolling mean reads the future — exactly the mistake to catch.
    panel = panel.sort_values(["lga", "date"]).copy()
    panel["rain_mm_sum7d"] = (
        panel.groupby("lga")["rain_mm"]
        .rolling(7, center=True, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    with pytest.raises(AssertionError, match="not causal"):
        assert_causal_features(panel, features, n_probes=12)


def test_max_lookback_tracks_the_longest_window():
    config = FeatureConfig(rain_accum_windows=(3, 90), api_window_days=30)
    assert config.max_lookback_days == 90
