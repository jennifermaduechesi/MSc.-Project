"""Tests for SFED → 4-class label construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lagos_flood.config import RISK_CLASSES, LabelConfig
from lagos_flood.labels import (
    LABEL_INT_COL,
    apply_thresholds,
    class_distribution,
    fit_thresholds,
    threshold_sensitivity,
)


@pytest.fixture
def sfed() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for lga, scale in (("Dry", 0.002), ("Wet", 0.05)):
        rows.append(
            pd.DataFrame(
                {
                    "lga": lga,
                    "date": pd.date_range("2020-01-01", periods=600, freq="D"),
                    "sfed_fraction": rng.gamma(0.7, scale, 600),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_absolute_thresholds_are_the_configured_constants(sfed):
    cfg = LabelConfig(strategy="absolute", absolute_cuts=(0.01, 0.05, 0.15))
    assert fit_thresholds(sfed, cfg).globally == (0.01, 0.05, 0.15)


def test_binning_is_left_closed_so_dry_days_are_low():
    df = pd.DataFrame({"lga": "X", "sfed_fraction": [0.0, 0.01, 0.011, 0.05, 0.2]})
    cfg = LabelConfig(strategy="absolute", absolute_cuts=(0.01, 0.05, 0.15))
    out = apply_thresholds(df, fit_thresholds(df, cfg))
    # 0.0 and exactly 0.01 are Low; 0.011 crosses into Medium.
    assert list(out[LABEL_INT_COL]) == [0, 0, 1, 1, 3]
    assert out["risk_class"].iloc[0] == "Low"
    assert out["risk_class"].iloc[-1] == "Critical"


def test_percentile_thresholds_split_roughly_as_requested(sfed):
    cfg = LabelConfig(strategy="percentile", percentile_cuts=(0.80, 0.95, 0.99))
    out = apply_thresholds(sfed, fit_thresholds(sfed, cfg))
    shares = class_distribution(out)["share"]
    assert shares["Low"] == pytest.approx(0.80, abs=0.02)
    assert shares["Critical"] == pytest.approx(0.01, abs=0.01)


def test_per_lga_thresholds_differ_between_wet_and_dry(sfed):
    cfg = LabelConfig(strategy="percentile", percentile_cuts=(0.80, 0.95, 0.99), per_lga=True)
    thresholds = fit_thresholds(sfed, cfg)
    assert thresholds.per_lga is not None
    assert thresholds.for_lga("Wet")[0] > thresholds.for_lga("Dry")[0]


def test_per_lga_thresholds_equalise_class_shares(sfed):
    """Per-LGA cuts should give each LGA a similar class mix; global cuts should not."""
    per_lga = apply_thresholds(
        sfed, fit_thresholds(sfed, LabelConfig("percentile", per_lga=True))
    )
    pooled = apply_thresholds(
        sfed, fit_thresholds(sfed, LabelConfig("percentile", per_lga=False))
    )

    def critical_share(df, lga):
        sub = df[df["lga"] == lga]
        return float((sub[LABEL_INT_COL] == 3).mean())

    per_lga_spread = abs(critical_share(per_lga, "Wet") - critical_share(per_lga, "Dry"))
    pooled_spread = abs(critical_share(pooled, "Wet") - critical_share(pooled, "Dry"))
    assert per_lga_spread < pooled_spread


def test_unknown_lga_raises_rather_than_guessing(sfed):
    thresholds = fit_thresholds(sfed, LabelConfig("percentile", per_lga=True))
    with pytest.raises(KeyError, match="no fitted threshold"):
        thresholds.for_lga("Ikorodu")


def test_tied_quantiles_are_nudged_apart():
    """A bone-dry LGA has identical quantiles; classes must not silently merge."""
    df = pd.DataFrame({"lga": "Dry", "sfed_fraction": np.zeros(500)})
    cuts = fit_thresholds(df, LabelConfig("percentile")).globally
    assert cuts is not None
    assert cuts[0] < cuts[1] < cuts[2]


def test_all_classes_are_representable(sfed):
    out = apply_thresholds(sfed, fit_thresholds(sfed, LabelConfig("percentile")))
    assert set(out["risk_class"].cat.categories) == set(RISK_CLASSES)


def test_sensitivity_table_covers_every_scheme(sfed):
    table = threshold_sensitivity(
        sfed,
        {
            "abs": LabelConfig("absolute"),
            "pct": LabelConfig("percentile"),
            "pct_lga": LabelConfig("percentile", per_lga=True),
        },
    )
    assert list(table.index) == ["abs", "pct", "pct_lga"]
    for cls in RISK_CLASSES:
        assert f"{cls}_share" in table.columns
    assert table[[f"{c}_share" for c in RISK_CLASSES]].sum(axis=1).round(3).eq(1.0).all()


def test_config_rejects_unusable_cut_points():
    with pytest.raises(ValueError, match="increasing"):
        LabelConfig(strategy="absolute", absolute_cuts=(0.1, 0.05, 0.2))
    with pytest.raises(ValueError, match="distinct"):
        LabelConfig(strategy="absolute", absolute_cuts=(0.1, 0.1, 0.2))
    with pytest.raises(ValueError, match="unknown labelling strategy"):
        LabelConfig(strategy="kmeans")
