"""Integration tests: the whole pipeline on a small synthetic panel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lagos_flood.config import Config, LabelConfig, ModelConfig, ValidationConfig
from lagos_flood.data.synthetic import make_synthetic_panel
from lagos_flood.evaluate import evaluate, per_lga_breakdown
from lagos_flood.labels import LABEL_INT_COL
from lagos_flood.models import PERSISTENCE_FEATURE
from lagos_flood.pipeline import TARGET_COL, prepare_dataset, run_experiment
from lagos_flood.validation import temporal_holdout


@pytest.fixture(scope="module")
def small_config() -> Config:
    return Config(
        label=LabelConfig(strategy="percentile", percentile_cuts=(0.80, 0.95, 0.99)),
        validation=ValidationConfig(test_start="2022-01-01", n_folds=3, lead_time_days=1),
        model=ModelConfig(n_estimators=40, random_state=0, n_jobs=1),
    )


@pytest.fixture(scope="module")
def prepared(small_config):
    dynamic, static = make_synthetic_panel("2018-01-01", "2022-12-31", seed=3, lgas=("Ikeja", "Eti-Osa", "Epe"))
    return prepare_dataset(dynamic, static, small_config, check_causality=False)


def test_synthetic_panel_has_the_expected_shape():
    dynamic, static = make_synthetic_panel("2020-01-01", "2020-12-31", seed=1)
    assert len(static) == 20
    assert len(dynamic) == 20 * 366
    assert dynamic["sfed_fraction"].between(0, 1).all()
    assert (dynamic["rain_mm"] >= 0).all()


def test_synthetic_rainfall_is_seasonal():
    dynamic, _ = make_synthetic_panel("2018-01-01", "2022-12-31", seed=1)
    by_month = dynamic.groupby(dynamic["date"].dt.month)["rain_mm"].mean()
    # The long rains must out-rain the dry season by a wide margin.
    assert by_month.loc[[6, 7, 8]].mean() > 3 * by_month.loc[[1, 2, 12]].mean()


def test_prepare_dataset_produces_a_forecast_target(prepared):
    panel, features = prepared
    assert TARGET_COL in panel.columns
    assert panel[TARGET_COL].between(0, 3).all()
    assert len(features) > 10
    assert panel[features].notna().all().all()


def test_labels_are_fitted_on_development_data_only():
    """Flooding the test period must not move the class boundaries."""
    config = Config(
        label=LabelConfig(strategy="percentile"),
        validation=ValidationConfig(test_start="2022-01-01", n_folds=2),
    )
    dynamic, static = make_synthetic_panel("2018-01-01", "2022-12-31", seed=5, lgas=("Ikeja",))
    baseline, _ = prepare_dataset(dynamic, static, config, check_causality=False)

    flooded = dynamic.copy()
    flooded.loc[flooded["date"] >= pd.Timestamp("2022-01-01"), "sfed_fraction"] = 0.9
    shifted, _ = prepare_dataset(flooded, static, config, check_causality=False)

    dev = pd.Timestamp("2022-01-01")
    # Compare the labels themselves, not the forward-shifted target: a target on
    # 2021-12-31 is by definition drawn from 2022-01-01, so it is *expected* to
    # move. The class boundaries are what must not.
    before_base = baseline[baseline["date"] < dev][LABEL_INT_COL].to_numpy()
    before_shift = shifted[shifted["date"] < dev][LABEL_INT_COL].to_numpy()
    assert np.array_equal(before_base, before_shift), (
        "development-period labels changed when only the test period changed — "
        "thresholds are leaking from the held-out data"
    )


def test_embargo_keeps_test_period_targets_out_of_training():
    """No training row may carry a label drawn from the held-out test period.

    A training row at date t carries the class at t + lead. Without an embargo
    that clears the lead time, rows near the cutoff pull their answer out of the
    test set — leakage through the target rather than through the features, so
    nothing that inspects X would catch it.
    """
    config = Config(
        label=LabelConfig(strategy="absolute"),
        validation=ValidationConfig(test_start="2022-01-01", n_folds=2, lead_time_days=14),
    )
    # The derived embargo must clear the lead time, not just the feature window.
    assert config.embargo_days >= config.validation.lead_time_days

    dynamic, static = make_synthetic_panel("2018-01-01", "2022-12-31", seed=9, lgas=("Ikeja",))
    panel, _ = prepare_dataset(dynamic, static, config, check_causality=False)

    dev_idx, _ = temporal_holdout(panel, config.validation.test_start, embargo_days=config.embargo_days)
    train_rows = panel.iloc[dev_idx]
    latest_target_date = train_rows["date"].max() + pd.Timedelta(days=config.validation.lead_time_days)

    assert latest_target_date < pd.Timestamp(config.validation.test_start), (
        f"a training row's target lands on {latest_target_date.date()}, inside the "
        f"test period starting {config.validation.test_start}"
    )


def test_run_experiment_completes_and_scores(prepared, small_config):
    panel, features = prepared
    result = run_experiment(panel, features, "random_forest", small_config)

    assert len(result.fold_results) >= 1
    assert 0.0 <= result.test_result.macro_f1 <= 1.0
    assert result.test_result.confusion.to_numpy().sum() == result.test_result.n_samples
    assert set(result.per_lga["lga"]) == {"Ikeja", "Eti-Osa", "Epe"}
    assert "macro_f1" in result.cv_summary.index
    assert result.report()


def test_model_beats_a_majority_class_baseline(prepared, small_config):
    """A sanity floor: the pipeline should carry real signal, not just noise."""
    panel, features = prepared
    result = run_experiment(panel, features, "random_forest", small_config)

    y_true = result.test_frame[TARGET_COL].to_numpy()
    majority = np.full_like(y_true, np.bincount(y_true).argmax())
    baseline = evaluate(y_true, majority)

    assert result.test_result.macro_f1 > baseline.macro_f1


def test_evaluate_keeps_absent_classes_in_the_macro_average():
    """A class that never appears must count as 0, not vanish from the denominator."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    result = evaluate(y_true, y_pred)

    # Perfect on the two classes present, but High and Critical score zero.
    assert result.macro_f1 == pytest.approx(0.5)
    assert result.per_class.loc["Critical", "f1"] == 0.0
    assert result.confusion.shape == (4, 4)


def test_compare_models_flags_whether_persistence_was_beaten(prepared, small_config):
    """The gate the brief calls essential has to be answered in the table itself."""
    from lagos_flood.pipeline import compare_models

    panel, features = prepared
    table, results = compare_models(
        panel, features, model_names=("persistence", "random_forest"), config=small_config
    )

    assert set(table["model"]) == {"persistence", "random_forest"}
    assert "beats_persistence" in table.columns

    # The persistence row holds NA, not a self-comparison.
    persistence_row = table[table["model"] == "persistence"]["beats_persistence"]
    assert persistence_row.isna().all()

    # And the RF verdict matches the macro-F1 numbers it was derived from.
    floor = results["persistence"].test_result.macro_f1
    rf = table[table["model"] == "random_forest"].iloc[0]
    assert bool(rf["beats_persistence"]) == (rf["test_macro_f1"] > floor)


def test_persistence_runs_through_the_full_pipeline(prepared, small_config):
    """Persistence must score on exactly the same split as every other model."""
    panel, features = prepared
    result = run_experiment(panel, features, "persistence", small_config)

    assert 0.0 <= result.test_result.macro_f1 <= 1.0
    assert result.test_result.n_samples > 0
    # It predicts the current class verbatim, so predictions match that column.
    current = result.test_frame[PERSISTENCE_FEATURE].to_numpy().round().astype(int)
    assert np.array_equal(result.test_frame["prediction"].to_numpy(), current)


def test_pipeline_rejects_a_panel_without_the_persistence_feature(prepared, small_config):
    panel, features = prepared
    stripped = [f for f in features if f != PERSISTENCE_FEATURE]
    with pytest.raises(ValueError, match="persistence baseline has nothing to read"):
        run_experiment(panel, stripped, "random_forest", small_config)


def test_risk_history_features_are_present_and_causal(prepared):
    """Both risk-history features are built from labels at or before time t."""
    panel, features = prepared
    assert PERSISTENCE_FEATURE in features
    assert "risk_days_elevated_30d" in features

    # risk_class_current is exactly the label at t, not the target at t+h.
    assert np.array_equal(
        panel[PERSISTENCE_FEATURE].to_numpy().round().astype(int),
        panel[LABEL_INT_COL].to_numpy().astype(int),
    )
    # The 30-day elevated count can never exceed its own window.
    assert panel["risk_days_elevated_30d"].between(0, 30).all()


def test_per_lga_breakdown_sorts_worst_first():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 1, 0, 3, 2])
    lgas = np.array(["Good"] * 4 + ["Bad"] * 4)
    table = per_lga_breakdown(y_true, y_pred, lgas)

    assert table.iloc[0]["lga"] == "Bad"
    assert table.iloc[-1]["lga"] == "Good"
    assert table.iloc[-1]["macro_f1"] == pytest.approx(1.0)
