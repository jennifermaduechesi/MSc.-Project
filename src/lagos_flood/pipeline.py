"""End-to-end experiment pipeline.

Step order here is load → engineer features → *split* → fit label thresholds →
label → build forecast target → cross-validate → test once. The position of the
split matters: percentile label thresholds are fitted from the data, so fitting
them before splitting would let the test period's flood distribution decide where
the class boundaries fall. That is leakage, and because it acts through the
labels rather than the features it survives every check applied to X.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEFAULT, Config
from .evaluate import EvaluationResult, aggregate_folds, evaluate, per_lga_breakdown
from .features.build import assert_causal_features, build_panel
from .labels import LABEL_INT_COL, apply_thresholds, class_distribution, fit_thresholds
from .models import PERSISTENCE_FEATURE, build_model, sample_weights_balanced
from .validation import (
    assert_no_temporal_leakage,
    build_forecast_target,
    expanding_window_folds,
    temporal_holdout,
)

log = logging.getLogger(__name__)

TARGET_COL = "target"


@dataclass
class ExperimentResult:
    model_name: str
    cv_summary: pd.DataFrame
    fold_results: list[EvaluationResult]
    test_result: EvaluationResult
    per_lga: pd.DataFrame
    feature_names: list[str]
    class_balance: pd.DataFrame
    fitted_model: object = None
    test_frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    def report(self) -> str:
        lines = [
            f"=== {self.model_name} ===",
            f"features: {len(self.feature_names)}",
            "",
            "class balance (development period):",
            self.class_balance.to_string(),
            "",
            f"cross-validation over {len(self.fold_results)} time-ordered folds:",
            self.cv_summary.round(4).to_string(),
            "",
            "held-out test period:",
            f"  {self.test_result.summary()}",
            "",
            "per class:",
            self.test_result.per_class.round(3).to_string(),
            "",
            "confusion matrix:",
            self.test_result.confusion.to_string(),
            "",
            "weakest LGAs by macro-F1:",
            self.per_lga.head(5).round(3).to_string(index=False),
        ]
        return "\n".join(lines)


def prepare_dataset(
    dynamic: pd.DataFrame,
    static: pd.DataFrame | None,
    config: Config = DEFAULT,
    *,
    check_causality: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Engineer features, label, and turn the labels into a forecast target."""
    panel, feature_names = build_panel(dynamic, static, config.features)
    log.info("panel: %d rows, %d features", len(panel), len(feature_names))

    if check_causality:
        assert_causal_features(panel, feature_names)
        log.info("causality check passed on sampled rows")

    # Fit thresholds on the development period only, then apply everywhere.
    dev_mask = pd.to_datetime(panel["date"]) < pd.Timestamp(config.validation.test_start)
    if not dev_mask.any():
        raise ValueError(
            f"no rows before test_start={config.validation.test_start}; "
            "check the date range of the input panel"
        )
    thresholds = fit_thresholds(panel.loc[dev_mask], config.label)
    panel = apply_thresholds(panel, thresholds)

    # Risk-history features, added after labelling because they are built from the
    # labels themselves. Both are known at time t and the target sits at t+h, so
    # both are causal: today's risk class is observed, tomorrow's is not.
    #
    # risk_class_current is what the persistence baseline predicts from. Giving it
    # to every model too is deliberate — the brief lists an "LGA risk-history flag"
    # as a planned feature, and it makes the comparison strictly fair, since the ML
    # models then hold everything persistence holds and more.
    panel = panel.sort_values(["lga", "date"]).reset_index(drop=True)
    panel[PERSISTENCE_FEATURE] = panel[LABEL_INT_COL].astype(float)

    elevated = (panel[LABEL_INT_COL] >= 2).astype(float)
    panel["risk_days_elevated_30d"] = (
        elevated.groupby(panel["lga"])
        .rolling(30, min_periods=30)
        .sum()
        .reset_index(level=0, drop=True)
    )

    feature_names = [*feature_names, PERSISTENCE_FEATURE, "risk_days_elevated_30d"]
    panel = panel.dropna(subset=feature_names).reset_index(drop=True)

    panel = build_forecast_target(
        panel, config.validation.lead_time_days, label_col=LABEL_INT_COL, target_col=TARGET_COL
    )
    panel[TARGET_COL] = panel[TARGET_COL].astype(int)
    return panel, feature_names


def run_experiment(
    panel: pd.DataFrame,
    feature_names: list[str],
    model_name: str = "random_forest",
    config: Config = DEFAULT,
) -> ExperimentResult:
    """Cross-validate on the development period, then score the test period once."""
    embargo = config.embargo_days
    dev_idx, test_idx = temporal_holdout(
        panel, config.validation.test_start, embargo_days=embargo
    )
    dev = panel.iloc[dev_idx].reset_index(drop=True)
    test = panel.iloc[test_idx].reset_index(drop=True)
    log.info("development: %d rows | test: %d rows | embargo %d days", len(dev), len(test), embargo)

    X_dev = dev[feature_names].to_numpy(dtype=float)
    y_dev = dev[TARGET_COL].to_numpy(dtype=int)

    if PERSISTENCE_FEATURE not in feature_names:
        raise ValueError(
            f"{PERSISTENCE_FEATURE!r} is missing from the feature list, so the "
            "persistence baseline has nothing to read. Build the panel with "
            "prepare_dataset()."
        )
    persistence_idx = feature_names.index(PERSISTENCE_FEATURE)

    fold_results: list[EvaluationResult] = []
    for fold in expanding_window_folds(dev, config.validation.n_folds, embargo_days=embargo):
        assert_no_temporal_leakage(dev, fold.train_idx, fold.valid_idx, embargo_days=embargo)
        model = build_model(model_name, config.model, persistence_column_index=persistence_idx)
        _fit(model, X_dev[fold.train_idx], y_dev[fold.train_idx], model_name)
        preds = model.predict(X_dev[fold.valid_idx])
        result = evaluate(y_dev[fold.valid_idx], preds)
        fold_results.append(result)
        log.info("fold %d | %s | %s", fold.index, fold, result.summary())

    if not fold_results:
        raise RuntimeError(
            "no usable folds — the development period is too short for "
            f"{config.validation.n_folds} folds with a {embargo}-day embargo"
        )

    # Refit on the whole development period, then touch the test set once.
    final_model = build_model(model_name, config.model, persistence_column_index=persistence_idx)
    _fit(final_model, X_dev, y_dev, model_name)

    X_test = test[feature_names].to_numpy(dtype=float)
    y_test = test[TARGET_COL].to_numpy(dtype=int)
    test_preds = final_model.predict(X_test)
    test_result = evaluate(y_test, test_preds)

    test_frame = test.copy()
    test_frame["prediction"] = test_preds

    return ExperimentResult(
        model_name=model_name,
        cv_summary=aggregate_folds(fold_results),
        fold_results=fold_results,
        test_result=test_result,
        per_lga=per_lga_breakdown(y_test, test_preds, test["lga"].to_numpy()),
        feature_names=feature_names,
        class_balance=class_distribution(dev.assign(risk_class=dev[TARGET_COL].map(_int_to_class()))),
        fitted_model=final_model,
        test_frame=test_frame,
    )


def compare_models(
    panel: pd.DataFrame,
    feature_names: list[str],
    model_names: tuple[str, ...] = (
        "persistence",
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    ),
    config: Config = DEFAULT,
) -> tuple[pd.DataFrame, dict[str, ExperimentResult]]:
    """Run several models over identical splits and tabulate them side by side.

    Persistence runs first and by default. The output carries a
    ``beats_persistence`` column so the comparison the brief calls essential is
    answered in the table itself rather than left to the reader — a model that
    cannot beat "assume tomorrow looks like today" has not earned a write-up,
    and that has to be visible.

    A model that is unavailable (optional dependency missing) is skipped with a
    warning rather than aborting the comparison.
    """
    results: dict[str, ExperimentResult] = {}
    rows = []
    for name in model_names:
        try:
            result = run_experiment(panel, feature_names, name, config)
        except ImportError as exc:
            log.warning("skipping %s: %s", name, exc)
            continue
        results[name] = result
        rows.append(
            {
                "model": name,
                "cv_macro_f1_mean": result.cv_summary.loc["macro_f1", "mean"],
                "cv_macro_f1_std": result.cv_summary.loc["macro_f1", "std"],
                "test_macro_f1": result.test_result.macro_f1,
                "test_critical_recall": result.test_result.critical_recall,
                "test_quadratic_kappa": result.test_result.quadratic_kappa,
            }
        )
    if not rows:
        raise RuntimeError("no models could be run — install at least scikit-learn")

    table = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    if "persistence" in results:
        floor = results["persistence"].test_result.macro_f1
        # Nullable boolean, not plain bool: the persistence row itself holds NA
        # rather than a self-comparison, and plain bool cannot store NA.
        table["beats_persistence"] = (table["test_macro_f1"] > floor).astype("boolean")
        table.loc[table["model"] == "persistence", "beats_persistence"] = pd.NA
    return table, results


def _fit(model, X: np.ndarray, y: np.ndarray, model_name: str) -> None:
    """Fit, routing imbalance handling to whichever mechanism the model supports."""
    if model_name.lower() in {"xgboost", "xgb"}:
        model.fit(X, y, sample_weight=sample_weights_balanced(y))
    else:
        model.fit(X, y)


def _int_to_class() -> dict[int, str]:
    from .config import INT_TO_CLASS

    return INT_TO_CLASS


__all__ = ["ExperimentResult", "TARGET_COL", "compare_models", "prepare_dataset", "run_experiment"]
