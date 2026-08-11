"""Tests for the model factory, focused on the persistence baseline.

Persistence is the gate every other model has to clear, so it needs to be
provably correct — a baseline that is accidentally too weak makes every ML model
look good, which is the failure mode the brief is guarding against.
"""
from __future__ import annotations

import numpy as np
import pytest

from lagos_flood.config import ModelConfig
from lagos_flood.models import (
    PERSISTENCE_FEATURE,
    PersistenceBaseline,
    build_model,
    sample_weights_balanced,
)


def test_persistence_repeats_the_current_class():
    model = PersistenceBaseline(column_index=2)
    X = np.array([[9.0, 9.0, 0.0], [9.0, 9.0, 3.0], [9.0, 9.0, 1.0]])
    model.fit(X, np.array([0, 3, 1]))
    assert list(model.predict(X)) == [0, 3, 1]


def test_persistence_ignores_every_other_column():
    """Changing the other features must not change the prediction."""
    model = PersistenceBaseline(column_index=0)
    base = np.array([[2.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    noisy = np.array([[2.0, 999.0, -50.0], [0.0, -3.0, 71.0]])
    model.fit(base, np.array([2, 0]))
    assert np.array_equal(model.predict(base), model.predict(noisy))


def test_persistence_rounds_float_columns():
    """The feature arrives as float via the numpy panel; classes are integers."""
    model = PersistenceBaseline(column_index=0).fit(np.zeros((1, 1)), np.array([0]))
    X = np.array([[2.0], [2.4], [0.6], [3.0]])
    assert list(model.predict(X)) == [2, 2, 1, 3]


def test_persistence_proba_is_one_hot_and_normalised():
    model = PersistenceBaseline(column_index=0).fit(np.zeros((1, 1)), np.array([0]))
    proba = model.predict_proba(np.array([[0.0], [3.0]]))
    assert proba.shape == (2, 4)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 0] == 1.0
    assert proba[1, 3] == 1.0


def test_persistence_is_perfect_on_a_constant_series():
    """If the class never changes, persistence cannot be beaten."""
    y = np.array([1] * 50)
    X = np.ones((50, 1))
    model = PersistenceBaseline(column_index=0).fit(X, y)
    assert np.array_equal(model.predict(X), y)


def test_persistence_fails_on_an_alternating_series():
    """And is useless when the class flips every step — the honest opposite case."""
    y_current = np.array([0, 1] * 25, dtype=float)
    y_next = np.array([1, 0] * 25)
    model = PersistenceBaseline(column_index=0).fit(y_current.reshape(-1, 1), y_next)
    preds = model.predict(y_current.reshape(-1, 1))
    assert (preds == y_next).sum() == 0


def test_build_model_returns_each_named_model():
    cfg = ModelConfig(n_estimators=5, n_jobs=1)
    assert isinstance(build_model("persistence", cfg), PersistenceBaseline)
    for name in ("logistic_regression", "random_forest"):
        model = build_model(name, cfg)
        assert hasattr(model, "fit") and hasattr(model, "predict")


def test_build_model_passes_the_persistence_column_through():
    model = build_model("persistence", persistence_column_index=7)
    assert isinstance(model, PersistenceBaseline)
    assert model.column_index == 7


def test_logistic_regression_is_scaled():
    """Unscaled, a building-density feature would swamp a rainfall one."""
    from sklearn.pipeline import Pipeline

    model = build_model("logistic_regression", ModelConfig(n_jobs=1))
    assert isinstance(model, Pipeline)
    assert any("scaler" in step.lower() for step in model.named_steps)


def test_logistic_regression_trains_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)  # 3 classes
    model = build_model("logistic_regression", ModelConfig(n_jobs=1))
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset(set(np.unique(y)))
    assert (preds == y).mean() > 0.6


def test_build_model_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("transformer")


def test_balanced_weights_favour_the_rare_class():
    y = np.array([0] * 90 + [1] * 9 + [3])
    weights = sample_weights_balanced(y)
    assert weights[y == 3][0] > weights[y == 1][0] > weights[y == 0][0]
    # Total weight is conserved across classes.
    assert weights.sum() == pytest.approx(len(y), rel=1e-6)


def test_persistence_feature_name_is_stable():
    """The pipeline looks this column up by name; renaming it silently breaks it."""
    assert PERSISTENCE_FEATURE == "risk_class_current"
