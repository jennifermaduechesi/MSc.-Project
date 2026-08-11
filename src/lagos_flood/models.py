"""Model factory for the multi-class flood-risk classifier.

Tree ensembles are the default family. Reviews of flood susceptibility modelling
put random forest and gradient boosting at the top consistently, and random
forest specifically dominates Sub-Saharan and West African studies where records
are short and gauge coverage is thin (Manyaka et al., 2026) — which describes
this dataset. Random forest is the baseline; XGBoost and LightGBM are the
comparators the dissertation reports against it.

On class imbalance: ``class_weight="balanced"`` is the default rather than SMOTE.
Synthetic oversampling of a spatiotemporal panel interpolates between rows that
are neighbours in time and space, inventing flood days that never occurred and
smearing the minority class across the calendar. Cost-sensitive weighting
achieves the same emphasis without fabricating events. SMOTE is still available
via :func:`resample_training_fold` for the comparison, with the constraint that
it may only ever touch a training fold.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RISK_CLASSES, ModelConfig

#: Feature holding the class observed at time t. Persistence predicts t+h from it.
PERSISTENCE_FEATURE = "risk_class_current"


class PersistenceBaseline:
    """Predict that the class at t+h is whatever it was at t.

    The brief calls this essential, and it is: if a random forest cannot beat
    "assume tomorrow looks like today", the honest conclusion is that the
    engineered features add nothing, and that conclusion has to be reached
    deliberately rather than discovered by an examiner.

    It is a genuine forecast, not a cheat. The class at t is known at forecast
    time; only the class at t+h is withheld. The model reads a single column and
    ignores everything else, so any ML model is being handed strictly more
    information and has no excuse for losing.
    """

    def __init__(self, column_index: int = 0):
        self.column_index = column_index
        self.classes_: np.ndarray | None = None

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> PersistenceBaseline:
        self.classes_ = np.unique(y) if y is not None else np.arange(len(RISK_CLASSES))
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.rint(np.asarray(X)[:, self.column_index]).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        preds = self.predict(X)
        out = np.zeros((len(preds), len(RISK_CLASSES)), dtype=float)
        out[np.arange(len(preds)), np.clip(preds, 0, len(RISK_CLASSES) - 1)] = 1.0
        return out


class Classifier(Protocol):  # pragma: no cover - structural typing only
    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any: ...
    def predict(self, X: Any) -> np.ndarray: ...
    def predict_proba(self, X: Any) -> np.ndarray: ...


def build_model(
    name: str,
    config: ModelConfig | None = None,
    *,
    persistence_column_index: int = 0,
) -> Classifier:
    """Construct a model by name.

    Available: ``persistence``, ``logistic_regression``, ``random_forest``,
    ``xgboost``, ``lightgbm``. The first two need only scikit-learn; XGBoost and
    LightGBM are optional dependencies, and if one is missing the error names the
    extra to install rather than surfacing a bare ImportError.
    """
    cfg = config or ModelConfig()
    key = name.lower().replace("-", "_")

    if key in {"persistence", "baseline"}:
        return PersistenceBaseline(column_index=persistence_column_index)

    if key in {"logistic_regression", "logreg", "lr"}:
        # Scaled, because the features span millimetres of rain, dimensionless
        # indices and building counts per km² — unscaled, the largest-magnitude
        # feature dominates the penalty term for reasons of unit choice alone.
        return make_pipeline(
            StandardScaler(),
            # No n_jobs: it has had no effect on LogisticRegression since
            # scikit-learn 1.8 and is removed in 1.10.
            LogisticRegression(
                max_iter=2000,
                class_weight=cfg.class_weight,
                random_state=cfg.random_state,
            ),
        )

    if key in {"random_forest", "rf"}:
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )

    if key in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "xgboost is not installed — run `pip install -e '.[boosting]'`"
            ) from exc
        return XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth or 6,
            learning_rate=cfg.learning_rate,
            objective="multi:softprob",
            tree_method="hist",
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )

    if key in {"lightgbm", "lgbm", "lgb"}:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "lightgbm is not installed — run `pip install -e '.[boosting]'`"
            ) from exc
        return LGBMClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth or -1,
            learning_rate=cfg.learning_rate,
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            verbose=-1,
        )

    raise ValueError(
        f"unknown model {name!r}; choose from random_forest, xgboost, lightgbm"
    )


def sample_weights_balanced(y: np.ndarray) -> np.ndarray:
    """Per-sample weights inversely proportional to class frequency.

    For estimators such as XGBoost that take ``sample_weight`` rather than a
    ``class_weight`` argument.
    """
    classes, counts = np.unique(y, return_counts=True)
    weight_for = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts, strict=True)}
    return np.array([weight_for[v] for v in y], dtype=float)


def resample_training_fold(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "smote",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a **training fold only** — never validation, never test.

    Applying this to data that is later scored produces optimistic and meaningless
    metrics, because synthetic minority points are interpolations of the very rows
    the model is being evaluated on.
    """
    key = method.lower()
    try:
        if key == "smote":
            from imblearn.over_sampling import SMOTE

            sampler = SMOTE(random_state=random_state)
        elif key == "adasyn":
            from imblearn.over_sampling import ADASYN

            sampler = ADASYN(random_state=random_state)
        elif key == "none":
            return X, y
        else:
            raise ValueError(f"unknown resampling method {method!r}")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "imbalanced-learn is not installed — run `pip install -e '.[imbalance]'`"
        ) from exc

    X_res, y_res = sampler.fit_resample(X, y)
    return np.asarray(X_res), np.asarray(y_res)


__all__ = [
    "PERSISTENCE_FEATURE",
    "Classifier",
    "PersistenceBaseline",
    "build_model",
    "resample_training_fold",
    "sample_weights_balanced",
]
