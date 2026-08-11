"""Evaluation metrics for the 4-class flood-risk task.

Accuracy is not reported as a headline number. With a class distribution where
``Low`` dominates, a model that predicts ``Low`` unconditionally scores highly on
accuracy while being useless for early warning — the ``Critical`` class it misses
is the entire point of the system. Macro-F1 weights every class equally
regardless of frequency and is the primary metric here (Hinojosa Lee et al.,
2024; Grandini et al., 2020).

Per-class recall is reported alongside it, because for a warning system the cost
of a missed ``Critical`` day is not symmetric with the cost of a false alarm, and
a single aggregate hides that asymmetry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import RISK_CLASSES


@dataclass
class EvaluationResult:
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    quadratic_kappa: float
    per_class: pd.DataFrame
    confusion: pd.DataFrame
    critical_recall: float
    n_samples: int
    extra: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"macro-F1 {self.macro_f1:.3f} | weighted-F1 {self.weighted_f1:.3f} | "
            f"balanced acc {self.balanced_accuracy:.3f} | quadratic κ {self.quadratic_kappa:.3f} | "
            f"Critical recall {self.critical_recall:.3f} | n={self.n_samples}"
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "quadratic_kappa": self.quadratic_kappa,
            "critical_recall": self.critical_recall,
            "n_samples": float(self.n_samples),
            **self.extra,
        }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: tuple[str, ...] = RISK_CLASSES,
) -> EvaluationResult:
    """Score predictions over the full label set.

    ``labels`` is passed explicitly everywhere so that a class absent from a
    particular fold still appears in the output, as a zero row, instead of
    silently shrinking the macro average's denominator and inflating it.
    """
    label_ids = np.arange(len(labels))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    per_class = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=list(labels),
    )

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    confusion = pd.DataFrame(
        cm,
        index=pd.Index(list(labels), name="actual"),
        columns=pd.Index(list(labels), name="predicted"),
    )

    critical_idx = len(labels) - 1
    return EvaluationResult(
        macro_f1=float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        # Quadratic weighting penalises Low→Critical far more than Low→Medium,
        # which matches how wrong those two mistakes actually are operationally.
        quadratic_kappa=float(cohen_kappa_score(y_true, y_pred, labels=label_ids, weights="quadratic")),
        per_class=per_class,
        confusion=confusion,
        critical_recall=float(recall[critical_idx]),
        n_samples=int(len(y_true)),
    )


def aggregate_folds(results: list[EvaluationResult]) -> pd.DataFrame:
    """Mean and standard deviation of each metric across folds.

    Report the spread, not just the mean. A model whose macro-F1 swings from 0.41
    to 0.78 across time-ordered folds is not a 0.6 model — it is an unstable one,
    and the standard deviation is what makes that visible.
    """
    if not results:
        raise ValueError("no fold results to aggregate")
    frame = pd.DataFrame([r.to_dict() for r in results])
    return pd.DataFrame({"mean": frame.mean(), "std": frame.std(ddof=0)})


def per_lga_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lgas: np.ndarray,
    *,
    labels: tuple[str, ...] = RISK_CLASSES,
) -> pd.DataFrame:
    """Macro-F1 and Critical recall for each LGA separately.

    An aggregate score can hide a model that works in Ikeja and fails in Ibeju-
    Lekki. For a per-LGA warning product that failure mode matters more than the
    headline number, so it is broken out by default.
    """
    label_ids = np.arange(len(labels))
    rows = []
    for lga in pd.unique(lgas):
        mask = lgas == lga
        yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
        _, recall, _, _ = precision_recall_fscore_support(
            yt, yp, labels=label_ids, zero_division=0
        )
        rows.append(
            {
                "lga": lga,
                "n": int(mask.sum()),
                "macro_f1": float(f1_score(yt, yp, labels=label_ids, average="macro", zero_division=0)),
                "critical_recall": float(recall[-1]),
                "n_critical_actual": int((yt == len(labels) - 1).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1").reset_index(drop=True)


__all__ = ["EvaluationResult", "aggregate_folds", "evaluate", "per_lga_breakdown"]
