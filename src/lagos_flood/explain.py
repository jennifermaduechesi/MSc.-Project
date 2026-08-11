"""SHAP explainability for the flood-risk classifier.

An early-warning product that cannot say *why* it raised a warning is hard for an
agency to act on and harder to trust. SHAP (Lundberg & Lee, 2017) gives both the
global picture — which drivers matter across Lagos — and the local one — why
Kosofe was flagged Critical on a particular date. The second is what a duty
officer actually needs.

``TreeExplainer`` is exact for the tree ensembles used here, so no sampling
approximation enters the attributions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RISK_CLASSES


@dataclass
class ShapResult:
    """SHAP values with the bookkeeping needed to interpret them.

    ``values`` has shape ``(n_samples, n_features, n_classes)``. Multi-class SHAP
    is per class: a feature can push towards Critical and away from Low at the
    same time, and collapsing that to one number throws away the direction.
    """

    values: np.ndarray
    feature_names: list[str]
    class_names: tuple[str, ...]
    base_values: np.ndarray

    def global_importance(self, class_name: str | None = None) -> pd.Series:
        """Mean absolute SHAP per feature, descending.

        With ``class_name`` given, importance for driving that one class;
        otherwise averaged over all classes.
        """
        if class_name is None:
            magnitude = np.abs(self.values).mean(axis=(0, 2))
        else:
            idx = self.class_names.index(class_name)
            magnitude = np.abs(self.values[:, :, idx]).mean(axis=0)
        return (
            pd.Series(magnitude, index=self.feature_names)
            .sort_values(ascending=False)
            .rename("mean_abs_shap")
        )

    def explain_row(self, i: int, class_name: str, top_n: int = 8) -> pd.Series:
        """Signed contributions for a single prediction — the local explanation.

        Positive values pushed the model towards ``class_name``, negative away.
        """
        idx = self.class_names.index(class_name)
        contributions = pd.Series(self.values[i, :, idx], index=self.feature_names)
        return contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(top_n)


def compute_shap(
    model,
    X: pd.DataFrame,
    *,
    class_names: tuple[str, ...] = RISK_CLASSES,
    max_samples: int | None = 5000,
    random_state: int = 42,
) -> ShapResult:
    """Compute SHAP values for a fitted tree model.

    ``max_samples`` subsamples for tractability on a long panel; attributions are
    averages, so a few thousand rows give a stable global picture. Pass ``None``
    to explain every row.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("shap is not installed — run `pip install -e '.[explain]'`") from exc

    if max_samples is not None and len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state).sort_index()

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)

    # SHAP's multi-class return type has varied across versions: older releases
    # give a list of per-class arrays, newer ones a single stacked array.
    if isinstance(raw, list):
        values = np.stack(raw, axis=-1)
    else:
        values = np.asarray(raw)
        if values.ndim == 2:  # binary or single-output
            values = values[:, :, None]

    base = np.atleast_1d(np.asarray(explainer.expected_value))
    return ShapResult(
        values=values,
        feature_names=list(X.columns),
        class_names=class_names[: values.shape[2]],
        base_values=base,
    )


def save_importance_table(result: ShapResult, path: Path | str) -> Path:
    """Write global importance per class to CSV for the results chapter."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"overall": result.global_importance()})
    for cls in result.class_names:
        table[cls] = result.global_importance(cls)
    table.to_csv(path)
    return path


def compare_to_literature(result: ShapResult, top_n: int = 5) -> str:
    """Narrative comparison of the top drivers against published findings.

    Choubin et al. (2025) report distance-to-stream, TWI and elevation as the
    dominant drivers of flood susceptibility under SHAP. Agreement supports the
    model; divergence is not automatically a fault, since this task is temporal
    forecasting rather than static susceptibility, and rainfall-history features
    have no counterpart in a static model. Either way it belongs in the
    discussion, which is what this helper drafts.
    """
    top = result.global_importance().head(top_n)
    literature = {"dist_to_river_km", "twi_mean", "elevation_mean", "hand_mean"}
    dynamic = [f for f in top.index if f not in literature]
    shared = [f for f in top.index if f in literature]

    lines = [f"Top {top_n} drivers by mean |SHAP|:"]
    lines += [f"  {i}. {name} ({value:.4g})" for i, (name, value) in enumerate(top.items(), 1)]
    lines.append("")
    if shared:
        lines.append(
            "Consistent with static susceptibility work (Choubin et al., 2025): "
            + ", ".join(shared)
        )
    if dynamic:
        lines.append(
            "Dynamic drivers with no counterpart in static susceptibility models: "
            + ", ".join(dynamic)
        )
    return "\n".join(lines)


__all__ = ["ShapResult", "compare_to_literature", "compute_shap", "save_importance_table"]
