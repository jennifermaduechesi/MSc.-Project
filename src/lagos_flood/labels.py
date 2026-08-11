"""Convert FloodScan SFED flooded fraction into 4-class flood-risk labels.

FloodScan's SFED product gives, per grid cell per day, the fraction of that cell
judged to be inundated. Aggregated to LGA level (area-weighted mean over the
cells intersecting the LGA polygon) it becomes a daily flooded-fraction series
per LGA — a continuous quantity. The supervised task is multi-class, so that
series has to be cut into Low / Medium / High / Critical.

No peer-reviewed study appears to have done this before, so the cut points are a
methodological choice the dissertation must defend rather than a convention it
can inherit. Two consequences shape this module:

1. Percentile thresholds are *fitted*, and fitting only ever sees training data.
   Deriving quantiles from the full series and then splitting would leak the test
   period's flood distribution into the labels.
2. :func:`threshold_sensitivity` exists so the choice can be reported as a
   sensitivity analysis instead of asserted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CLASS_TO_INT, RISK_CLASSES, LabelConfig

SFED_COL = "sfed_fraction"
LABEL_COL = "risk_class"
LABEL_INT_COL = "risk_class_id"


@dataclass(frozen=True)
class Thresholds:
    """Fitted cut points. ``per_lga`` maps LGA name to its own three cut points."""

    globally: tuple[float, float, float] | None = None
    per_lga: dict[str, tuple[float, float, float]] | None = None

    def for_lga(self, lga: str) -> tuple[float, float, float]:
        if self.per_lga is not None:
            try:
                return self.per_lga[lga]
            except KeyError as exc:
                raise KeyError(
                    f"no fitted threshold for LGA {lga!r}; it was absent from the "
                    "training period. Refit on a window that covers every LGA."
                ) from exc
        if self.globally is None:
            raise ValueError("Thresholds carries neither global nor per-LGA cut points")
        return self.globally


def fit_thresholds(
    df: pd.DataFrame,
    config: LabelConfig,
    *,
    sfed_col: str = SFED_COL,
    lga_col: str = "lga",
) -> Thresholds:
    """Derive cut points from ``df``, which must contain **training rows only**.

    For the ``absolute`` strategy this just echoes the configured constants; the
    call is still required so that the pipeline has one code path regardless of
    strategy.
    """
    if config.strategy == "absolute":
        return Thresholds(globally=tuple(config.absolute_cuts))  # type: ignore[arg-type]

    qs = list(config.percentile_cuts)
    if not config.per_lga:
        cuts = tuple(float(x) for x in df[sfed_col].quantile(qs).to_numpy())
        return Thresholds(globally=_dedupe(cuts))

    per_lga: dict[str, tuple[float, float, float]] = {}
    for lga, grp in df.groupby(lga_col, sort=False):
        cuts = tuple(float(x) for x in grp[sfed_col].quantile(qs).to_numpy())
        per_lga[str(lga)] = _dedupe(cuts)
    return Thresholds(per_lga=per_lga)


def _dedupe(cuts: tuple[float, ...]) -> tuple[float, float, float]:
    """Nudge apart ties so that the cut points stay strictly increasing.

    In a dry LGA the 80th and 95th percentiles of flooded fraction can both be
    0.0. Left alone that silently collapses two classes into one; nudging keeps
    the boundaries distinct and leaves the empty class visibly empty in the
    confusion matrix, which is the honest outcome.
    """
    out = list(cuts)
    eps = 1e-9
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return (out[0], out[1], out[2])


def apply_thresholds(
    df: pd.DataFrame,
    thresholds: Thresholds,
    *,
    sfed_col: str = SFED_COL,
    lga_col: str = "lga",
) -> pd.DataFrame:
    """Attach ``risk_class`` and ``risk_class_id`` columns.

    Bins are left-closed: ``Low`` is ``sfed <= c0``, ``Medium`` is
    ``c0 < sfed <= c1``, and so on, so a completely dry day is always ``Low``.
    """
    out = df.copy()
    ids = np.empty(len(out), dtype=np.int8)

    if thresholds.per_lga is not None:
        for lga, idx in out.groupby(lga_col, sort=False).indices.items():
            cuts = thresholds.for_lga(str(lga))
            ids[idx] = _digitise(out[sfed_col].to_numpy()[idx], cuts)
    else:
        ids[:] = _digitise(out[sfed_col].to_numpy(), thresholds.for_lga(""))

    out[LABEL_INT_COL] = ids
    out[LABEL_COL] = pd.Categorical.from_codes(ids, categories=list(RISK_CLASSES), ordered=True)
    return out


def _digitise(values: np.ndarray, cuts: tuple[float, float, float]) -> np.ndarray:
    return np.searchsorted(np.asarray(cuts), values, side="left").astype(np.int8)


def class_distribution(df: pd.DataFrame, *, label_col: str = LABEL_COL) -> pd.DataFrame:
    """Class counts and shares — the imbalance the modelling has to survive."""
    counts = df[label_col].value_counts().reindex(list(RISK_CLASSES)).fillna(0).astype(int)
    return pd.DataFrame({"count": counts, "share": counts / max(int(counts.sum()), 1)})


def threshold_sensitivity(
    df: pd.DataFrame,
    candidate_configs: dict[str, LabelConfig],
    *,
    sfed_col: str = SFED_COL,
    lga_col: str = "lga",
) -> pd.DataFrame:
    """Class balance produced by each candidate labelling scheme.

    Report this in the methodology chapter. It shows the reader how much of the
    class distribution is a property of Lagos and how much is an artefact of
    where the cut points were placed.
    """
    rows = []
    for name, cfg in candidate_configs.items():
        thresholds = fit_thresholds(df, cfg, sfed_col=sfed_col, lga_col=lga_col)
        labelled = apply_thresholds(df, thresholds, sfed_col=sfed_col, lga_col=lga_col)
        dist = class_distribution(labelled)
        row: dict[str, object] = {"scheme": name, "strategy": cfg.strategy}
        for cls in RISK_CLASSES:
            row[f"{cls}_n"] = int(dist.loc[cls, "count"])
            row[f"{cls}_share"] = round(float(dist.loc[cls, "share"]), 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("scheme")


__all__ = [
    "CLASS_TO_INT",
    "LABEL_COL",
    "LABEL_INT_COL",
    "SFED_COL",
    "Thresholds",
    "apply_thresholds",
    "class_distribution",
    "fit_thresholds",
    "threshold_sensitivity",
]
