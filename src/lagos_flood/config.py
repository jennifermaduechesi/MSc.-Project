"""Project-wide configuration for the Lagos per-LGA flood-risk classifier.

Everything that a reviewer might want to challenge — the label thresholds, the
forecast lead time, the feature lookback windows — is declared here rather than
buried in the pipeline, so a single file documents the whole experimental setup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTEFACTS = ROOT / "artefacts"

# --------------------------------------------------------------------------- study area
#: The 20 Local Government Areas of Lagos State. Names follow the GRID3 Nigeria
#: LGA boundary layer; keep this spelling when joining to any external table.
LAGOS_LGAS: tuple[str, ...] = (
    "Agege",
    "Ajeromi-Ifelodun",
    "Alimosho",
    "Amuwo-Odofin",
    "Apapa",
    "Badagry",
    "Epe",
    "Eti-Osa",
    "Ibeju-Lekki",
    "Ifako-Ijaiye",
    "Ikeja",
    "Ikorodu",
    "Kosofe",
    "Lagos Island",
    "Lagos Mainland",
    "Mushin",
    "Ojo",
    "Oshodi-Isolo",
    "Shomolu",
    "Surulere",
)

#: Risk classes, ordered. The ordering is meaningful — macro-F1 treats them as
#: nominal, but the confusion matrix and any ordinal analysis rely on this order.
RISK_CLASSES: tuple[str, ...] = ("Low", "Medium", "High", "Critical")
CLASS_TO_INT = {c: i for i, c in enumerate(RISK_CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}


# --------------------------------------------------------------------------- labelling
@dataclass(frozen=True)
class LabelConfig:
    """How a FloodScan SFED flooded-fraction series becomes a 4-class label.

    There is no published convention for this conversion — the literature review
    found no peer-reviewed use of SFED as a supervised-ML labelling source — so
    the choice is part of the contribution and must be reported, not assumed.

    Two strategies are supported:

    ``absolute``
        Fixed flooded-fraction cut points, identical for every LGA. Interpretable
        and comparable across LGAs, but a rural LGA such as Epe and a dense urban
        one such as Mushin reach a given fraction under very different rainfall.

    ``percentile``
        Cut points taken from quantiles of the *training-period* SFED distribution.
        Set ``per_lga=True`` to compute them separately for each LGA, which
        controls for the exposure differences above at the cost of cross-LGA
        comparability.

    Quantiles must be fitted on training data only — see
    :func:`lagos_flood.labels.fit_percentile_thresholds`.
    """

    strategy: str = "absolute"          # "absolute" | "percentile"
    absolute_cuts: tuple[float, float, float] = (0.01, 0.05, 0.15)
    percentile_cuts: tuple[float, float, float] = (0.80, 0.95, 0.99)
    per_lga: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in {"absolute", "percentile"}:
            raise ValueError(f"unknown labelling strategy: {self.strategy!r}")
        cuts = self.absolute_cuts if self.strategy == "absolute" else self.percentile_cuts
        if list(cuts) != sorted(cuts):
            raise ValueError(f"cut points must be strictly increasing, got {cuts}")
        if len(set(cuts)) != 3:
            raise ValueError(f"cut points must be distinct, got {cuts}")


# --------------------------------------------------------------------------- features
@dataclass(frozen=True)
class FeatureConfig:
    """Feature-engineering windows.

    ``max_lookback_days`` is derived from the other windows and is what the
    temporal splitter uses to size its embargo gap. If you add a feature with a
    longer window, this value must grow with it or the gap will be too short and
    training data will leak into the validation fold.
    """

    rain_lag_days: tuple[int, ...] = (1, 2, 3, 5, 7)
    rain_accum_windows: tuple[int, ...] = (3, 7, 14, 30)
    soil_moisture_lag_days: tuple[int, ...] = (1, 3, 7)
    #: Antecedent Precipitation Index decay constant (Heggen, 2001). 0.85–0.95 is
    #: the usual range; lower values forget past rainfall faster.
    api_decay_k: float = 0.90
    api_window_days: int = 30
    #: Seasonal flags. Lagos has a bimodal regime — the long rains (JJA) and the
    #: shorter SON peak both drive inundation.
    add_season_flags: bool = True

    @property
    def max_lookback_days(self) -> int:
        return max(
            max(self.rain_lag_days, default=0),
            max(self.rain_accum_windows, default=0),
            max(self.soil_moisture_lag_days, default=0),
            self.api_window_days,
        )


# --------------------------------------------------------------------------- validation
@dataclass(frozen=True)
class ValidationConfig:
    """Temporal validation settings.

    Random k-fold cross-validation is *not* offered anywhere in this codebase.
    Daily per-LGA flood data is strongly autocorrelated in both time and space,
    and random folds on autocorrelated data inflate measured skill — see Roberts
    et al. (2017) and Bergmeir & Benítez (2012). Every split here is time-ordered.
    """

    #: Everything strictly before this date is available for training; everything
    #: from this date onward is held out and touched exactly once, at the end.
    test_start: str = "2023-01-01"
    #: Number of expanding-window folds carved out of the training period.
    n_folds: int = 5
    #: Days held out between a fold's train end and its validation start. Leave as
    #: None to derive it — see Config.embargo_days, which sizes it from both the
    #: feature lookback and the lead time. Set it explicitly only to make it
    #: *longer*; a shorter value reintroduces the leakage it exists to prevent.
    embargo_days: int | None = None
    #: Forecast lead time: predict the class at t+h from features known at t.
    lead_time_days: int = 1


# --------------------------------------------------------------------------- models
@dataclass(frozen=True)
class ModelConfig:
    """Model defaults.

    ``class_weight="balanced"`` is the default imbalance strategy rather than
    SMOTE. Synthetic oversampling of a spatiotemporal panel interpolates between
    rows that are neighbours in time and space, which manufactures flood days
    that never happened; cost-sensitive weighting avoids that. SMOTE remains
    available in ``lagos_flood.models`` for the comparison the dissertation
    reports, but it must be applied inside the training fold only.
    """

    random_state: int = 42
    n_estimators: int = 500
    max_depth: int | None = None
    learning_rate: float = 0.05
    class_weight: str | None = "balanced"
    n_jobs: int = -1


@dataclass(frozen=True)
class Config:
    label: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def embargo_days(self) -> int:
        """Effective gap between a fold's training end and its validation start.

        Two separate overlaps have to be excluded, and the gap must clear the
        larger of them. Writing D for the boundary date:

        *Features reaching back.* A validation row at ``D + gap`` reads a window
        back to ``D + gap - lookback``. That must not touch training, so
        ``gap > max_lookback_days``.

        *Targets reaching forward.* A training row at ``D`` carries the label
        from ``D + lead``. That must not touch validation, so
        ``gap > lead_time_days``.

        Taking the maximum satisfies both — they are alternative constraints on
        the same gap, not costs that accumulate, so summing them would only
        discard usable data.
        """
        if self.validation.embargo_days is not None:
            return self.validation.embargo_days
        return max(self.features.max_lookback_days, self.validation.lead_time_days)


DEFAULT = Config()
