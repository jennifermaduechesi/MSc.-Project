"""Synthetic Lagos panel for testing the pipeline without credentials.

The real inputs need registered access (FloodScan via HDX, ERA5 via the Copernicus
CDS). This module generates a panel with the same shape and broadly the same
statistical character so the pipeline can be exercised end to end, tests can run
in CI, and the code path is proven before any download happens.

It is a stand-in for plumbing, not a simulation of Lagos hydrology. Numbers
produced from it are meaningless as findings and must never appear in the
dissertation as results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LAGOS_LGAS
from ..features.build import RAINFALL_COL, SOIL_MOISTURE_COL

#: Coastal and low-lying LGAs, which the generator makes flood more readily.
_COASTAL = {"Eti-Osa", "Ibeju-Lekki", "Amuwo-Odofin", "Apapa", "Ojo", "Badagry", "Lagos Island"}


def make_synthetic_panel(
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    *,
    seed: int = 42,
    lgas: tuple[str, ...] = LAGOS_LGAS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(dynamic, static)`` frames mimicking the real inputs.

    ``dynamic`` is one row per (LGA, date) with rainfall, soil moisture and an
    SFED flooded fraction. ``static`` is one row per LGA with terrain and
    exposure attributes.

    Rainfall follows the bimodal Lagos regime (JJA long rains, SON secondary
    peak). Flooded fraction responds to accumulated recent rainfall filtered
    through each LGA's terrain, so antecedent-wetness features carry real signal
    and the causality checks have something to bite on.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()

    # Bimodal seasonal rainfall climatology.
    seasonal = (
        11.0 * np.exp(-0.5 * ((doy - 180) / 38.0) ** 2)   # JJA long rains
        + 6.0 * np.exp(-0.5 * ((doy - 285) / 26.0) ** 2)  # SON secondary peak
        + 0.35
    )

    static = _make_static(lgas, rng)
    frames = []

    for lga in lgas:
        attrs = static.loc[static["lga"] == lga].iloc[0]
        # Wet-day occurrence then intensity — a gamma mixture, not a bare normal,
        # so the series keeps the heavy right tail that drives flood days.
        wet = rng.random(len(dates)) < np.clip(seasonal / 14.0, 0.02, 0.72)
        intensity = rng.gamma(shape=1.25, scale=seasonal * 1.5)
        rain = np.where(wet, intensity, 0.0)

        # Soil moisture: decaying store topped up by rainfall.
        soil = np.zeros(len(dates))
        store = 0.25
        for i, p in enumerate(rain):
            store = np.clip(0.94 * store + 0.012 * p, 0.02, 1.0)
            soil[i] = store

        # Flooded fraction responds to 7-day accumulation, amplified by low
        # elevation, high imperviousness and proximity to drainage.
        accum7 = pd.Series(rain).rolling(7, min_periods=1).sum().to_numpy()
        susceptibility = (
            0.55 * (1.0 - attrs["elevation_mean"] / 60.0)
            + 0.30 * attrs["impervious_frac"]
            + 0.15 * attrs["coastal_flag"]
        )
        drive = 0.0016 * accum7 * (0.5 + susceptibility) + 0.35 * soil * susceptibility * 0.02
        noise = rng.gamma(shape=0.6, scale=0.004, size=len(dates))
        sfed = np.clip(drive + noise - 0.008, 0.0, 1.0)

        frames.append(
            pd.DataFrame(
                {
                    "lga": lga,
                    "date": dates,
                    RAINFALL_COL: np.round(rain, 3),
                    SOIL_MOISTURE_COL: np.round(soil, 4),
                    "sfed_fraction": np.round(sfed, 5),
                }
            )
        )

    dynamic = pd.concat(frames, ignore_index=True).sort_values(["lga", "date"]).reset_index(drop=True)
    return dynamic, static


def _make_static(lgas: tuple[str, ...], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for lga in lgas:
        coastal = lga in _COASTAL
        elevation = float(rng.uniform(1.5, 12.0) if coastal else rng.uniform(8.0, 45.0))
        rows.append(
            {
                "lga": lga,
                "elevation_mean": round(elevation, 2),
                "slope_mean": round(float(rng.uniform(0.1, 2.4)), 3),
                "twi_mean": round(float(rng.uniform(6.0, 14.5)), 2),
                "hand_mean": round(float(max(0.4, elevation * rng.uniform(0.3, 0.8))), 2),
                "dist_to_river_km": round(float(rng.uniform(0.2, 18.0)), 2),
                "building_density": round(float(rng.uniform(120, 4200)), 1),
                "impervious_frac": round(float(rng.uniform(0.12, 0.93)), 3),
                "coastal_flag": int(coastal),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["make_synthetic_panel"]
