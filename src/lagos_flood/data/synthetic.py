"""Synthetic Lagos panel for testing the pipeline without downloads.

Mirrors the shape of the real inputs — FloodScan SFED joined to NASA POWER
weather on ``lga_id`` — so the pipeline, and CI, can run before any credentials
or files exist.

It is a stand-in for plumbing, not a simulation of Lagos hydrology. Numbers
produced from it are meaningless as findings and must never appear in the
dissertation as results.

One real property is reproduced deliberately: **LGAs sharing a NASA POWER grid
cell get identical weather.** That is true of the real data — all 20 LGAs fall
into 2 cells — and if the synthetic data pretended otherwise, tests would pass
against a world that does not exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lga_reference import lga_reference_frame
from .nasapower import grid_cell

#: Coastal and low-lying LGAs, which the generator makes flood more readily.
_COASTAL = {
    "Eti-Osa", "Ibeju-Lekki", "Amuwo-Odofin", "Apapa", "Ojo", "Badagry", "Lagos Island",
}


def make_synthetic_panel(
    start: str = "2018-01-01",
    end: str = "2025-12-31",
    *,
    seed: int = 42,
    lgas: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(dynamic, static)`` frames mimicking the real inputs.

    ``dynamic`` is one row per (lga_id, date) with NASA POWER weather and
    FloodScan SFED. ``static`` is one row per LGA with terrain and exposure.

    Rainfall follows the bimodal Lagos regime — the JJA long rains and the
    shorter SON peak — and is generated **per grid cell**, then shared by every
    LGA in that cell. Flood extent responds to accumulated rainfall filtered
    through each LGA's terrain, so SFED still varies LGA by LGA even where the
    weather does not.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()

    reference = lga_reference_frame()
    if lgas is not None:
        reference = reference[reference["lga"].isin(lgas)].reset_index(drop=True)

    cells = reference.apply(lambda r: grid_cell(r["latitude"], r["longitude"]), axis=1)
    reference = reference.assign(
        grid_lat=[c[0] for c in cells], grid_lon=[c[1] for c in cells]
    )

    # Bimodal seasonal rainfall climatology.
    seasonal = (
        11.0 * np.exp(-0.5 * ((doy - 180) / 38.0) ** 2)   # JJA long rains
        + 6.0 * np.exp(-0.5 * ((doy - 285) / 26.0) ** 2)  # SON secondary peak
        + 0.35
    )

    # One weather series per grid cell, exactly as NASA POWER would serve it.
    weather_by_cell: dict[tuple[float, float], pd.DataFrame] = {}
    for cell in reference[["grid_lat", "grid_lon"]].drop_duplicates().itertuples(index=False):
        key = (cell.grid_lat, cell.grid_lon)
        wet = rng.random(len(dates)) < np.clip(seasonal / 14.0, 0.02, 0.72)
        rain = np.where(wet, rng.gamma(shape=1.25, scale=seasonal * 1.5), 0.0)
        weather_by_cell[key] = pd.DataFrame(
            {
                "date": dates,
                "rain_mm": np.round(rain, 2),
                "temp_c": np.round(27.5 - 0.004 * seasonal * 10 + rng.normal(0, 0.9, len(dates)), 2),
                "humidity_pct": np.round(
                    np.clip(74 + 1.4 * seasonal + rng.normal(0, 3.0, len(dates)), 40, 100), 2
                ),
            }
        )

    static = _make_static(reference, rng)
    frames = []

    for row in reference.itertuples(index=False):
        weather = weather_by_cell[(row.grid_lat, row.grid_lon)]
        attrs = static.loc[static["lga_id"] == row.lga_id].iloc[0]
        rain = weather["rain_mm"].to_numpy()

        # Flood extent responds to 7-day accumulation, amplified by low elevation,
        # imperviousness and coastal position.
        accum7 = pd.Series(rain).rolling(7, min_periods=1).sum().to_numpy()
        susceptibility = (
            0.55 * (1.0 - attrs["elevation_mean"] / 60.0)
            + 0.30 * attrs["impervious_frac"]
            + 0.15 * attrs["coastal_flag"]
        )
        drive = 0.0016 * accum7 * (0.5 + susceptibility)
        noise = rng.gamma(shape=0.6, scale=0.004, size=len(dates))
        sfed = np.clip(drive + noise - 0.008, 0.0, 1.0)

        frame = weather.copy()
        frame.insert(0, "lga", row.lga)
        frame.insert(0, "lga_id", row.lga_id)
        frame["sfed_fraction"] = np.round(sfed, 5)
        frames.append(frame)

    dynamic = pd.concat(frames, ignore_index=True)

    # Day-of-year baseline, as the HDX export supplies.
    dynamic["sfed_baseline"] = dynamic.groupby(
        ["lga_id", dynamic["date"].dt.dayofyear]
    )["sfed_fraction"].transform("mean").round(5)
    dynamic["sfed_deviation"] = (
        dynamic["sfed_fraction"] - dynamic["sfed_baseline"]
    ).round(5)

    dynamic = dynamic.sort_values(["lga_id", "date"]).reset_index(drop=True)
    return dynamic, static


def _make_static(reference: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for row in reference.itertuples(index=False):
        coastal = row.lga in _COASTAL
        elevation = float(rng.uniform(1.5, 12.0) if coastal else rng.uniform(8.0, 45.0))
        rows.append(
            {
                "lga_id": row.lga_id,
                "lga": row.lga,
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
