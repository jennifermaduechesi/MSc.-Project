"""FloodScan SFED ingestion and aggregation to LGA level.

FloodScan is a passive-microwave flood-extent product from Atmospheric and
Environmental Research. Its SFED (Standardized Flood Extent Depiction) layer
gives a daily flooded fraction per ~90 m grid cell. The methodology is documented
in a peer-reviewed book chapter (Galantowicz & Picton, 2021) and in AER/NASA
technical reports (Galantowicz & Frey, 2002); **there is no dedicated
peer-reviewed validation paper for FloodScan itself**. Cite it on that basis and
treat the absence as a limitation to state, not one to paper over.

Access: the daily product is distributed through HDX and requires registration.
Downloading is deliberately not automated here — credentials belong in the
environment, not in the repository.

    https://data.humdata.org/dataset/floodscan

The aggregation below is area-weighted: each LGA's daily value is the mean
flooded fraction over the grid cells intersecting its polygon, weighted by
intersection area. A plain unweighted mean biases large, mostly-dry LGAs such as
Epe upwards relative to small dense ones.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SFED_VARIABLE = "SFED"
DEFAULT_CRS = "EPSG:4326"
#: Equal-area projection for Nigeria — use a projected CRS when computing the
#: intersection areas that become aggregation weights, never degrees.
AREA_CRS = "EPSG:26332"


def load_lga_boundaries(path: Path | str, *, lga_col: str = "lga"):
    """Load Lagos LGA polygons and normalise the name column.

    Source: GRID3 Nigeria — Operational LGA Boundaries (https://grid3.org).
    Requires geopandas.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("geopandas is required — `pip install -e '.[geo]'`") from exc

    gdf = gpd.read_file(path)
    for candidate in ("lga", "LGA", "lganame", "NAME_2", "admin2Name"):
        if candidate in gdf.columns:
            gdf = gdf.rename(columns={candidate: lga_col})
            break
    else:
        raise KeyError(f"no recognisable LGA name column in {sorted(gdf.columns)}")

    gdf[lga_col] = gdf[lga_col].astype(str).str.strip()
    return gdf.to_crs(DEFAULT_CRS)


def aggregate_sfed_to_lga(
    sfed_path: Path | str,
    boundaries,
    *,
    variable: str = SFED_VARIABLE,
    lga_col: str = "lga",
    date_col: str = "date",
    value_col: str = "sfed_fraction",
) -> pd.DataFrame:
    """Area-weighted daily mean SFED per LGA.

    ``sfed_path`` points at a NetCDF/GeoTIFF stack with a time dimension.
    Returns tidy rows of ``(lga, date, sfed_fraction)``.
    """
    try:
        import xarray as xr
        from exactextract import exact_extract
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "xarray and exactextract are required — `pip install -e '.[geo]'`"
        ) from exc

    stack = xr.open_dataset(sfed_path)
    if variable not in stack:
        raise KeyError(f"variable {variable!r} not in {sfed_path}; have {list(stack.data_vars)}")

    boundaries = boundaries.to_crs(AREA_CRS)
    records: list[pd.DataFrame] = []

    for timestamp in pd.to_datetime(stack["time"].values):
        layer = stack[variable].sel(time=timestamp).rio.reproject(AREA_CRS)
        stats = exact_extract(layer, boundaries, ops=["mean"], output="pandas")
        records.append(
            pd.DataFrame(
                {
                    lga_col: boundaries[lga_col].to_numpy(),
                    date_col: timestamp,
                    value_col: stats["mean"].to_numpy(),
                }
            )
        )

    out = pd.concat(records, ignore_index=True)
    out[value_col] = out[value_col].clip(lower=0.0, upper=1.0)
    return out.sort_values([lga_col, date_col]).reset_index(drop=True)


def validate_sfed_series(df: pd.DataFrame, *, value_col: str = "sfed_fraction") -> pd.DataFrame:
    """Sanity checks worth running before anything is labelled.

    Returns a per-LGA table of coverage and extremes. Long runs of exact zeros
    can be genuine dry periods or a masked/absent input — the two look identical
    downstream once labels are cut, so check here where they are still separable.
    """
    grouped = df.groupby("lga")[value_col]
    report = pd.DataFrame(
        {
            "n_days": grouped.size(),
            "n_missing": grouped.apply(lambda s: int(s.isna().sum())),
            "pct_exact_zero": grouped.apply(lambda s: round(float((s == 0).mean() * 100), 2)),
            "max": grouped.max().round(5),
            "p99": grouped.quantile(0.99).round(5),
        }
    )
    return report.sort_values("pct_exact_zero", ascending=False)


__all__ = [
    "AREA_CRS",
    "SFED_VARIABLE",
    "aggregate_sfed_to_lga",
    "load_lga_boundaries",
    "validate_sfed_series",
]
