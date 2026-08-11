"""Rainfall and soil-moisture ingestion, with the bias caveats attached.

Products used
-------------
``ERA5`` / ``ERA5-Land``
    ECMWF reanalysis at 0.25° and 0.1°. ERA5-Land supplies volumetric soil water,
    which is the antecedent-wetness input. Both have documented biases over
    tropical Africa: ERA5 reduces but does not remove the wet bias (Steinkopf &
    Engelbrecht, 2022), and reproduces Guinea-Coast rainfall with regional error
    (Gbode et al., 2023). Bodjrènou et al. (2025) evaluate exactly these two
    resolutions over West Africa.

``CHIRPS`` / ``IMERG``
    Satellite-gauge blended rainfall. Ganiyu et al. (2025) evaluate IMERG v07 and
    CHIRPS 2.0 against Nigerian gauges and are the reference for how much to trust
    them in-country.

The resolution caveat matters for Lagos specifically: these grids are coarse
relative to the convective cells that drive urban flash flooding, so a real
storm over one LGA is smeared across several. That limitation is a finding to
report, not a defect to hide — see :func:`resolution_caveat`.

Access needs credentials (Copernicus CDS for ERA5, NASA Earthdata for IMERG), so
download is left to the operator; these functions handle what happens after.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ERA5_RESOLUTION_DEG = 0.25
ERA5_LAND_RESOLUTION_DEG = 0.1
CHIRPS_RESOLUTION_DEG = 0.05
IMERG_RESOLUTION_DEG = 0.1

#: Lagos State bounding box (lon_min, lat_min, lon_max, lat_max), WGS84.
LAGOS_BBOX = (2.65, 6.35, 4.35, 6.75)


def aggregate_gridded_to_lga(
    path: Path | str,
    boundaries,
    variable: str,
    *,
    lga_col: str = "lga",
    date_col: str = "date",
    value_col: str | None = None,
    resample: str | None = None,
) -> pd.DataFrame:
    """Zonal-mean a gridded time series onto LGA polygons.

    ``resample`` accepts a pandas offset alias; pass ``"D"`` with hourly ERA5 to
    get daily totals. Rainfall is summed on resample, everything else averaged.
    """
    try:
        import xarray as xr
        from exactextract import exact_extract
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "xarray and exactextract are required — `pip install -e '.[geo]'`"
        ) from exc

    from .floodscan import AREA_CRS

    ds = xr.open_dataset(path)
    if variable not in ds:
        raise KeyError(f"variable {variable!r} not in {path}; have {list(ds.data_vars)}")
    da = ds[variable]

    if resample is not None:
        is_rain = any(t in variable.lower() for t in ("precip", "rain", "tp"))
        da = da.resample(time=resample).sum() if is_rain else da.resample(time=resample).mean()

    boundaries = boundaries.to_crs(AREA_CRS)
    out_col = value_col or variable
    records = []
    for timestamp in pd.to_datetime(da["time"].values):
        layer = da.sel(time=timestamp).rio.reproject(AREA_CRS)
        stats = exact_extract(layer, boundaries, ops=["mean"], output="pandas")
        records.append(
            pd.DataFrame(
                {
                    lga_col: boundaries[lga_col].to_numpy(),
                    date_col: timestamp,
                    out_col: stats["mean"].to_numpy(),
                }
            )
        )
    return pd.concat(records, ignore_index=True).sort_values([lga_col, date_col]).reset_index(drop=True)


def compare_rainfall_products(
    products: dict[str, pd.DataFrame],
    *,
    lga_col: str = "lga",
    date_col: str = "date",
    value_col: str = "rain_mm",
) -> pd.DataFrame:
    """Cross-product agreement, as the robustness check the write-up needs.

    Returns pairwise correlation, mean bias and RMSE per LGA. Where products
    disagree sharply, any conclusion resting on a single one is fragile and
    should be reported with that caveat.
    """
    names = sorted(products)
    if len(names) < 2:
        raise ValueError("need at least two products to compare")

    merged = None
    for name in names:
        frame = products[name][[lga_col, date_col, value_col]].rename(columns={value_col: name})
        merged = frame if merged is None else merged.merge(frame, on=[lga_col, date_col], how="inner")
    assert merged is not None

    rows = []
    for lga, grp in merged.groupby(lga_col, sort=False):
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                x, y = grp[a].to_numpy(), grp[b].to_numpy()
                ok = ~(np.isnan(x) | np.isnan(y))
                if ok.sum() < 2:
                    continue
                rows.append(
                    {
                        "lga": lga,
                        "product_a": a,
                        "product_b": b,
                        "n": int(ok.sum()),
                        "pearson_r": round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 4),
                        "mean_bias": round(float((x[ok] - y[ok]).mean()), 4),
                        "rmse": round(float(np.sqrt(((x[ok] - y[ok]) ** 2).mean())), 4),
                    }
                )
    return pd.DataFrame(rows)


def resolution_caveat(resolution_deg: float, lga_area_km2: float) -> str:
    """Draft the limitations sentence for a given product resolution.

    Makes the mismatch concrete: how many LGAs fit inside one grid cell.
    """
    cell_km = resolution_deg * 111.0
    cell_area = cell_km ** 2
    ratio = cell_area / max(lga_area_km2, 1e-9)
    return (
        f"At {resolution_deg}° (~{cell_km:.0f} km, ~{cell_area:.0f} km² per cell) a single "
        f"grid cell covers roughly {ratio:.1f}× the area of a {lga_area_km2:.0f} km² LGA. "
        "Convective storms driving urban flash flooding in Lagos are typically smaller than "
        "one cell, so rainfall is spatially smoothed and short-duration intensity is "
        "underestimated — a limitation on the fine-scale skill of any model built on it."
    )


__all__ = [
    "CHIRPS_RESOLUTION_DEG",
    "ERA5_LAND_RESOLUTION_DEG",
    "ERA5_RESOLUTION_DEG",
    "IMERG_RESOLUTION_DEG",
    "LAGOS_BBOX",
    "aggregate_gridded_to_lga",
    "compare_rainfall_products",
    "resolution_caveat",
]
