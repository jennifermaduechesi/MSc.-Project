"""NASA POWER daily weather for the 20 Lagos LGAs.

Replaces the earlier ERA5/CHIRPS module. The brief names NASA POWER as the
weather source, and this is it — free, open, no key required.

    https://power.larc.nasa.gov/api/temporal/daily/point

**The constraint that shapes the whole dissertation.** NASA POWER daily data is
served on the MERRA-2 grid: 0.5 degrees of latitude by 0.625 of longitude, which
is roughly 55 km by 69 km. Lagos State is about 100 km across. So the 20 LGAs do
not get 20 different weather series — they collapse onto a handful of grid cells,
and LGAs sharing a cell receive *identical* rainfall, temperature and humidity.

Two consequences to state in the methodology:

1. Weather features act as a **shared seasonal driver**, not a per-LGA signal.
   Whatever separates Ikeja from Ibeju-Lekki on a given day comes from FloodScan
   SFED and from static geography, not from NASA POWER.
2. Feature-importance results must be read with this in mind. A weather feature
   ranking highly is telling you about season, not about place.

:func:`grid_collapse_report` quantifies it for the write-up.

Because LGAs share cells, :func:`fetch_lagos_weather` fetches once per *cell* and
then expands the result back out to every LGA in that cell. For 20 LGAs that is
typically 2–4 requests instead of 20.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .lga_reference import lga_reference_frame

log = logging.getLogger(__name__)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

#: MERRA-2 native grid spacing used by NASA POWER daily products.
GRID_LAT_DEG = 0.5
GRID_LON_DEG = 0.625

#: NASA POWER parameter codes → the column names used in this project.
PARAMETERS: dict[str, str] = {
    "PRECTOTCORR": "rain_mm",      # bias-corrected total precipitation, mm/day
    "T2M": "temp_c",               # mean temperature at 2 m, degrees C
    "T2M_MAX": "temp_max_c",
    "T2M_MIN": "temp_min_c",
    "RH2M": "humidity_pct",        # relative humidity at 2 m, percent
}

#: NASA POWER writes this instead of null. Left in place it silently poisons
#: every rolling sum it touches.
MISSING_SENTINEL = -999.0


def grid_cell(latitude: float, longitude: float) -> tuple[float, float]:
    """Snap a coordinate to the NASA POWER grid point that serves it."""
    return (
        round(round(latitude / GRID_LAT_DEG) * GRID_LAT_DEG, 4),
        round(round(longitude / GRID_LON_DEG) * GRID_LON_DEG, 4),
    )


def grid_collapse_report(reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """How many LGAs share each NASA POWER grid cell.

    Put this table in the methodology chapter. It is the evidence for the
    resolution limitation, and an examiner will ask about it.
    """
    ref = reference if reference is not None else lga_reference_frame()
    cells = ref.apply(lambda r: grid_cell(r["latitude"], r["longitude"]), axis=1)
    out = ref.assign(
        grid_lat=[c[0] for c in cells],
        grid_lon=[c[1] for c in cells],
    )
    grouped = (
        out.groupby(["grid_lat", "grid_lon"])
        .agg(n_lgas=("lga", "size"), lgas=("lga", lambda s: ", ".join(sorted(s))))
        .reset_index()
        .sort_values("n_lgas", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def fetch_point(
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    *,
    parameters: dict[str, str] | None = None,
    community: str = "AG",
    max_retries: int = 4,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch one grid point's daily series.

    Retries with exponential backoff — the API is free and occasionally refuses
    a burst, and a half-finished pull is worse than a slow one.

    ``start`` and ``end`` are ``YYYYMMDD`` or anything pandas can parse.
    """
    params = parameters or PARAMETERS
    query = urllib.parse.urlencode(
        {
            "parameters": ",".join(params),
            "community": community,
            "longitude": f"{longitude:.4f}",
            "latitude": f"{latitude:.4f}",
            "start": pd.Timestamp(start).strftime("%Y%m%d"),
            "end": pd.Timestamp(end).strftime("%Y%m%d"),
            "format": "JSON",
        }
    )
    url = f"{BASE_URL}?{query}"

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.load(response)
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            wait = 2 ** (attempt + 1)
            log.warning("NASA POWER attempt %d/%d failed (%s); retrying in %ds",
                        attempt + 1, max_retries, exc, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(
            f"NASA POWER failed after {max_retries} attempts for "
            f"({latitude}, {longitude}): {last_error}"
        )

    try:
        block = payload["properties"]["parameter"]
    except KeyError as exc:
        raise ValueError(f"unexpected NASA POWER response shape: {list(payload)}") from exc

    frame = pd.DataFrame({column: pd.Series(block[code]) for code, column in params.items()
                          if code in block})
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame = frame.rename_axis("date").reset_index()
    frame.insert(0, "grid_lon", grid_cell(latitude, longitude)[1])
    frame.insert(0, "grid_lat", grid_cell(latitude, longitude)[0])
    return clean_sentinels(frame)


def clean_sentinels(df: pd.DataFrame, sentinel: float = MISSING_SENTINEL) -> pd.DataFrame:
    """Turn NASA POWER's -999 fill value into proper NaN.

    Uses a tolerance rather than equality because the fill value comes back as a
    float and can arrive as -999.0000001.
    """
    out = df.copy()
    for column in out.select_dtypes(include=[np.number]).columns:
        if column in {"grid_lat", "grid_lon", "lga_id"}:
            continue
        out.loc[np.isclose(out[column], sentinel, atol=1e-3), column] = np.nan
    return out


def fetch_lagos_weather(
    start: str,
    end: str,
    *,
    reference: pd.DataFrame | None = None,
    cache_path: Path | str | None = None,
    parameters: dict[str, str] | None = None,
    pause_seconds: float = 1.0,
) -> pd.DataFrame:
    """Daily weather for all 20 LGAs, one row per (lga_id, date).

    Fetches once per distinct grid cell, then expands to the LGAs sharing it.
    ``cache_path`` writes each cell's series to disk as it arrives, so an
    interrupted pull resumes instead of restarting.

    The returned frame carries ``grid_lat``/``grid_lon`` so it stays visible
    which LGAs are sharing a weather series.
    """
    ref = reference if reference is not None else lga_reference_frame()
    ref = ref.copy()
    cells = ref.apply(lambda r: grid_cell(r["latitude"], r["longitude"]), axis=1)
    ref["grid_lat"] = [c[0] for c in cells]
    ref["grid_lon"] = [c[1] for c in cells]

    unique_cells = ref[["grid_lat", "grid_lon"]].drop_duplicates()
    log.info("%d LGAs collapse onto %d NASA POWER grid cell(s)", len(ref), len(unique_cells))

    cache_dir = Path(cache_path) if cache_path else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for _, cell in unique_cells.iterrows():
        lat, lon = float(cell["grid_lat"]), float(cell["grid_lon"])
        cached = cache_dir / f"power_{lat}_{lon}_{start}_{end}.csv" if cache_dir else None

        if cached and cached.exists():
            log.info("cell (%.4f, %.4f) already cached, skipping fetch", lat, lon)
            frames.append(pd.read_csv(cached, parse_dates=["date"]))
            continue

        log.info("fetching cell (%.4f, %.4f)", lat, lon)
        series = fetch_point(lat, lon, start, end, parameters=parameters)
        if cached:
            series.to_csv(cached, index=False)
        frames.append(series)
        time.sleep(pause_seconds)

    weather = pd.concat(frames, ignore_index=True)
    panel = ref.merge(weather, on=["grid_lat", "grid_lon"], how="left")

    columns = ["lga_id", "lga", "date", "grid_lat", "grid_lon"]
    columns += [c for c in (parameters or PARAMETERS).values() if c in panel.columns]
    return panel[columns].sort_values(["lga_id", "date"]).reset_index(drop=True)


def load_power_csv(path: Path | str, *, name_col: str | None = None) -> pd.DataFrame:
    """Load a CSV produced by an existing NASA POWER pull script.

    Provided so Phase 1 output can feed this pipeline directly rather than being
    re-fetched. Renames NASA POWER parameter codes to project column names,
    cleans sentinels, and attaches ``lga_id`` if a name column is present.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={code: col for code, col in PARAMETERS.items() if code in df.columns})

    for candidate in ("date", "DATE", "YYYYMMDD"):
        if candidate in df.columns:
            df["date"] = pd.to_datetime(df[candidate].astype(str), errors="coerce")
            break
    else:
        raise KeyError(f"no date column found in {path}; have {list(df.columns)}")

    df = clean_sentinels(df)

    if "lga_id" not in df.columns and name_col:
        from .lga_reference import attach_lga_id

        df, unmatched = attach_lga_id(df, name_col)
        if unmatched:
            log.warning("unmatched LGA names in %s: %s", path, unmatched)

    return df.sort_values(["lga_id", "date"]).reset_index(drop=True)


__all__ = [
    "BASE_URL",
    "GRID_LAT_DEG",
    "GRID_LON_DEG",
    "MISSING_SENTINEL",
    "PARAMETERS",
    "clean_sentinels",
    "fetch_lagos_weather",
    "fetch_point",
    "grid_cell",
    "grid_collapse_report",
    "load_power_csv",
]
