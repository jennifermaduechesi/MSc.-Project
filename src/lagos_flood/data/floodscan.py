"""FloodScan SFED loading from HDX exports.

Replaces the earlier raster module. The brief takes FloodScan from HDX as
Excel/CSV at admin-2 level, which means **it already arrives per LGA** — there
is no GIS step, no polygons, no zonal statistics. The earlier version solved a
problem this project does not have.

    https://data.humdata.org/dataset/floodscan

Two quirks of HDX files this module handles.

**HXL tag rows.** HDX files carry a machine-readable row directly beneath the
header, holding tags like ``#date`` and ``#adm2+name``. Loaded naively it
becomes the first data row, and every numeric column silently turns into text.

**Name spellings.** admin-2 names will not match Lagos's official LGA names.
Everything is resolved to ``lga_id`` on load, and anything unmatched is reported
rather than dropped in silence.

On provenance: FloodScan has **no dedicated peer-reviewed validation paper**.
The citable basis is Galantowicz & Picton (2021) plus AER/NASA technical
reports. State that in the methodology rather than implying validation that does
not exist.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .lga_reference import attach_lga_id, coverage_report

log = logging.getLogger(__name__)

#: Patterns matched case-insensitively against column names, in priority order.
COLUMN_PATTERNS: dict[str, tuple[str, ...]] = {
    "date": (r"^date$", r"^time$", r"date"),
    "admin1": (r"^adm1[_ ]?name$", r"^admin1$", r"^state$", r"adm1"),
    "admin2": (r"^adm2[_ ]?name$", r"^admin2$", r"^lga$", r"adm2"),
    "sfed_fraction": (r"^sfed[_ ]?mean$", r"^sfed$", r"sfed(?!.*base)"),
    "sfed_baseline": (r"sfed.*base", r"base.*sfed"),
    "return_period": (r"return[_ ]?period", r"^rp$"),
}

LAGOS_ALIASES = {"lagos", "lagos state"}


def _is_hxl_row(row: pd.Series) -> bool:
    """True if a row looks like an HXL tag row rather than data."""
    values = [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]
    if not values:
        return False
    return sum(v.startswith("#") for v in values) / len(values) >= 0.5


def _match_columns(columns: list[str]) -> dict[str, str]:
    """Map canonical names to whichever column in the file holds them."""
    found: dict[str, str] = {}
    for canonical, patterns in COLUMN_PATTERNS.items():
        for pattern in patterns:
            for column in columns:
                if re.search(pattern, str(column).strip(), flags=re.IGNORECASE):
                    found[canonical] = column
                    break
            if canonical in found:
                break
    return found


def inspect_file(path: Path | str, n_rows: int = 8) -> None:
    """Print a file's shape, columns, matched columns and a preview.

    Run this first on any new HDX download. Column names change between
    releases, and it is much cheaper to look than to debug a silent mismatch.
    """
    raw = _read_raw(path, nrows=n_rows + 2)
    print(f"file: {path}")
    print(f"columns ({len(raw.columns)}): {list(raw.columns)}")
    if len(raw) and _is_hxl_row(raw.iloc[0]):
        print(f"HXL tag row detected: {list(raw.iloc[0])}")
    matched = _match_columns(list(raw.columns))
    print("matched columns:")
    for canonical in COLUMN_PATTERNS:
        print(f"  {canonical:<16} -> {matched.get(canonical, '*** NOT FOUND ***')}")
    print("\npreview:")
    print(raw.head(n_rows).to_string())


def _read_raw(path: Path | str, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows)
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def load_floodscan(
    path: Path | str,
    *,
    admin1_filter: str | None = "Lagos",
    strict_names: bool = False,
    sfed_as_fraction: bool = True,
) -> pd.DataFrame:
    """Load an HDX FloodScan export into a tidy per-LGA daily frame.

    Returns columns ``lga_id``, ``lga``, ``date``, ``sfed_fraction``, plus
    ``sfed_baseline`` and ``return_period`` when the file carries them.

    ``sfed_as_fraction`` converts a 0–100 percentage column to a 0–1 fraction.
    The check is on the observed range, because HDX has shipped both.
    """
    raw = _read_raw(path)

    if len(raw) and _is_hxl_row(raw.iloc[0]):
        log.info("dropping HXL tag row")
        raw = raw.iloc[1:].reset_index(drop=True)

    matched = _match_columns(list(raw.columns))
    for required in ("date", "admin2", "sfed_fraction"):
        if required not in matched:
            raise KeyError(
                f"could not find a {required!r} column in {path}. "
                f"Columns present: {list(raw.columns)}. "
                "Run inspect_file() and extend COLUMN_PATTERNS if the export changed."
            )

    df = raw.rename(columns={v: k for k, v in matched.items()})
    keep = [c for c in COLUMN_PATTERNS if c in df.columns]
    df = df[keep].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    if bad_dates:
        log.warning("dropping %d row(s) with unparseable dates", bad_dates)
        df = df.loc[df["date"].notna()]

    if admin1_filter and "admin1" in df.columns:
        before = len(df)
        mask = df["admin1"].astype(str).str.strip().str.lower().isin(
            LAGOS_ALIASES if admin1_filter.lower() == "lagos" else {admin1_filter.lower()}
        )
        df = df.loc[mask]
        log.info("admin1 filter %r: %d -> %d rows", admin1_filter, before, len(df))
        if df.empty:
            raise ValueError(
                f"no rows left after filtering admin1 to {admin1_filter!r}. "
                f"Values present: {sorted(set(raw[matched['admin1']].astype(str)))[:10]}"
            )

    for column in ("sfed_fraction", "sfed_baseline", "return_period"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df, unmatched = attach_lga_id(df, "admin2", strict=strict_names)
    if unmatched:
        log.warning("%d admin2 name(s) not matched and dropped: %s", len(unmatched), unmatched)

    if sfed_as_fraction:
        for column in ("sfed_fraction", "sfed_baseline"):
            if column in df.columns and df[column].max(skipna=True) > 1.5:
                log.info("%s looks like a percentage; dividing by 100", column)
                df[column] = df[column] / 100.0

    for column in ("sfed_fraction", "sfed_baseline"):
        if column in df.columns:
            df[column] = df[column].clip(lower=0.0, upper=1.0)

    out_columns = ["lga_id", "lga", "date", "sfed_fraction"]
    out_columns += [c for c in ("sfed_baseline", "return_period") if c in df.columns]
    return (
        df[out_columns]
        .drop_duplicates(subset=["lga_id", "date"])
        .sort_values(["lga_id", "date"])
        .reset_index(drop=True)
    )


def validate_sfed(df: pd.DataFrame, *, value_col: str = "sfed_fraction") -> pd.DataFrame:
    """Per-LGA data-quality summary. Run before deriving any labels.

    Long runs of exact zeros are the thing to look at. They may be genuine dry
    periods or a masked/absent input, and once labels are cut the two are
    indistinguishable — so check here, where they are still separable.
    """
    grouped = df.groupby(["lga_id", "lga"])[value_col]
    report = pd.DataFrame(
        {
            "n_days": grouped.size(),
            "n_missing": grouped.apply(lambda s: int(s.isna().sum())),
            "pct_exact_zero": grouped.apply(lambda s: round(float((s == 0).mean() * 100), 2)),
            "mean": grouped.mean().round(5),
            "p99": grouped.quantile(0.99).round(5),
            "max": grouped.max().round(5),
        }
    ).reset_index()
    return report.sort_values("pct_exact_zero", ascending=False).reset_index(drop=True)


def add_sfed_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``sfed_deviation`` = SFED minus its baseline.

    The brief lists this as an engineered feature. If the export has no baseline
    column, one is computed as the day-of-year mean per LGA — but note that a
    baseline computed from the full series has seen the test period, so when
    self-computed it must be fitted on training data only.
    """
    out = df.copy()
    if "sfed_baseline" not in out.columns:
        log.warning(
            "no sfed_baseline column; computing day-of-year means. Fit these on "
            "training data only, or the baseline leaks the test period."
        )
        doy = out["date"].dt.dayofyear
        out["sfed_baseline"] = out.groupby(["lga_id", doy])["sfed_fraction"].transform("mean")
    out["sfed_deviation"] = out["sfed_fraction"] - out["sfed_baseline"]
    return out


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Which of the 20 LGAs this export actually covers."""
    return coverage_report(df)


def empirical_return_period(df: pd.DataFrame, *, value_col: str = "sfed_fraction") -> pd.DataFrame:
    """Add an empirical return period per LGA, in years.

    Rank-based: the largest observed value in an n-year record gets a return
    period of about n years. Rough by construction, and only meaningful once the
    record is long — with ~24 years of FloodScan it is usable, with 5 it is not.
    """
    out = df.copy()
    pieces = []
    for _lga_id, group in out.groupby("lga_id", sort=False):
        years = max((group["date"].max() - group["date"].min()).days / 365.25, 1e-9)
        ranks = group[value_col].rank(ascending=False, method="average")
        group = group.copy()
        group["return_period_years"] = np.where(
            group[value_col] > 0, (len(group) + 1) / ranks * (years / len(group)), np.nan
        )
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


__all__ = [
    "COLUMN_PATTERNS",
    "add_sfed_deviation",
    "coverage",
    "empirical_return_period",
    "inspect_file",
    "load_floodscan",
    "validate_sfed",
]
