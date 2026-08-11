"""The 20 Lagos LGAs, and the numeric key everything joins on.

LGA names are spelled differently by every source. FloodScan writes
"Ajeromi/Ifelodun", NiHSA writes "Ajeromi-Ifelodun", other exports write
"AJEROMI IFELODUN". Joining tables on that text silently drops rows: the merge
succeeds, the row count falls, and the missing LGAs look like missing data
rather than a broken join.

So nothing joins on names. Every source is mapped to a stable numeric
``lga_id`` first, and the join happens on that. The canonical name is carried
alongside for display only.

Centroids are used to pick NASA POWER grid points. They are approximate,
hand-entered values good to about a kilometre — fine for choosing a ~55 km
weather grid cell, not fine for anything cartographic. Replace them with GRID3
polygon centroids before any map goes in the dissertation.
"""
from __future__ import annotations

import re

import pandas as pd

#: (lga_id, canonical name, latitude, longitude)
LGA_REFERENCE: tuple[tuple[int, str, float, float], ...] = (
    (1, "Agege", 6.6155, 3.3242),
    (2, "Ajeromi-Ifelodun", 6.4667, 3.3333),
    (3, "Alimosho", 6.6000, 3.2500),
    (4, "Amuwo-Odofin", 6.4500, 3.2833),
    (5, "Apapa", 6.4500, 3.3667),
    (6, "Badagry", 6.4315, 2.8876),
    (7, "Epe", 6.5833, 3.9833),
    (8, "Eti-Osa", 6.4500, 3.5500),
    (9, "Ibeju-Lekki", 6.4667, 3.9333),
    (10, "Ifako-Ijaiye", 6.6500, 3.3167),
    (11, "Ikeja", 6.6018, 3.3515),
    (12, "Ikorodu", 6.6194, 3.5106),
    (13, "Kosofe", 6.5667, 3.4000),
    (14, "Lagos Island", 6.4550, 3.4060),
    (15, "Lagos Mainland", 6.5000, 3.3833),
    (16, "Mushin", 6.5333, 3.3500),
    (17, "Ojo", 6.4667, 3.1833),
    (18, "Oshodi-Isolo", 6.5500, 3.3167),
    (19, "Shomolu", 6.5400, 3.3800),
    (20, "Surulere", 6.5000, 3.3500),
)

#: Spellings that survive normalisation but still differ. Keys are already
#: normalised; values are canonical names.
ALIASES: dict[str, str] = {
    "somolu": "Shomolu",
    "shomulu": "Shomolu",
    "ifako ijaye": "Ifako-Ijaiye",
    "ifako ijaiye": "Ifako-Ijaiye",
    "ifako": "Ifako-Ijaiye",
    "amuwo odofin": "Amuwo-Odofin",
    "ajeromi ifelodun": "Ajeromi-Ifelodun",
    "ajeromi": "Ajeromi-Ifelodun",
    "eti osa": "Eti-Osa",
    "etiosa": "Eti-Osa",
    "ibeju lekki": "Ibeju-Lekki",
    "ibeju": "Ibeju-Lekki",
    "lekki": "Ibeju-Lekki",
    "oshodi isolo": "Oshodi-Isolo",
    "oshodi": "Oshodi-Isolo",
    "badagri": "Badagry",
    "lagos island": "Lagos Island",
    "island": "Lagos Island",
    "lagos mainland": "Lagos Mainland",
    "mainland": "Lagos Mainland",
    "surulere lagos": "Surulere",
    "epe lagos": "Epe",
}

_CANONICAL = {name: lga_id for lga_id, name, _, _ in LGA_REFERENCE}


def normalise_name(name: str) -> str:
    """Reduce an LGA name to a comparable form.

    Lowercases, turns slashes, hyphens and underscores into spaces, strips a
    trailing "LGA"/"Local Government Area", and collapses whitespace.
    """
    text = str(name).strip().lower()
    text = re.sub(r"[/_\-]+", " ", text)
    text = re.sub(r"\b(local government area|local government|lga|l\.g\.a\.?)\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_lga_id(name: str) -> int | None:
    """Map any spelling to its ``lga_id``. Returns None if unrecognised.

    Returning None rather than guessing is deliberate — a wrong match is far
    worse than a reported miss, because it silently attributes one LGA's floods
    to another.
    """
    normalised = normalise_name(name)

    for canonical, lga_id in _CANONICAL.items():
        if normalise_name(canonical) == normalised:
            return lga_id

    if normalised in ALIASES:
        return _CANONICAL[ALIASES[normalised]]

    return None


def lga_reference_frame() -> pd.DataFrame:
    """The reference table: lga_id, name, latitude, longitude."""
    return pd.DataFrame(
        LGA_REFERENCE, columns=["lga_id", "lga", "latitude", "longitude"]
    )


def attach_lga_id(
    df: pd.DataFrame,
    name_col: str,
    *,
    drop_unmatched: bool = True,
    strict: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Add ``lga_id`` and canonical ``lga`` columns by resolving ``name_col``.

    Returns the frame and the list of names that could not be resolved. With
    ``strict=True`` an unresolved name raises instead — use that once the
    name-matching pass is finished and any miss means a real problem.
    """
    resolved = df[name_col].map(resolve_lga_id)
    unmatched = sorted(set(df.loc[resolved.isna(), name_col].astype(str)))

    if unmatched and strict:
        raise ValueError(
            f"{len(unmatched)} LGA name(s) could not be matched: {unmatched}. "
            "Add them to ALIASES in lga_reference.py."
        )

    out = df.copy()
    out["lga_id"] = resolved
    if drop_unmatched:
        out = out.loc[out["lga_id"].notna()].copy()
    out["lga_id"] = out["lga_id"].astype("Int64")

    names = {lga_id: name for lga_id, name, _, _ in LGA_REFERENCE}
    out["lga"] = out["lga_id"].map(names)
    return out.reset_index(drop=True), unmatched


def coverage_report(df: pd.DataFrame, *, lga_id_col: str = "lga_id") -> pd.DataFrame:
    """Which of the 20 LGAs are present, and which are missing.

    Run this after every load. A source covering 18 of 20 LGAs is a finding that
    belongs in the methodology, not something to discover during modelling.
    """
    present = set(df[lga_id_col].dropna().astype(int))
    rows = [
        {"lga_id": lga_id, "lga": name, "present": lga_id in present}
        for lga_id, name, _, _ in LGA_REFERENCE
    ]
    return pd.DataFrame(rows)


__all__ = [
    "ALIASES",
    "LGA_REFERENCE",
    "attach_lga_id",
    "coverage_report",
    "lga_reference_frame",
    "normalise_name",
    "resolve_lga_id",
]
