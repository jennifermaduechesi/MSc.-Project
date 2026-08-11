"""Tests for the data layer: LGA identity, NASA POWER, FloodScan HDX loading.

The join key and the loaders are where a silent failure does the most damage.
A dropped LGA looks like missing data, not a broken merge, so these are tested
against deliberately messy inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lagos_flood.data.floodscan import (
    add_sfed_deviation,
    load_floodscan,
    validate_sfed,
)
from lagos_flood.data.lga_reference import (
    LGA_REFERENCE,
    attach_lga_id,
    coverage_report,
    lga_reference_frame,
    normalise_name,
    resolve_lga_id,
)
from lagos_flood.data.nasapower import (
    MISSING_SENTINEL,
    clean_sentinels,
    grid_cell,
    grid_collapse_report,
)

# --------------------------------------------------------------- LGA identity


def test_there_are_exactly_twenty_lgas():
    assert len(LGA_REFERENCE) == 20
    ids = [row[0] for row in LGA_REFERENCE]
    names = [row[1] for row in LGA_REFERENCE]
    assert len(set(ids)) == 20
    assert len(set(names)) == 20


def test_lagos_centroids_are_inside_lagos_state():
    frame = lga_reference_frame()
    assert frame["latitude"].between(6.3, 6.8).all()
    assert frame["longitude"].between(2.6, 4.4).all()


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("Ikeja", "Ikeja"),
        ("IKEJA", "Ikeja"),
        ("  ikeja  ", "Ikeja"),
        ("Ikeja LGA", "Ikeja"),
        ("Ajeromi/Ifelodun", "Ajeromi-Ifelodun"),
        ("Ajeromi Ifelodun", "Ajeromi-Ifelodun"),
        ("AJEROMI-IFELODUN", "Ajeromi-Ifelodun"),
        ("Eti Osa", "Eti-Osa"),
        ("Eti-Osa", "Eti-Osa"),
        ("Somolu", "Shomolu"),
        ("Shomolu", "Shomolu"),
        ("Ifako Ijaye", "Ifako-Ijaiye"),
        ("Ifako-Ijaiye", "Ifako-Ijaiye"),
        ("Ibeju/Lekki", "Ibeju-Lekki"),
        ("Oshodi Isolo", "Oshodi-Isolo"),
        ("Badagri", "Badagry"),
        ("Lagos Island", "Lagos Island"),
        ("lagos-mainland", "Lagos Mainland"),
        ("Amuwo Odofin", "Amuwo-Odofin"),
    ],
)
def test_name_spellings_resolve_to_the_right_lga(written, expected):
    """Every spelling variant seen across FloodScan, NiHSA and GRID3."""
    names = {lga_id: name for lga_id, name, _, _ in LGA_REFERENCE}
    resolved = resolve_lga_id(written)
    assert resolved is not None, f"{written!r} did not resolve"
    assert names[resolved] == expected


def test_unknown_names_return_none_rather_than_guessing():
    """A wrong match silently attributes one LGA's floods to another."""
    assert resolve_lga_id("Ibadan North") is None
    assert resolve_lga_id("") is None
    assert resolve_lga_id("Zzzz") is None


def test_normalise_strips_lga_suffixes_and_punctuation():
    assert normalise_name("Eti-Osa L.G.A.") == "eti osa"
    assert normalise_name("IFAKO/IJAIYE Local Government Area") == "ifako ijaiye"


def test_attach_lga_id_reports_and_drops_unmatched():
    df = pd.DataFrame({"admin2": ["Ikeja", "Somolu", "Ibadan North"], "value": [1, 2, 3]})
    out, unmatched = attach_lga_id(df, "admin2")

    assert unmatched == ["Ibadan North"]
    assert len(out) == 2
    assert set(out["lga"]) == {"Ikeja", "Shomolu"}
    assert out["lga_id"].notna().all()


def test_attach_lga_id_can_fail_loudly():
    df = pd.DataFrame({"admin2": ["Ikeja", "Nowhere"]})
    with pytest.raises(ValueError, match="could not be matched"):
        attach_lga_id(df, "admin2", strict=True)


def test_coverage_report_flags_missing_lgas():
    df = pd.DataFrame({"lga_id": [1, 2, 3]})
    report = coverage_report(df)
    assert len(report) == 20
    assert report["present"].sum() == 3
    assert not report.loc[report["lga"] == "Surulere", "present"].iloc[0]


# ----------------------------------------------------------------- NASA POWER


def test_grid_cell_snaps_to_the_merra2_grid():
    lat, lon = grid_cell(6.6018, 3.3515)
    assert lat == pytest.approx(6.5)
    assert lon == pytest.approx(3.125)


def test_lagos_collapses_onto_a_handful_of_weather_cells():
    """The constraint the methodology has to state.

    All 20 LGAs share only a couple of NASA POWER grid cells, so weather is a
    shared seasonal driver rather than a per-LGA signal.
    """
    report = grid_collapse_report()
    assert len(report) <= 3, "expected Lagos to collapse onto very few cells"
    assert report["n_lgas"].sum() == 20
    # And the collapse is lopsided — most LGAs share a single cell.
    assert report["n_lgas"].max() >= 10


def test_sentinel_values_become_nan():
    df = pd.DataFrame(
        {
            "grid_lat": [6.5, 6.5],
            "rain_mm": [3.2, MISSING_SENTINEL],
            "temp_c": [MISSING_SENTINEL, 27.4],
        }
    )
    out = clean_sentinels(df)
    assert np.isnan(out.loc[1, "rain_mm"])
    assert np.isnan(out.loc[0, "temp_c"])
    assert out.loc[0, "rain_mm"] == 3.2
    # Grid coordinates are never treated as data.
    assert out["grid_lat"].notna().all()


def test_sentinel_cleaning_survives_float_drift():
    df = pd.DataFrame({"rain_mm": [-999.0000001, -998.9999, 5.0]})
    out = clean_sentinels(df)
    assert out["rain_mm"].isna().sum() == 2
    assert out.loc[2, "rain_mm"] == 5.0


# ------------------------------------------------------------------ FloodScan


HXL_CSV = """date,ADM1_NAME,ADM2_NAME,SFED_MEAN,SFED_BASELINE
#date,#adm1+name,#adm2+name,#indicator+sfed,#indicator+baseline
2023-06-01,Lagos,Ikeja,0.021,0.015
2023-06-02,Lagos,Ikeja,0.044,0.015
2023-06-01,Lagos,Somolu,0.011,0.008
2023-06-02,Lagos,Somolu,0.019,0.008
2023-06-01,Lagos,Ajeromi/Ifelodun,0.055,0.030
2023-06-01,Ogun,Abeokuta North,0.900,0.100
"""


@pytest.fixture
def hdx_csv(tmp_path):
    path = tmp_path / "floodscan_lagos.csv"
    path.write_text(HXL_CSV)
    return path


def test_hxl_tag_row_is_dropped_and_numbers_stay_numeric(hdx_csv):
    """Loaded naively the HXL row becomes data and turns SFED into text."""
    df = load_floodscan(hdx_csv)
    assert not df["date"].isna().any()
    assert pd.api.types.is_numeric_dtype(df["sfed_fraction"])
    assert "#date" not in df["date"].astype(str).tolist()


def test_lagos_filter_removes_other_states(hdx_csv):
    df = load_floodscan(hdx_csv)
    assert len(df) == 5
    assert "Abeokuta North" not in set(df["lga"])


def test_messy_admin2_names_map_to_canonical_lgas(hdx_csv):
    df = load_floodscan(hdx_csv)
    assert set(df["lga"]) == {"Ikeja", "Shomolu", "Ajeromi-Ifelodun"}
    assert df["lga_id"].notna().all()


def test_output_is_sorted_and_unique_per_lga_date(hdx_csv):
    df = load_floodscan(hdx_csv)
    assert not df.duplicated(subset=["lga_id", "date"]).any()
    assert df.equals(df.sort_values(["lga_id", "date"]).reset_index(drop=True))


def test_percentage_sfed_is_converted_to_a_fraction(tmp_path):
    path = tmp_path / "pct.csv"
    path.write_text(
        "date,ADM1_NAME,ADM2_NAME,SFED\n"
        "2023-06-01,Lagos,Ikeja,12.5\n"
        "2023-06-02,Lagos,Ikeja,44.0\n"
    )
    df = load_floodscan(path)
    assert df["sfed_fraction"].between(0, 1).all()
    assert df["sfed_fraction"].iloc[0] == pytest.approx(0.125)


def test_missing_required_column_fails_with_a_useful_message(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,ADM1_NAME\n2023-06-01,Lagos\n")
    with pytest.raises(KeyError, match="admin2"):
        load_floodscan(path)


def test_sfed_deviation_uses_the_supplied_baseline(hdx_csv):
    df = add_sfed_deviation(load_floodscan(hdx_csv))
    row = df[(df["lga"] == "Ikeja") & (df["date"] == "2023-06-02")].iloc[0]
    assert row["sfed_deviation"] == pytest.approx(0.044 - 0.015)


def test_validate_sfed_reports_per_lga_quality(hdx_csv):
    report = validate_sfed(load_floodscan(hdx_csv))
    assert set(report["lga"]) == {"Ikeja", "Shomolu", "Ajeromi-Ifelodun"}
    assert {"n_days", "n_missing", "pct_exact_zero", "max"}.issubset(report.columns)
