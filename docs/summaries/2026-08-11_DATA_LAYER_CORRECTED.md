# Summary — Data layer corrected to match the brief

**Date:** 11 August 2026
**Phase:** 1–2 (data collection and loading)

---

## The problem

Two files were written before I had your project brief. They pointed at the
wrong data sources:

| File | Was built for | Your brief says |
|---|---|---|
| `data/climate.py` | ERA5 / CHIRPS / IMERG | **NASA POWER** |
| `data/floodscan.py` | Map-tile files needing GIS software | **HDX Excel/CSV** |

Neither had ever been run, so nothing was broken — but neither was usable.

---

## What was done

### `data/climate.py` — deleted

Replaced by `data/nasapower.py`. **Tested against the live NASA POWER API** — it
really does pull Lagos weather. Sample from June 2024:

| date | rain_mm | temp_c | humidity_pct |
|---|---|---|---|
| 2024-06-01 | 3.91 | 28.08 | 84.60 |
| 2024-06-06 | 17.95 | 27.19 | 90.48 |

278 mm of rain that month, temperatures 27–28 °C, humidity 84–90%. Those are
sensible numbers for Lagos in the middle of the rainy season.

It handles the things that break a weather pull:
- **−999 cleanup.** NASA POWER writes −999 instead of blank. Left alone it
  poisons every rolling total it touches.
- **Retries** with increasing waits, so a dropped connection doesn't leave a
  half-finished pull.
- **Resume.** Each grid cell is cached as it arrives; an interrupted run picks
  up where it stopped.

### `data/floodscan.py` — rewritten

Your FloodScan data comes from HDX as Excel/CSV **already broken down per LGA**.
There is no map/GIS step. The new loader handles:
- **HXL tag rows.** HDX files have a hidden row under the header with tags like
  `#adm2+name`. Loaded carelessly it becomes your first row of data and turns
  every number into text.
- **Flexible column matching**, so a renamed column in a new HDX release doesn't
  break the load.
- **Lagos filtering** and mapping every spelling to `lga_id`.

### `data/lga_reference.py` — new

The 20 LGAs, each with a permanent number (`lga_id`), coordinates, and a name
resolver. This is your Phase 1 decision, now enforced in code.

Names are never used to join tables. Nineteen spelling variants are tested:
`Somolu` → Shomolu, `Ajeromi/Ifelodun` → Ajeromi-Ifelodun, `Ifako Ijaye` →
Ifako-Ijaiye, `Badagri` → Badagry, and so on.

An unrecognised name returns **nothing** rather than a best guess. A wrong guess
would quietly credit one LGA's floods to another.

---

## ⚠️ Important correction to your notes

Your notes say the 20 LGAs collapse into **~3** NASA POWER grid cells.

Measured: **2 cells.** And it's lopsided.

| Grid cell | LGAs | Which |
|---|---|---|
| 6.5, 3.125 | **16** | Agege, Ajeromi-Ifelodun, Alimosho, Amuwo-Odofin, Apapa, Badagry, Ifako-Ijaiye, Ikeja, Kosofe, Lagos Island, Lagos Mainland, Mushin, Ojo, Oshodi-Isolo, Shomolu, Surulere |
| 6.5, 3.750 | 4 | Epe, Eti-Osa, Ibeju-Lekki, Ikorodu |

**16 of your 20 LGAs get identical weather data.** Same rainfall, same
temperature, same humidity, every single day.

### What this means

Rainfall **cannot** tell Ikeja apart from Surulere. It is physically the same
number. Anything that distinguishes those two LGAs has to come from FloodScan
SFED or from fixed geography.

Three consequences:

1. Say this plainly in your methodology. An examiner will spot it.
2. When you read feature importance, a high-ranking rainfall feature is telling
   you about **season**, not about **place**.
3. Your per-LGA results breakdown matters more than ever — the model has almost
   nothing to separate the 16 LGAs in that big cell.

This is a genuine limitation, not a mistake. It is the honest consequence of
using a free global weather product over a small state. Reporting it well is
worth more than hiding it.

Run `nasapower.grid_collapse_report()` to regenerate that table.

*(Caveat: my LGA centroids are approximate. Recheck once you have GRID3
centroids — the count may shift to 3, but the picture won't change.)*

---

## Features now match the brief

Went from 30 to **33 features**. Added:

- `sfed_fraction` and lags at **7, 14, 30 days** (your brief asked for these)
- `sfed_deviation` — SFED minus its day-of-year baseline
- `temp_c`, `humidity_pct`

Removed `soil_moisture` — that came from ERA5-Land, which isn't one of your
sources. The antecedent precipitation index already covers antecedent wetness,
and it's built from rainfall you actually have.

---

## Testing

**97 tests passing** (was 59). Lint clean.

| Module | Coverage |
|---|---|
| `lga_reference.py` | 100% |
| `synthetic.py` | 100% |
| `floodscan.py` | 72% |
| `nasapower.py` | 35% |

NASA POWER's low number is the network-calling code, which tests don't hit. I
verified that part by running it against the real API instead.

---

## What's next

**Your line of action, in order:**

1. **Download one FloodScan file from HDX** for Lagos. Run
   `floodscan.inspect_file(path)` on it — it prints the columns and shows which
   ones it matched. That tells us in seconds whether the loader works on the
   real thing.
2. **Send me your Phase 1 scripts** so we can compare. Your `03loadfloodscan.py`
   already handles HXL rows; if yours is better, we keep yours. I'm not
   attached to my version.
3. **Pull NASA POWER for real** — `fetch_lagos_weather('2018-01-01', '2025-12-31')`.
   Because of the 2-cell collapse this is only **2 API calls**, so it takes
   seconds, not hours.
4. **Then Phase 2:** look at the real SFED distribution and set the
   Low/Medium/High/Critical cut points. Nothing meaningful can be decided until
   we see real numbers.

**Still waiting on your supervisor:** full ~24 years of FloodScan or a shorter
window, and whether NASA POWER goes back to 1998.
