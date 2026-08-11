# Summary — Real FloodScan data: three problems

**Date:** 11 August 2026
**Status:** ⚠️ Needs a supervisor conversation before Phase 2 continues

---

## What was done

Downloaded the real FloodScan file from HDX and ran it through the loader.

- Dataset: **FloodScan: Near Real-Time and Historical Flood Mapping**
- Publisher: JANUS Atmospheric and Environmental Research (AER)
- File: `hdx_floodscan_zonal_stats.xlsx`, 16.5 MB
- Licence: **CC BY-IGO** — free to use with credit. Good for a dissertation.
- No login needed.

**The loader works.** It read the real file correctly on the first proper run,
and all 17 LGA spellings resolved — including the awkward ones, `Ibeju/Lekki`
and `Ifako-Ijaye`.

Three bugs were found and fixed by using the real file:

1. The workbook has three sheets (`Readme`, `admin1`, `admin2`). Reading the
   default first sheet returns the Readme.
2. The file holds **12 countries**, so it needs a country filter, not just a
   state filter.
3. `RP` (return period) is **text**, not numbers — `"1-1.5"`, `">10"`.
   Converting it to numbers blanked the entire column.

---

## ⚠️ Problem 1 — the file only holds 89 days

| | |
|---|---|
| Date range | **2026-05-11 to 2026-08-07** |
| Days of data | **89** |

It is a **rolling 90-day window**, not a historical archive. Every update
replaces the oldest day.

Your brief assumes years of history to train on. This file cannot provide it.

**Why:** AER sells the historical record. The FloodScan interface lets you browse
for free, but bulk historical data is a paid product — subscriptions are
advertised at **$10,000** for the first year and **$25,000/year** after.

This also threatens originality claim #3 in your brief — *"fully reproducible,
an examiner can rerun the whole pipeline"* — because an examiner cannot rerun
something behind a paywall.

---

## ⚠️ Problem 2 — three LGAs are missing entirely

FloodScan has data for **17 of your 20 LGAs**. Missing:

- **Agege**
- **Ajeromi-Ifelodun**
- **Lagos Island**

The dataset's own Readme explains it: *"some administrative boundaries may not
have summary statistics available. This happens when administrative polygons are
sufficiently small relative to the size of the input raster dataset."*

FloodScan's grid squares are about 10 km across. Those three LGAs are smaller
than that. They are also among the densest, most built-up parts of Lagos — so
the areas where flooding affects the most people are the ones the data cannot
see.

---

## ⚠️ Problem 3 — this is the serious one

Almost every reading is **zero**.

| LGA | Days with exactly zero flooding | Highest reading |
|---|---|---|
| Badagry | **100%** | 0.00000 |
| Ifako-Ijaiye | **100%** | 0.00000 |
| Ojo | 98.9% | 0.00006 |
| Shomolu | 98.9% | 0.00035 |
| Mushin | 98.9% | 0.00035 |
| Oshodi-Isolo | 98.9% | 0.00017 |
| Alimosho | 97.8% | 0.00040 |
| Apapa | 96.6% | 0.00339 |
| Amuwo-Odofin | 96.6% | 0.00207 |
| Lagos Mainland | 96.6% | 0.00339 |
| Surulere | 96.6% | 0.00339 |
| Ikeja | 93.3% | 0.15074 |
| Eti-Osa | 88.8% | 0.00021 |
| Kosofe | 84.3% | 0.05025 |
| Ibeju-Lekki | 80.9% | 0.00024 |
| Ikorodu | 48.3% | 0.00570 |
| **Epe** | **4.5%** | 0.01090 |

**Two LGAs recorded no flooding at all across all 89 days. Eleven are 96% zero
or worse.**

And this window covers **May to August — the peak of the rainy season.** It is
not an unrepresentative dry stretch.

### Why this matters

You cannot split a column that is 98% zeros into Low / Medium / High / Critical.
There is nothing there to separate. For Badagry and Ifako-Ijaiye there is
literally no signal to model.

### Why it is happening

Look at which LGAs *do* show variation: **Epe, Ikorodu, Kosofe** — the large,
rural, riverine ones. The dense urban ones are flat zero.

FloodScan measures **large-scale inland flooding** — rivers spilling across
floodplains, seen in 10 km squares. Lagos's flooding is mostly **urban drainage
flooding**: blocked drains, streets and streams inside a built-up city. At 10 km
resolution, a flooded street network does not register as a flooded fraction.

**FloodScan may simply be the wrong instrument for the kind of flooding this
project is about.** That is a genuine methodological finding, not a failure on
your part — but it needs deciding now, not in week 9.

---

## Options

**A. Ask AER for free academic access.** Costs one email. Many data providers
grant student licences. Would fix Problem 1, and possibly nothing else.

**B. Switch flood-extent source to NASA MODIS NRT Global Flood Product.**
- Free, no purchase, daily, **2012 to present**
- About **250 m** resolution — roughly 40× finer than FloodScan, so it would
  likely see all 20 LGAs
- Already in your reading list (Policelli et al., 2017; Nigro et al., 2014)
- Downside: optical, so cloud blocks it. Nigro et al. found 44% of events well
  detected, rising to 66% once cloud-obscured cases are excluded.

**C. Sentinel-1 radar via Google Earth Engine.** Free, 2014 to present, ~10 m,
and radar sees through cloud. Best data quality; most processing work.

**D. Keep FloodScan but narrow the study** to the LGAs where it shows signal
(Epe, Ikorodu, Kosofe, Ikeja). Abandons the all-20-LGAs design.

**E. Change the target.** Predict a rainfall-driven risk level checked against
NiHSA and news reports, instead of deriving labels from SFED.

---

## Recommended next steps

1. **Take this to your supervisor this week.** It affects whether the project
   works as designed. It is much better raised now than discovered in week 9.
2. **Email AER** asking for academic access to the historical record. One email,
   possible large payoff.
3. **In parallel, look at MODIS.** Option B keeps your whole design intact —
   same target, same models, same validation — and only swaps where the labels
   come from. It is the smallest change that fixes all three problems.

**Do not spend more time on modelling until the label source is settled.** Every
model result depends on it.

---

## Good news

- The loader works on real data, and the name matching is right.
- The NASA POWER weather side is working and tested against the live API.
- The models, splitting, metrics and explanations are built and tested.
- None of that work is wasted by switching label source — everything downstream
  of "daily flood value per LGA" stays exactly the same.

**What changes is one input. What stays is everything else.**
