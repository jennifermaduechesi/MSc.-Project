# Lagos Flood Risk Prediction — MSc Data Science

Student: Maduechesi Chidiebere Jennifer (25120133019)
Supervisor: Solomon Alili, Pan Atlantic University
Timeline: 12 weeks

---

## Project facts — locked

The uploaded Project Brief is the source of truth. Do not propose a different
project direction unless Jennifer explicitly asks.

- **Target:** 4-class flood risk (Low / Medium / High / Critical) per Lagos LGA,
  mirroring NiHSA's colour-coded scale. Classification, not regression.
- **Spatial unit:** all 20 Lagos LGAs.
- **Data sources (all public, no employer data):**
  - **FloodScan SFED** via HDX/OCHA — primary flood extent. Admin-2 level,
    Excel/CSV with HXL tag rows. ~2001–2025 available.
  - **NASA POWER API** — daily rainfall, temperature, humidity. *This is the
    weather source. Not ERA5, not CHIRPS, not IMERG.*
  - **NiHSA** — comparison benchmark only. Snapshot source, not a historical
    archive. Cannot be a primary label source.
- **Labels** are derived from FloodScan SFED thresholds, validated against NiHSA.
- **Models:** Persistence baseline → Logistic Regression → Random Forest → XGBoost.
- **Validation:** time-based train/test split. Never random shuffling.
- **Metrics:** macro F1 primary, per-class recall second. Accuracy is not a
  headline number — Critical is rare and high-stakes.
- **Deliverable:** Streamlit dashboard (LGA + date → risk band + top 2–3 drivers).

---

## How to explain things to Jennifer

- Pitch at a 12–15-year-old comprehension level. Simple language.
- Work through material sequentially. Do not skip steps.
- Direct answers. No filler, no "great question", no unnecessary hedging.
- Visual aids (tables, diagrams) where they help.
- Organise options by difficulty tier.
- She values honest assessment over validation. Flag problems early and plainly.

---

## Standing rules

1. **After every completed task or section, write a summary MD file.** Do not wait
   to be asked. Convention: `docs/summaries/YYYY-MM-DD_TOPIC.md`. Cover: what was
   done, key decisions, what's next.
2. **Never introduce a random-shuffle split.** Not in code, not in a notebook,
   not "just to check".
3. **Fit label thresholds on training data only.** Deriving cut points from the
   full series leaks the test period into the labels.
4. **Report macro F1 with its spread across folds**, not just the mean.
5. **Break results down per LGA.** An aggregate score hides a model that works in
   Ikeja and fails in Ibeju-Lekki.

---

## Key architectural decisions already made

- **`lga_id` (stable numeric) is the join key across all sources.** LGA name
  spellings differ between FloodScan, NASA POWER and NiHSA. 23 spelling variants
  are handled in `lgautils.py`. Never join on name text.
- Scripts are built defensively: retries, resume-on-restart, flexible column
  matching, validation against messy inputs.

---

## Known data constraints — state these in the methodology

- **NASA POWER resolution collapses the study area.** All 20 LGAs fall into only
  ~3 distinct NASA POWER grid cells (~50–55 km). Weather features therefore act
  as a *shared seasonal driver*; SFED carries most of the per-LGA signal. This
  must be addressed explicitly in methodology and feature-importance analysis.
- **NiHSA is a snapshot, not an archive.** Must be logged as dated snapshots.
- **FloodScan history is ~24 years, longer than the brief assumed (~10).** More
  rare Critical events available, but more cleaning.
- **Satellite rainfall is weak at daily resolution in Nigeria.** Best product
  tested against Nigerian gauges reached median r = 0.33 daily vs 0.86 monthly
  (Ganiyu et al., 2025). Favour accumulated windows over same-day rainfall.

---

## Current status

**Phase 1 (weeks 1–2) — underway.** These scripts exist on Jennifer's local
machine and are **not yet in this repo**:
`01lgareference.py`, `lgautils.py`, `02pullnasapower.py`, `03loadfloodscan.py`,
`04loadnihsa.py`, `READMEdatacollection.md`, `requirements.txt`, `PHASE1SUMMARY.md`

**In this repo:** verified literature (`docs/literature/`) and a modelling
scaffold (`src/lagos_flood/`, 42 tests passing).

**Scaffold caveat:** `data/floodscan.py` and `data/climate.py` were written
before the brief was available and assume the wrong sources (raster GIS,
ERA5/CHIRPS). Neither has ever been run. They need replacing with NASA POWER and
HDX Excel loaders. Everything else in the scaffold matches the brief.

**Open questions for the supervisor:**
- Full ~24 years of FloodScan, or a narrower window?
- NASA POWER from 2018, or extend back to 1998?

---

## Proposed refinements — need Jennifer's sign-off

Not part of the original brief. Adopt, reject, or amend.

1. **Gate the ML models behind the persistence baseline.** The brief calls
   persistence essential but doesn't make it blocking. Rule: no RF/XGBoost result
   gets written up until it has been compared against persistence on the same
   split. "The ML model didn't beat persistence" is a legitimate finding; not
   having checked is not.

2. **Make the label-threshold sensitivity table a required methodology artifact.**
   There is no published convention for turning SFED into 4 classes — the
   literature review found no peer-reviewed precedent. So the cut points are a
   choice to defend, not inherit. `python -m lagos_flood labels-sweep` produces
   the table. On synthetic data the Critical class share swung from 2.5% to 22%
   depending on cut points alone. That is worth showing an examiner before they
   ask.

3. **Decide the binary-fallback trigger now, not later.** The brief offers
   "elevated vs not" as a fallback if classes are too skewed. Set the trigger in
   advance (e.g. "if the test period contains fewer than 30 Critical days, report
   the binary framing alongside the 4-class one"). Deciding after seeing results
   looks like fishing.

4. **State the forecast lead time explicitly and report it.** Research question 4
   asks about lead time. The model must predict t+h from features known at t, with
   h named in the results. The scaffold already has `lead_time_days`. Suggest
   reporting at least two horizons (e.g. 1 day and 7 days) so the lead-time
   question gets a real answer.

5. **Pin versions and seeds.** Originality claim #3 is "fully reproducible — an
   examiner can rerun the whole pipeline". That needs pinned library versions and
   fixed random seeds, or reruns will differ.

6. **Run the leakage checklist before any result is written up:** temporal split
   verified · embargo ≥ max(feature lookback, lead time) · label thresholds fitted
   on training only · no random shuffle anywhere. The scaffold has automated
   checks for all four.
