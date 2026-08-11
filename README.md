# Lagos Flood Risk Prediction

Per-LGA multi-class flood-risk forecasting for Lagos State, Nigeria.
MSc Data Science dissertation, Pan Atlantic University.

The task: for each of the 20 Lagos Local Government Areas, forecast a daily
flood-risk class — **Low / Medium / High / Critical** — at a configurable lead
time, using FloodScan SFED as the labelling source, rainfall and soil-moisture
reanalysis as drivers, and SHAP for explainability.

The literature review found no published study doing this. Existing Lagos work is
static susceptibility or vulnerability mapping — GIS/MCDA or a one-off random
forest — not a temporally validated classifier across all 20 LGAs. See
[`docs/literature/`](docs/literature/) for the verified evidence base.

---

## Quick start

```bash
pip install -e '.[dev]'          # core + tests
pip install -e '.[all]'          # everything, including geo and SHAP

python -m lagos_flood demo               # full pipeline on synthetic data
python -m lagos_flood demo --compare     # RF vs XGBoost vs LightGBM
python -m lagos_flood labels-sweep       # label-threshold sensitivity table
pytest                                   # 42 tests
```

`demo` runs on **synthetic data** so the pipeline can be exercised without
credentials. Those numbers are a plumbing check and mean nothing as findings.

---

## Three commitments the code enforces

**No random cross-validation, anywhere.** Daily per-LGA flood data is strongly
autocorrelated in time and space; random folds on autocorrelated data inflate
measured skill (Roberts et al., 2017; Bergmeir & Benítez, 2012). Every split in
`validation.py` is time-ordered, and there is deliberately no shuffled splitter
to reach for.

**The embargo is derived, not guessed.** Two overlaps have to be excluded at each
train/validation boundary: validation *features* reaching back into training, and
training *targets* reaching forward into validation. `Config.embargo_days` sizes
the gap from `max(feature lookback, lead time)` so both are covered.
`assert_no_temporal_leakage` re-checks it on every fold.

**Label thresholds are fitted on training data only.** Percentile cut points are
derived from data, so fitting them before splitting would let the test period's
flood distribution decide where the class boundaries fall. That is leakage acting
through the labels rather than the features, so nothing that inspects `X` would
catch it — `test_labels_are_fitted_on_development_data_only` catches it instead.

Two further guards: `assert_causal_features` rebuilds features from truncated
history and confirms they are unchanged, which catches look-ahead features such
as a centred rolling window; and macro-F1 is the headline metric rather than
accuracy, since a model that always predicts `Low` scores ~0.80 on accuracy while
being useless for warning.

---

## Layout

```
src/lagos_flood/
  config.py            study area, label thresholds, feature windows, embargo
  labels.py            SFED flooded fraction → 4 classes (+ sensitivity analysis)
  validation.py        temporal holdout, expanding-window folds, leakage guards
  models.py            RF / XGBoost / LightGBM; class weighting; SMOTE (fold-only)
  evaluate.py          macro-F1, per-class, quadratic κ, per-LGA breakdown
  explain.py           SHAP global + local attribution
  pipeline.py          load → features → split → label → CV → test once
  cli.py               command-line entry points
  data/
    floodscan.py       SFED ingestion, area-weighted zonal aggregation
    climate.py         ERA5 / ERA5-Land / CHIRPS / IMERG + product comparison
    synthetic.py       synthetic panel for tests and demos
  features/
    antecedent.py      API (Heggen, 2001), lags, rolling accumulations
    build.py           panel assembly + causality check
docs/literature/       verified bibliography (see below)
tests/                 42 tests, weighted towards the leakage guarantees
```

---

## Data sources

None are committed — all require registration, and the rasters are large.

| Source | Use | Access |
|---|---|---|
| FloodScan SFED | flood-extent labels | [HDX](https://data.humdata.org/dataset/floodscan) |
| ERA5 / ERA5-Land | rainfall, soil moisture | Copernicus CDS |
| CHIRPS 2.0 | rainfall cross-check | UCSB CHC |
| GPM IMERG v07 | rainfall cross-check | NASA Earthdata |
| GRID3 Nigeria | LGA boundaries, building density | [grid3.org](https://grid3.org) |
| SRTM / MERIT Hydro | elevation, slope, TWI, HAND | USGS / MERIT |

Put credentials in the environment (`.cdsapirc`, `.netrc`) — both are gitignored.

**Two limitations to state rather than smooth over.** FloodScan has no dedicated
peer-reviewed validation article; cite Galantowicz & Picton (2021) and the AER/NASA
technical reports, and say so plainly. And ERA5 at 0.25° smooths the convective
cells that drive Lagos flash flooding across several LGAs —
`climate.resolution_caveat()` makes the mismatch concrete for the write-up.

---

## Literature

- [`docs/literature/references-apa6.md`](docs/literature/references-apa6.md) —
  63 verified references in APA 6th, alphabetised.
- [`docs/literature/verification-report.md`](docs/literature/verification-report.md) —
  what was checked and what was wrong.
- [`docs/literature/annotated-bibliography-source.md`](docs/literature/annotated-bibliography-source.md) —
  the original annotated bibliography, unedited.

Every DOI was resolved against Crossref. All 43 that were given are real, but
**20 entries carried wrong metadata** and four were serious misattributions —
two had both the wrong first author and the wrong journal, and one named a first
author who does not appear on the paper at all. One entry was dropped as
unverifiable. The verification report lists each correction.

The report is bibliographic only. Numerical claims quoted in the original
annotations — benefit–cost ratios, reported accuracies — are flagged there as
unchecked, and several sit on entries whose authorship was wrong.
