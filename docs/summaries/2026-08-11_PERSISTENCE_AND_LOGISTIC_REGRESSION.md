# Summary — Persistence baseline and Logistic Regression

**Date:** 11 August 2026
**Phase:** 3 (models) — partial, built ahead of schedule while Phase 1/2 are blocked

---

## What was done

Added the two models from the brief that were missing:

1. **Persistence baseline** — predicts that tomorrow's risk class is the same as
   today's. The brief calls this essential.
2. **Logistic Regression** — the interpretable baseline, with feature scaling.

Both now run through exactly the same pipeline, the same time-based split, and
the same metrics as Random Forest and XGBoost. That is the point: a fair
comparison needs identical splits.

Test count went from 42 to 59. Lint clean.

---

## Two new features

The brief lists an "LGA risk-history flag" as a planned input. Added two:

| Feature | What it is |
|---|---|
| `risk_class_current` | The risk class observed today (0–3) |
| `risk_days_elevated_30d` | How many of the last 30 days were High or Critical |

Both are **causal** — they use only what is known today. The answer being
predicted sits at tomorrow (or 7 days out), so nothing leaks.

`risk_class_current` is what persistence reads. Every other model gets it too.
That is deliberate: it means the ML models hold everything persistence holds,
**plus** rainfall, soil moisture and terrain. If they still lose, they have no
excuse.

---

## ⚠️ The important result

I ran all five models on **fake data**. The numbers below say nothing about
Lagos. But the pattern they show is one you need to watch for.

### At 1-day lead time — persistence WINS

| Model | Macro F1 | Critical recall | Beats persistence? |
|---|---|---|---|
| **persistence** | **0.784** | 0.717 | — |
| lightgbm | 0.776 | 0.684 | ✗ No |
| xgboost | 0.771 | 0.678 | ✗ No |
| random_forest | 0.761 | 0.540 | ✗ No |
| logistic_regression | 0.756 | 0.849 | ✗ No |

Every machine learning model lost to "assume tomorrow looks like today".

**Why:** flooding changes slowly. Over one day, the risk class usually does not
move. So copying today's answer is very hard to beat.

### At 7-day lead time — persistence COLLAPSES

| Model | Macro F1 | Critical recall | Beats persistence? |
|---|---|---|---|
| lightgbm | 0.468 | 0.191 | ✓ Yes |
| xgboost | 0.467 | 0.309 | ✓ Yes |
| random_forest | 0.451 | 0.072 | ✓ Yes |
| logistic_regression | 0.441 | 0.717 | ✓ Yes |
| **persistence** | **0.407** | 0.092 | — |

Persistence falls from 0.784 to 0.407. Copying today's answer stops working when
you look a week ahead. Now every ML model wins.

### What this means for your dissertation

**The lead time you choose decides whether you have a result at all.** At 1 day,
the honest finding may be "machine learning adds nothing". At 7 days, it clearly
does. Research Question 4 asks about lead time — this is the answer to it, and it
needs to be reported at more than one horizon.

---

## A second thing worth noticing

Look at the **Critical recall** column, not just macro F1.

At 7-day lead:
- Random Forest has the **best-looking** overall score of the tree models, but
  catches only **7%** of Critical days.
- Logistic Regression has the **worst** macro F1, but catches **72%** of them.

For an early warning system, missing a Critical day is the expensive mistake. So
the "best" model on the headline metric is close to the worst at the job the
system exists to do.

This is exactly why the brief says per-class recall matters more than accuracy.
Do not pick the final model on macro F1 alone.

---

## Key decisions made

- **Persistence is now the default first model** in every comparison run.
- The comparison table has a **`beats_persistence`** column, so the check cannot
  be skipped or forgotten — it is answered in the output.
- Logistic Regression is wrapped in a scaler. Without scaling, building density
  (values in the thousands) would swamp rainfall (values under 100) purely
  because of the units they are measured in.

---

## Bug found and fixed

The `beats_persistence` column was a plain true/false column, and the persistence
row needs to hold "not applicable". Plain boolean columns cannot store a blank
value, so the comparison crashed. Fixed by switching to a nullable boolean type.

The tests had not covered `compare_models` at all. They do now.

---

## What's next

**Still blocked on:**
- Your Phase 1 scripts are not in this repo.
- No real data has been through this pipeline.
- Supervisor decisions on the FloodScan window and NASA POWER date range.

**Not yet built:**
- NASA POWER loader (my `data/climate.py` still assumes ERA5/CHIRPS — wrong source)
- HDX Excel loader (my `data/floodscan.py` still assumes map-tile files — wrong format)
- SFED baseline and return-period features
- NiHSA comparison
- Streamlit dashboard

**Recommendation:** when real data arrives, run the comparison at 1 day and 7 days
before doing anything else. If persistence wins at both, that is your finding, and
it is better to know in week 5 than week 11.
