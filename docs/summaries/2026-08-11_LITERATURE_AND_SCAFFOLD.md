# Summary — Literature verification and code scaffold

**Date:** 11 August 2026
**Student:** Maduechesi Chidiebere Jennifer (25120133019)
**Supervisor:** Solomon Alili, Pan Atlantic University

---

## What was done

### 1. Checked every citation in the annotated bibliography

I looked up all 64 references against the publishers' own records, using the
Crossref database, PubMed Central, and arXiv.

**Good news:** every DOI in the list is real. None were invented.

**Bad news:** 20 entries had the wrong information attached to them. Four were
serious — the reference pointed at a real paper, but named the wrong author or
the wrong journal:

| Entry | What the bibliography said | Who actually wrote it |
|---|---|---|
| 10 | Bello et al., *Sustainable Cities and Society* | Kafi, Ponrahono, Ash'aari & Barau, *J. Climate Change and Health* |
| 42 | Camici et al., *HESS* | Brocca et al., *Scientific Reports* (Camici is 6th author) |
| 62 | Shukla et al. | Rai, van den Homberg, Ghimire & McQuistan |
| 13 | *Natural Hazards Research* | *Watershed Ecology and the Environment* |

One entry was dropped because I could not confirm who wrote it.

### 2. Built a clean APA 6th reference list

63 verified references, in alphabetical order, formatted for APA 6th edition.
File: `docs/literature/references-apa6.md`

### 3. Checked the numbers quoted in the bibliography

23 figures were quoted (accuracy scores, benefit-cost ratios, and so on). I
checked each against the actual paper. **20 were correct.** Three were not:

- One accuracy was misquoted (94.3% should be 94.73%).
- One set of scores was unlabelled — they are ROC-AUC, not accuracy.
- One accuracy of 0.941 is internal testing only; independent testing gave 0.642.

### 4. Built a code scaffold

A working pipeline with 42 passing tests. It runs end to end on fake data.

---

## Key decisions made

**Time-based splitting is enforced in code, not left to discipline.** There is no
random-shuffle option anywhere in the codebase. You cannot accidentally use one.

**A gap ("embargo") sits between training and testing data.** Two things can leak
across a simple date cutoff:
- A test row's 30-day rainfall window can reach back into training data.
- A training row's answer sits in the future, and can reach forward into test data.

The gap is set automatically to whichever of those two is longer.

**Label thresholds are worked out from training data only.** If you calculate the
Low/Medium/High/Critical cut-off points using all your data, the test period
helps decide where the boundaries go. That is cheating, and it is invisible —
it hides in the labels, not the inputs. A test checks for it.

**Macro F1 is the headline number, not accuracy.** About 80% of days are Low. A
model that always says "Low" scores 80% on accuracy and is useless.

---

## Important finding for the methodology chapter

The best satellite rainfall product tested against Nigerian rain gauges reaches a
correlation of only **0.33 at daily resolution** — but 0.86 at monthly
(Ganiyu et al., 2025).

In plain terms: satellite rainfall for a single day in Nigeria is not very
accurate. Added up over a week or a month, it gets much better.

This supports leaning on **accumulated rainfall features** (7-day, 14-day,
30-day totals) rather than same-day rainfall — which the brief already plans.

---

## ⚠️ Where this code does NOT match the project brief

I wrote the scaffold before I had the brief. Some of it assumes the wrong data
sources. This must be fixed before the code is used.

| Part | What I built | What the brief says |
|---|---|---|
| Rainfall | ERA5, CHIRPS, IMERG | **NASA POWER API** |
| FloodScan loading | Map-tile files needing GIS software | **Excel/CSV from HDX**, already per-LGA |
| Joining tables | LGA name text | **Numeric `lga_id`** (your Phase 1 decision) |
| Baseline model | None | **Persistence model** (brief calls it essential) |
| Simple model | None | **Logistic Regression** |
| Missing features | — | SFED baseline, return period, temperature, humidity |
| Dashboard | None | **Streamlit** |

**What is still good and worth keeping:**

- Time-based splitting and leakage checks (`validation.py`)
- SFED → 4-class label logic, including a threshold sensitivity table (`labels.py`)
- Macro F1, per-class recall, per-LGA breakdown (`evaluate.py`)
- Lagged and rolling features, antecedent wetness index (`features/`)
- SHAP explanations (`explain.py`)
- All the literature work

The two files to replace are `data/floodscan.py` and `data/climate.py`. Neither
has ever been run, so nothing is lost by rewriting them.

---

## Where we are on the 12-week plan

| Phase | Weeks | Status |
|---|---|---|
| 1 — Data collection | 1–2 | Underway (scripts on your local machine, not in this repo) |
| 2 — Cleaning, labels, EDA | 3–4 | Not started. Label logic exists but is untested on real data |
| 3 — Features + models | 5–7 | Partly scaffolded. No persistence baseline, no Logistic Regression |
| 4 — Tuning, NiHSA comparison | 8–9 | Not started |
| 5 — Dashboard, write-up | 10–12 | Not started |

---

## What's next

**Blocking decision:** your Phase 1 scripts (`01lgareference.py`,
`02pullnasapower.py`, `03loadfloodscan.py`, `04loadnihsa.py`) are on your local
machine and are not in this repository. They need to come in, so everything lives
in one place and I can build Phase 2 on top of them.

**Still waiting on your supervisor:**
- Use all ~24 years of FloodScan, or a shorter window?
- NASA POWER from 2018, or extend back to 1998?

**First real milestone:** get one month of real SFED data through the pipeline and
look at the actual numbers for the 20 LGAs. The Low/Medium/High/Critical cut-off
points cannot be chosen sensibly until we see the real distribution — and that
single choice shapes the whole results chapter.
