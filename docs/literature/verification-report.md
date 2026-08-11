# Citation Verification Report

**Project:** Lagos Flood Risk Prediction — MSc Data Science, Pan Atlantic University
**Source document:** `annotated-bibliography-source.md` (64 numbered entries + grey literature)
**Verified:** 11 August 2026
**Method:** Crossref REST API (`api.crossref.org`) for DOI resolution and publisher metadata;
NCBI eutils for PubMed Central identifiers; arXiv abstract pages for preprints; publisher
pages for the remainder. Numerical claims quoted in the annotations were checked separately
against full text where open access, publisher abstracts otherwise.

---

## Summary

| Outcome | Count |
|---|---|
| Verified, cite as-is (metadata was already correct) | 43 |
| Verified **with corrections** (author, journal, year, volume or title was wrong) | 20 |
| **Could not verify — recommend dropping** | 1 |
| Grey literature verified (no DOI, cite as reports) | 3 |
| Numerical claims checked — correct as quoted | 20 of 23 |
| Numerical claims wrong, mislabelled or incomplete | 3 (entries 34, 10, 35) |

Every entry that carried an explicit DOI resolved successfully — **all 43 DOIs in the source
document are real and point at real papers**. The problems are not invented DOIs; they are
wrong *author attributions* and *journal names* attached to otherwise real work. Four of these
are serious enough that citing them unchanged would misattribute published research.

---

## Critical corrections

These four would be flagged in a viva. The DOI was right but the human-readable citation was wrong.

### Entry 10 — wrong first author **and** wrong journal
- **Source said:** Bello, et al. (2025). *Sustainable Cities and Society*.
- **Actually:** Kafi, K. M., Ponrahono, Z., Ash'aari, Z. H., & Barau, A. S. (2025).
  *The Journal of Climate Change and Health*, *26*, Article 100490.
- **DOI:** 10.1016/j.joclim.2025.100490
- No author named Bello appears on this paper. The journal is not *Sustainable Cities and Society*.

### Entry 42 — wrong first author **and** wrong journal
- **Source said:** Camici, S., et al. (2020). *Hydrology and Earth System Sciences*.
- **Actually:** Brocca, L., Massari, C., Pellarin, T., Filippucci, P., Ciabatta, L., Camici, S.,
  Kerr, Y. H., & Fernández-Prieto, D. (2020). *Scientific Reports*, *10*(1), Article 12517.
- **DOI:** 10.1038/s41598-020-69343-x
- Camici is the **sixth** author, not the first. The paper is in *Scientific Reports*, not HESS.

### Entry 62 — first author does not exist on the paper
- **Source said:** Shukla, et al. (2020).
- **Actually:** Rai, R. K., van den Homberg, M. J. C., Ghimire, G. P., & McQuistan, C. (2020).
- **DOI:** 10.1016/j.ijdrr.2020.101534 (journal, volume and article number were correct)
- There is no author named Shukla. The source document had already flagged this one for
  confirmation — the flag was justified.

### Entry 13 — wrong journal
- **Source said:** *Natural Hazards Research* (2026).
- **Actually:** Chowdhury, S. T., Yasmin, F., Tanni, M. M. P., Rahman, M., Sarwar, H., & Tin, T. T.
  (2026). *Watershed Ecology and the Environment*, *8*, 185–207.
- **DOI:** 10.1016/j.wsee.2026.03.002

---

## Other corrections

| # | What was wrong | Correction |
|---|---|---|
| 7 | Year given as 2024 | Published **2025**; *Critical Issue of Sustainable Future*, *2*(1), 80–98. Authors: Abdulhamid, K. A., & Manurung, P. DOI now confirmed: 10.61511/crsusf.v2i1.1825 |
| 29 | Title and authorship | Actual title is "Real **time** monitoring of flooding from microwave satellite **observations**" (not "Real-Time … Radiometry"), and there is a **second author, Herb Frey** |
| 38 | Year 2022, volume 39 | Published **2023**, volume **40**, pp. 570–586. Authors: Gbode, I. E., Babalola, T. E., Diro, G. T., & Intsiful, J. D. |
| 41 | Listed as an arXiv preprint | **Now peer-reviewed** — *Theoretical and Applied Climatology*, *156*(5), Article 241 (2025). Cite the published version. Authors: Bagiliko, J., Stern, D., Ndanguza, D., & Torgbor, F. F. |
| 52 | No authors given | **Jafarigol, E., Trafalis, T., & Mohammadi, N.** (2023) |
| 57 | Pages given as 4765–4777 | This conflates two real paginations. *Advances in NIPS 30* (Curran) gives **4765–4774**; the ACM Digital Library version gives 4768–4777. Neither is 4765–4777. Recommended: **pp. 4765–4774** |
| 58 | Journal unidentified | *Results in Engineering*, *27*, Article 105976. Authors: Choubin, B., Jaafari, A., Henareh, J., Karimi, O., & Sajedi Hosseini, F. |
| 6 | Authors unknown, DOI unconfirmed | Nkeki, F. N., Bello, E. I., & Agbaje, I. G. (2022). *IJDRR*, *77*, Article 103097. DOI: 10.1016/j.ijdrr.2022.103097 |
| 8 | Authors unknown | Aiyelokun, O. O., Aiyelokun, O. D., & Agbede, O. A. — *Natural Hazards*, *119*(3), 2179–2195 |
| 37 | Author guessed as "surname begins S" | Correct: **Steinkopf, J., & Engelbrecht, F.** |
| 61 | Authors unconfirmed | Clark, S. R., Fu, G., & Janardhanan, S. — *WRR*, *61*(10), Article e2025WR041303 |
| 63 | No author given | Single author: **Van Houtven, G.** |
| 14, 19, 39, 43, 44, 47, 56 | DOI unconfirmed (cited only by Elsevier PII) | All seven resolved — see the reference list for full details |
| 3, 5, 15, 16, 20, 32, 40, 59, 60 | Authors unknown or partial | All resolved from Crossref |
| 9, 11, 40 | Minor title wording drift | Titles corrected to the published wording (e.g. entry 11 is "flood **event** along **Lower** Niger", singular, no "the") |
| 26 | Year 2020 vs 2021 | Cite **2021** — online-first 2020, but volume 14(1) is the 2021 issue. Same for entry 25 (online 2021, issue 37(19) is 2022) and entry 31 (online 2016, book published 2017) |

---

## Recommend dropping

### Entry 2 — *American Journal of Geographic Information System*, 14(1), pp. 1–13
The article exists at `http://article.sapub.org/10.5923.j.ajgis.20251401.01.html`, but:
- the DOI-like string in the URL is **not registered with Crossref** and does not resolve;
- the publisher page returned HTTP 503 on verification, so the **author list could not be confirmed**;
- SAP/SciEP is a low-prestige publisher, as the source document itself noted.

Per the source document's own Recommendation #5 — drop anything whose authorship cannot be
confirmed on a genuine publisher page — **this entry should be dropped**. Entry 7 (Abdulhamid &
Manurung, 2025) is now fully verified and covers similar Lagos remote-sensing ground, so nothing
substantive is lost.

---

## Grey literature — verified, cite as reports

| # | Status |
|---|---|
| 29 | Verified on NASA NTRS. Report No. AER-P870-FR-I-20020906, 9 September 2002, Contractor Report, NASA Goddard. **Two authors: Galantowicz & Frey** |
| 36 | Verified. Full title: "NASA/DFO MODIS near real-time (NRT) global flood mapping product evaluation of flood and permanent water detection", Nigro, Slayback, Policelli & Brakenridge (2014). No DOI |
| NiHSA | Annual Flood Outlook — agency publication, cite by year and URL (`https://nihsa.gov.ng`) |

---

## Numerical claims — checked against the papers

Every figure quoted in the source document's annotations was checked against the paper itself
(full text where open, publisher abstract otherwise). **Twenty of twenty-three claims are
correct.** One is wrong, one is mislabelled, and one is correct but materially incomplete.

### Wrong

**Entry 34 (Tiwari et al., 2020) — the accuracies are misquoted.**

| | Source document | Paper |
|---|---|---|
| 9 August 2018 | 94.3% | **94.73%** |
| 21 August 2018 | 94.1% | **94.71%** |
| Kappa | 0.87 / 0.88 | 0.87 / 0.88 ✓ |

The paper states: *"The overall accuracy of the classification was found to be 94.73% and 94.71%
with a kappa coefficient of 0.87 and 0.88 respectively."* The kappas are right; the accuracies
each dropped a digit.

### Mislabelled

**Entry 10 (Kafi et al., 2025) — the numbers are right, the metric is unnamed.**
The source calls 0.93 / 0.92 / 0.84 a "score". They are **ROC-AUC** values. Accuracy is a
different and lower set of numbers, so an unlabelled quotation is ambiguous:

| Model | ROC-AUC | Accuracy | Kappa |
|---|---|---|---|
| Random Forest | **0.93** | 0.857 | 0.714 |
| XGBoost | **0.92** | 0.847 | 0.70 |
| SVM | **0.84** | 0.757 | 0.515 |

RF as top performer is confirmed. Quote these as ROC-AUC or quote the accuracies — do not mix.

Worth noting for your own feature set: this paper's SHAP analysis ranks **settlement formality
highest** (0.22), above elevation (0.20), population density (0.13) and rainfall (0.12).
Informal settlement is a Nigerian urban-flood driver with an obvious Lagos analogue and no
counterpart in your current static features.

### Correct but incomplete

**Entry 35 (Alonso-Sarria et al., 2025) — 0.941 is internal, not independent.**
The mean accuracy of 0.941 (SD 0.048) is confirmed, as is a mean F1 of 0.931. But when the same
method was validated against *independent* Sentinel-2 optical imagery, accuracy fell to **0.642**,
recovering to 0.886 only after a corrective procedure. The study covers 19 flood events in Campo
de Cartagena, Spain.

Citing 0.941 alone overstates how well SAR flood detection validates out of sample. If you use
this paper to support SAR as a validation reference for SFED labels, the 0.642 figure belongs in
the same sentence.

### Confirmed as stated

| # | Claim | Status |
|---|---|---|
| 62 | Benefit–cost ratio 24–73 depending on scenario | ✓ |
| 62 | "Improving the forecast lead time by 1 h can increase the current savings by 1.83 times" | ✓ verbatim |
| 62 | 453 household surveys, 30 focus groups, 40 key-informant interviews | ✓ |
| 62 | NPR 117,027 / USD 1,083 saved per household | ✓ |
| 64 | EFAS returns "of the order of 400 Euro for every 1 Euro invested" | ✓ — the source was right to reject the 159:1 figure |
| 9 | XGBoost best at 91% overall accuracy | ✓ — ROC-AUC 0.89; SVM 88%/0.82, ANN 85%/0.86 |
| 9 | 20 conditioning factors, 1,164 flood/non-flood points, 1998–2023 | ✓ |
| 27 | 913 large flood events, 2000–2018 | ✓ |
| 27 | ~2.23 million km² total inundation | ✓ |
| 27 | 255–290 million people directly affected | ✓ |
| 27 | 20–24% rise in proportion exposed, ten times previous estimates | ✓ |
| 58 | XGBoost best, RMSE 0.333, AUC 0.890 | ✓ |
| 58 | Distance-to-stream, then TWI, then elevation most important | ✓ rank order confirmed |
| 43 | 7-year rainfall on saturated soil → 100-year flood | ✓ |
| 43 | 200-year rainfall on dry soil → 15-year flood | ✓ |
| 36 | 44% good/excellent/almost perfect; 66% excluding cloud-obscured cases | ✓ — 33% were "too many clouds" |
| 40 | GPM-IMERG v07, CHIRPS 2.0, CPC-CMORPH, PERSIANN-CDR vs Nigerian gauges | ✓ |
| 40 | Niger Central Hydrological Area, 2013–2022 | ✓ — five gauging stations |

### One finding the source document missed, and you should not

Entry 40 is your most important rainfall citation, and its headline result is a warning about
this dissertation's own design. The best-performing product was **PERSIANN-CDR**, not IMERG or
CHIRPS — and its median correlation against Nigerian gauges was **0.33 at daily resolution**,
rising to 0.86 only at monthly resolution (POD 85%).

A daily correlation of 0.33 is the operating regime your model sits in. It does not invalidate
daily modelling, but it does mean satellite rainfall error is a first-order limitation rather
than a footnote, it strengthens the case for the multi-product robustness check, and it predicts
that predictive skill will be materially better for accumulated antecedent features (7-, 14-,
30-day windows) than for same-day rainfall. Say so in the limitations section before an examiner
says it for you.

---

## What this report still does not cover

The FloodScan position stands: there is no dedicated peer-reviewed FloodScan validation article.
Galantowicz & Picton (2021) is the correct peer-reviewed basis, supported by the NTRS report
(now correctly attributed to **Galantowicz & Frey**) and the HDX product page.

Entry 2 remains unverifiable and dropped. Entries 51 and 52 remain unpublished preprints.
