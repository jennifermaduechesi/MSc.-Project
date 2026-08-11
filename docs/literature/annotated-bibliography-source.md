# Annotated Bibliography — "Lagos Flood Risk Prediction" (MSc Data Science, Pan Atlantic University)

## TL;DR
- I assembled ~60 real, thematically grouped references; every entry states what I could verify, flags open-access status and Nigeria/West-Africa relevance, and explicitly marks each DOI I could **not** confirm rather than fabricating one.
- The most defensible citations are the foundational method papers (Breiman 2001; Chen & Guestrin 2016; Lundberg & Lee 2017; Chawla et al. 2002; Roberts et al. 2017) plus several Nigeria/West-Africa-specific validation papers; the single weakest evidence base is **FloodScan itself**, which has **no dedicated peer-reviewed validation article** — it must be cited via a book chapter and grey literature.
- Biggest defensible novelty: no published study performs **multi-class (Low/Medium/High/Critical) per-LGA flood-risk forecasting for Lagos using FloodScan SFED labels with strict temporal validation** — the student should position their contribution here.

## Key Findings
- FloodScan/SFED is documented mainly through a peer-reviewed Elsevier book chapter (Galantowicz & Picton 2021) and NASA/AER technical reports — cite it honestly on that basis, not as if a validation paper exists.
- Coarse reanalysis/satellite rainfall products (ERA5, IMERG, CHIRPS) have documented biases over West Africa/Nigeria; multiple Nigeria-specific validation papers exist to justify the NASA POWER/MERRA-2 robustness checks and the convective-rainfall caveat.
- Tree ensembles (RF, XGBoost) are repeatedly identified as the strongest ML family for flood susceptibility, and RF specifically dominates in Sub-Saharan/West African studies — firm ground for the model choice.

## Details

### Theme 1 — Flooding in Lagos and Nigeria

1. **Adeaga, O., Oyeneye, O. T., & Akinbaloye, O. (2020). Urban flood vulnerability mapping of part of the Lagos metropolis.** *Proceedings of IAHS (PIAHS)*, 383, 249–254. DOI: 10.5194/piahs-383-249-2020. **[OPEN ACCESS] [NIGERIA/LAGOS]** — GIS + cellular-automata flood simulation for Lagos; supports the drainage-failure/low-elevation framing and gives a local flood-extent baseline.

2. **"Flood Susceptibility Mapping and Urban Resilience Assessment in Lagos State Using a Machine Learning Approach" (2025).** *American Journal of Geographic Information System*, 14(1), article 01. URL: http://article.sapub.org/10.5923.j.ajgis.20251401.01.html. **[OPEN ACCESS] [NIGERIA/LAGOS]** — Regularized Random Forest with elevation, slope, TWI, HAND and rainfall for Lagos; precedents the RF approach and static feature set. DOI not verified (SAP is a low-prestige publisher) — cite by URL and use as context, not a methodological authority.

3. **"Drowning in urban growth: rethinking flood resilience and spatial equity in Lagos, Nigeria" (2025).** *Frontiers in Sustainable Resource Management*, DOI: 10.3389/fsrma.2025.1659930. **[OPEN ACCESS] [NIGERIA/LAGOS]** — Spatial analysis of 35 recorded Lagos flood events; documents drivers (drainage, governance) and hotspots (Lekki, Kosofe, Ikeja, Agege) useful for interpreting per-LGA labels.

4. **"Vulnerability, Resilience and Adaptation of Lagos Coastal Communities to Flooding" (2024).** *Earth Science, Systems and Society*, DOI: 10.3389/esss.2024.10087. **[OPEN ACCESS] [NIGERIA/LAGOS]** — Reviews urbanisation, drainage-blocking pollution and sea-level drivers; supports the impervious-surface/building-density feature justification.

5. **"Integrating geospatial mapping and stakeholders' perception on sustainable flood solutions in a typical coastal megacity of Nigeria: a SWOT-AHP approach" (2025).** *Frontiers in Sustainable Cities*, DOI: 10.3389/frsc.2025.1663269. **[OPEN ACCESS] [NIGERIA/LAGOS]** — Links JJA and SON rainfall seasons to high inundation; supports rainy-season flags and seasonal-accumulation features.

6. **"Flood risk mapping and urban infrastructural susceptibility assessment using a GIS and analytic hierarchical raster fusion approach in the Ona River Basin, Nigeria" (2022).** *International Journal of Disaster Risk Reduction* (ScienceDirect pii S2212420922003168). **[NIGERIA]** — Multi-criteria flood risk incorporating building/road density; supports the GRID3 building-density-as-exposure proxy. *DOI to confirm on publisher page.*

7. **"Flood risk assessment and regional detailed spatial planning in Lagos State: A remote sensing perspective" (2024).** *Critical Issue of Sustainable Future (CRSUSF)*. URL: https://journal-iasssf.com/index.php/CRSUSF/article/view/1825. **[NIGERIA/LAGOS]** — PRISMA review of 40 papers on remote sensing for Lagos flooding; good for a literature-context paragraph. *DOI unverified.*

8. **(authors to confirm on publisher page) (2023). Application of random forest (RF) for flood levels prediction in Lower Ogun Basin, Nigeria.** *Natural Hazards* (Springer). DOI: 10.1007/s11069-023-06211-7. **[NIGERIA — high-value local]** — RF flood-level prediction in the Ogun basin (the very river system driving much Lagos flooding); strong local precedent for RF and the Ogun River driver.

9. **Adeyemi, A. B., & Komolafe, A. A. (2025). Flood hazard zones prediction using a machine-learning-based geospatial approach in the lower Niger River basin, Nigeria.** *Natural Hazards Research*. DOI: 10.1016/j.nhres.2025.01.002. **[NIGERIA]** — Compares SVM, XGBoost and ANN using 20 conditioning factors on 1,164 flooded/non-flooded points (1998–2023); direct in-country precedent for the multi-model comparison. (Note: an "XGBoost ≈91% best" claim seen in snippets was **not** confirmed on the abstract — re-verify before citing any accuracy figure.)

10. **Bello, et al. (2025). Flood risk prediction and modeling in Bauchi: Leveraging machine learning models and explainable AI for urban resilience.** *Sustainable Cities and Society* (PMC12851268). **[NIGERIA]** — RF/XGBoost/SVM + SHAP for a Nigerian city; precedents the exact ML-plus-XAI stack, with RF top-performing (score 0.93, vs XGBoost 0.92 and SVM 0.84). *Confirm full author list and DOI on the publisher page.*

11. **Adedeji, O., Olusola, A., Babamaaji, R., & Adelabu, S. (2021). An assessment of flood events along the lower Niger using Sentinel-1 imagery.** *Environmental Monitoring and Assessment*, 193(12). DOI: 10.1007/s10661-021-09647-1. **[NIGERIA]** — SAR flood mapping in Nigeria; supports SAR-derived extent as an independent validation reference for SFED-derived labels.

*Grey-literature context (cite as agency material, not peer-reviewed):* **NiHSA Annual Flood Outlook (AFO) 2021–2026** (https://nihsa.gov.ng) — the source of the colour-coded High/Medium/Low scale the dissertation mirrors and benchmarks against; note NiHSA's own move toward AI/deep-learning forecasting on ~200 years of data.

### Theme 2 — Machine learning for flood prediction and susceptibility

12. **Aghenda, M., Labbaci, A., Bouchaou, L., et al. (2025). Flood prediction using machine learning and deep learning models: a systematic review.** *Mediterranean Geoscience Reviews*, 7, 1149–1167. DOI: 10.1007/s42990-025-00201-6. — Two-decade PRISMA review; anchors the "state of ML flood prediction" section.

13. **"A systematic review of flood prediction (2018–2025): Flood categories, input features, and Machine Learning, Deep Learning and hybrid approaches" (2026).** *Natural Hazards Research* (ScienceDirect pii S2589471426000045). — Recent SLR of inputs/metrics/model families; supports feature-selection and metric choices. *DOI to confirm.*

14. **"Integrating remote sensing and machine learning for flood modelling: A systematic literature review" (2026).** (ScienceDirect pii S1474706526000483). — Notes RF dominance in Sub-Saharan/West Africa flood studies; supports RF choice for data-scarce Nigeria. *DOI to confirm.*

15. **"Flood susceptibility assessment using three machine learning techniques and comparison of their performance" (2026).** *Scientific Reports*, article s41598-026-38391-0. **[OPEN ACCESS]** — RF vs Gradient Boosting vs XGBoost in a data-scarce, topographically complex watershed; direct model-comparison precedent.

16. **"Comparative assessment of machine learning models for flood susceptibility mapping" (2026).** *International Journal of Energy and Water Resources* (Springer), DOI: 10.1007/s42108-026-00478-9. — RF vs SVM vs DT across 17 variables; comparative-study support.

17. **"Robustness of machine learning algorithms to generate flood susceptibility maps for watersheds in Jordan" (2024).** *Geomatics, Natural Hazards and Risk*, 15(1). DOI: 10.1080/19475705.2024.2378991. **[OPEN ACCESS]** — RF/SVM/ANN comparison; supports robustness-checking design.

18. **"Flood Risk Assessment Combining Machine Learning with Multi-criteria Decision Analysis in Jiangxi Province, China" (2025).** *International Journal of Disaster Risk Science* (Springer), DOI: 10.1007/s13753-025-00669-8. — RF, XGBoost and LightGBM as base classifiers; precedent for exactly the tree-ensemble trio.

19. **"A comparison of performance measures of three machine learning algorithms for flood susceptibility mapping of river Silabati (India)" (2022).** *Physics and Chemistry of the Earth* (ScienceDirect pii S1474706522000912). **[INDIA — tropical river]** — RF, Naïve Bayes and XGBoost comparison in a tropical river basin. *DOI to confirm.*

20. **"Data-driven flood susceptibility assessment using hybrid machine learning and optimization techniques: Sedrata Watershed, NE Algeria" (2026).** *Scientific Reports*, article s41598-026-43262-9. **[OPEN ACCESS] [NORTH AFRICA]** — RF optimisation; reinforces RF as a reliable baseline.

21. **Farhadi, H., & Najafzadeh, M. (2021). Flood risk mapping by remote sensing data and random forest technique.** *Water*, 13(21), 3115. DOI: 10.3390/w13213115. **[OPEN ACCESS]** — RF + remote sensing flood mapping; a clean methodological template.

### Theme 3 — Random Forest & XGBoost / gradient boosting (foundations + hydrology)

22. **Breiman, L. (2001). Random Forests.** *Machine Learning*, 45(1), 5–32. DOI: 10.1023/A:1010933404324. — The canonical RF citation; mandatory for the RF model.

23. **Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.** *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. DOI: 10.1145/2939672.2939785. — The canonical XGBoost citation; mandatory for the XGBoost model.

24. **Ma, M., Zhao, G., He, B., Li, Q., Dong, H., Wang, S., & Wang, Z. (2021). XGBoost-based method for flash flood risk assessment.** *Journal of Hydrology*, 598, 126382. DOI: 10.1016/j.jhydrol.2021.126382. — Direct XGBoost-for-flood-risk precedent.

25. **Abedi, R., Costache, R., Shafizadeh-Moghadam, H., & Pham, Q. B. (2022). Flash-flood susceptibility mapping based on XGBoost, random forest and boosted regression trees.** *Geocarto International*, 37(19), 5479–5496. DOI: 10.1080/10106049.2021.1920636. — Head-to-head RF vs XGBoost for floods.

26. **Mirzaei, S., Vafakhah, M., Pradhan, B., & Alavi, S. J. (2021). Flood susceptibility assessment using extreme gradient boosting (EGB).** *Earth Science Informatics*, 14, 51–67. DOI: 10.1007/s12145-020-00530-0. — Applied XGBoost flood susceptibility; *verify exact pagination on publisher page.*

### Theme 4 — Satellite-derived flood extent products and validation

27. **Tellman, B., Sullivan, J. A., Kuhn, C., Kettner, A. J., Doyle, C. S., Brakenridge, G. R., Erickson, T. A., & Slayback, D. A. (2021). Satellite imaging reveals increased proportion of population exposed to floods.** *Nature*, 596, 80–86. DOI: 10.1038/s41586-021-03695-w. — The Global Flood Database, built from 913 large flood events (2000–2018), total inundation ~2.23 million km², 255–290 million people directly affected, and a 20–24% rise in the proportion of the global population exposed (ten times higher than previous estimates). Foundational for satellite flood-extent labels and their documented limitations (cloud cover, MODIS misses).

28. **Galantowicz, J. F., & Picton, J. (2021). Flood Mapping with Passive Microwave Remote Sensing: Current Capabilities and Directions for Future Development.** In G. J.-P. Schumann (Ed.), *Earth Observation for Flood Applications*, pp. 39–60. Elsevier. DOI: 10.1016/B978-0-12-819412-6.00003-1. **[FLOODSCAN — closest peer-reviewed documentation]** — Describes the microwave flood-fraction methodology underlying FloodScan SFED; the primary citable peer-reviewed basis for the label source.

29. **Galantowicz, J. F. (2002). Real-Time Monitoring of Flooding from Microwave Satellite Radiometry.** AER technical report, NASA Technical Reports Server (doc ID 20020079434). URL: https://ntrs.nasa.gov/api/citations/20020079434/downloads/20020079434.pdf. **[GREY LITERATURE — NO DOI]** — Foundational FloodScan/AER algorithm origins.

30. **Brakenridge, R., & Anderson, E. (2006). MODIS-Based Flood Detection, Mapping and Measurement: The Potential for Operational Hydrological Applications.** In *Transboundary Floods: Reducing Risks Through Flood Management*, NATO Science Series IV, vol. 72, pp. 1–12. Springer. DOI: 10.1007/1-4020-4902-1_1. — Foundational MODIS flood mapping (Dartmouth Flood Observatory).

31. **Policelli, F., Slayback, D., Brakenridge, B., et al. (2017). The NASA Global Flood Mapping System.** In *Remote Sensing of Hydrological Extremes*, ch. 3, pp. 47–63. Springer. DOI: 10.1007/978-3-319-43744-6_3. — Operational MODIS NRT flood product; relevant to positioning the dashboard alongside NiHSA's real-time layer.

32. **"Mapping global floods with 10 years of satellite radar data" (2025).** *Nature Communications*, article s41467-025-60973-1. **[OPEN ACCESS]** — Deep learning + Sentinel-1 SAR longitudinal global flood dataset; state-of-the-art SAR flood extent and its validation discussion.

33. **"Global flood extent segmentation in optical satellite images" (2023).** *Scientific Reports*, article s41598-023-47595-7. **[OPEN ACCESS]** — WorldFloods optical segmentation; useful for discussing optical-vs-SAR trade-offs and cloud limitations.

34. **Tiwari, V., et al. (2020). Flood inundation mapping — Kerala 2018: Harnessing the power of SAR, automatic threshold detection method and Google Earth Engine.** *PLOS ONE* (PMC7437901). — Otsu-thresholded Sentinel-1 flood maps with overall accuracy of **94.3% (9 Aug) and 94.1% (21 Aug), kappa 0.87 and 0.88**; supports threshold-based extent-to-label conversion for SFED. *Confirm DOI on publisher page.*

35. **"Detecting Flooded Areas Using Sentinel-1 SAR Imagery" (2025).** *Remote Sensing*, 17(8), 1368. DOI: 10.3390/rs17081368. **[OPEN ACCESS]** — Random Forest on SAR metrics for flood detection (mean accuracy 0.941); ties SAR validation to the RF method.

36. **Nigro, J., Slayback, D., Policelli, F., & Brakenridge, G. R. (2014). NASA/DFO MODIS Near Real-Time Global Flood Mapping Product Evaluation.** NASA Goddard Space Flight Center technical report. **[GREY LITERATURE — NO DOI]** — Accuracy evaluation of the MODIS NRT product; supports the satellite-vs-ground validation discussion.

### Theme 5 — Reanalysis & satellite rainfall product validation over West Africa/Nigeria

37. **(first author surname begins "S"; confirm) (2022). Verification of ERA5 and ERA-Interim precipitation over Africa at intra-annual and interannual timescales.** *Atmospheric Research*, 280, 106427. DOI: 10.1016/j.atmosres.2022.106427. **[AFRICA]** — Documents that ERA5 substantially reduces (but does not eliminate) the wet bias over tropical Africa; justifies ERA5 use with an explicit bias caveat. *Confirm author list.*

38. **(authors to confirm) (2022). Assessment of ERA5 and ERA-Interim in Reproducing Mean and Extreme Climates over West Africa.** *Advances in Atmospheric Sciences*, 39. DOI: 10.1007/s00376-022-2161-8. **[WEST AFRICA]** — ERA5 reproduces Guinea-Coast/Savannah rainfall reasonably but with regional biases; supports the ERA5-vs-MERRA-2 robustness check.

39. **"Revealing the spatiotemporal precipitation patterns of ECMWF fifth-generation reanalyses since the mid-20th century in West Africa" (2025).** (Elsevier pii S2666765725000286). **[WEST AFRICA]** — Directly evaluates ERA5 (0.25°) and ERA5-Land (0.1°) over West Africa — the exact products/resolutions used in the dissertation. *DOI to confirm.*

40. **(authors to confirm) (2025). Comprehensive evaluation of satellite precipitation products over a sparsely gauged river basin in Nigeria.** *Theoretical and Applied Climatology*. DOI: 10.1007/s00704-025-05388-0. **[NIGERIA — highest-value rainfall citation]** — Evaluates GPM-IMERG v07, CHIRPS 2.0, CPC-CMORPH and PERSIANN-CDR against Nigerian gauges (Niger Central Hydrological Area, 2013–2022); the single best Nigeria-specific rainfall-validation citation.

41. **"Validation of satellite and reanalysis rainfall products against rain gauge observations in Ghana and Zambia" (2025).** arXiv:2501.14829 (preprint). **[WEST/SOUTHERN AFRICA — PREPRINT, flag as non-peer-reviewed]** — Compares ERA5/AgERA5, CHIRPS, CHIRP and TAMSAT and the effect of coarse resolution; supports the convective-rainfall coarse-resolution limitation argument.

42. **Camici, S., et al. (2020). River flow prediction in data-scarce regions: soil-moisture-integrated satellite rainfall products outperform rain-gauge observations in West Africa.** *Hydrology and Earth System Sciences* (PMC7385167). **[WEST AFRICA]** — Shows integrated satellite products beat gauges/ERA5 for runoff in West Africa; supports the antecedent-wetness/soil-moisture proxy. *Confirm exact journal/DOI on publisher page.*

### Theme 6 — Soil moisture, antecedent wetness and rainfall intensity as flood predictors

43. **"Assessment of antecedent moisture condition on flood frequency: An experimental study in Napa River Basin, CA" (2019).** *Journal of Hydrology: Regional Studies* (ScienceDirect pii S2214581818303768). — Shows a 7-year rainfall on saturated soil can produce a 100-year flood while a 200-year rainfall on dry soil yields only a ~15-year flood; the strongest justification for antecedent-rainfall/soil-saturation features. *DOI to confirm.*

44. **"Monitoring soil moisture at the catchment scale — A novel approach combining antecedent precipitation index and radar-derived rainfall data" (2020).** *Journal of Hydrology* (pii S0022169420306156). — API-based soil-moisture estimation; supports API/lagged-rainfall feature engineering. *DOI to confirm.*

45. **Heggen, R. J. (2001). Normalized Antecedent Precipitation Index.** *Journal of Hydrologic Engineering*, 6(5), 377–381. DOI: 10.1061/(ASCE)1084-0699(2001)6:5(377). — The methodological citation for the antecedent precipitation index. *Verify DOI on ASCE Library.*

46. **"Satellite- and Ground-Soil-Moisture Synchronization and Rainfall Index Linkage for Developing Early-Warning Thresholds for Flash Floods in Korean Dam Basins" (2026).** *Water*, 18(8), 909. DOI: 10.3390/w18080909. **[OPEN ACCESS]** — Random Forest + antecedent wetness for flash-flood thresholds; precedents the SFED-threshold + rainfall-index design.

47. **"Residual-Oriented Optimization of Antecedent Precipitation Index and Its Impact on Flood Prediction Uncertainty" (2022).** *Water*, 14 (DOAJ 7072b27e…). **[OPEN ACCESS]** — RAPI as a soil-moisture proxy for flood prediction; supports antecedent-index features. *DOI to confirm on MDPI.*

### Theme 7 — Class imbalance and rare-event classification

48. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique.** *Journal of Artificial Intelligence Research*, 16, 321–357. DOI: 10.1613/jair.953. — The canonical resampling citation; central to the class-imbalance discussion. *Verify DOI on JAIR.*

49. **He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning.** *Proceedings of IJCNN 2008 (IEEE)*, 1322–1328. DOI: 10.1109/IJCNN.2008.4633969. — A resampling alternative to SMOTE. *Verify DOI on IEEE Xplore.*

50. **"Performance Metrics for Multilabel Emotion Classification: Comparing Micro, Macro, and Weighted F1-Scores" (2024).** *Applied Sciences*, 14(21), 9863. DOI: 10.3390/app14219863. **[OPEN ACCESS]** — Directly compares F1 variants under imbalance; supports macro-F1 as the primary metric over accuracy.

51. **Grandini, M., Bagli, E., & Visani, G. (2020). Metrics for Multi-Class Classification: an Overview.** arXiv:2008.05756. **[PREPRINT — flag]** — Widely cited reference for macro-averaged metrics and confusion-matrix/per-class interpretation.

52. **"A Review of Machine Learning Techniques in Imbalanced Data and Future Trends" (2023).** arXiv:2310.07917. **[PREPRINT — flag]** — Survey of resampling vs cost-sensitive learning; context for the imbalance-handling and class-weighting-vs-SMOTE choice.

### Theme 8 — Temporal validation and data leakage in spatiotemporal ML

53. **Roberts, D. R., Bahn, V., Ciuti, S., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.** *Ecography*, 40(8), 913–929. DOI: 10.1111/ecog.02881. — The definitive demonstration that random CV inflates performance on autocorrelated data and that blocked/temporal folds give honest error estimates; core justification for the strict time-based split.

54. **Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation.** *Information Sciences*, 191, 192–213. DOI: 10.1016/j.ins.2011.12.028. — Foundational argument for blocked/temporal evaluation of time-series models. *Verify DOI on publisher page.*

55. **Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models.** *Methods in Ecology and Evolution*, 12(9), 1620–1633. DOI: 10.1111/2041-210X.13650. — "Area of applicability"; supports honest reporting of spatiotemporal generalisation limits. *Verify DOI.*

56. **"Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks" (2022).** *ISPRS Open Journal of Photogrammetry and Remote Sensing* (pii S2667393222000072). **[OPEN ACCESS]** — Empirical demonstration of leakage from autocorrelation; reinforces the no-random-shuffle rule. *DOI to confirm.*

### Theme 9 — Explainable AI in hydrology and flood risk

57. **Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions.** *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4777. arXiv:1705.07874. — The canonical SHAP citation; mandatory for the SHAP explainability component.

58. **"Explainable artificial intelligence (XAI) for interpreting predictive models and key variables in flood susceptibility" (2025).** (ScienceDirect pii S2590123025020481). — SHAP on an XGBoost flood model (best RMSE 0.333, AUC 0.890; distance-to-stream, TWI, elevation most important); a direct template for the SHAP analysis. *DOI to confirm.*

59. **"Interpretable machine learning for flood susceptibility mapping in the metropolitan region of São Paulo, Southeast Brazil" (2025).** *Discover Geoscience* (Springer), DOI: 10.1007/s44288-025-00362-9. **[OPEN ACCESS] [megacity analogue]** — SHAP + LIME for an urban megacity; a strong analogue for Lagos interpretability.

60. **"Challenges and opportunities of ML and explainable AI in large-sample hydrology" (2025).** *Philosophical Transactions of the Royal Society A*, 383(2302), 20240287. DOI: 10.1098/rsta.2024.0287. — Argues for interpretability in operational hydrology and uses SHAP interaction values across flood sizes; supports the early-warning transparency case. *Verify DOI.*

61. **Clark, S., et al. (2025). Explainable AI for Interpreting Spatiotemporal Groundwater Predictions.** *Water Resources Research*, DOI: 10.1029/2025WR041303. — SHAP local/global attribution in a spatiotemporal hydrology setting; methodological support for feature-attribution reporting. *Verify DOI/authors.*

### Theme 10 — Flood early warning systems, lead time and operational decision-making

62. **Shukla, et al. (2020). Cost-benefit analysis of flood early warning system in the Karnali River Basin of Nepal.** *International Journal of Disaster Risk Reduction*, 47, 101534. DOI: 10.1016/j.ijdrr.2020.101534. — Reports a benefit-cost ratio of **24–73** depending on scenario and that **a one-hour lead-time gain increases savings 1.83×** (based on 453 household surveys, 30 focus groups, 40 key-informant interviews; per-household savings ~NPR 117,027 / USD 1,083); quantifies the value of the forecasting layer. *Confirm exact author list on publisher page.*

63. **"Economic Value of Flood Forecasts and Early Warning Systems: A Review" (2024).** *Natural Hazards Review*, 25(4). DOI: 10.1061/NHREFO.NHENG-2094. — Value-of-information framing (probabilistic > deterministic; lead-time vs reliability trade-off); supports positioning the dashboard as a decision-support tool.

64. **Pappenberger, F., Cloke, H. L., Parker, D. J., Wetterhall, F., Richardson, D. S., & Thielen, J. (2015). The monetary benefit of early flood warnings in Europe.** *Environmental Science & Policy*, 51, 278–291. DOI: 10.1016/j.envsci.2015.04.016. — The standard citation for early-warning economic value: the European Flood Awareness System returns of **the order of €400 for every €1 invested** (not the ~159:1 figure sometimes quoted — use 400:1). *Verify DOI.*

## Recommendations
1. **Cite FloodScan honestly.** Use Galantowicz & Picton (2021, entry 28) as the peer-reviewed basis, plus the NTRS report (29) and the HDX product page, and state explicitly there is no dedicated peer-reviewed FloodScan validation article. Frame this absence as part of your contribution.
2. **Lead with the bulletproof foundational papers** (22, 23, 27, 48, 53, 57) — these are verified and required for the methods chapter.
3. **Prioritise the Nigeria/West-Africa entries** (1, 3–11, 37–42) for local-context chapters; the Lower Ogun-basin RF paper (8), the Lower-Niger ML paper (9), the Bauchi RF+XAI paper (10) and the Nigeria rainfall-validation paper (40) are the highest-value local citations.
4. **Before final submission, open the publisher page for every entry marked "DOI to confirm" or "authors to confirm"** and paste the exact DOI, author order and pagination. Do not submit any of those until confirmed.
5. **Threshold that changes the decision:** if you cannot confirm a paper's DOI/authors on a genuine publisher page, **drop it** rather than risk a hallucinated citation — accuracy outweighs hitting exactly 60.

## Caveats
- Several 2025–2026 entries carry very recent (occasionally future-styled) volume numbers; they are real but recent — verify each DOI directly.
- Preprints (41, 51, 52) and non-DOI grey literature (29, 36, NiHSA AFO) must be labelled as such and not used as primary methodological authority.
- Author names for a few Elsevier/Springer 2022–2025 papers (entries 6, 8, 37, 38, 42, 61, 62) could not be fully verified from search results and are flagged for confirmation.
- Entry 2 is from a low-prestige publisher (SAP/SciEP); use it only as descriptive local context, not as a methods citation.

## Notable literature gaps the student can position against
- **No multi-class per-LGA flood-risk *forecasting* for Lagos** — existing Lagos work is static susceptibility/vulnerability mapping (GIS/MCDA or one-off RF), not a temporally validated Low/Medium/High/Critical classifier across all 20 LGAs. This is the clearest novelty.
- **No published use of FloodScan SFED as a labelling source for supervised ML in Nigeria** — SFED is used operationally/humanitarianly but there is essentially no peer-reviewed ML-labelling literature, so the threshold-choice methodology is genuinely under-documented.
- **Sparse Nigeria-specific rainfall-product-into-flood-model validation** — entry 40 is nearly alone in evaluating IMERG/CHIRPS against Nigerian gauges *for flood contexts*; the ERA5/ERA5-Land soil-moisture products are almost unvalidated for Nigerian urban flooding specifically.
- **Little work couples strict temporal (non-shuffled) validation with SHAP explainability in an operational African early-warning setting** — the transparency-for-decision-making angle (Theme 9 + 10) is thinly covered for Sub-Saharan cities, giving room for a methodological contribution.