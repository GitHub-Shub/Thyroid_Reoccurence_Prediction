# Thyroid Cancer Recurrence Prediction Dataset

This README documents the features and their distribution/association with recurrence (the target variable) within a dataset of 383 thyroid cancer patients.

---

## 🔍 Dataset Overview

The dataset is focused on identifying factors that predict **Recurrence** (defined as cancer return after initial treatment). The overall recurrence rate in this cohort is **28%** (n=108 recurred vs. n=275 did not).

## ✨ Feature Descriptions and Association with Recurrence

| Feature | Description | Distribution (n=383) | Association with Recurrence (Yes Rate; Chi-square p-value) |
| :--- | :--- | :--- | :--- |
| **Age** | Patient's age at diagnosis. Continuous variable influencing prognosis. | Mean: 40.9 years; Std: 15.1; Range: 15–82; Median: 37 | 26% higher correlation with Yes (older patients recur more; r=0.26) |
| **Gender** | Biological sex. Females are more common, but males often have more aggressive disease. | Female: 312 (81%); Male: 71 (19%) | **Male: 59%** vs. Female: 21% (**p<0.0001**; **strongest link** for demographics) |
| **Smoking** | Current tobacco use (Yes/No). Modifiable risk factor. | No: 334 (87%); Yes: 49 (13%) | **Yes: 67%** vs. No: 22% (**p<0.0001**; ~3x higher risk for smokers) |
| **Hx Smoking** | History of past smoking (Yes/No). Cumulative exposure. | No: 355 (93%); Yes: 28 (7%) | Yes: 50% vs. No: 26% (p=0.0145; moderate association) |
| **Hx Radiotherapy** | Prior radiation therapy exposure (Yes/No). Known risk factor for secondary cancers. | No: 376 (98%); Yes: 7 (2%) | **Yes: 86%** vs. No: 27% (p=0.0028; strong, despite small Yes sample) |
| **Thyroid Function** | Thyroid hormone status (TSH/T4 levels). | **Euthyroid**: 332 (87%). Others: Clinical Hyper (5%), Subclinical Hypo (4%), etc. | Low overall association (p=0.2724); Highest in Subclinical Hypo (36%), lowest in Subclinical Hyper (0%) |
| **Physical Exam** | Clinical findings on neck palpation (goiter types). | Multinodular goiter: 140 (37%); Single nodular-right: 140 (37%); Single nodular-left: 89 (23%), etc. | **Multinodular: 37%** vs. Single-right: 20% (p=0.0114; multinodular higher risk) |
| **Adenopathy** | Lymph node enlargement location, signaling local metastasis. | **No**: 277 (72%). Others: Right (13%), Bilateral (8%), Left (4%), etc. | **Bilateral/Extensive/Posterior: 84%–100%** vs. No: 11% (**p<0.0001**; **strongest clinical predictor**) |
| **Pathology** | Histological cancer type from biopsy/surgery. | **Papillary**: 287 (75%); Micropapillary: 48 (13%); Follicular: 28 (7%); Hurthle cell: 20 (5%) | **Follicular: 43%** vs. Micropapillary: 0% (**p<0.0001**; Follicular/Hurthle worse prognosis) |
| **Focality** | Tumor site(s) within the thyroid. | Uni-Focal: 247 (65%) vs. **Multi-Focal**: 136 (35%) | **Multi-Focal: 51%** vs. Uni-Focal: 15% (**p<0.0001**; multifocal ~3x risk) |
| **Risk** | ATA risk stratification (Low/Intermediate/High). | **Low**: 249 (65%). Others: Intermediate (27%), High (8%) | **High: 100%**; Intermediate: 63%; Low: 5% (**p<0.0001**; highly predictive) |
| **T** | Primary tumor size/invasion (TNM staging component). | **T2**: 151 (39%). Others range from T1a (13%) to T4b (2%). | **T4b: 100%**; T4a: 95%; T3b: 88% vs. T1a: 2% (**p<0.0001**; higher T stage recur more) |
| **N** | Regional lymph node metastasis (TNM staging component). | **N0**: 268 (70%). Others: N1b (24%), N1a (6%). | **N1b: 76%** vs. N0: 10% (**p<0.0001**; nodal involvement is a key predictor) |
| **M** | Distant metastasis (TNM staging component). | **M0**: 365 (95%) vs. M1: 18 (5%) | **M1: 100%** vs. M0: 25% (**p<0001**; distant spread almost guarantees recurrence) |
| **Stage** | Overall TNM-derived stage. | **Stage I**: 333 (87%). Others range from Stage II (8%) to Stage IVB (3%). | **Advanced Stages (IVB/IVA/III): 100%**; Stage II: 78% vs. Stage I: 20% (**p<0001**; advanced stages recur near-certainly) |
| **Response** | Post-therapy response via scans/levels. | **Excellent**: 208 (54%). Others: Structural Incomplete (24%), Indeterminate (16%), Biochemical Incomplete (6%). | **Structural Incomplete: 98%**; Biochemical: 48% vs. Excellent: 0.5% (**p<0.0001**; **most predictive factor**) |
| **Recurred** | **Target Variable**: Cancer return after treatment (Yes/No). | No: 275 (72%); Yes: 108 (28%) | N/A (Target variable) |

---

## 🔑 Key Predictive Factors

Based on the p-values and recurrence rates, the strongest predictors of recurrence in this dataset are:

1.  **Post-Therapy Response** (p<0.0001): **Structural Incomplete Response** is associated with a 98% recurrence rate.
2.  **TNM Components / Stage / Risk** (p<0.0001): **M1** (Distant Metastasis), **N1b** (Lateral Neck Nodes), and **High/Intermediate ATA Risk** are highly predictive, with recurrence rates up to 100%.
3.  **Adenopathy** (p<0.0001): **Bilateral or Extensive Lymph Node Enlargement** indicates a very high risk of recurrence (84% - 100%).
4.  **Focality** (p<0.0001): **Multi-Focal** disease shows a recurrence rate more than three times higher than Uni-Focal (51% vs. 15%).
5.  **Gender** (p<0.0001): **Male Patients** have a significantly higher recurrence rate (59%) compared to Females (21%).
6.  **Current Smoking** (p<0.0001): **Smokers** recur 67% of the time.

These features should be prioritized in any predictive modeling efforts.