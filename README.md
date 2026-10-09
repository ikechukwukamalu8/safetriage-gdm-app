## SafeTriage-GDM: An Uncertainty-Quantified Gestational Diabetes Mellitus (GDM) Risk Triage Framework with Algorithmic Fairness Auditing & Conformal Prediction

**Version:** Research Prototype  
**Author:** Ikechukwu Okechi Kamalu

[![Launch App](https://img.shields.io/badge/Launch-SafeTriage--GDM%20App-brightgreen?logo=streamlit)](https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/)

## 🚀 Live Application

**Try the SafeTriage-GDM research prototype:**  
https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/

SafeTriage-GDM is a research prototype for uncertainty-aware GDM risk triage, conformal prediction, algorithmic fairness auditing, population-stability monitoring, and model-based explainability.

It supports **two explicitly selectable prediction configurations**:

- **3-predictor configuration:** maternal age + pre-pregnancy BMI + parity
- **4-predictor configuration:** maternal age + pre-pregnancy BMI + parity + **Maternal Multiple Micronutrient Supplementation (MMS) started by week 28**

The researcher selects the intended configuration in the application sidebar. The model is **not inferred from the presence of extra columns**. Extra columns are ignored unless they belong to the selected configuration.

> **Important:** SafeTriage-GDM is a research prototype. It is not a medical device, diagnostic system, treatment recommendation system, or substitute for professional clinical decision-making. The system has not been clinically validated or approved for clinical use.

---
## Research Contribution

SafeTriage-GDM is designed as a GDM-specific responsible-AI framework rather than solely as a predictive model. Its contribution is the integration of:

- uncertainty-aware prediction using conformal prediction sets;
- algorithmic fairness auditing across maternal age groups;
- population-shift monitoring using the Population Stability Index (PSI);
- probability calibration and decision-threshold analysis;
- model-based explainability;
- development-weighted ensemble modelling; and
- a leakage-conscious workflow separating model development, calibration,
  final testing, and independent external validation.

The framework is intended to demonstrate how predictive performance can be
evaluated together with uncertainty, fairness, explainability, and
population stability rather than relying on discrimination metrics alone.

The contribution is methodological and framework-oriented. SafeTriage-GDM
does not claim to introduce a new machine-learning algorithm, nor does it
claim to be the first GDM prediction system to use any individual component.
Independent clinical external validation remains necessary before any
clinical-use claims can be made.

## Research Dataset: Cambridge Baby Growth Study (CBGS)

The research work underlying this prototype uses data from the **Cambridge Baby Growth Study (CBGS)**, a prospective pregnancy cohort based at the Rosie Maternity Hospital in Cambridge, United Kingdom.

Participants were recruited during pregnancy, with recruitment occurring around 12 weeks of gestation. Gestational diabetes mellitus was assessed using a 75-g oral glucose tolerance test at approximately 28 weeks of gestation.

The repository includes a copy of the CBGS research dataset used for the model-development workflow:

```text
dataCBGS_dataset.xlsx
```

This is **third-party research data**, not data created by the author of SafeTriage-GDM. The dataset was originally deposited through the University of Cambridge repository. Users should consult the original repository for the authoritative dataset, metadata, provenance, and licensing information.

**Original Cambridge repository:**  
https://www.repository.cam.ac.uk/items/ca23c466-948b-4981-a415-c74c6ef139dc

**DOI:** 10.17863/CAM.54014

The Cambridge repository identifies the dataset as available under **CC BY 4.0, except where otherwise noted**. Users who reuse or redistribute the dataset should comply with the original repository's licence and attribution requirements.

### CBGS dataset used by this project

The working Excel file included in this repository contains:

- **970 observations**
- **34 columns**

The CBGS file is included here for reproducible research and software demonstration. It remains third-party research data and should be attributed to its original source.

---

## What does MMS mean?

**MMS** means **Multiple Micronutrient Supplementation**.

Where MMS-start information is available, SafeTriage-GDM uses the prediction-time variable:

> **Maternal Multiple Micronutrient Supplementation (MMS) started by week 28**

This is represented as:

```text
1 = MMS started on or before week 28
0 = MMS started after week 28
```

If a dataset provides the raw MMS start week instead of the binary variable, the application can derive this landmark-safe indicator when the column is recognized.

The application does **not** use MMS stop timing, total MMS duration, or whole-pregnancy MMS exposure as substitutes for the MMS-start predictor.

---

## Predictor configurations

### 3-predictor configuration

Select this configuration when you want the model to use only maternal age, pre-pregnancy BMI, and parity. Any MMS columns present in the uploaded dataset are ignored.

| Predictor | Description |
|---|---|
| Maternal age | Mother's age in years |
| Pre-pregnancy BMI | Mother's pre-pregnancy body mass index in kg/m² |
| Parity | Parity/history of previous births represented in the source dataset |

### 4-predictor configuration

Select this configuration when the dataset contains usable MMS-start information. The application requires the fourth predictor and will not silently substitute a 3-predictor model.

| Predictor | Description |
|---|---|
| Maternal age | Mother's age in years |
| Pre-pregnancy BMI | Mother's pre-pregnancy body mass index in kg/m² |
| Parity | Parity/history of previous births represented in the source dataset |
| MMS started by week 28 | Whether Multiple Micronutrient Supplementation started on or before week 28 |

The two configurations are separate model specifications.

### Temporal design

The predictor definitions are aligned with an approximately **28-week prediction landmark**. Variables that are inherently post-delivery or dependent on final pregnancy outcomes are excluded from the prediction-time feature set.

Examples of excluded variables include birth outcomes and newborn anthropometric measurements.

---

## Supported MMS input forms

The application can recognize the canonical MMS variable and common equivalent names, including:

```text
MMS_started_by_28_weeks
MMS started by 28 weeks
Maternal Multiple Micronutrient Supplementation started by week 28
Multiple Micronutrient Supplementation started by week 28
```

It can also derive the binary predictor from a recognized raw start-week field such as:

```text
Relative to the start of pregnancy, when did multiple micronutrient supplementation start?
```

For a numeric start week:

```text
start week <= 28  ->  1
start week > 28   ->  0
```

Ambiguous or unavailable timing is treated as missing rather than guessed.

---

## How to use the application

### Build & Evaluate Model

1. Select **Build & Evaluate Model**.
2. Choose either the **3-predictor** or **4-predictor** configuration in the sidebar.
3. Upload a labelled research dataset containing the GDM outcome and the predictors required by the selected configuration.
4. Review the column mapping and run the model-development pipeline.

For the CBGS research dataset, the presence of MMS columns does **not** force 4-predictor mode. You can explicitly select the 3-predictor model and the additional MMS columns will be ignored.

> **One dataset is sufficient for Build & Evaluate Model.** You do not need two datasets unless you specifically want to use the External Validation workflow.

### External Validation

1. Select **External Validation**.
2. Choose the predictor configuration that corresponds to the reference model you want to evaluate.
3. Upload the **reference/development dataset** used to develop the reference model.
4. Upload a **separate independent external dataset** containing the same required predictor concepts.
5. The reference model is developed and then frozen before predictions are generated for the external dataset.
6. The external dataset is not used for retraining, rebalancing, threshold optimization, or conformal recalibration.

### What the two external-validation datasets mean

The **reference/development dataset** is used to develop and freeze the reference model, including preprocessing, model tuning, ensemble weighting, decision-threshold selection, and conformal calibration.

The **external dataset** is then passed through that frozen workflow without retraining, rebalancing, threshold optimization, or conformal recalibration.

Therefore:

```text
Reference dataset
        ↓
Model development + calibration
        ↓
Frozen reference model
        ↓
Independent external dataset
        ↓
External evaluation
```

For **outcome-based external performance evaluation**, the external dataset should contain an observed GDM outcome with both classes represented. If the external GDM outcome is unavailable, the application can still generate predictions and uncertainty/conformal outputs, but outcome-based performance metrics cannot be calculated.

A synthetic dataset in this repository may be used to **test the software workflow**, but synthetic data do not constitute clinical external validation.

---

## Methodological Pipeline

### 1. Data validation

The application validates the uploaded Excel dataset, identifies the GDM outcome, and checks whether the selected 3-predictor or 4-predictor configuration can be supported by the uploaded columns.

### 2. Leakage-aware split

Known GDM outcomes are divided using a stratified development/test split followed by a stratified development/calibration split, giving an approximate:

```text
60% training
15% calibration
25% test
```

The exact row counts depend on the number of observations with valid GDM outcomes.

The final test partition is kept separate from model fitting, threshold selection, and conformal calibration.

### 3. Missing-data handling

The preprocessing pipeline uses MICE-style iterative imputation for numeric variables and most-frequent imputation for categorical variables. Imputation is fitted within the development/training workflow rather than using information from the final test partition.

### 4. Training imbalance handling

The training partition may use a custom **ROSE-style smoothed minority oversampling** procedure. Synthetic minority observations are generated only within training data.

Calibration and final test partitions retain their natural outcome distribution.

The implementation is ROSE-style and is not claimed to be an exact reproduction of the R `ROSE` package.

### 5. Models

The application uses three complementary model families:

1. Random Forest
2. XGBoost
3. Logistic Regression

Their predicted probabilities are combined using a development-data-weighted ensemble.

### 6. Development-only tuning

Hyperparameters are tuned using stratified 5-fold cross-validation inside the training partition. Each inner fold independently performs training-only ROSE-style balancing and preprocessing.

Calibration and test partitions are excluded from hyperparameter tuning.

### 7. Development-only ensemble weighting

Out-of-fold training predictions are used to select non-negative ensemble weights on a simplex. Precision-recall AUC is the primary weight-selection criterion, with ROC-AUC used as a tie-breaker.

The resulting weights are frozen before calibration and test evaluation.

### 8. Threshold selection

The decision threshold is selected using the calibration partition only. The research prototype maximizes **F2**, giving greater weight to sensitivity, subject to a prespecified **minimum specificity of 60%**.

This is a methodological research constraint, not a clinical recommendation.

### 9. Conformal prediction

A separate calibration partition is used to construct a **90% conformal prediction set**.

Possible outputs include:

- `{GDM}`
- `{No GDM}`
- `{GDM, No GDM}`

A `{GDM, No GDM}` set indicates that both classes remain plausible at the specified conformal level; it is not a diagnosis.

### 10. Uncertainty quantification

The application reports:

- epistemic uncertainty from model-probability dispersion;
- aleatoric uncertainty using Bernoulli entropy;
- predictive entropy;
- mutual information.

These measures should not be interpreted as clinical certainty.

### 11. Fairness auditing

Fairness analysis is performed across maternal age groups:

```text
<25
25–34
35–44
45+
```

The application reports selection-rate information and false-positive-rate disparities when the required information is available. Small groups are flagged rather than treated as reliable evidence of disparity.

The reported FPR disparity is not a complete Equalized Odds assessment.

### 12. BMI population stability

Population stability is monitored using the Population Stability Index (PSI) across BMI categories:

```text
<18.5
18.5–24.9
25.0–29.9
30.0–39.9
>=40
```

A PSI value of 0.20 is used as the application's substantial-shift monitoring threshold.

PSI does not alter the fitted model, decision threshold, or conformal quantile.

### 13. Explainability

The application summarizes model-based feature contributions using model-specific feature importance and coefficients, with encoded contributions aggregated back to original predictors where applicable.

These are predictive/model-based explanations and **are not causal effects**.

---

## Public Datasets for Software Testing and Demonstration

The repository contains synthetic datasets for software testing and demonstration. They are **not the original Cambridge Baby Growth Study observations**, are not patient records, and do not establish clinical validity.

### CBGS research dataset

#### `dataCBGS_dataset.xlsx`

This is the third-party CBGS research dataset used for the primary model-development workflow.

Use it to test:

- the 3-predictor configuration;
- the 4-predictor configuration;
- model development;
- uncertainty quantification;
- conformal prediction;
- fairness auditing;
- BMI population-stability monitoring; and
- model-based explainability.

### 3-predictor synthetic test datasets

#### `SafeTriage_GDM_external_synthetic_test.xlsx`

```text
maternal_age
prepreg_bmi
parity
gdm_status
```

Contains labelled and unlabelled synthetic observations for testing external-validation workflows.

#### `SafeTriage_GDM_external_renamed_schema_test.xlsx`

Uses `parity_count` instead of `parity` to test schema recognition/mapping.

#### `SafeTriage_GDM_fairness_psi_stress_test.xlsx`

Contains intentionally shifted synthetic age/BMI distributions for testing population-stability and fairness-monitoring workflows.

### 4-predictor synthetic test datasets

#### `SafeTriage_GDM_4predictor_synthetic_test.xlsx`

```text
maternal_age
prepreg_bmi
parity
MMS_started_by_28_weeks
gdm_status
```

This file is designed to test the explicitly selected **4-predictor configuration**.

#### `SafeTriage_GDM_4predictor_raw_MMS_start_test.xlsx`

Contains the raw MMS start-week concept instead of the binary derived field. This file is intended to test derivation of **MMS started by week 28**.

All synthetic outcomes and predictor relationships are artificial and must not be interpreted as clinical truth.

---

## Recommended Dataset Testing Sequence

### Test 1 — CBGS 3-predictor development

```text
Dataset: dataCBGS_dataset.xlsx
Mode: 3-predictor
```

### Test 2 — CBGS 4-predictor development

```text
Dataset: dataCBGS_dataset.xlsx
Mode: 4-predictor
```

Compare the two configurations.

### Test 3 — Raw MMS processing

```text
Dataset: SafeTriage_GDM_4predictor_raw_MMS_start_test.xlsx
Mode: 4-predictor
```

Confirm that raw MMS start-week information can be transformed into the week-28 indicator.

### Test 4 — Four-predictor synthetic workflow

```text
Dataset: SafeTriage_GDM_4predictor_synthetic_test.xlsx
Mode: 4-predictor
```

Confirm that the complete 4-predictor software pipeline runs correctly.

### Test 5 — Synthetic external-validation workflow

Reference/development dataset:

```text
dataCBGS_dataset.xlsx
```

External dataset:

```text
SafeTriage_GDM_external_synthetic_test.xlsx
```

Configuration:

```text
3-predictor
```

This tests the external-validation software workflow. It is **not clinical external validation**.

### Test 6 — Renamed-schema workflow

Reference/development dataset:

```text
dataCBGS_dataset.xlsx
```

External dataset:

```text
SafeTriage_GDM_external_renamed_schema_test.xlsx
```

Configuration:

```text
3-predictor
```

This tests schema recognition/mapping.

### Test 7 — Fairness and population-shift stress test

```text
Dataset: SafeTriage_GDM_fairness_psi_stress_test.xlsx
Mode: 3-predictor
```

Inspect fairness metrics and PSI/population-shift outputs.

---

## Important Distinction: Development, Software Testing, and External Validation

This project separates three types of evidence:

### 1. Model-development evidence

The CBGS dataset is used for the primary research/model-development workflow.

### 2. Software-testing evidence

The synthetic datasets are used to demonstrate and stress-test:

- predictor handling;
- MMS processing;
- schema recognition;
- external-validation workflow;
- fairness auditing; and
- population-stability monitoring.

Synthetic results should not be presented as clinical validation.

### 3. Genuine external-validation evidence

A genuine external validation study requires an **independent real-world clinical cohort** that was not used for model training, hyperparameter tuning, threshold selection, conformal calibration, or feature-selection decisions.

---

## Privacy

Do not upload names, medical record numbers, addresses, or other directly identifiable patient information.

The CBGS dataset included in this repository is third-party research data obtained from the publicly deposited Cambridge University repository. It should not be modified or redistributed independently of the licence and attribution requirements specified by the original source.

---

## Installation

Recommended environment:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run app.py
```

---

## Repository Structure

```text
safetriage-gdm-app/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── datasets/
    ├── README.md
    ├── dataCBGS_dataset.xlsx
    ├── SafeTriage_GDM_external_synthetic_test.xlsx
    ├── SafeTriage_GDM_external_renamed_schema_test.xlsx
    ├── SafeTriage_GDM_fairness_psi_stress_test.xlsx
    ├── SafeTriage_GDM_4predictor_synthetic_test.xlsx
    └── SafeTriage_GDM_4predictor_raw_MMS_start_test.xlsx
```

---

## Research and Clinical Disclaimer

SafeTriage-GDM is a research prototype intended for methodological experimentation in uncertainty quantification, conformal prediction, algorithmic fairness, and population-shift monitoring for GDM risk prediction.

It has **not** been established as a clinically validated prediction model and must not be used to diagnose GDM, determine treatment, or make clinical decisions about an individual.

All predictions should be interpreted as research outputs requiring independent scientific and clinical validation.

The inclusion of the CBGS dataset in this repository does not imply endorsement of SafeTriage-GDM by the University of Cambridge, the Cambridge Baby Growth Study investigators, or any associated clinical institution.

---

## 👨‍💻 Author

**Ikechukwu Okechi Kamalu**

ikechukwukamalu8@gmail.com 

Research interests include:

- Responsible AI
- Trustworthy Machine Learning
- Explainable AI
- Fairness-Aware Machine Learning
- Causal Inference
- Uncertainty Quantification
- Computational Health
- Biostatistics
- Psychometric Modelling
- Scientific Machine Learning
- Human-Centered AI
- Knowledge-Grounded Machine Learning
- Information Retrieval
- Natural Language Processing

---

## 🌐 Project Links

**Live Application:**  
https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/

**GitHub Repository:**  
https://github.com/ikechukwukamalu8/safetriage-gdm-app

---

## 📄 Citation

If this prototype contributes to your research, please cite the associated project or repository.

```text
Kamalu, I. O.
SafeTriage-GDM: An Uncertainty-Quantified GDM Risk Triage Framework
with Algorithmic Fairness Auditing and Conformal Prediction.
GitHub repository.
```

---

## 🔒 Public Repository Data Policy

For the public GitHub repository:

- Commit source code and documentation.
- Commit only synthetic demonstration datasets intended for public release.
- Do not commit identifiable patient information.
- Do not commit API keys, passwords, access tokens, or Streamlit secrets.
- Do not commit private research files merely because they are required for local reproduction.

The public synthetic datasets are intended to make the application demonstrable without redistributing restricted research data.

---

## 📜 License

This project is provided for research and educational purposes.

Please review the repository license before using, modifying, or redistributing the software or associated datasets.
