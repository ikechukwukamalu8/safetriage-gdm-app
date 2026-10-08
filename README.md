# SafeTriage-GDM

**Uncertainty-Quantified Clinical Triage System for Gestational Diabetes Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds**

[![Launch App](https://img.shields.io/badge/Launch-SafeTriage--GDM%20App-brightgreen?logo=streamlit)](https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/)

## 🚀 Live Application

**Try the SafeTriage-GDM research prototype:**  
https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/


## Uncertainty-Quantified Clinical Triage System for Gestational Diabetes Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds

**Version:** Research Prototype  
**Author:** Ikechukwu Okechi Kamalu

SafeTriage-GDM is a research prototype for uncertainty-aware GDM risk triage, conformal prediction, algorithmic fairness auditing, population-stability monitoring, and model-based explainability.

It supports **two automatically detected prediction configurations**:

- **3-predictor configuration:** maternal age + pre-pregnancy BMI + parity
- **4-predictor configuration:** maternal age + pre-pregnancy BMI + parity + **Maternal Multiple Micronutrient Supplementation (MMS) started by week 28**

The application detects the available predictor configuration from the uploaded dataset. If valid MMS-start information is present, the 4-predictor specification is used. If it is absent, the 3-predictor specification is used.

> **Important:** SafeTriage-GDM is a research prototype. It is not a medical device, diagnostic system, treatment recommendation system, or substitute for professional clinical decision-making. The system has not been clinically validated or approved for clinical use.

---

## Research Dataset: Cambridge Baby Growth Study (CBGS)

The research work underlying this prototype uses data from the **Cambridge Baby Growth Study (CBGS)**, a prospective pregnancy cohort based at the Rosie Maternity Hospital in Cambridge, United Kingdom.

Participants were recruited during pregnancy, with recruitment occurring around 12 weeks of gestation. Gestational diabetes mellitus was assessed using a 75-g oral glucose tolerance test at approximately 28 weeks of gestation.

The original individual-level CBGS research dataset is **not distributed in this public repository**.

### What does MMS mean?

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

Used automatically when no usable MMS-start variable is available.

| Predictor | Description |
|---|---|
| Maternal age | Mother's age in years |
| Pre-pregnancy BMI | Mother's pre-pregnancy body mass index in kg/m² |
| Parity | Parity/history of previous births represented in the source dataset |

### 4-predictor configuration

Used automatically when a usable MMS-start variable is present.

| Predictor | Description |
|---|---|
| Maternal age | Mother's age in years |
| Pre-pregnancy BMI | Mother's pre-pregnancy body mass index in kg/m² |
| Parity | Parity/history of previous births represented in the source dataset |
| MMS started by week 28 | Whether Multiple Micronutrient Supplementation started on or before week 28 |

The two configurations are separate model specifications. The application does not insert an MMS value into a 3-predictor model or remove MMS from a 4-predictor model after fitting.

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

## Methodological Pipeline

### 1. Data validation

The application validates the uploaded Excel dataset, identifies the GDM outcome, and automatically determines whether the 3-predictor or 4-predictor configuration can be used.

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

Their predicted probabilities are combined using a development-data-weighted ensemble. The model family is the same across the two predictor configurations, while the feature set is determined automatically from the uploaded data.

### 6. Development-only tuning

Hyperparameters are tuned using stratified 5-fold cross-validation inside the training partition. Each inner fold independently performs training-only ROSE-style balancing and preprocessing.

Calibration and test partitions are excluded from hyperparameter tuning.

### 7. Development-only ensemble weighting

Out-of-fold training predictions are used to select non-negative ensemble weights on a simplex. Precision-recall AUC is the primary weight-selection criterion, with ROC-AUC used as a tie-breaker.

The resulting weights are frozen before calibration and test evaluation.

### 8. Threshold selection

The decision threshold is selected using the calibration partition only. The research prototype maximizes **F2**, giving greater weight to sensitivity, subject to a prespecified **minimum specificity of 60%**.

This is a methodological research constraint, not a clinical recommendation. The final test partition is never used to choose the threshold.

### 9. Conformal prediction

A separate calibration partition is used to construct a **90% conformal prediction set**.

Possible outputs include:

- `{GDM}`
- `{No GDM}`
- `{GDM, No GDM}`

A `{GDM, No GDM}` set indicates that both classes remain plausible at the specified conformal level; it is not a diagnosis.

### 10. Uncertainty quantification

The application reports model/predictive uncertainty measures including:

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

## Public Synthetic Demonstration Datasets

The repository contains synthetic datasets for software testing and demonstration. They are **not the original Cambridge Baby Growth Study observations**, are not patient records, and do not establish clinical validity.

### 3-predictor test datasets

#### `SafeTriage_GDM_external_synthetic_test.xlsx`

```text
maternal_age
prepreg_bmi
parity
gdm_status
```

Contains labeled and unlabeled synthetic observations for testing external-validation workflows.

#### `SafeTriage_GDM_external_renamed_schema_test.xlsx`

Uses `parity_count` instead of `parity` to test schema recognition/mapping.

#### `SafeTriage_GDM_fairness_psi_stress_test.xlsx`

Contains intentionally shifted synthetic age/BMI distributions for testing population-stability and fairness-monitoring workflows.

### 4-predictor test datasets

#### `SafeTriage_GDM_4predictor_synthetic_test.xlsx`

```text
maternal_age
prepreg_bmi
parity
MMS_started_by_28_weeks
gdm_status
```

This file is designed to trigger the **4-predictor configuration** automatically.

#### `SafeTriage_GDM_4predictor_raw_MMS_start_test.xlsx`

Contains the raw MMS start-week concept instead of the binary derived field. This file is intended to test derivation of **MMS started by week 28**.

All synthetic outcomes and predictor relationships are artificial and must not be interpreted as clinical truth.

---

## Privacy

Do not upload names, medical record numbers, addresses, or other directly identifiable patient information.

The original individual-level CBGS dataset should remain private and should **not** be committed to a public GitHub repository.

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
