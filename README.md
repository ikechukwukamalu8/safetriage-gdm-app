# SafeTriage-GDM

## Uncertainty-Quantified Clinical Triage System for Gestational Diabetes Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds

**Research Prototype**

SafeTriage-GDM is a research-oriented Streamlit application for uncertainty-aware GDM risk triage, conformal prediction, population-shift monitoring, algorithmic fairness auditing, and model explainability.

> **Important:** SafeTriage-GDM is a research prototype. It is **not a medical device** and must not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

---

# Predictor schema

The application uses a parsimonious set of **10 approved antepartum predictors**. Predictor selection is based on the available CBGS data structure and leakage-aware modelling considerations.



The current application uses only the following antepartum predictors:

```text
Evidence of maternal anaemia?

Did the mother supplement with multiple micronutrients during pregnancy?

Relative to the start of pregnancy, when did multiple micronutrient supplementation start?

Relative to the start of pregnancy, when did multiple micronutrient supplementation stop?

Did the mothers just supplement with multiple micronutrients during pregnancy and nothing else?

For how many weeks were multiple micronutrients taken?

Mother's pre-pregnancy BMI (kg/m2)

Mother's age (years)

Did the mother smoke during pregnancy?

Parity
```

The GDM outcome column is:

```text
Gestational diabetes?
```

where `No = 0` and `Yes = 1`.

---

# Why BMI is retained instead of height and weight

The uploaded CBGS dataset was checked directly. For observations with complete height, pre-pregnancy weight, and BMI data, BMI agrees with:

\[
BMI = \frac{weight\;(kg)}{height\;(m)^2}
\]

with only negligible floating-point/rounding differences.

Therefore, including height, pre-pregnancy weight, and BMI simultaneously would provide redundant information. The application retains the clinically interpretable **pre-pregnancy BMI** variable and removes the two components from which it is mathematically constructed.

---

# Variables deliberately excluded

The application deliberately avoids variables that are constant, redundant, post-delivery, or strongly outcome-adjacent for the intended antepartum prediction task.

Examples of excluded CBGS variables include:

- Multiple-micronutrient data availability — constant in the supplied CBGS dataset
- Twin pregnancy — constant in the supplied CBGS dataset
- Maternal height — redundant with BMI and pre-pregnancy weight
- Pre-pregnancy weight — redundant with BMI and height
- Gestational hypertension
- Evidence of pre-eclampsia
- Premature birth
- Low birth weight
- Small-for-gestational-age status
- Baby sex
- Baby anthropometric measurements
- Baby gestational age at birth
- Other measurements collected at or after birth

These exclusions are intended to reduce leakage and keep the predictive feature set aligned with an antepartum research setting.

---

# Public synthetic demonstration datasets

The repository should contain **synthetic demonstration datasets only**. The original individual-level CBGS dataset should not be committed to a public GitHub repository unless explicit redistribution permission exists.

Recommended structure:

```text
safetriage-gdm-app/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── datasets/
    ├── README.md
    ├── SafeTriage_GDM_external_synthetic_test.xlsx
    ├── SafeTriage_GDM_external_renamed_schema_test.xlsx
    └── SafeTriage_GDM_fairness_psi_stress_test.xlsx
```

## 1. SafeTriage_GDM_external_synthetic_test.xlsx

Synthetic external-validation demonstration dataset.

- 1,100 rows
- 1,000 labeled observations
- 100 unlabeled observations
- 150 GDM / 850 No GDM among labeled observations
- 10 approved predictors
- Canonical CBGS-style predictor names
- Synthetic observations only

Use it to demonstrate external validation, inference, dataset mapping, conformal prediction, uncertainty estimation, fairness monitoring, BMI population-stability monitoring, and export.

The supplementation start/stop variables are represented as **numeric weeks relative to the start of pregnancy**, matching the data type used by the CBGS reference dataset.

## 2. SafeTriage_GDM_external_renamed_schema_test.xlsx

Synthetic dataset with alternative variable names to test the application's flexible schema mapping.

Main names include:

```text
maternal_anemia
mmn_supplementation
mmn_start
mmn_stop
only_mmn
mmn_duration
prepreg_bmi
maternal_age
smoking
parity
gdm_status
```

It contains 1,100 rows, with 1,000 labeled and 100 unlabeled observations. The supplementation start/stop variables are numeric week values rather than trimester labels so that they are compatible with the reference preprocessing schema.

## 3. SafeTriage_GDM_fairness_psi_stress_test.xlsx

Synthetic stress-test dataset designed for fairness and population-stability demonstrations.

- 1,500 rows
- 1,000 labeled observations
- 500 unlabeled observations
- 10 approved predictors
- shifted age/BMI distribution relative to the main synthetic dataset
- intended for age-group fairness and BMI PSI monitoring demonstrations

The stress dataset is synthetic and **does not represent a real clinical population**.

---

# Original CBGS data policy

Do **not** upload the original `dataCBGS_dataset.xlsx` or other restricted patient-level research data to the public repository unless you have explicit permission to redistribute the data.

The application is designed so that researchers can upload their permitted research dataset at runtime.

Users are responsible for ensuring that uploaded data are appropriately de-identified and that their use complies with applicable ethical, institutional, legal, and data-protection requirements.

Do not upload:

- Names
- Medical record numbers
- Addresses
- Telephone numbers
- Email addresses
- National identification numbers
- Other direct identifiers

---

# External validation workflow

External Validation is intended to evaluate a dataset against a frozen reference model. The workflow is:

```text
Reference/development dataset
        │
        ▼
Train + calibration + reference test
        │
        ▼
Frozen preprocessing + models
        │
        ├── Frozen decision threshold
        └── Reference conformal calibration
                 │
                 ▼
        Independent external dataset
                 │
                 ▼
          External predictions
```

To use this mode, upload the dataset used to develop the reference model first (for example, a permitted copy of the CBGS research dataset), and then upload the independent external dataset. The external dataset is not used to retrain, rebalance, optimize the decision threshold, or recalibrate the conformal layer.

The public synthetic datasets are intended to demonstrate this workflow; they are not replacement training data for the CBGS research analysis.

# Methodological architecture

## 1. Data validation

The application checks the uploaded dataset for the required target and the 10 approved predictors.

Additional columns may be present and are not used as model predictors.

## 2. Target handling

Rows with a known GDM outcome are eligible for model development/evaluation. Rows without an outcome can still be used for inference after the model has been trained.

## 3. Leakage-aware split

Labeled observations are split using stratification:

```text
60% training
15% calibration
25% untouched test
```

The calibration partition is used for threshold and conformal calibration. The test partition remains untouched until final evaluation.

## 4. MICE-style missing-data handling

The application uses an iterative imputation strategy for numeric variables and most-frequent imputation for categorical variables within the modelling workflow.

Numeric variables are subsequently standardized.

Categorical variables are one-hot encoded with unknown categories safely ignored.

## 5. Training-only ROSE-style balancing

Because GDM is less prevalent than No GDM, a ROSE-style smoothed minority oversampling procedure is applied **only to the training partition**.

The implementation:

- separates minority and majority classes;
- samples minority observations with replacement;
- perturbs numeric minority features using controlled Gaussian noise;
- samples categorical minority values from observed minority categories;
- approximately balances the training classes;
- shuffles the resulting training data.

This is a **ROSE-style implementation**, not a claim of exact equivalence to the R `ROSE` package.

Calibration and test data retain their natural class distribution.

The application does **not** use:

- Class weighting
- `scale_pos_weight`
- SMOTETomek
- Test-set oversampling
- Test-set threshold optimization
- Post-hoc sigmoid probability calibration

---

# Predictive models

Exactly three models are used:

1. **Random Forest**
2. **XGBoost**
3. **Logistic Regression**

The ensemble is an equal-weight probability average:

\[
P(GDM)=\frac{P_{RF}(GDM)+P_{XGB}(GDM)+P_{LR}(GDM)}{3}
\]

No single model is presented as a clinical gold standard.

---

# Decision threshold

The classification threshold is selected from the **calibration partition**, using balanced accuracy.

The test partition is not used to select the threshold.

Therefore, a threshold such as 0.36 should not be interpreted as a universal clinical risk threshold. It is a research-model decision threshold derived from the calibration sample.

---

# Test-set evaluation

The untouched test set is evaluated using:

- ROC-AUC
- PR-AUC
- Accuracy
- Balanced accuracy
- Sensitivity / recall
- Specificity
- Precision
- F1-score
- Brier score
- Log loss
- Confusion matrix

Because GDM is relatively uncommon, PR-AUC and sensitivity should be interpreted alongside ROC-AUC rather than relying on accuracy alone.

---

# 90% conformal prediction

The calibration partition is used to construct a 90% conformal prediction rule.

Possible prediction sets include:

```text
{GDM}
{No GDM}
{GDM, No GDM}
```

`{GDM, No GDM}` means that both outcomes remain plausible under the conformal procedure. It is an **uncertainty statement**, not a diagnosis of both conditions.

The conformal quantile is derived from the calibration data and is not dynamically modified using BMI PSI.

---

# Ensemble uncertainty

SafeTriage-GDM reports several uncertainty quantities:

### Epistemic uncertainty

Estimated from disagreement among the three model probabilities.

### Aleatoric uncertainty

Estimated using mean Bernoulli entropy.

### Predictive entropy

Entropy of the ensemble probability.

### Mutual information

A disagreement/information quantity derived from predictive and aleatoric entropy.

These measures describe model/predictive uncertainty and should not be interpreted as clinical safety guarantees.

---

# Algorithmic fairness audit

The application audits age groups:

```text
<25
25–34
35–44
45+
```

Reported quantities include:

- Selection rate
- False-positive rate
- Age-group FPR disparity

The current audit is **not a complete Equalized Odds evaluation** and does not establish fairness across every protected attribute.

---

# BMI population-stability monitoring

BMI population stability is assessed using Population Stability Index (PSI).

Bins are:

```text
<18.5
18.5–24.9
25.0–29.9
30.0–39.9
≥40
```

The monitoring threshold used by the application is:

```text
PSI ≥ 0.20
```

PSI is a monitoring statistic. It does not change the trained models, decision threshold, or conformal quantile.

---

# Model explainability

The application aggregates model-specific importance/contribution information back to the original predictor concepts.

- Random Forest: feature importance
- XGBoost: feature importance
- Logistic Regression: absolute coefficient magnitude

The aggregated display is intended for **model interpretation**, not causal inference.

Feature importance does not prove that a predictor causes GDM.

---

# Interactive visualizations

The application includes Plotly visualizations for:

- Dataset validation
- Class distribution
- Data split
- ROSE training balance
- Calibration threshold selection
- Test-set performance
- Model comparison
- ROC curves
- Precision-recall curves
- Confusion matrix
- Conformal prediction sets
- Ensemble uncertainty
- Probability distribution
- Age-group fairness
- BMI PSI
- Feature importance

The visualizations are generated from the existing model outputs and do not alter the modelling pipeline.

---

# External validation

The application supports a separate external-validation workflow.

The external dataset is processed using the reference modelling pipeline, while the following quantities remain frozen from model development:

- trained models
- preprocessing transformations
- decision threshold
- conformal calibration quantity

External BMI PSI is reported as a monitoring statistic rather than being used to alter the model.

---

# Export

The application exports row-level predictions to:

```text
safetriage_gdm_predictions.csv
```

The export includes the ensemble GDM probability, risk classification, conformal prediction set, uncertainty measures, and individual model probabilities.

---

# Limitations

## Synthetic-data limitations

The public demonstration datasets are synthetic. Their performance results must not be interpreted as evidence of clinical performance.

## Sample size

The CBGS dataset contains a relatively small number of GDM-positive observations. Consequently, sensitivity, PR-AUC, F1-score, and subgroup metrics may have considerable sampling variability.

## Class imbalance

ROSE-style training augmentation addresses class imbalance during model fitting but does not guarantee better external generalization.

## External validity

Performance on CBGS data does not establish performance in another population, country, healthcare system, demographic group, or clinical setting.

## Conformal assumptions

Conformal prediction depends on assumptions such as exchangeability between calibration and future observations. Distribution shift can affect empirical coverage.

## Fairness

The current audit is limited and should not be interpreted as proof that the system is universally fair.

## Explainability

Feature importance and model contributions are not causal effects.

## Clinical use

This application is not validated for clinical deployment and should not be used to diagnose, treat, or triage actual patients.

---

# Quick start

## 1. Clone the repository

```bash
git clone https://github.com/ikechukwukamalu8/safetriage-gdm-app.git
cd safetriage-gdm-app
```

## 2. Create a virtual environment

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run locally

```bash
streamlit run app.py
```

---

# Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, and the `datasets/` directory to GitHub.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository.
4. Set the main file to:

```text
app.py
```

5. Deploy.

The application does not require a pre-trained `.pkl` or `.joblib` model artifact. Models are trained dynamically from the uploaded research dataset.

---

# Requirements

Recommended `requirements.txt`:

```text
streamlit==1.65.0
scikit-learn==1.6.1
numpy
pandas
xgboost
openpyxl
xlrd>=2.0.1,<3
plotly>=5.24,<7
```

---

# Research workflow

```text
Upload dataset
      │
      ▼
Validate target + 10 approved predictors
      │
      ▼
Identify labeled observations
      │
      ▼
60/15/25 stratified split
      │
      ├───────────────┐
      ▼               ▼
Training          Calibration       Test
      │               │                │
      ▼               │                │
ROSE-style           │                │
training-only        │                │
      │               │                │
      ▼               │                │
MICE-style           │                │
preprocessing        │                │
      │               │                │
      ▼               │                │
RF + XGBoost + LR    │                │
      │               │                │
      └──────┬────────┘                │
             ▼                         │
      Equal-weight ensemble            │
             │                         │
             ▼                         │
     Calibration threshold             │
             │                         │
             ├───────────────┐         │
             ▼               ▼         ▼
       Conformal         Uncertainty  Final test
       prediction          analysis   evaluation
             │               │         │
             └───────┬───────┴─────────┘
                     ▼
              Fairness audit
                     │
                     ▼
                 BMI PSI
                     │
                     ▼
               Explainability
                     │
                     ▼
             Dataset-level inference
                     │
                     ▼
                  CSV export
```

---

# Research interpretation

The principal objective of SafeTriage-GDM is not to maximize a single performance metric at any cost. The application is designed to demonstrate a more transparent research workflow combining:

- leakage-aware predictor selection;
- missing-data handling;
- training-only imbalance correction;
- multiple predictive models;
- uncertainty quantification;
- conformal prediction;
- fairness auditing;
- population-shift monitoring; and
- model explainability.

The deployed schema is designed to be parsimonious and avoids constant variables and variables that are mathematically redundant with BMI.

Predictive performance should be interpreted from the **untouched test evaluation**, rather than assumed from predictor selection alone.

---

# Citation / acknowledgement

The modelling workflow is based on research using the Cambridge Baby Growth Study dataset. Users should follow the dataset's applicable access, attribution, ethical, and redistribution conditions when using the underlying research data.

---

## Final disclaimer

> **SafeTriage-GDM is a research prototype. It is not a medical device, has not been clinically validated for deployment, and must not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.**
