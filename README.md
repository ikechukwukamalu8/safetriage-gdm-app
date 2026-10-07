# SafeTriage-GDM

## Uncertainty-Quantified Clinical Triage System for Gestational Diabetes Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds

**Version:** Research Prototype  
**Author:** Ikechukwu Okechi Kamalu

SafeTriage-GDM is a research prototype for uncertainty-aware prediction of gestational diabetes mellitus (GDM) risk, conformal prediction, algorithmic fairness auditing, population-stability monitoring, and model explainability.

It is designed around a strict **antepartum prediction-time feature set** and uses three complementary machine-learning models:

1. Random Forest
2. XGBoost
3. Logistic Regression

The application combines their predicted probabilities into an ensemble and reports predictive performance, uncertainty, conformal prediction sets, age-group fairness measures, BMI population stability, and model-based feature importance.

> **Important:** SafeTriage-GDM is a research prototype. It is not a medical device, diagnostic system, treatment recommendation system, or substitute for professional clinical decision-making. The system has not been clinically validated or approved for clinical use.

---

## Research Dataset: Cambridge Baby Growth Study

The research work underlying this prototype uses data from the **Cambridge Baby Growth Study (CBGS)**.

The CBGS was a prospective pregnancy cohort based at the Rosie Maternity Hospital in Cambridge, United Kingdom. Participants were recruited during pregnancy, with recruitment occurring around 12 weeks of gestation. Gestational diabetes mellitus was assessed using a 75-g oral glucose tolerance test at approximately 28 weeks of gestation.

The original individual-level CBGS research dataset is **not distributed in this public repository**.

For the local research workflow, the dataset may be supplied as:

```text
dataCBGS_dataset
```

The application is designed to derive the prediction-time MMS feature directly from the original CBGS supplementation-start variable rather than requiring users to modify the research dataset manually.

### Prediction-time feature set

The final strict antepartum model uses four predictors:

| Predictor | Description |
|---|---|
| `Mother's age (years)` | Maternal age |
| `Mother's pre-pregnancy BMI (kg/m2)` | Maternal pre-pregnancy body mass index |
| `Parity` | Number/history of previous births as represented in the CBGS data |
| `MMS started by 28 weeks` | Derived from the CBGS MMS start timing variable; 1 if supplementation started at or before week 28, 0 if after week 28 |

The application derives `MMS started by 28 weeks` from:

```text
Relative to the start of pregnancy, when did multiple micronutrient supplementation start?
```

Missing MMS-start information remains missing and is handled by the model's preprocessing pipeline.

### Variables deliberately excluded from the strict baseline model

The strict prediction-time schema does not use whole-pregnancy or potentially post-landmark information such as:

- MMS stop timing
- Total MMS duration
- Whole-pregnancy MMS exposure indicators
- Pregnancy weight gain
- Gestational hypertension recorded as a final pregnancy status
- Evidence of pre-eclampsia recorded as a final pregnancy status
- Birth outcomes
- Newborn anthropometric measurements
- Other postnatal variables
- Maternal height and pre-pregnancy weight as separate predictors when BMI is available
- Sparse predictors that do not provide adequate variation

This design is intended to reduce temporal leakage and keep the prediction problem aligned with an approximately 28-week prediction landmark.

---

## Methodological Pipeline

### 1. Data validation

The application checks the uploaded Excel dataset, maps compatible column names, validates the GDM outcome, and constructs the derived MMS landmark feature when the original CBGS MMS-start column is available.

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

The training partition may use a custom **ROSE-style smoothed minority oversampling** procedure. Synthetic minority observations are generated only within the training data.

The calibration and final test partitions retain their natural outcome distribution.

The implementation is ROSE-style and is not claimed to be an exact reproduction of the R `ROSE` package.

### 5. Models

#### Random Forest

```text
n_estimators = 300
min_samples_leaf = 3
```

#### XGBoost

```text
n_estimators = 300
max_depth = 3
learning_rate = 0.03
subsample = 0.90
colsample_bytree = 0.90
objective = binary:logistic
eval_metric = logloss
```

#### Logistic Regression

```text
max_iter = 5000
solver = lbfgs
```

The ensemble probability is:

\[
P(GDM)=\frac{P_{RF}(GDM)+P_{XGB}(GDM)+P_{LR}(GDM)}{3}.
\]

### 6. Threshold selection

The decision threshold is selected using development/calibration information and is not optimized on the final test set.

### 7. Conformal prediction

The application uses a separate calibration partition to construct a **90% conformal prediction set**.

Possible outputs include:

- `{GDM}`
- `{No GDM}`
- `{GDM, No GDM}`
- an uncertain/empty set in cases where no class meets the conformal criterion

A `{GDM, No GDM}` set means that both classes remain plausible at the specified conformal confidence level; it does not mean that a person simultaneously has and does not have GDM.

### 8. Uncertainty quantification

The application reports:

- Epistemic uncertainty from the dispersion of the three model probabilities
- Aleatoric uncertainty using mean Bernoulli entropy
- Predictive entropy
- Mutual information as an epistemic-information measure

These quantities describe model/predictive uncertainty and should not be interpreted as clinical certainty.

### 9. Fairness auditing

Fairness analysis is performed across maternal age groups:

```text
<25
25–34
35–44
45+
```

The application reports selection-rate information and false-positive-rate disparities when the required age and outcome information are available.

The reported FPR disparity is **not a complete Equalized Odds assessment**.

### 10. BMI population stability

Population stability is monitored using the Population Stability Index (PSI) across BMI categories:

```text
<18.5
18.5–24.9
25.0–29.9
30.0–39.9
>=40
```

A PSI value of 0.20 is used as the application's substantial-shift monitoring threshold.

PSI is a monitoring statistic. It does not alter the fitted model, decision threshold, or conformal quantile.

### 11. Explainability

The application summarizes model-based feature contributions using:

- Random Forest feature importance
- XGBoost feature importance
- Absolute Logistic Regression coefficients

One-hot encoded contributions are aggregated back to their original predictors where applicable.

These explanations are predictive/model-based explanations and **are not causal effects**.

---

# Public Synthetic Demonstration Datasets

The repository can include synthetic datasets for software testing and demonstration. They are **not the original Cambridge Baby Growth Study observations** and do not represent clinical truth.

All three files use the current four-predictor model schema.

## 1. `SafeTriage_GDM_external_synthetic_test.xlsx`

Main synthetic external-validation demonstration dataset.

- 1,100 rows
- 1,000 observations with known GDM outcomes
- 100 observations without a GDM outcome
- 4 model predictors + GDM outcome column
- Synthetic observations only

Schema:

```text
maternal_age
prepreg_bmi
parity
mms_started_by_28_weeks
gdm_status
```

Recommended uses:

- External-validation workflow testing
- Labeled/unlabeled inference testing
- Schema validation
- Model prediction demonstration

## 2. `SafeTriage_GDM_external_renamed_schema_test.xlsx`

Synthetic external-validation dataset using alternative predictor naming.

Schema:

```text
maternal_age
prepreg_bmi
parity
mmn_started_by_28_weeks
gdm_status
```

Recommended uses:

- Alternative schema mapping
- External dataset integration testing
- Predictor-name matching
- Target-column mapping

## 3. `SafeTriage_GDM_fairness_psi_stress_test.xlsx`

Synthetic dataset designed to exercise fairness and BMI population-stability monitoring.

- 1,500 rows
- 1,000 observations with known GDM outcomes
- 500 observations without a GDM outcome
- Shifted age/BMI distributions for stress testing
- 4 model predictors + GDM outcome column

Schema:

```text
maternal_age
prepreg_bmi
parity
mms_started_by_28_weeks
gdm_status
```

Recommended uses:

- Age-group fairness auditing
- Selection-rate analysis
- False-positive-rate analysis
- BMI PSI monitoring
- Population-shift stress testing
- Labeled/unlabeled inference testing

### Synthetic-data disclaimer

These datasets are synthetic and intended for:

- Software testing
- Interface testing
- Research-method demonstration
- External-validation workflow demonstration
- Fairness testing
- Population-stability testing
- Schema-mapping testing

They are **not clinical datasets**, do not establish clinical performance, and must not be used for medical decision-making.

---

# Original CBGS Data Policy

The original individual-level Cambridge Baby Growth Study dataset is not included in the public GitHub repository.

Do not commit the private research dataset under names such as:

```text
dataCBGS_dataset.xlsx
dataCBGS_dataset(5).xlsx
```

unless you have explicit permission to redistribute the data.

Do not commit:

- Individual-level patient/research records
- Names or direct identifiers
- Medical record numbers
- Addresses or contact information
- API keys
- Passwords
- Access tokens
- Streamlit secrets
- Other confidential research files

The public repository is intended to contain application code, documentation, and synthetic demonstration data only.

---

# Repository Structure

```text
safetriage-gdm-app/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── datasets/
    ├── README.md
    ├── SafeTriage_GDM_external_synthetic_test.xlsx
    ├── SafeTriage_GDM_external_renamed_schema_test.xlsx
    └── SafeTriage_GDM_fairness_psi_stress_test.xlsx
```

The original CBGS research dataset should remain outside the public repository.

---

# Installation

## Requirements

The main dependencies are:

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

## Local installation

```bash
git clone https://github.com/ikechukwukamalu8/safetriage-gdm-app.git
cd safetriage-gdm-app
python -m venv venv
```

### Windows

```powershell
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# Using `dataCBGS_dataset`

For the main research workflow, upload the compatible Excel version of the CBGS dataset.

The application expects the GDM outcome and the four final antepartum predictors, but the MMS landmark predictor can be **derived automatically** from the original CBGS MMS-start field.

You therefore do **not** need to edit `dataCBGS_dataset` manually to add:

```text
MMS started by 28 weeks
```

The application creates this feature internally from the raw MMS-start timing variable.

---

# External Validation

The external-validation workflow is designed to keep the reference model frozen while evaluating an independent dataset.

The workflow:

1. Upload the reference/development dataset.
2. Upload the independent external dataset.
3. Map compatible predictor and outcome columns when necessary.
4. Train the reference models using the reference data.
5. Freeze the reference threshold and conformal calibration.
6. Transform the external dataset using the frozen preprocessing pipeline.
7. Generate external predictions without retraining or re-optimizing on the external data.

The synthetic external datasets in `datasets/` can be used to demonstrate this workflow.

---

# Streamlit Deployment

The application can be deployed using Streamlit Community Cloud.

Set the main application file to:

```text
app.py
```

and ensure that `requirements.txt` is present in the repository.

The application trains models from the uploaded research dataset and does not require a committed `.pkl` or `.joblib` model artifact.

Do not upload the original restricted CBGS dataset to a public GitHub repository or permanently store it in a public application deployment unless the applicable data permissions explicitly allow this.

---

# Responsible AI Considerations

SafeTriage-GDM incorporates several responsible machine-learning principles:

- **Leakage control:** prediction-time features are separated from post-landmark or outcome-related information.
- **Test isolation:** the final test set is not used for threshold optimization or model fitting.
- **Uncertainty awareness:** predictions are accompanied by uncertainty measures and conformal prediction sets.
- **Fairness monitoring:** performance-related disparities are examined across maternal age groups.
- **Population monitoring:** BMI PSI is used to monitor distributional change.
- **Explainability:** model-based feature importance is reported.
- **Privacy:** users are instructed not to upload directly identifiable information.

These mechanisms are research safeguards and do not establish clinical safety or regulatory compliance.

---

# Disclaimer

**SafeTriage-GDM is a research prototype for research and educational purposes.**

It is not:

- A medical device
- A diagnostic system
- A treatment recommendation system
- A clinical decision-making system
- A replacement for a healthcare professional

Predictions, probabilities, uncertainty estimates, conformal prediction sets, fairness metrics, population-stability statistics, and model explanations must not be used as a substitute for professional medical assessment.

The system has not been clinically validated, prospectively evaluated, or approved for clinical use.

---

# Project Links

**Live application:**

https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/

**GitHub repository:**

https://github.com/ikechukwukamalu8/safetriage-gdm-app

---

# Citation

If this prototype contributes to research or software development, cite the project as:

```text
Kamalu, I. O.
SafeTriage-GDM: Uncertainty-Quantified Clinical Triage System for
Gestational Diabetes Mellitus (GDM) Risk with Algorithmic Fairness
Auditing & Conformal Safety Bounds.
GitHub repository.
```

---

# License

This project is provided for research and educational purposes. Review the repository license before using, modifying, or redistributing the software or associated datasets.
