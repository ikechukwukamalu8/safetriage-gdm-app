# 🛡️ SafeTriage-GDM

## Uncertainty-Quantified Clinical Triage System for Gestational Diabetes Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-green.svg)](https://xgboost.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-blue.svg)](https://plotly.com/python/)

**SafeTriage-GDM** is a research prototype for uncertainty-aware Gestational Diabetes Mellitus (GDM) risk triage, conformal prediction, algorithmic fairness auditing, population-shift monitoring, and explainable machine learning.

The application trains exactly three predictive models:

1. Random Forest
2. XGBoost
3. Logistic Regression

Their predicted probabilities are combined using an equal-weight ensemble:

```text
P(GDM) = [P_RF(GDM) + P_XGB(GDM) + P_LR(GDM)] / 3
```

> **Important research disclaimer:** SafeTriage-GDM is not a medical device, diagnostic system, treatment recommendation system, or autonomous clinical decision-making system. It must not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

---

# 🌐 Live Application

**SafeTriage-GDM:**

https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/

The application accepts Excel datasets (`.xlsx` and `.xls`) and provides:

- Dataset validation
- Leakage-aware predictor selection
- 60/15/25 stratified train/calibration/test splitting
- MICE-style iterative imputation
- Training-only ROSE-style balancing
- Random Forest, XGBoost, and Logistic Regression
- Equal-weight ensemble prediction
- Calibration-derived decision threshold
- Untouched test-set evaluation
- 90% conformal prediction
- Ensemble uncertainty quantification
- Age-group fairness auditing
- BMI population-stability monitoring using PSI
- Model explainability
- Dataset-level predictions
- CSV export
- Interactive Plotly visualizations

---

# 🔐 Privacy and Data Protection

Do **not** upload or commit directly identifiable patient information, including:

- Names
- Medical record numbers
- Addresses
- Telephone numbers
- Email addresses
- National identification numbers
- Other direct identifiers

Users are responsible for ensuring that uploaded datasets are appropriately de-identified and that their use complies with applicable ethical, institutional, legal, and data-protection requirements.

The original Cambridge Baby Growth Study individual-level dataset is **not included in this public repository**.

---

# 🧬 Predictor Schema

The updated application uses **12 approved predictors**.

The important change in this version is that:

```text
Mother's height (cm)
Mother's weight before pregnancy (kg)
```

have been removed from the predictive feature set because the dataset's recorded pre-pregnancy BMI is mathematically consistent with those two quantities. Retaining BMI together with its component height and weight measurements would introduce redundant predictors without adding an independent measurement dimension.

The model therefore retains **pre-pregnancy BMI alone**.

## Approved predictors

```text
Evidence of maternal anaemia?

Do we have data related to multiple micronutrient supplementation?

Did the mother supplement with multiple micronutrients during pregnancy?

Relative to the start of pregnancy, when did multiple micronutrient supplementation start?

Relative to the start of pregnancy, when did multiple micronutrient supplementation stop?

Did the mothers just supplement with multiple micronutrients during pregnancy and nothing else?

For how many weeks were multiple micronutrients taken?

Mother's pre-pregnancy BMI (kg/m2)

Mother's age (years)

Did the mother smoke during pregnancy?

Twin pregnancy?

Parity
```

## Why BMI is retained instead of height + weight

BMI is represented as:

```text
BMI = pre-pregnancy weight / (height in metres)^2
```

For the supplied CBGS dataset, reconstruction from the recorded maternal height and pre-pregnancy weight produced essentially identical BMI values, with differences attributable to numerical precision/rounding.

Therefore, the updated model uses:

```text
Pre-pregnancy BMI
```

rather than simultaneously using:

```text
Pre-pregnancy BMI + height + pre-pregnancy weight
```

This is a feature-engineering and redundancy decision. It should **not** be interpreted as evidence that the removed variables were measured after pregnancy.

---

# 🚫 Leakage Control

The application deliberately excludes variables that are clearly post-delivery or strongly outcome-adjacent from the predictive feature schema.

Examples include:

- Baby birth weight
- Baby BMI at birth
- Baby gestational age at birth
- Baby head circumference
- Baby skinfold measurements
- Baby length at birth
- Baby ponderal index
- Low birth weight
- Premature birth
- Small-for-gestational-age status
- Baby sex

Gestational hypertension and pre-eclampsia variables are also excluded from the baseline predictor schema because their availability depends on the intended clinical prediction time point.

The objective is to create an antepartum-oriented research pipeline rather than allowing information generated at or after delivery to enter the predictive model.

---

# 🧠 Model Architecture

## Random Forest

```text
n_estimators = 300
min_samples_leaf = 3
random_state = 42
n_jobs = -1
```

## XGBoost

```text
n_estimators = 300
max_depth = 3
learning_rate = 0.03
subsample = 0.90
colsample_bytree = 0.90
objective = binary:logistic
eval_metric = logloss
random_state = 42
n_jobs = -1
```

## Logistic Regression

```text
max_iter = 5000
solver = lbfgs
random_state = 42
```

## Equal-weight ensemble

```text
P_ensemble = (P_RF + P_XGB + P_LR) / 3
```

No single model receives a larger ensemble weight.

---

# 🔀 Train / Calibration / Test Design

Only observations with known GDM outcomes are used for supervised model development and final evaluation.

| Partition | Approx. proportion | Purpose |
|---|---:|---|
| Training | 60% | Model fitting |
| Calibration | 15% | Threshold and conformal calibration |
| Test | 25% | Final performance evaluation |

The split is stratified by GDM outcome.

The test partition remains untouched during model fitting, threshold selection, and calibration.

The exact row counts can differ by one observation because stratified splitting operates on finite samples.

---

# ⚖️ Class-Imbalance Strategy

SafeTriage-GDM uses a **training-only ROSE-style smoothed minority oversampling** procedure.

The workflow is:

```text
Labeled observations
        │
        ▼
Stratified split
        │
        ├───────────────┐
        │               │
     Training      Calibration/Test
        │               │
        ▼               │
ROSE-style balancing   Natural distribution
        │               │
        ▼               │
     Model fitting      │
                        │
                        ▼
                 Final evaluation
```

The Python implementation is a **ROSE-style approximation**, not a claim of exact equivalence to the R `ROSE` package.

The application does **not** use:

- Class weighting
- `scale_pos_weight`
- SMOTETomek
- Test-set oversampling
- Calibration-set oversampling
- Synthetic/random labels

The test set therefore retains its natural class distribution.

---

# 🧩 Missing Data and Preprocessing

## Numeric variables

The application uses an iterative-imputation procedure designed as a MICE-style preprocessing approach:

```text
Numeric variables
       │
       ▼
Iterative imputation
       │
       ▼
Standardization
```

Configuration includes:

```text
max_iter = 20
initial_strategy = median
random_state = 42
```

## Categorical variables

```text
Categorical variables
       │
       ▼
Most-frequent imputation
       │
       ▼
One-hot encoding
```

Unknown categorical levels during inference are handled with:

```python
handle_unknown = "ignore"
```

Preprocessing is fitted on the training workflow and then applied to calibration, test, and inference observations.

---

# 🎯 Calibration-Derived Decision Threshold

The classification threshold is selected using the dedicated calibration partition rather than the final test set.

The test set is therefore not used to choose a threshold that maximizes its performance.

The threshold is selected using balanced accuracy on the calibration data.

This is intended to avoid the pathological situation in which a fixed threshold such as `0.50` produces extremely low sensitivity in an imbalanced problem.

> A calibration-derived threshold can improve the operating point for sensitivity/specificity trade-offs, but it does not guarantee better external performance.

---

# 🛡️ 90% Conformal Prediction

SafeTriage-GDM constructs split-conformal prediction sets using the dedicated calibration partition.

For each calibration observation:

```text
S_i = 1 - P(Y_i | X_i)
```

where `P(Y_i | X_i)` is the ensemble probability assigned to the observed class.

The application uses:

```text
90% conformal confidence
```

Possible prediction sets include:

```text
{GDM}
{No GDM}
{GDM, No GDM}
```

`{GDM, No GDM}` means that both classes remain plausible under the conformal procedure. It is **not** a diagnosis and should not be interpreted as a clinical safety guarantee.

The conformal calibration quantity is not dynamically modified by BMI PSI.

---

# 🌫️ Uncertainty Quantification

The application reports four research-oriented uncertainty quantities.

## Epistemic uncertainty

Model disagreement is summarized using the standard deviation of the three model probabilities:

```text
SD(P_RF, P_XGB, P_LR)
```

## Aleatoric uncertainty proxy

Binary entropy is calculated for each model probability:

```text
H(p) = -p log2(p) - (1-p) log2(1-p)
```

and averaged across models.

## Predictive entropy

```text
H_predictive = -p log2(p) - (1-p) log2(1-p)
```

where `p` is the ensemble probability.

## Mutual information proxy

```text
MI = max(H_predictive - H_aleatoric, 0)
```

These measures are model-based uncertainty proxies, not direct clinical certainty measures.

---

# 📈 Test-Set Performance Evaluation

The final test partition is evaluated using:

- ROC-AUC
- PR-AUC
- Accuracy
- Balanced Accuracy
- Sensitivity / Recall
- Specificity
- Precision
- F1-score
- Brier score
- Log loss
- Confusion matrix

The application also displays:

- Individual model performance
- ROC curves
- Precision-recall curves
- Threshold-performance curves
- Confusion matrix visualization
- Ensemble probability distributions

Because GDM-positive cases can be relatively uncommon, PR-AUC, sensitivity, F1, and related metrics may have substantial sampling uncertainty.

---

# ⚖️ Algorithmic Fairness Audit

The application evaluates maternal-age groups:

```text
<25
25–34
35–44
45+
```

The audit reports:

- Group size
- Selection rate
- False-positive rate

The principal disparity measure is the range of observed group FPRs:

```text
FPR disparity = max(FPR_group) - min(FPR_group)
```

> This is an age-group fairness audit, not a complete Equalized Odds assessment. It does not establish fairness across every protected attribute or fairness criterion.

---

# 📊 BMI Population-Stability Monitoring

BMI is monitored using Population Stability Index (PSI).

The predefined BMI bins are:

```text
<18.5
18.5–24.9
25.0–29.9
30.0–39.9
40+
```

The application compares the reference training BMI distribution with an external/current BMI distribution.

Interpretation used by the application:

| PSI | Interpretation |
|---:|---|
| < 0.10 | Minimal shift |
| 0.10–<0.20 | Moderate shift |
| ≥ 0.20 | Substantial shift |

PSI is a monitoring statistic. It does not modify the trained model, decision threshold, or conformal calibration.

---

# 🔎 Explainability

Feature contributions are aggregated back to the original predictor variables after one-hot encoding.

The application uses:

- Random Forest feature importance
- XGBoost feature importance
- Absolute Logistic Regression coefficients

The model-level contributions are aggregated and summarized for comparison.

> Feature importance describes predictive model behaviour. It does not establish causality.

---

# 🧪 Public Synthetic Demonstration Datasets

The repository should contain **synthetic demonstration datasets only**.

Recommended structure:

```text
safetriage-gdm-app/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── datasets/
    ├── README.md
    ├── SafeTriage_GDM_external_synthetic_test.xlsx
    ├── SafeTriage_GDM_external_renamed_schema_test.xlsx
    └── SafeTriage_GDM_fairness_psi_stress_test.xlsx
```

## 1. `SafeTriage_GDM_external_synthetic_test.xlsx`

Synthetic canonical-schema demonstration dataset.

- 1,100 rows
- 1,000 observations with known GDM outcomes
- 100 observations without a GDM outcome
- 12 approved predictors + target
- Synthetic observations only

Recommended uses:

- External-validation workflow testing
- Labeled/unlabeled inference
- Dataset validation
- Schema compatibility testing

## 2. `SafeTriage_GDM_external_renamed_schema_test.xlsx`

Synthetic dataset using alternative variable names:

```text
maternal_anemia
mmn_data_available
mmn_supplementation
mmn_start
mmn_stop
only_mmn
mmn_duration
prepreg_bmi
maternal_age
smoking
twins
parity
gdm_status
```

It contains the same 12 predictor concepts but uses simplified names to test the application's flexible schema-mapping workflow.

## 3. `SafeTriage_GDM_fairness_psi_stress_test.xlsx`

Synthetic dataset designed to exercise:

- Age-group fairness analysis
- Selection-rate analysis
- False-positive-rate analysis
- BMI PSI monitoring
- Labeled/unlabeled inference

It contains 1,500 rows and the updated 12-predictor canonical schema plus the GDM target.

## 🚫 Original CBGS Dataset

Do **not** upload the original:

```text
dataCBGS_dataset.xlsx
```

to a public GitHub repository unless you have explicit legal and ethical permission to redistribute it.

The public repository is intended to contain:

```text
Application code
       +
Documentation
       +
Synthetic demonstration data
```

not the restricted individual-level research observations.

---

# 🚀 Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/ikechukwukamalu8/safetriage-gdm-app.git
cd safetriage-gdm-app
```

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run locally

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

# ☁️ Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to GitHub.
2. Open Streamlit Community Cloud.
3. Select the repository.
4. Set the main file to:

```text
app.py
```

5. Deploy.

The application does not depend on a pre-trained `.pkl` or `.joblib` model file. Models are trained dynamically from the uploaded research dataset.

For a public demonstration, use one of the synthetic datasets in `datasets/`.

---

# 📦 Recommended `requirements.txt`

Use:

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

Plotly is required because the current application uses interactive visualizations.

Matplotlib and Altair are not required by the current application.

---

# 📂 Repository Structure

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

---

# 🔁 External Validation Workflow

The application supports an external-validation mode in which the uploaded external dataset is mapped to the SafeTriage-GDM schema.

The application can work with alternative column names through the schema-mapping interface.

The external workflow reuses the reference model's fitted preprocessing/model objects and does not retrain the model on external unlabeled observations.

The external analysis includes:

1. Predictor mapping
2. Target mapping where available
3. External probability prediction
4. Frozen decision threshold
5. Frozen conformal calibration reference
6. Uncertainty analysis
7. Fairness analysis
8. BMI PSI monitoring
9. Dataset-level prediction export

---

# 🧪 Reproducibility

The principal random seed is:

```python
RANDOM_STATE = 42
```

The modelling design is deliberately deterministic where supported by the underlying libraries.

The core development procedure is:

```text
Upload dataset
     │
     ▼
Validate schema
     │
     ▼
Select observations with known GDM outcomes
     │
     ▼
60% Training ──► ROSE-style balancing ──► Model fitting
     │
     ├── 15% Calibration ──► threshold + conformal calibration
     │
     └── 25% Test ─────────► untouched final evaluation
```

---

# 🛡️ Methodological Safeguards

SafeTriage-GDM deliberately avoids several practices that could produce optimistic evaluation:

- No synthetic/random labels
- No test-set oversampling
- No test-set threshold optimization
- No test-set probability calibration
- No class weighting
- No `scale_pos_weight`
- No SMOTETomek
- No post-hoc sigmoid calibration
- No baby-at-birth variables as predictors
- No training on unlabeled external target observations
- No dynamic modification of conformal `q` using BMI PSI

---

# 📋 End-to-End Research Workflow

```text
1. Upload research dataset
             │
             ▼
2. Validate target and 12 predictors
             │
             ▼
3. Identify labeled observations
             │
             ▼
4. Stratified 60/15/25 split
             │
             ▼
5. Training-only ROSE-style balancing
             │
             ▼
6. MICE-style preprocessing
             │
             ▼
7. Train three models
   ├── Random Forest
   ├── XGBoost
   └── Logistic Regression
             │
             ▼
8. Equal-weight ensemble
             │
             ▼
9. Calibration-derived threshold
             │
             ▼
10. Untouched test-set evaluation
             │
             ├── ROC-AUC
             ├── PR-AUC
             ├── Sensitivity
             ├── Specificity
             ├── Precision
             ├── F1
             ├── Brier
             └── Log Loss
             │
             ▼
11. 90% conformal prediction
             │
             ▼
12. Uncertainty quantification
             │
             ▼
13. Fairness audit
             │
             ▼
14. BMI PSI monitoring
             │
             ▼
15. Explainability
             │
             ▼
16. Dataset-level inference
             │
             ▼
17. CSV export
```

---

# 📊 Interpreting the Results

The application should be interpreted as a research evaluation rather than a clinical validation study.

A higher ROC-AUC indicates better ranking discrimination across thresholds.

PR-AUC is particularly informative when the positive class is relatively uncommon.

Sensitivity measures the proportion of true GDM cases identified as positive under the selected threshold.

Specificity measures the proportion of true non-GDM cases identified as negative.

Brier score and log loss evaluate probabilistic prediction quality.

Balanced accuracy gives equal weight to sensitivity and specificity.

No single metric is sufficient to establish clinical usefulness.

---

# ⚠️ Limitations

## Dataset limitations

Predictive performance depends on the quality, completeness, size, representativeness, and provenance of the uploaded dataset.

## Class imbalance

GDM-positive observations may be substantially fewer than GDM-negative observations. Training-only ROSE-style balancing addresses the training imbalance but does not guarantee improved generalization.

## Small positive test sample

When the test set contains relatively few GDM-positive observations, sensitivity, F1, PR-AUC, and related estimates can be unstable.

## External validity

Performance on one cohort does not establish performance in another population, country, healthcare system, demographic group, or clinical setting.

## Prediction timing

Some variables may become available at different times during pregnancy. The current schema is intended as an antepartum-oriented research feature set, but real clinical deployment would require a precisely defined prediction time point and prospective validation.

## Fairness limitations

The current audit focuses on maternal age groups, selection rates, and false-positive rates. It is not a complete fairness assessment and does not establish Equalized Odds or fairness across all protected characteristics.

## Conformal limitations

Conformal prediction depends on assumptions such as exchangeability between calibration and future observations. Distribution shift can affect empirical coverage. A conformal prediction set is not a guarantee of clinical safety.

## Explainability limitations

Feature importance and model contribution measures describe predictive behaviour. They do not establish causal effects, treatment effects, or clinical mechanisms.

## Synthetic demonstration datasets

The public synthetic datasets are for software and methodological demonstration only. Their results must not be presented as evidence of performance on a real-world clinical population.

---

# 📜 Research Status

**Status:** Research Prototype

**Primary application:** Responsible and uncertainty-aware machine-learning research for GDM risk modelling.

**Not approved for clinical use.**

---

# 👤 Author

**Ikechukwu Okechi Kamalu**

Research interests include:

- Responsible AI
- Trustworthy AI
- Explainable AI
- Fairness-aware machine learning
- Causal inference
- Interpretable machine learning
- Biostatistics
- Computational health
- Uncertainty quantification
- Clinical decision-support research

---

# 📄 License and Data Notice

Choose an appropriate software license for the source code before publishing the repository.

Do not assume that a software license automatically grants redistribution rights for third-party research datasets.

The synthetic datasets included with this project are demonstration data and should be clearly labelled as synthetic.

---

# ⚠️ Final Disclaimer

SafeTriage-GDM is a research prototype. It is intended to demonstrate an uncertainty-aware, leakage-conscious, fairness-audited machine-learning workflow for GDM risk research.

It is **not** a medical device and must not be used as a substitute for professional diagnosis, treatment, or clinical decision-making.
