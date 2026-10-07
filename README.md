# 🛡️ SafeTriage-GDM

## Uncertainty-Quantified Clinical Triage System for Gestational Diabetes Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-green.svg)](https://xgboost.readthedocs.io/)
[![Altair](https://img.shields.io/badge/Visualization-Altair-blue.svg)](https://altair-viz.github.io/)

**SafeTriage-GDM** is a standalone research prototype for uncertainty-aware Gestational Diabetes Mellitus (GDM) risk triage, conformal prediction, algorithmic fairness auditing, population-shift monitoring, and explainable machine learning.

The system combines three complementary machine-learning models:

- Random Forest
- XGBoost
- Logistic Regression

The three models produce individual GDM probabilities, which are combined using an equal-weight probability ensemble:

```text
P(GDM) = [P_RF(GDM) + P_XGB(GDM) + P_LR(GDM)] / 3
```

SafeTriage-GDM is designed for **research and methodological evaluation**.

> **Important:** SafeTriage-GDM is not a medical device, diagnostic system, treatment recommendation system, or autonomous clinical decision-making system. It must not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

---

# 🌐 Live Application

**[Open SafeTriage-GDM](https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/)**

The application allows researchers to upload a compatible Excel dataset and execute the complete modelling workflow, including:

- Dataset validation
- Leakage-aware predictor selection
- MICE-style imputation
- Training-only ROSE-style balancing
- Random Forest modelling
- XGBoost modelling
- Logistic Regression modelling
- Equal-weight ensemble prediction
- Calibration-based threshold selection
- Conformal prediction
- Uncertainty quantification
- Performance evaluation
- Fairness auditing
- BMI population-shift monitoring
- Explainability analysis
- Dataset-level prediction export

> **Privacy notice:** Do not upload names, medical record numbers, addresses, telephone numbers, email addresses, national identification numbers, or other directly identifiable patient information.

---

# 🎯 Research Objectives

SafeTriage-GDM investigates several dimensions of responsible machine learning for GDM risk modelling:

1. GDM risk prediction using heterogeneous machine-learning algorithms.
2. Comparison of nonlinear and linear predictive models.
3. Equal-weight probability ensembling.
4. Uncertainty quantification through model disagreement and entropy.
5. Conformal prediction using a dedicated calibration partition.
6. Algorithmic fairness assessment across maternal age groups.
7. Population-shift monitoring using maternal pre-pregnancy BMI.
8. Explainable feature contribution analysis.
9. Leakage-aware predictor selection for antepartum modelling.
10. Evaluation of predictive performance under class imbalance.

---

# 🧠 Model Architecture

SafeTriage-GDM uses exactly **three predictive models**.

## 1. Random Forest

Random Forest is used to capture nonlinear relationships and interactions between maternal and pregnancy-related variables.

Configuration:

```text
n_estimators = 300
min_samples_leaf = 3
random_state = 42
```

## 2. XGBoost

XGBoost provides a gradient-boosted tree model capable of representing nonlinear predictive relationships.

Configuration:

```text
n_estimators = 300
max_depth = 3
learning_rate = 0.03
subsample = 0.90
colsample_bytree = 0.90
objective = binary:logistic
```

## 3. Logistic Regression

Logistic Regression provides a comparatively interpretable linear modelling baseline.

Configuration:

```text
max_iter = 5000
```

## Ensemble Probability

The final research probability is calculated as:

```text
P_ensemble = (P_RF + P_XGB + P_LR) / 3
```

where:

- `P_RF` = Random Forest probability of GDM
- `P_XGB` = XGBoost probability of GDM
- `P_LR` = Logistic Regression probability of GDM

All three models contribute equally to the final ensemble probability.

---

# 📊 Complete Data Processing Pipeline

```text
Excel Dataset
      │
      ▼
Dataset Validation
      │
      ▼
GDM Ground-Truth Identification
      │
      ├── Labeled observations
      │        │
      │        └── Train / Calibration / Test
      │
      └── Unlabeled observations
               │
               └── Inference population
      │
      ▼
Antepartum Predictor Selection
      │
      ▼
60 / 15 / 25 Stratified Split
      │
      ├── 60% Training
      ├── 15% Calibration
      └── 25% Test
      │
      ▼
Training-Only ROSE-Style Balancing
      │
      ▼
MICE-Style Iterative Imputation
      │
      ▼
Categorical Imputation + One-Hot Encoding
      │
      ▼
Standardization
      │
      ▼
┌─────────────────────────┐
│ Random Forest           │
│ XGBoost                 │
│ Logistic Regression     │
└─────────────────────────┘
      │
      ▼
Equal-Weight Probability Ensemble
      │
      ├── Calibration Threshold
      ├── Test Performance
      ├── Conformal Prediction
      ├── Uncertainty Quantification
      ├── Fairness Auditing
      ├── BMI Population Shift
      └── Explainability
      │
      ▼
Dataset-Level Predictions
      │
      ▼
CSV Export
```

---

# 🔀 Train / Calibration / Test Design

Only observations with known GDM outcomes are used for supervised model development and evaluation.

| Partition | Proportion | Purpose |
|---|---:|---|
| Training | 60% | Model fitting |
| Calibration | 15% | Threshold and conformal calibration |
| Test | 25% | Final performance evaluation |

The split is stratified by GDM outcome.

The test partition remains untouched for final predictive performance evaluation.

> The test partition is not used to select the classification threshold.

---

# 🧬 Predictor Selection and Leakage Control

SafeTriage-GDM uses a deliberately restricted **antepartum predictor schema**.

The approved predictors are:

```text
Evidence of maternal anaemia?

Do we have data related to multiple micronutrient supplementation?

Did the mother supplement with multiple micronutrients during pregnancy?

Relative to the start of pregnancy, when did multiple micronutrient supplementation start?

Relative to the start of pregnancy, when did multiple micronutrient supplementation stop?

Did the mothers just supplement with multiple micronutrients during pregnancy and nothing else?

For how many weeks were multiple micronutrients taken?

Mother's pre-pregnancy BMI (kg/m2)

Mother's height (cm)

Mother's weight before pregnancy (kg)

Mother's age (years)

Did the mother smoke during pregnancy?

Twin pregnancy?

Parity
```

## Variables deliberately excluded

Variables representing information that is clearly available only after delivery are excluded from the predictive feature schema.

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

Gestational hypertension and pre-eclampsia variables are also excluded from the baseline predictor schema because their availability depends on the clinical prediction time point.

This predictor design is intended to reduce obvious outcome leakage and make the modelling framework more consistent with an antepartum prediction setting.

---

# ⚖️ Class Imbalance Strategy

GDM-positive observations may be substantially less frequent than GDM-negative observations.

SafeTriage-GDM uses a **training-only ROSE-style smoothed minority oversampling procedure**.

The procedure:

1. Separates minority and majority observations.
2. Samples minority observations with replacement.
3. Applies small Gaussian perturbations to numeric minority features.
4. Samples categorical values from observed minority values.
5. Generates additional minority observations until the training classes are approximately balanced.
6. Shuffles the resulting training data.

The ROSE-style procedure is applied **only to the training partition**.

```text
Training
    │
    └── ROSE-style balancing
            │
            ▼
       Model fitting

Calibration
    │
    └── Natural distribution

Test
    │
    └── Natural distribution
```

The application does **not** use:

- Class weighting
- `scale_pos_weight`
- SMOTETomek
- Test-set oversampling
- Calibration-set oversampling

> The Python implementation is a ROSE-style approximation using smoothed minority resampling. It is not presented as an exact reimplementation of the R `ROSE` package.

---

# 🧩 Missing Data and Preprocessing

The application uses a MICE-style iterative imputation approach for numeric variables.

The numeric preprocessing workflow is:

```text
Numeric Variables
      │
      ▼
Iterative Imputation
      │
      ▼
Standardization
```

Categorical variables use:

```text
Categorical Variables
      │
      ▼
Most-Frequent Imputation
      │
      ▼
One-Hot Encoding
```

Unknown categorical levels encountered during inference are handled using:

```python
handle_unknown="ignore"
```

The preprocessing transformation is fitted on the training data and subsequently applied to calibration, test, and inference observations.

---

# 🛡️ Conformal Prediction

SafeTriage-GDM uses a dedicated calibration partition to construct a split-conformal prediction threshold.

For each calibration observation, the nonconformity score is:

```text
S_i = 1 - P(Y_i | X_i)
```

where:

- `Y_i` is the observed GDM class.
- `X_i` represents the predictor variables.
- `P(Y_i | X_i)` is the ensemble probability assigned to the true class.

The application uses:

```text
90% conformal confidence
```

The conformal threshold is estimated from the dedicated calibration partition.

Population-shift monitoring does **not** dynamically modify the conformal threshold.

> **Important:** Conformal prediction provides a statistical coverage framework under its underlying assumptions. It does not establish clinical safety, diagnostic validity, or medical reliability.

---

# 🔬 Uncertainty Quantification

SafeTriage-GDM reports several uncertainty-related quantities.

## Model-Disagreement / Epistemic Uncertainty

The standard deviation of the three model probabilities is used as a model-disagreement proxy:

```text
σ_models = SD(P_RF, P_XGB, P_LR)
```

A larger value indicates greater disagreement between the constituent models.

## Aleatoric Entropy

Binary entropy is calculated for each model probability:

```text
H(p) = -p log2(p) - (1-p) log2(1-p)
```

The model-specific entropy values are averaged to produce an aleatoric-uncertainty proxy.

## Predictive Entropy

Predictive entropy is calculated from the ensemble probability:

```text
H_predictive = -p log2(p) - (1-p) log2(1-p)
```

where `p` is the ensemble probability of GDM.

## Mutual Information

The application estimates:

```text
MI = max(H_predictive - H_aleatoric, 0)
```

These quantities are intended for research-oriented uncertainty analysis and should not be interpreted as direct measures of clinical certainty.

---

# 📈 Predictive Performance Evaluation

Final predictive performance is evaluated on the untouched test partition.

The application reports:

- ROC-AUC
- PR-AUC
- Accuracy
- Balanced Accuracy
- Sensitivity
- Specificity
- Precision
- F1 Score
- Brier Score
- Log Loss

## ROC-AUC

ROC-AUC evaluates the ability of the model to rank positive observations above negative observations across classification thresholds.

## PR-AUC

PR-AUC summarizes the precision-recall relationship and is particularly informative when the positive class is relatively uncommon.

## Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

## Balanced Accuracy

```text
Balanced Accuracy = (Sensitivity + Specificity) / 2
```

## Sensitivity

```text
Sensitivity = TP / (TP + FN)
```

## Specificity

```text
Specificity = TN / (TN + FP)
```

## Precision

```text
Precision = TP / (TP + FP)
```

## F1 Score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

## Brier Score

```text
Brier = (1/n) × Σ(p_i - y_i)²
```

Lower values indicate better probabilistic accuracy.

## Log Loss

```text
LogLoss = -(1/n) × Σ[y_i log(p_i) + (1-y_i) log(1-p_i)]
```

Lower values indicate better probabilistic performance.

---

# 🎚️ Calibration-Derived Decision Threshold

The application does not automatically use a fixed `0.50` threshold.

Instead, the ensemble decision threshold is selected using the **calibration partition only**.

Candidate thresholds are evaluated using balanced accuracy:

```text
Balanced Accuracy
=
(Sensitivity + Specificity) / 2
```

The threshold with the highest calibration balanced accuracy is selected.

The test partition is not used for threshold selection.

---

# 📊 Model Performance Visualizations

The Streamlit application provides interactive visualizations for research analysis.

These include:

### Model Performance Comparison

Compares:

- Accuracy
- Balanced Accuracy
- Sensitivity
- Specificity
- Precision
- F1

across:

- Random Forest
- XGBoost
- Logistic Regression

### ROC Curves

Interactive ROC curves are displayed for:

- Random Forest
- XGBoost
- Logistic Regression
- Equal-weight Ensemble

### Precision-Recall Curves

Interactive precision-recall curves are provided for all three individual models and the ensemble.

### Threshold Analysis

The application displays sensitivity, specificity, and balanced accuracy across candidate classification thresholds.

### Confusion Matrix

The final ensemble confusion matrix is displayed both as a table and as a visual chart.

### Uncertainty Visualization

The application displays distributions of:

- Epistemic uncertainty
- Aleatoric uncertainty
- Predictive entropy
- Mutual information

### Fairness Visualization

Age-group selection rates and false-positive rates are visualized.

### BMI Distribution

Training and uploaded BMI distributions are compared visually.

### Feature Contribution

The most influential predictive variables are displayed as interactive horizontal feature-importance bars.

---

# ⚖️ Fairness Auditing

SafeTriage-GDM includes a dedicated fairness analysis across maternal age groups.

The age groups are:

| Age group |
|---|
| <25 |
| 25–34 |
| 35–44 |
| 45+ |

## Selection Rate

```text
Selection Rate_g
=
Number classified as elevated risk
/
Total observations in group g
```

The application reports the difference between the highest and lowest group-level selection rates.

## False Positive Rate

For observations with known GDM ground truth:

```text
FPR_g = FP_g / (FP_g + TN_g)
```

The application reports:

```text
FPR Disparity
=
Maximum group FPR
-
Minimum group FPR
```

This is explicitly labelled **FPR disparity**.

It is **not** described as Equalized Odds disparity because Equalized Odds requires assessment of both false-positive and true-positive rates.

---

# 📉 Population Shift Monitoring

SafeTriage-GDM monitors the distribution of maternal pre-pregnancy BMI using the Population Stability Index (PSI).

BMI categories are:

| Category | Range |
|---|---|
| Underweight | <18.5 |
| Normal | 18.5–24.9 |
| Overweight | 25.0–29.9 |
| Obesity I | 30.0–39.9 |
| Obesity II+ | ≥40 |

PSI is calculated as:

```text
PSI = Σ(A_j - E_j) × ln(A_j / E_j)
```

where:

- `A_j` is the current/uploaded proportion in category `j`.
- `E_j` is the training/reference proportion in category `j`.

The research monitoring threshold is:

```text
PSI = 0.20
```

Interpretation:

```text
PSI < 0.10
    Minimal population shift

0.10 ≤ PSI < 0.20
    Moderate population shift

PSI ≥ 0.20
    Substantial population shift
```

A PSI warning does **not** automatically modify model predictions or the conformal threshold.

---

# 🔎 Explainable AI

SafeTriage-GDM provides global model-based feature contribution analysis.

The application uses:

- Random Forest feature importance
- XGBoost feature importance
- Absolute Logistic Regression coefficients

The contribution values are aggregated across the three models at the level of the original predictor variables.

The application displays the highest-ranked predictors as interactive feature-weight bars.

> **Important:** Feature importance describes predictive model behaviour. It is not causal inference. A highly influential predictive variable does not necessarily mean that changing that variable will cause a change in GDM risk.

---

# 📁 Accepted Data Format

The application accepts:

```text
.xlsx
.xls
```

The uploaded dataset should contain the target column:

```text
Gestational diabetes?
```

and all required approved predictor variables.

The application distinguishes between:

```text
All uploaded observations
        │
        ├── Known GDM outcome
        │       │
        │       └── Training / Calibration / Test
        │
        └── Missing GDM outcome
                │
                └── Inference population
```

Observations without a known GDM outcome can remain in the uploaded dataset for inference, while supervised performance evaluation uses only observations with known GDM outcomes.

---

# 🔐 Privacy and Data Protection

SafeTriage-GDM is intended for research datasets.

**Do not upload directly identifiable patient information.**

Do not upload:

- Names
- Medical record numbers
- Addresses
- Telephone numbers
- Email addresses
- National identification numbers
- Other direct identifiers

Users are responsible for ensuring that datasets uploaded to the application are appropriately de-identified and that their use complies with applicable ethical, institutional, and data-protection requirements.

> For a public GitHub repository, do not commit private patient-level datasets unless you have explicit permission and the data are legally shareable.

---

# 🚀 Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/ikechukwukamalu8/safetriage-gdm-app.git
cd safetriage-gdm-app
```

## 2. Create a Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Application

```bash
streamlit run app.py
```

---

# ☁️ Streamlit Cloud Deployment

SafeTriage-GDM can be deployed using Streamlit Community Cloud.

### Deployment steps

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository.
4. Set the main file to:

```text
app.py
```

5. Deploy the application.
6. Ensure that `requirements.txt` is present in the repository.

The application does not require a pre-trained `.pkl` or `.joblib` model artifact.

Models are trained dynamically from the uploaded research dataset.

---

# 📦 Requirements

The main dependencies are:

```text
streamlit==1.65.0
scikit-learn==1.6.1
numpy
pandas
xgboost
openpyxl
joblib
imbalanced-learn
altair
```

The current application uses Altair for interactive visualization and does not require Matplotlib.

---

# 📂 Repository Structure

```text
safetriage-gdm-app/
│
├── app.py
├── requirements.txt
└── README.md
```

## `app.py`

Main Streamlit application containing:

- Dataset validation
- Predictor validation
- Leakage-aware feature selection
- Data splitting
- ROSE-style training balancing
- MICE-style imputation
- Categorical encoding
- Standardization
- Random Forest
- XGBoost
- Logistic Regression
- Equal-weight ensemble
- Threshold selection
- Conformal prediction
- Uncertainty quantification
- Test-set evaluation
- ROC curves
- Precision-recall curves
- Threshold analysis
- Confusion matrix
- Fairness auditing
- BMI PSI monitoring
- Feature contribution analysis
- Dataset-level inference
- CSV export

## `requirements.txt`

Python dependency specification.

## `README.md`

Project documentation.

---

# 🔬 Reproducibility

The application uses a fixed random seed:

```python
RANDOM_STATE = 42
```

The model development procedure follows:

```text
60% Training
15% Calibration
25% Testing
```

The calibration partition is used for:

- Decision-threshold selection
- Conformal calibration

The final test partition is reserved for performance evaluation.

---

# 🧪 Methodological Safeguards

The application deliberately avoids several practices that can lead to optimistic or misleading evaluation.

SafeTriage-GDM does **not** use:

- Synthetic labels
- Test-set oversampling
- Test-set threshold optimization
- Class weighting
- `scale_pos_weight`
- SMOTETomek
- Post-hoc sigmoid calibration
- Test-set probability calibration
- Baby-at-birth variables as predictors
- Dynamic modification of conformal `q` using BMI drift
- Training of models on unlabeled target observations

The test partition retains its natural class distribution.

---

# 📋 Research Workflow Summary

```text
1. Upload research dataset
             │
             ▼
2. Validate target and predictors
             │
             ▼
3. Identify labeled observations
             │
             ▼
4. Stratified 60/15/25 split
             │
             ▼
5. ROSE-style balancing
   └── Training only
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
9. Select threshold
   └── Calibration only
             │
             ▼
10. Evaluate untouched test set
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
12. Uncertainty analysis
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

# ⚠️ Limitations

SafeTriage-GDM is a research prototype and has important limitations.

## Dataset Limitations

Predictive performance depends strongly on the characteristics, quality, size, completeness, and representativeness of the supplied research dataset.

## Class Imbalance

GDM-positive observations may be substantially fewer than GDM-negative observations.

Training-only ROSE-style balancing is used to address the modelling challenge, but balancing does not guarantee improved generalization.

## Small Test-Set Uncertainty

When the number of positive GDM cases is small, sensitivity, specificity, PR-AUC, F1, and related metrics can have substantial sampling variability.

Point estimates should therefore be interpreted cautiously.

## External Validity

Performance on one dataset does not establish performance in other:

- Populations
- Healthcare systems
- Countries
- Demographic groups
- Clinical settings

External validation is required before drawing conclusions about generalizability.

## Prediction Timing

Some pregnancy-related variables become available only at particular points during pregnancy.

The predictor schema therefore deliberately excludes variables that are clearly post-delivery or strongly outcome-adjacent.

## Fairness Limitations

The current fairness audit focuses on:

- Maternal age groups
- Selection rates
- False-positive rates

It is not a complete fairness evaluation.

It does not establish Equalized Odds, calibration fairness, demographic parity, or fairness across every protected characteristic.

## Conformal Prediction Limitations

Conformal prediction relies on assumptions such as exchangeability between calibration and future observations.

Violation of these assumptions can affect empirical coverage.

Conformal prediction should not be interpreted as a guarantee of clinical safety.

## Explainability Limitations

Feature importance and model contribution measures describe predictive model behaviour.

They do not establish:

- Causality
- Biological mechanisms
- Clinical effectiveness
- Treatment effects

## Clinical Validity

SafeTriage-GDM has not been established as a clinically validated diagnostic or triage instrument.

It should therefore be used only as a research and educational prototype.

---

# 🧪 Research Interpretation

SafeTriage-GDM should be interpreted as an experimental framework for studying the interaction between:

```text
Predictive Modelling
        +
Uncertainty Quantification
        +
Conformal Methods
        +
Algorithmic Fairness
        +
Population Shift Monitoring
        +
Explainable AI
        +
Leakage-Aware Modelling
```

The system is intended to support methodological research rather than replace clinical expertise.

---

# 🔭 Potential Research Extensions

Possible future research directions include:

- External validation on independent cohorts
- Prospective evaluation
- Temporal validation
- Geographic validation
- Calibration analysis
- Confidence intervals for performance metrics
- Bootstrap uncertainty estimates
- Decision-curve analysis
- Subgroup calibration
- Additional protected-group fairness analysis
- Fairness-aware model optimization
- Counterfactual fairness analysis
- Causal modelling
- Conformal risk-control methods
- Distribution-free uncertainty evaluation
- Advanced drift detection
- Federated learning
- Privacy-preserving machine learning
- Prospective clinical workflow evaluation

These extensions are research directions and are not currently implemented in the application.

---

# 📊 Interpreting Model Performance

A higher ROC-AUC does not necessarily mean that the model is clinically useful.

In an imbalanced GDM prediction setting, multiple dimensions of performance should be considered together:

```text
Discrimination
    │
    ├── ROC-AUC
    └── PR-AUC

Classification
    │
    ├── Sensitivity
    ├── Specificity
    ├── Precision
    └── F1

Probabilistic Quality
    │
    ├── Brier Score
    └── Log Loss

Uncertainty
    │
    ├── Epistemic
    ├── Aleatoric
    ├── Predictive Entropy
    └── Mutual Information

Reliability / Robustness
    │
    ├── Conformal Coverage
    ├── Fairness
    └── Population Stability
```

No single metric should be interpreted in isolation.

---

# 🧭 Responsible AI Considerations

SafeTriage-GDM is designed around several responsible machine-learning principles:

### Transparency

The modelling architecture and evaluation metrics are explicitly documented.

### Leakage Control

Post-delivery variables are excluded from the antepartum predictor schema.

### Test Isolation

The final test partition is not used for model fitting or threshold selection.

### Uncertainty Awareness

The system reports several uncertainty-related quantities rather than presenting predictions without context.

### Fairness Monitoring

Performance-related disparities are examined across maternal age groups.

### Population Monitoring

BMI population stability is monitored to identify potential distributional changes.

### Explainability

Feature contribution information is presented to support model inspection.

### Privacy

Users are instructed not to upload directly identifiable patient information.

---

# 📜 Disclaimer

**SafeTriage-GDM is a research prototype.**

It is not:

- A medical device
- A diagnostic system
- A treatment recommendation system
- A clinical decision-making system
- A replacement for a healthcare professional

Predictions, probabilities, uncertainty estimates, conformal prediction sets, fairness metrics, population-shift statistics, and explanations should not be used as a substitute for assessment by qualified healthcare professionals.

The system has not been clinically validated, externally validated, prospectively evaluated, or approved for clinical use.

---

# 👨‍💻 Author

**Ikechukwu Okechi Kamalu**

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

# 🌐 Project Links

**Live Application:**  
https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/

**GitHub Repository:**  
https://github.com/ikechukwukamalu8/safetriage-gdm-app

---

# 📄 Citation

If this prototype contributes to your research, please cite the associated project or repository.

```text
Kamalu, I. O.
SafeTriage-GDM: Uncertainty-Quantified Clinical Triage System
for Gestational Diabetes Mellitus Risk with Algorithmic Fairness
Auditing and Conformal Safety Bounds.
GitHub repository.
```

---

# 📜 License

This project is provided for research and educational purposes.

Please review the repository license before using, modifying, or redistributing the software or associated datasets.

---

## ⭐ Project Summary

SafeTriage-GDM integrates:

```text
Leakage-Aware Predictors
          +
MICE-Style Imputation
          +
Training-Only ROSE-Style Balancing
          +
Random Forest
          +
XGBoost
          +
Logistic Regression
          +
Equal-Weight Ensemble
          +
Calibration-Derived Threshold
          +
90% Conformal Prediction
          +
Uncertainty Quantification
          +
Fairness Auditing
          +
BMI Population Stability
          +
Explainable AI
          +
Interactive Research Dashboard
```

**SafeTriage-GDM is intended as a transparent, uncertainty-aware, and fairness-conscious research framework for studying machine-learning approaches to GDM risk prediction.**
