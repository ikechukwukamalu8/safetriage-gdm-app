# 🛡️ SafeTriage-GDM

## Uncertainty-Quantified Clinical Triage System for Gestational Diabetes Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds

**SafeTriage-GDM** is a standalone research prototype for uncertainty-aware risk triage of Gestational Diabetes Mellitus (GDM).

The system combines a heterogeneous three-model machine learning ensemble with conformal safety assessment, uncertainty quantification, algorithmic fairness auditing, population-shift monitoring, and global Explainable AI (XAI).

> **SafeTriage-GDM is a research prototype. It is not a medical device, has not been clinically validated, and must not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.**

---

## 🚀 Live Application

The deployed Streamlit application is publicly accessible here:

**[Launch SafeTriage-GDM](https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/)**

Users can upload a compatible `.xlsx` or `.xls` dataset directly through the web application.

The application validates the uploaded dataset, identifies the approved predictor variables, trains or loads the SafeTriage-GDM pipeline, and provides risk, uncertainty, fairness, population-shift, and XAI analyses.

---

## 🌟 Key Features

### Three-Model Ensemble

SafeTriage-GDM uses exactly three machine learning models:

- Random Forest
- XGBoost
- Logistic Regression

The ensemble GDM probability is calculated as:

$$
P(GDM) =
\frac{
P_{RF}(GDM) +
P_{XGB}(GDM) +
P_{LR}(GDM)
}{3}
$$

---

### Antepartum Predictor Schema

The application uses an explicit predictor whitelist to reduce information leakage.

Approved variables include maternal and pregnancy-related information such as:

- Maternal anaemia
- Multiple micronutrient supplementation
- Timing of micronutrient supplementation
- Duration of micronutrient supplementation
- Pre-pregnancy BMI
- Maternal height
- Pre-pregnancy weight
- Pregnancy weight gain
- Maternal age
- Smoking during pregnancy
- Twin pregnancy
- Parity

Post-delivery variables are deliberately excluded.

These include:

- Baby sex
- Baby birth weight
- Baby BMI at birth
- Baby gestational age at birth
- Premature birth
- Low birth weight
- Baby anthropometric measurements
- Small-for-gestational-age status

This predictor filtering is designed to reduce post-outcome information leakage.

---

## 🛡️ Conformal Safety Assessment

A dedicated calibration set is used for conformal prediction.

The supervised dataset is divided into:

| Partition | Proportion | Purpose |
|---|---:|---|
| Training | 60% | Model fitting |
| Calibration | 15% | Conformal calibration |
| Test | 25% | Final evaluation |

The conformal confidence level used by the application is:

**90%**

For each calibration observation, the nonconformity score is calculated as:

$$
s_i = 1 - P(Y_i \mid X_i)
$$

The resulting finite-sample conformal quantile is used to determine whether a prediction satisfies the predefined conformal safety bound.

The conformal threshold is estimated from the calibration data and is not dynamically modified using population drift.

---

## 🔬 Uncertainty Quantification

SafeTriage-GDM provides several uncertainty measures.

### Model-Disagreement Uncertainty

The standard deviation of the three model probabilities is used as a model-disagreement proxy:

$$
\sigma_{models}
=
SD(P_{RF}, P_{XGB}, P_{LR})
$$

### Aleatoric Entropy

Binary entropy is calculated for each model probability and averaged across the ensemble.

### Predictive Entropy

The entropy of the ensemble probability is calculated as:

$$
H(P)
=
-P\log(P)
-(1-P)\log(1-P)
$$

### Mutual Information

The application calculates:

$$
MI =
\max
\left(
H_{predictive}
-
H_{aleatoric},
0
\right)
$$

These quantities are intended for research-oriented uncertainty analysis and should not be interpreted as clinical certainty.

---

## 📈 Predictive Performance

Performance is evaluated on an untouched test set.

The application reports:

- ROC-AUC
- PR-AUC
- Brier Score
- Log Loss
- Accuracy
- Sensitivity
- Specificity
- Precision
- F1-score
- Confusion Matrix

The application also provides:

- ROC curve
- Precision–Recall curve
- Descriptive calibration curve
- Individual model performance
- Ensemble performance

The test set is not used for model fitting or conformal calibration.

---

## ⚖️ Algorithmic Fairness Auditing

SafeTriage-GDM includes an age-stratified fairness analysis.

Age groups are defined as:

| Age Group | Definition |
|---|---|
| `<25` | Age below 25 |
| `25–34` | Age 25–34 |
| `35–44` | Age 35–44 |
| `45+` | Age 45 and above |

### Selection Rate

The application calculates the proportion of observations receiving a positive prediction within each age group.

This provides a demographic-parity-style analysis.

### False Positive Rate

For observations with known ground truth:

$$
FPR =
\frac{FP}{FP+TN}
$$

The application reports FPR disparity as:

$$
FPR\ Disparity =
\max(FPR_g) - \min(FPR_g)
$$

The application deliberately calls this **FPR disparity**, rather than Equalized Odds Disparity, because Equalized Odds requires assessment of both false-positive and true-positive rates.

---

## 📊 Population-Shift Monitoring

SafeTriage-GDM monitors the distribution of maternal pre-pregnancy BMI using the Population Stability Index (PSI).

BMI is grouped into:

- Underweight
- Normal
- Overweight
- Obesity I
- Obesity II+

The PSI is calculated as:

$$
PSI =
\sum_i
(A_i-E_i)
\ln
\left(
\frac{A_i}{E_i}
\right)
$$

where:

- $E_i$ represents the baseline training proportion.
- $A_i$ represents the current population proportion.

The research monitoring threshold is:

**PSI = 0.20**

A PSI of 0.20 or greater generates a population-shift warning.

### Important

BMI population-shift monitoring does not:

- Retrain the model
- Modify model probabilities
- Modify the conformal threshold
- Automatically alter predictions

It is a monitoring and research diagnostic.

---

## 🔎 Explainable AI (XAI)

SafeTriage-GDM provides global feature-contribution analysis across the three models.

### Random Forest

Uses model feature importance.

### XGBoost

Uses model feature importance.

### Logistic Regression

Uses the absolute magnitude of standardized model coefficients.

The contributions are normalized and aggregated across one-hot encoded variables to produce contributions at the original clinical-variable level.

The application displays a global feature-contribution ranking.

> Feature contribution is not equivalent to causal inference. A high contribution does not mean that changing a variable will necessarily change GDM risk.

---

## 🩺 Patient Triage Console

The Patient Triage Console allows an observation to be selected from the uploaded dataset.

The application displays:

- Ensemble GDM probability
- Random Forest probability
- XGBoost probability
- Logistic Regression probability
- Model-disagreement uncertainty
- Predictive entropy
- Mutual information
- Conformal nonconformity
- Conformal safety status
- Patient predictor values

The research classification threshold is:

**P(GDM) ≥ 0.50**

This threshold is an operational research setting and is **not a clinically validated diagnostic threshold**.

---

## 📁 Repository Structure

```text
safetriage-gdm-app/
│
├── app.py
├── dataCBGS_dataset.xlsx
├── requirements.txt
└── README.md


