# 🛡️ SafeTriage-GDM

## Uncertainty-Quantified Clinical Triage System for Gestational Diabetes Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-green.svg)](https://xgboost.readthedocs.io/)

**SafeTriage-GDM** is a standalone research prototype for uncertainty-aware Gestational Diabetes Mellitus (GDM) risk triage, conformal safety assessment, algorithmic fairness auditing, population-shift monitoring, and explainable machine learning.

The system combines three complementary machine-learning models:

- Random Forest
- XGBoost
- Logistic Regression

Their predicted GDM probabilities are combined into a simple ensemble:

$$
P(\mathrm{GDM}) =
\frac{
P_{\mathrm{RF}} +
P_{\mathrm{XGB}} +
P_{\mathrm{LR}}
}{3}
$$

The application is designed for **research and methodological evaluation**. It is **not a medical device** and must not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

---

## 🌐 Live Application

The deployed Streamlit application is available here:

**[Open SafeTriage-GDM](https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/)**

The application allows users to upload a compatible research dataset in Excel format and perform model training, prediction, uncertainty analysis, fairness auditing, population-shift monitoring, and explainability analysis.

> **Privacy notice:** Do not upload names, medical record numbers, addresses, telephone numbers, email addresses, national identification numbers, or other directly identifiable patient information.

---

## 🎯 Research Objectives

SafeTriage-GDM investigates several aspects of responsible clinical machine learning:

1. GDM risk prediction using heterogeneous machine-learning models.
2. Uncertainty quantification through model disagreement and entropy.
3. Conformal safety assessment using a dedicated calibration set.
4. Algorithmic fairness assessment across maternal age groups.
5. Population-shift monitoring using maternal BMI.
6. Explainable feature contribution analysis.
7. Leakage-aware predictor selection for antepartum modelling.

---

## 🧠 Model Architecture

SafeTriage-GDM uses exactly three predictive models:

### 1. Random Forest

Random Forest is used to capture nonlinear relationships and interactions between maternal and pregnancy-related variables.

### 2. XGBoost

XGBoost provides a gradient-boosted tree model capable of representing nonlinear predictive relationships.

### 3. Logistic Regression

Logistic Regression provides a comparatively interpretable linear modelling baseline.

### Ensemble Probability

The final research risk score is calculated as the arithmetic mean of the three model probabilities:

$$
P_{\mathrm{ensemble}}
=
\frac{
P_{\mathrm{RF}} +
P_{\mathrm{XGB}} +
P_{\mathrm{LR}}
}{3}
$$

---

## 📊 Data Processing Pipeline

The application follows the following general workflow:

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
      │       │
      │       └── Train / Calibration / Test
      │
      └── Unlabeled observations
              │
              └── Inference population
      │
      ▼
Antepartum Predictor Selection
      │
      ▼
One-Hot Encoding
      │
      ▼
Iterative Imputation
      │
      ▼
Standardization
      │
      ▼
┌───────────────────────┐
│ Random Forest         │
│ XGBoost               │
│ Logistic Regression   │
└───────────────────────┘
      │
      ▼
Probability Ensemble
      │
      ├── Risk Classification
      ├── Uncertainty Quantification
      ├── Conformal Safety
      ├── Fairness Auditing
      ├── BMI Population Shift
      └── XAI Attribution
```

---

## 🔀 Train / Calibration / Test Design

Labeled observations are divided into three independent partitions:

| Partition | Proportion | Purpose |
|---|---:|---|
| Training | 60% | Model fitting |
| Calibration | 15% | Conformal calibration |
| Test | 25% | Final performance evaluation |

The test set remains untouched during model fitting and conformal calibration.

This separation is important because final predictive performance should be evaluated on observations that were not used to fit the predictive models.

---

## 🧬 Predictor Selection and Leakage Control

SafeTriage-GDM uses an antepartum predictor schema.

Examples include:

- Maternal age
- Maternal pre-pregnancy BMI
- Maternal height
- Maternal pre-pregnancy weight
- Pregnancy weight gain
- Parity
- Twin pregnancy
- Maternal anaemia
- Smoking during pregnancy
- Micronutrient supplementation variables

Variables representing information that is only available after delivery are excluded.

Examples include:

- Baby birth weight
- Baby BMI at birth
- Baby gestational age at birth
- Baby head circumference
- Baby skinfold measurements
- Low birth weight
- Premature birth
- Small-for-gestational-age status
- Baby sex

Gestational hypertension and pre-eclampsia variables are also excluded from the baseline predictor schema because their availability depends on the clinical prediction time point.

This design is intended to reduce obvious outcome leakage.

---

## 🛡️ Conformal Safety Assessment

SafeTriage-GDM uses a dedicated calibration set to estimate a conformal nonconformity threshold.

For a calibration observation, the nonconformity score is defined as:

$$
s_i
=
1 -
P(Y_i \mid X_i)
$$

where:

- $Y_i$ is the observed GDM class.
- $X_i$ represents the predictor variables.
- $P(Y_i \mid X_i)$ is the ensemble probability assigned to the true class.

The finite-sample conformal quantile is then calculated from the calibration scores.

The application uses a research confidence setting of **90%**.

The conformal threshold is estimated from the dedicated calibration data and is **not dynamically modified using population drift**.

> **Important:** Conformal coverage is a statistical property under the assumptions of the conformal procedure. It should not be interpreted as a clinical safety or diagnostic guarantee.

---

## 🔬 Uncertainty Quantification

SafeTriage-GDM reports several uncertainty-related quantities.

### Model-Disagreement Uncertainty

The standard deviation of the three model probabilities is used as a model-disagreement proxy:

$$
\sigma_{\mathrm{models}}
=
\operatorname{SD}
\left(
P_{\mathrm{RF}},
P_{\mathrm{XGB}},
P_{\mathrm{LR}}
\right)
$$

A larger value indicates greater disagreement between the constituent models.

### Aleatoric Entropy

Binary entropy is calculated for each model probability:

$$
H(p)
=
-p\log(p)
-
(1-p)\log(1-p)
$$

The model-specific entropy values are averaged to obtain an aleatoric-uncertainty proxy.

### Predictive Entropy

Predictive entropy is calculated using the ensemble probability:

$$
H_{\mathrm{predictive}}
=
-p\log(p)
-
(1-p)\log(1-p)
$$

where $p$ is the ensemble probability of GDM.

### Mutual Information

The application estimates a model-disagreement-related mutual-information quantity as:

$$
MI
=
\max
\left(
H_{\mathrm{predictive}}
-
H_{\mathrm{aleatoric}},
0
\right)
$$

These quantities are intended for research-oriented uncertainty analysis. They should not be interpreted as measures of clinical certainty.

---

## 📈 Predictive Performance

Performance is evaluated exclusively on the untouched test set.

The application reports:

- ROC-AUC
- PR-AUC
- Brier Score
- Log Loss
- Accuracy
- Sensitivity
- Specificity
- Precision
- F1 Score

### ROC-AUC

ROC-AUC evaluates the ability of the model to rank positive observations above negative observations across classification thresholds.

### PR-AUC

PR-AUC summarizes the precision-recall relationship and is particularly informative when the positive class is relatively uncommon.

### Brier Score

The Brier score evaluates the squared difference between predicted probability and observed binary outcome:

$$
\mathrm{Brier}
=
\frac{1}{n}
\sum_{i=1}^{n}
(p_i-y_i)^2
$$

where:

- $p_i$ is the predicted probability.
- $y_i$ is the observed binary outcome.

### Log Loss

Binary log loss is calculated as:

$$
\mathrm{LogLoss}
=
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
$$

Lower values indicate better probabilistic performance.

---

## ⚖️ Fairness Auditing

SafeTriage-GDM includes a dedicated fairness analysis.

Maternal age is divided into the following groups:

| Age group |
|---|
| <25 |
| 25–34 |
| 35–44 |
| 45+ |

### Selection Rate

The selection rate is the proportion of observations classified as elevated risk within each age group:

$$
\mathrm{SelectionRate}_g
=
\frac{
\sum_i I(\hat{Y}_i=1)
}{
N_g
}
$$

The application reports the difference between the highest and lowest group-level selection rates.

### False Positive Rate

For observations with known GDM ground truth, the false positive rate is:

$$
\mathrm{FPR}_g
=
\frac{
FP_g
}{
FP_g + TN_g
}
$$

The application reports:

$$
\mathrm{FPR\ Disparity}
=
\max_g(\mathrm{FPR}_g)
-
\min_g(\mathrm{FPR}_g)
$$

This metric is explicitly labelled **FPR disparity**.

It is **not** described as Equalized Odds Disparity because Equalized Odds requires assessment of both false-positive and true-positive rates.

---

## 📉 Population Shift Monitoring

The application monitors the distribution of maternal pre-pregnancy BMI using the Population Stability Index (PSI).

BMI categories are:

| Category | Range |
|---|---|
| Underweight | <18.5 |
| Normal | 18.5–24.9 |
| Overweight | 25.0–29.9 |
| Obesity I | 30.0–39.9 |
| Obesity II+ | ≥40 |

PSI is calculated as:

$$
PSI
=
\sum_j
(A_j-E_j)
\ln
\left(
\frac{A_j}{E_j}
\right)
$$

where:

- $A_j$ is the observed/current proportion in category $j$.
- $E_j$ is the baseline/training proportion in category $j$.

The application uses a research monitoring threshold of:

$$
PSI = 0.20
$$

A PSI warning does **not** automatically modify model predictions or the conformal threshold.

---

## 🔎 Explainable AI

SafeTriage-GDM provides global feature contribution analysis.

The application combines:

- Random Forest feature importance
- XGBoost feature importance
- Absolute Logistic Regression coefficients

Each model's contribution values are normalized before being aggregated.

The resulting contribution score is intended to identify variables that have greater influence on the predictive models.

> **Important:** Feature contribution is not causal inference. A highly influential predictive feature does not necessarily mean that changing the feature will cause a change in GDM risk.

---

## 📁 Accepted Data Format

The application accepts:

- `.xlsx`
- `.xls`

Excel files are uploaded directly through the Streamlit interface.

A compatible dataset should contain the GDM outcome column:

```text
Gestational diabetes?
```

and an appropriate subset of the approved predictor variables.

The application distinguishes between:

```text
All uploaded observations
        │
        ├── Valid GDM outcome
        │       └── Training / calibration / test evaluation
        │
        └── Missing GDM outcome
                └── Inference population
```

Therefore, unlabeled observations can remain in the uploaded dataset for inference, while supervised performance evaluation uses only observations with known GDM outcomes.

---

## 🔐 Privacy and Data Protection

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

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ikechukwukamalu8/safetriage-gdm-app.git
cd safetriage-gdm-app
```

### 2. Create a Virtual Environment

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

The main dependencies are:

```text
streamlit==1.65.0
scikit-learn==1.6.1
numpy
pandas
xgboost
openpyxl
```

---

## 📂 Repository Structure

```text
safetriage-gdm-app/
│
├── app.py
├── dataCBGS_dataset.xlsx
├── requirements.txt
└── README.md
```

### `app.py`

Main Streamlit application containing:

- Dataset validation
- Data preprocessing
- Model training
- Ensemble prediction
- Conformal calibration
- Uncertainty quantification
- Performance evaluation
- Fairness auditing
- BMI population-shift monitoring
- XAI feature contribution analysis

### `dataCBGS_dataset.xlsx`

Example research dataset used for local demonstration.

> Ensure that any dataset committed to a public repository is legally shareable and contains no directly identifiable information.

### `requirements.txt`

Python dependency specification.

### `README.md`

Project documentation.

---

## 🔬 Reproducibility

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

The calibration partition is used for conformal calibration, while the final test partition remains untouched for performance evaluation.

The application does not use:

- Synthetic labels
- Test-set oversampling
- Test-set threshold optimization
- Post-hoc probability calibration on the test set
- Baby-at-birth variables as predictors
- Dynamic modification of conformal q using BMI drift

---

## ⚠️ Limitations

SafeTriage-GDM is a research prototype and has important limitations.

### Dataset Limitations

Predictive performance depends strongly on the characteristics, quality, size, and representativeness of the supplied research dataset.

### Class Imbalance

GDM-positive observations may be substantially fewer than GDM-negative observations.

### Small Test-Set Uncertainty

When the number of positive GDM cases is small, sensitivity, specificity, PR-AUC, and related metrics can have substantial sampling variability.

### External Validity

Performance on one dataset does not establish performance in other populations, healthcare systems, countries, or demographic groups.

### Prediction Timing

Some pregnancy-related variables may only become available at particular points during pregnancy. The predictor schema therefore deliberately excludes variables that are clearly post-delivery or outcome-adjacent.

### Clinical Validity

The application has not been established as a clinically validated diagnostic or triage instrument.

---

## 🧪 Research Interpretation

SafeTriage-GDM should be interpreted as an experimental framework for studying the interaction between:

```text
Predictive Modelling
        +
Uncertainty Quantification
        +
Conformal Methods
        +
Fairness Auditing
        +
Population Shift Monitoring
        +
Explainable AI
```

The system is intended to support methodological research rather than replace clinical expertise.

---

## 📜 Disclaimer

**SafeTriage-GDM is a research prototype.**

It is not a medical device, diagnostic system, treatment recommendation system, or autonomous clinical decision-making system.

Predictions, uncertainty estimates, fairness metrics, conformal outputs, and explanations should not be used as a substitute for assessment by qualified healthcare professionals.

---

## 👨‍💻 Author

**Ikechukwu Okechi Kamalu**

Research interests include:

- Responsible AI
- Trustworthy Machine Learning
- Explainable AI
- Fairness-Aware Machine Learning
- Causal Inference
- Uncertainty Quantification
- Computational Health
- Psychometric Modelling
- Scientific Machine Learning
- Human-Centered AI

---

## 🌐 Project Links

**Live Application:**  
[SafeTriage-GDM Streamlit App](https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/)

**GitHub Repository:**  
[SafeTriage-GDM GitHub Repository](https://github.com/ikechukwukamalu8/safetriage-gdm-app)

---

## 📄 Citation

If this prototype contributes to your research, please cite the associated project or repository.

```text
Kamalu, I. O. SafeTriage-GDM:
Uncertainty-Quantified Clinical Triage System for Gestational Diabetes
Mellitus Risk with Algorithmic Fairness Auditing and Conformal Safety Bounds.
GitHub repository.
```

---

## 📜 License

This project is provided for research and educational purposes.

Please review the repository license before using, modifying, or redistributing the software or associated datasets.
