# 🛡️ SafeTriage-GDM: Uncertainty-Quantified Triage System

**SafeTriage-GDM** is a production-grade clinical decision support and triage system designed to predict Gestational Diabetes Mellitus (GDM) risks. It combines an ensemble machine learning architecture (Random Forest, XGBoost, and Logistic Regression) with **Conformal Prediction** (95% safety coverage guarantees), **Unsupervised Domain Adaptation (UDA)** for population drift monitoring, **Algorithmic Fairness Auditing** across demographic cohorts, and **Explainable AI (XAI)** metrics.

---

## 🌟 Key Features

* **Heterogeneous Ensemble Engine:** Aggregates risk probability outputs across Random Forest, XGBoost, and Logistic Regression models.
* **Conformal Uncertainty Bounds:** Generates mathematically calibrated 95% confidence intervals (`Healthy`, `GDM High Risk`, or `Human Audit Required`).
* **Real-Time Drift Telemetry:** Monitors Population Stability Index (PSI) on clinical indicators (such as maternal BMI) to detect out-of-distribution demographic shifts.
* **Autonomous Parameter Sifting:** Automatically recalibrates safety bounds ($q$-threshold) when elevated population drift or high aleatoric entropy is detected.
* **Algorithmic Fairness Auditing:** Dynamically evaluates Demographic Parity Disparity and Equalized Odds (FPR Uniformity) across maternal age groups (<35 vs. ≥35).
* **Explainable AI (XAI):** Displays global biometric feature attributions and patient-specific risk breakdowns.

---

## 📁 Repository Structure

```text
├── app.py                             # Main Streamlit web application & triage engine
├── safetriage_gdm_mixed_pipeline.pkl  # Serialized pipeline payload (Imputer, Scaler, Models, Metadata)
├── requirements.txt                   # Dependency environment specification
└── README.md                          # System documentation

🚀 Quickstart: Local Setup

1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

2. Clone the Repository
git clone [https://github.com/ikechukwukamalu8/safetriage-gdm-app.git](https://github.com/ikechukwukamalu8E/safetriage-gdm-app.git)
cd safetriage-gdm-app

3. Install Dependencies
It is recommended to use a virtual environment:
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
