# SafeTriage-GDM

**SafeTriage-GDM: Uncertainty-Quantified Clinical Triage System for Gestational Diabetes Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds**

## Research status
SafeTriage-GDM is a research prototype for uncertainty-aware GDM risk triage, conformal safety assessment, population-shift monitoring, and algorithmic fairness auditing. It is not a medical device and must not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

## Strict 28-week prediction landmark
The application is designed around the approximate 28-week GDM assessment landmark. The predictor schema is intentionally restricted to four variables:

1. Maternal age
2. Maternal pre-pregnancy BMI
3. Parity
4. **MMS started by 28 weeks**

The fourth variable is a landmark-safe transformation of the CBGS supplementation-start timing field. If the uploaded CBGS-format field contains a numeric gestational start week, the app derives `MMS started by 28 weeks = 1` when start week is <= 28 and `0` when it is > 28. The raw whole-pregnancy MMS indicator, MMS stop, total pregnancy MMS duration, and final pregnancy/birth variables are not used as predictors.

## Locked modelling architecture
- Stratified 60/15/25 development/calibration/test split
- MICE-style iterative imputation
- Training-only ROSE-style smoothed minority oversampling
- Random Forest
- XGBoost
- Logistic Regression
- Equal-weight probability ensemble
- Decision threshold selected on the calibration partition only
- 90% conformal prediction using a separate calibration partition
- Ensemble uncertainty estimates
- Age-group fairness audit
- BMI population-stability monitoring (PSI)
- Model-based feature importance / explainability

The test partition is not used for threshold selection, calibration, rebalancing, or model fitting.

## Data privacy
Do not upload names, medical record numbers, addresses, or other directly identifiable patient information. Do not publish the original CBGS participant-level dataset in this repository.

## Synthetic test datasets
The `datasets/` directory contains synthetic Excel files for software testing only. They do not represent clinical observations or clinical ground truth.

- `SafeTriage_GDM_external_synthetic_test.xlsx`: standard canonical schema; 1,100 rows, 1,000 labeled and 100 unlabeled.
- `SafeTriage_GDM_external_renamed_schema_test.xlsx`: same synthetic data with alternative column names for schema-mapping tests.
- `SafeTriage_GDM_fairness_psi_stress_test.xlsx`: synthetic shifted-population data for fairness and BMI-PSI stress testing; 1,500 rows, 1,000 labeled and 500 unlabeled.

## Repository structure
```text
safetriage-gdm-app/
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

## Local execution
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Interpretation
Low discrimination or zero sensitivity at a frozen threshold is not automatically a software error. The app reports the calibration-derived threshold, test performance, and a diagnostic count of positive predictions for each model and the ensemble. The threshold is not adjusted using test-set results.
