# SafeTriage-GDM

**SafeTriage-GDM: Uncertainty-Quantified Clinical Triage System for Gestational Diabetes Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds**

**Status:** Research Prototype

SafeTriage-GDM is a Streamlit research prototype for uncertainty-aware GDM risk triage, conformal prediction, population-shift monitoring, and algorithmic fairness auditing. It is **not a medical device** and must not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

## Live application

https://safetriage-gdm-app-3v5zd7afo2bxevoifgfjmw.streamlit.app/

## Research and temporal-validity principle

The model is framed around a **~28-week GDM prediction landmark**, corresponding to the approximate timing of the 75-g OGTT used to classify GDM in the Cambridge Baby Growth Study (CBGS).

The central leakage-control rule is:

> Only information that would be available at or before the prediction landmark may be used as a predictor.

Whole-pregnancy or post-delivery variables can introduce temporal leakage (lookahead bias). Therefore the deployed predictor schema is deliberately restricted to four features:

1. **MMS started by 28 weeks** — derived from the CBGS supplementation-start week; values at or before week 28 are coded 1 and starts after week 28 are coded 0.
2. **Mother's pre-pregnancy BMI (kg/m2)**
3. **Mother's age (years)**
4. **Parity**

Raw whole-pregnancy MMS exposure, MMS stop timing, total MMS duration, final pregnancy outcomes, birth measurements, gestational age at birth, and similar post-landmark variables are not used as predictors.

For raw CBGS-style uploads, the application automatically derives `MMS started by 28 weeks` from:

`Relative to the start of pregnancy, when did multiple micronutrient supplementation start?`

The application does **not** treat a final whole-pregnancy MMS indicator as equivalent to a pre-28-week exposure.

## Methodology

### Data split

Known GDM outcomes are divided using a stratified:

- **60% training**
- **15% calibration**
- **25% untouched test**

Stratification preserves the observed class proportions; it does not balance the test set.

### Missing-data handling

Numeric variables use an MICE-style `IterativeImputer` pipeline followed by standardization. Categorical variables, if introduced in future compatible schemas, use most-frequent imputation and one-hot encoding.

### Training-only imbalance handling

The training partition uses a **ROSE-style smoothed minority oversampling** procedure. Synthetic minority observations are generated only from the training data. Calibration and test data remain in their natural distributions.

No class weighting, `scale_pos_weight`, SMOTETomek, or post-hoc sigmoid calibration is used.

### Locked model ensemble

Exactly three models are trained, in this order:

1. Random Forest
2. XGBoost
3. Logistic Regression

The ensemble probability is the unweighted mean:

\[
P(GDM)=\frac{P_{RF}+P_{XGB}+P_{LR}}{3}.
\]

### Threshold selection

The classification threshold is selected using the calibration partition only. The test set is not used to optimize the threshold. The selected threshold is applied unchanged to the equal-weight ensemble and, for transparent model comparison, to the three individual models. Individual-model rows therefore represent performance at the common ensemble-selected threshold, not individually optimized thresholds.

The app also reports the number of GDM and No GDM predictions produced on the untouched test set at that frozen threshold. This makes cases such as zero sensitivity directly auditable rather than silently changing the threshold after inspecting test results.

### Conformal prediction

A separate calibration partition is used to construct **90% conformal prediction sets**. A set such as `{GDM, No GDM}` means that both classes remain plausible under the conformal procedure; it is not a diagnosis of two simultaneous conditions.

### Uncertainty quantification

The application reports:

- epistemic uncertainty from disagreement among the three model probabilities;
- aleatoric uncertainty using Bernoulli entropy;
- predictive entropy;
- mutual information.

These are research-oriented uncertainty measures and are not clinical confidence scores.

### Fairness auditing

Age-group auditing uses:

- `<25`
- `25–34`
- `35–44`
- `45+`

The app reports selection rates and false-positive-rate disparity where the required information is available. This is **not** a complete Equalized Odds assessment.

### BMI population stability

BMI Population Stability Index (PSI) is calculated using:

`[-inf, 18.5, 24.9, 29.9, 39.9, inf]`

A PSI of 0.20 is used as the substantial-shift monitoring threshold. PSI is a monitoring statistic and does not change the model, decision threshold, or conformal quantile.

### Explainability

Random Forest and XGBoost use model feature importance; Logistic Regression uses absolute coefficients. Feature importance describes predictive association within the fitted system and is **not causal evidence**.

## External validation

The app includes a separate **External Validation** workflow. The reference/development dataset determines:

- preprocessing;
- training-only ROSE balancing;
- model parameters;
- calibration-derived threshold;
- conformal calibration.

The independent external dataset is then transformed with the frozen reference pipeline. It is **not retrained, rebalanced, threshold-optimized, or conformally recalibrated** on the external data.

External datasets may use different column names through the schema-mapping interface. The MMS feature can be supplied directly as `MMS started by 28 weeks` or derived from a numeric supplementation-start week.

## Synthetic demonstration datasets

The repository includes synthetic datasets for software and methodological testing only. They contain **no real patient records** and must not be interpreted as clinical evidence.

- `datasets/SafeTriage_GDM_external_synthetic_test.xlsx` — 1,100 rows; 1,000 labeled and 100 unlabeled rows.
- `datasets/SafeTriage_GDM_external_renamed_schema_test.xlsx` — same synthetic observations with alternative predictor and outcome names for schema-mapping tests.
- `datasets/SafeTriage_GDM_fairness_psi_stress_test.xlsx` — 1,500 rows; 1,000 labeled and 500 unlabeled rows with deliberately shifted age/BMI distributions for fairness and PSI stress testing.

The synthetic outcomes are generated for testing and are **not clinical truth**.

## Repository structure

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
    └── SafeTriage_GDM_fairness_psi_stress_test.xlsx
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

## Expected requirements

- Python 3.8+
- Streamlit 1.65+
- scikit-learn 1.6.1
- NumPy
- pandas
- XGBoost
- openpyxl
- xlrd >=2.0.1,<3
- Plotly >=5.24,<7

## Data privacy

Do not upload names, medical record numbers, addresses, dates of birth, or other directly identifiable patient information.

The original CBGS individual-level research dataset should **not** be committed to this public repository unless explicit permission to redistribute it exists.

## Important limitations

- This is a research prototype, not a validated clinical decision-support system.
- The CBGS dataset is relatively small and has class imbalance.
- Synthetic oversampling can alter the training distribution and does not create new clinical evidence.
- External validation requires a genuinely independent dataset.
- Fairness results are dataset-dependent and should not be interpreted as proof of fairness in clinical deployment.
- Conformal prediction provides statistical coverage under its assumptions; it does not establish clinical safety.
- The current 28-week feature schema intentionally excludes variables whose final CBGS semantics cannot be demonstrated to be available by the prediction landmark.

## License and data

Code licensing and data redistribution permissions should be specified separately. Do not redistribute restricted CBGS individual-level data without appropriate permission.
