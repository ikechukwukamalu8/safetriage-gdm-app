# SafeTriage-GDM Test Datasets

These Excel files are **synthetic test datasets** created for testing the public SafeTriage-GDM application. They are not derived from real patient records and must not be interpreted as clinical validation data.

## Predictor configurations

The application supports two explicitly selectable configurations. Select the intended configuration in the app sidebar before uploading or validating a dataset.

### 3 predictors

1. Maternal age
2. Pre-pregnancy BMI
3. Parity

### 4 predictors

1. Maternal age
2. Pre-pregnancy BMI
3. Parity
4. **Maternal Multiple Micronutrient Supplementation (MMS) started by week 28**

**MMS** means **Multiple Micronutrient Supplementation**.

For the 4-predictor configuration:

```text
1 = MMS started on or before week 28
0 = MMS started after week 28
```

The application can also derive this indicator from a recognized raw MMS start-week field.

## Files

### `SafeTriage_GDM_external_synthetic_test.xlsx`
- 1,100 rows
- 1,000 labeled observations and 100 unlabeled observations
- 3 predictors + GDM outcome

### `SafeTriage_GDM_external_renamed_schema_test.xlsx`
- Same 3-predictor synthetic structure
- Uses `parity_count` instead of `parity`
- Tests schema mapping

### `SafeTriage_GDM_fairness_psi_stress_test.xlsx`
- 1,500 rows
- 1,000 labeled observations and 500 unlabeled observations
- Shifted synthetic age/BMI distributions
- Intended for fairness and PSI monitoring tests

### `SafeTriage_GDM_4predictor_synthetic_test.xlsx`
- 1,100 rows
- 1,000 labeled observations and 100 unlabeled observations
- Includes `MMS_started_by_28_weeks`
- Intended to test the explicitly selected 4-predictor configuration

### `SafeTriage_GDM_4predictor_raw_MMS_start_test.xlsx`
- Same general synthetic population as the 4-predictor test
- Contains a raw MMS start-week field instead of `MMS_started_by_28_weeks`
- Intended to test derivation of the week-28 MMS indicator within the explicitly selected 4-predictor configuration

## Important

All values and outcomes in these workbooks are synthetic. They are provided for software and methodology testing only and do not represent clinical truth, patient records, or external clinical validation.

Do not add the original individual-level CBGS dataset to a public GitHub repository.
