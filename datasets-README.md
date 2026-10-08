# SafeTriage-GDM Datasets

This directory contains the **public CBGS research dataset** used for reproducible model-development demonstrations and several **synthetic datasets** used to test SafeTriage-GDM software functionality.

## 1. Cambridge Baby Growth Study (CBGS) dataset

### `dataCBGS_dataset.xlsx`

This is the publicly deposited Cambridge Baby Growth Study dataset associated with the research publication on multiple micronutrient supplementation during pregnancy and offspring growth.

**Official Cambridge repository record:**  
https://www.repository.cam.ac.uk/items/ca23c466-948b-4981-a415-c74c6ef139dc

**Repository DOI:**  
https://doi.org/10.17863/CAM.54014

**Dataset title:**  
*Data related to "Multiple micronutrient supplementation during pregnancy and increased birth weight and skinfold thicknesses in the offspring: the Cambridge Baby Growth Study"*

**Repository-listed authors:** Clive Petry, Kenneth Ong, Ieuan Hughes, and David Dunger.

The University of Cambridge repository describes the file as an Excel dataset containing data relevant to the associated publication and states that, except where otherwise noted, the item's licence is **Attribution 4.0 International (CC BY 4.0)**.

The CBGS dataset is **third-party research data**. It is not owned or created by the SafeTriage-GDM project.

For the authoritative licence, metadata, provenance, and citation information, consult the original Cambridge repository record.

## 2. Predictor configurations

SafeTriage-GDM supports two explicitly selectable configurations.

### 3-predictor configuration

1. Maternal age
2. Mother's pre-pregnancy BMI
3. Parity

MMS columns may be present in the dataset but are ignored when this configuration is selected.

### 4-predictor configuration

1. Maternal age
2. Mother's pre-pregnancy BMI
3. Parity
4. **Maternal Multiple Micronutrient Supplementation (MMS) started by week 28**

**MMS** means **Multiple Micronutrient Supplementation**.

For the 4-predictor configuration:

```text
1 = MMS started on or before week 28
0 = MMS started after week 28
```

The application can derive this binary variable from a recognized raw MMS start-week field.

## 3. Synthetic software-test datasets

### `SafeTriage_GDM_external_synthetic_test.xlsx`

- 1,100 rows
- 1,000 labelled observations and 100 unlabelled observations
- 3 predictors + GDM outcome
- Intended to test the 3-predictor workflow

### `SafeTriage_GDM_external_renamed_schema_test.xlsx`

- 1,100 rows
- 3-predictor synthetic structure
- Uses an alternative parity field such as `parity_count`
- Intended to test schema mapping and column-name flexibility

### `SafeTriage_GDM_fairness_psi_stress_test.xlsx`

- 1,500 rows
- 1,000 labelled observations and 500 unlabelled observations
- Contains shifted synthetic age/BMI distributions
- Intended to exercise fairness and PSI monitoring

### `SafeTriage_GDM_4predictor_synthetic_test.xlsx`

- 1,100 rows
- 1,000 labelled observations and 100 unlabelled observations
- Contains `MMS_started_by_28_weeks`
- Intended to test the explicitly selected 4-predictor configuration

### `SafeTriage_GDM_4predictor_raw_MMS_start_test.xlsx`

- Synthetic 4-predictor test population
- Contains raw MMS start-week information rather than the binary indicator
- Intended to test derivation of the week-28 MMS indicator

## 4. How to use these datasets

### Build & Evaluate Model

Use the CBGS dataset when you want to reproduce the research model-development workflow.

Use the synthetic datasets to test the application without relying on an additional real-world cohort.

### External Validation

One labelled dataset is sufficient for **Build & Evaluate Model**. The **External Validation** workflow requires two datasets: a reference/development dataset used to develop and freeze the reference model, and a separate independent external dataset used for evaluation.

For genuine external validation, use an independent real-world clinical dataset that was not used for model training, preprocessing fitting, hyperparameter tuning, ensemble-weight selection, decision-threshold selection, or conformal calibration of the reference model.

The synthetic datasets can be uploaded to **test the external-validation software workflow**, but their resulting performance metrics must **not** be described as clinical external validation.

## 5. Important limitations

All synthetic values and outcomes in the synthetic workbooks are artificial.

They do not represent:

- real patients,
- clinical measurements,
- clinical outcomes,
- a representative external population,
- or evidence of clinical model validity.

The CBGS dataset, by contrast, is a publicly deposited third-party research dataset. Its use and redistribution remain subject to the original Cambridge repository licence and terms.

## 6. Data provenance

For the CBGS source, always refer to:

https://www.repository.cam.ac.uk/items/ca23c466-948b-4981-a415-c74c6ef139dc

and:

https://doi.org/10.17863/CAM.54014
