# ==============================================================================
# SafeTriage-GDM
# Uncertainty-Quantified Clinical Triage System for Gestational Diabetes
# Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds
#
# Research Prototype — NOT a medical device
#
# Pipeline:
#   1. Leakage-aware antepartum predictors
#   2. 60/15/25 stratified train/calibration/test split
#   3. MICE-style iterative imputation
#   4. Training-only ROSE-style smoothed minority oversampling
#   5. Random Forest
#   6. XGBoost
#   7. Logistic Regression
#   8. Equal-weight probability ensemble
#   9. Calibration-only threshold selection
#  10. 90% split conformal prediction
#  11. Ensemble uncertainty quantification
#  12. Fairness audit
#  13. BMI population stability monitoring
#  14. Model-based explainability
#
# Author: Ikechukwu Okechi Kamalu
# ==============================================================================

import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ==============================================================================
# APPLICATION CONSTANTS
# ==============================================================================

APP_TITLE = "SafeTriage-GDM"

APP_SUBTITLE = (
    "Uncertainty-Quantified Clinical Triage System for "
    "Gestational Diabetes Mellitus (GDM) Risk with "
    "Algorithmic Fairness Auditing & Conformal Safety Bounds"
)

APP_VERSION = "Research Prototype"

TARGET_COLUMN = "Gestational diabetes?"

BMI_COLUMN = "Mother's pre-pregnancy BMI (kg/m2)"

RANDOM_STATE = 42

CONFORMAL_CONFIDENCE = 0.90

PSI_THRESHOLD = 0.20

EXPECTED_MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
]

EXPECTED_MODEL_KEYS = set(
    EXPECTED_MODEL_ORDER
)

AGE_GROUPS = [
    "<25",
    "25–34",
    "35–44",
    "45+",
]

BMI_BINS = [
    -np.inf,
    18.5,
    24.9,
    29.9,
    39.9,
    np.inf,
]


# ==============================================================================
# LEAKAGE-AWARE ANTEPARTUM PREDICTORS
# ==============================================================================

APPROVED_PREDICTORS = [
    "Evidence of maternal anaemia?",
    "Do we have data related to multiple micronutrient supplementation?",
    "Did the mother supplement with multiple micronutrients during pregnancy?",
    "Relative to the start of pregnancy, when did multiple micronutrient supplementation start?",
    "Relative to the start of pregnancy, when did multiple micronutrient supplementation stop?",
    "Did the mothers just supplement with multiple micronutrients during pregnancy and nothing else?",
    "For how many weeks were multiple micronutrients taken?",
    "Mother's pre-pregnancy BMI (kg/m2)",
    "Mother's height (cm)",
    "Mother's weight before pregnancy (kg)",
    "Mother's age (years)",
    "Did the mother smoke during pregnancy?",
    "Twin pregnancy?",
    "Parity",
]


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# CSS
# ==============================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.35rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        font-size: 1.02rem;
        color: #777;
        margin-bottom: 0.8rem;
    }

    .research-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: #f0f2f6;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# HEADER
# ==============================================================================

st.markdown(
    f'<div class="main-title">🧬 {APP_TITLE}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="subtitle">{APP_SUBTITLE}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<span class="research-badge">{APP_VERSION}</span>',
    unsafe_allow_html=True,
)

st.warning(
    "SafeTriage-GDM is a research prototype for uncertainty-aware GDM risk "
    "triage, conformal safety assessment, population-shift monitoring, and "
    "algorithmic fairness auditing. It is not a medical device and must not "
    "be used as a substitute for professional medical diagnosis, treatment, "
    "or clinical decision-making."
)

st.info(
    "Upload a compatible research dataset in Excel format. "
    "Do not upload names, medical record numbers, addresses, or other "
    "directly identifiable patient information."
)


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:

    st.header("Pipeline")

    st.markdown(
        """
        **Locked architecture**

        1. Leakage-aware predictors
        2. 60/15/25 stratified split
        3. MICE-style imputation
        4. Training-only ROSE-style balancing
        5. Random Forest
        6. XGBoost
        7. Logistic Regression
        8. Equal-weight ensemble
        9. Calibration threshold
        10. 90% conformal prediction
        11. Fairness audit
        12. BMI PSI monitoring
        13. Explainability
        """
    )

    st.divider()

    st.caption(
        "ROSE-style balancing is applied only to the training partition. "
        "Calibration and test data retain their natural class distribution."
    )

    st.caption(
        "No class weighting, SMOTETomek, or post-hoc sigmoid calibration "
        "is used."
    )


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def clean_dataframe(df):
    """Standardize column names and common missing-value representations."""

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    df = df.replace(
        {
            "": np.nan,
            " ": np.nan,
            "NA": np.nan,
            "N/A": np.nan,
            "na": np.nan,
            "n/a": np.nan,
            "NULL": np.nan,
            "null": np.nan,
        }
    )

    return df


def normalize_binary_target(series):
    """Convert GDM target values to 0/1."""

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    for index, value in series.items():

        if pd.isna(value):
            continue

        text = str(value).strip().lower()

        if text in {
            "yes",
            "y",
            "1",
            "true",
            "positive",
        }:
            result.loc[index] = 1.0

        elif text in {
            "no",
            "n",
            "0",
            "false",
            "negative",
        }:
            result.loc[index] = 0.0

    return result


# ==============================================================================
# ROSE-STYLE TRAINING BALANCING
# ==============================================================================

def rose_style_balance(
    X,
    y,
    random_state=42,
    noise_fraction=0.10,
):
    """
    ROSE-style smoothed minority oversampling.

    This is a Python approximation of the R ROSE methodology.
    It is intentionally applied only to the training partition.

    Numeric minority observations receive Gaussian perturbations.
    Categorical minority values are sampled from observed minority values.
    """

    rng = np.random.default_rng(
        random_state
    )

    X = X.copy()
    y = pd.Series(y).copy()

    y_numeric = pd.to_numeric(
        y,
        errors="coerce",
    )

    minority_mask = (
        y_numeric == 1
    )

    majority_mask = (
        y_numeric == 0
    )

    X_minority = X.loc[
        minority_mask
    ].copy()

    X_majority = X.loc[
        majority_mask
    ].copy()

    y_minority = y_numeric.loc[
        minority_mask
    ].copy()

    y_majority = y_numeric.loc[
        majority_mask
    ].copy()

    minority_count = len(
        X_minority
    )

    majority_count = len(
        X_majority
    )

    if (
        minority_count == 0
        or majority_count == 0
    ):

        return (
            X,
            y_numeric,
            {
                "original_minority": minority_count,
                "original_majority": majority_count,
                "synthetic_minority": 0,
                "final_minority": minority_count,
                "final_majority": majority_count,
            },
        )

    synthetic_needed = max(
        0,
        majority_count - minority_count,
    )

    if synthetic_needed == 0:

        return (
            X,
            y_numeric,
            {
                "original_minority": minority_count,
                "original_majority": majority_count,
                "synthetic_minority": 0,
                "final_minority": minority_count,
                "final_majority": majority_count,
            },
        )

    numeric_columns = (
        X_minority
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in X_minority.columns
        if column not in numeric_columns
    ]

    numeric_std = {}

    for column in numeric_columns:

        values = pd.to_numeric(
            X_minority[column],
            errors="coerce",
        )

        std = values.std()

        if pd.isna(std):
            std = 0.0

        numeric_std[column] = float(
            std
        )

    synthetic_rows = []

    for _ in range(
        synthetic_needed
    ):

        base_index = rng.integers(
            0,
            len(X_minority),
        )

        base_row = (
            X_minority
            .iloc[base_index]
            .copy()
        )

        for column in numeric_columns:

            value = pd.to_numeric(
                base_row[column],
                errors="coerce",
            )

            if pd.isna(value):
                continue

            noise_sd = (
                noise_fraction
                * numeric_std[column]
            )

            if noise_sd > 0:

                value = (
                    value
                    + rng.normal(
                        0,
                        noise_sd,
                    )
                )

            base_row[column] = value

        for column in categorical_columns:

            values = (
                X_minority[column]
                .dropna()
            )

            if len(values) > 0:

                base_row[column] = (
                    rng.choice(
                        values.to_numpy()
                    )
                )

        synthetic_rows.append(
            base_row
        )

    X_synthetic = pd.DataFrame(
        synthetic_rows,
        columns=X.columns,
    )

    y_synthetic = pd.Series(
        np.ones(
            len(X_synthetic)
        ),
        name=y.name,
    )

    X_balanced = pd.concat(
        [
            X_majority,
            X_minority,
            X_synthetic,
        ],
        ignore_index=True,
    )

    y_balanced = pd.concat(
        [
            y_majority.reset_index(
                drop=True
            ),
            y_minority.reset_index(
                drop=True
            ),
            y_synthetic.reset_index(
                drop=True
            ),
        ],
        ignore_index=True,
    )

    shuffle_index = rng.permutation(
        len(X_balanced)
    )

    X_balanced = (
        X_balanced
        .iloc[shuffle_index]
        .reset_index(drop=True)
    )

    y_balanced = (
        y_balanced
        .iloc[shuffle_index]
        .reset_index(drop=True)
    )

    information = {
        "original_minority": minority_count,
        "original_majority": majority_count,
        "synthetic_minority": synthetic_needed,
        "final_minority": int(
            (y_balanced == 1).sum()
        ),
        "final_majority": int(
            (y_balanced == 0).sum()
        ),
    }

    return (
        X_balanced,
        y_balanced,
        information,
    )


# ==============================================================================
# PREPROCESSOR
# ==============================================================================

def make_preprocessor(X):

    numeric_columns = (
        X.select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "mice",
                IterativeImputer(
                    max_iter=20,
                    initial_strategy="median",
                    sample_posterior=False,
                    skip_complete=True,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


# ==============================================================================
# MODELS
# ==============================================================================

def create_models():

    return {

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
    }


# ==============================================================================
# TRAINING
# ==============================================================================

def train_models(
    X_train,
    y_train,
    X_calibration,
    X_test,
):

    (
        X_train_balanced,
        y_train_balanced,
        rose_information,
    ) = rose_style_balance(
        X_train,
        y_train,
        random_state=RANDOM_STATE,
        noise_fraction=0.10,
    )

    preprocessor = make_preprocessor(
        X_train_balanced
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train_balanced
        )
    )

    X_calibration_processed = (
        preprocessor.transform(
            X_calibration
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    models = create_models()

    fitted_models = {}
    probabilities = {}

    for model_name in EXPECTED_MODEL_ORDER:

        model = clone(
            models[model_name]
        )

        model.fit(
            X_train_processed,
            y_train_balanced,
        )

        fitted_models[
            model_name
        ] = model

        probabilities[
            model_name
        ] = {
            "calibration": (
                model
                .predict_proba(
                    X_calibration_processed
                )[:, 1]
            ),
            "test": (
                model
                .predict_proba(
                    X_test_processed
                )[:, 1]
            ),
        }

    return (
        preprocessor,
        fitted_models,
        probabilities,
        rose_information,
    )


# ==============================================================================
# ENSEMBLE
# ==============================================================================

def ensemble_probability(
    probabilities
):

    return np.mean(
        np.column_stack(
            [
                probabilities[
                    model_name
                ]
                for model_name
                in EXPECTED_MODEL_ORDER
            ]
        ),
        axis=1,
    )


# ==============================================================================
# THRESHOLD SELECTION
# ==============================================================================

def select_threshold(
    y_true,
    probability,
):

    thresholds = np.linspace(
        0.05,
        0.95,
        181,
    )

    records = []

    y_true = np.asarray(
        y_true
    ).astype(int)

    for threshold in thresholds:

        prediction = (
            probability
            >= threshold
        ).astype(int)

        sensitivity = recall_score(
            y_true,
            prediction,
            zero_division=0,
        )

        specificity = recall_score(
            y_true,
            prediction,
            pos_label=0,
            zero_division=0,
        )

        balanced_accuracy = (
            sensitivity
            + specificity
        ) / 2

        records.append(
            {
                "threshold": threshold,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy":
                    balanced_accuracy,
            }
        )

    table = pd.DataFrame(
        records
    )

    best = table.loc[
        table[
            "balanced_accuracy"
        ].idxmax()
    ]

    return (
        float(
            best["threshold"]
        ),
        table,
    )


# ==============================================================================
# CONFORMAL PREDICTION
# ==============================================================================

def conformal_prediction(
    calibration_probability,
    calibration_y,
    test_probability,
    confidence=0.90,
):

    calibration_probability = np.asarray(
        calibration_probability
    )

    calibration_y = np.asarray(
        calibration_y
    ).astype(int)

    p_true = np.where(
        calibration_y == 1,
        calibration_probability,
        1.0 - calibration_probability,
    )

    scores = 1.0 - p_true

    n = len(scores)

    q_level = min(
        1.0,
        np.ceil(
            (n + 1)
            * confidence
        ) / n,
    )

    q = np.quantile(
        scores,
        q_level,
        method="higher",
    )

    prediction_sets = []

    for probability in np.asarray(
        test_probability
    ):

        labels = []

        if (
            1.0 - probability
            <= q
        ):
            labels.append(
                "GDM"
            )

        if probability <= q:
            labels.append(
                "No GDM"
            )

        if not labels:
            labels = [
                "Uncertain"
            ]

        prediction_sets.append(
            "{"
            + ", ".join(labels)
            + "}"
        )

    return (
        np.asarray(
            prediction_sets
        ),
        float(q),
    )


# ==============================================================================
# UNCERTAINTY
# ==============================================================================

def binary_entropy(probability):

    probability = np.clip(
        probability,
        1e-12,
        1.0 - 1e-12,
    )

    return -(
        probability
        * np.log2(probability)
        +
        (1.0 - probability)
        * np.log2(
            1.0 - probability
        )
    )


def calculate_uncertainty(
    probabilities
):

    matrix = np.column_stack(
        [
            probabilities[
                model_name
            ]
            for model_name
            in EXPECTED_MODEL_ORDER
        ]
    )

    ensemble = matrix.mean(
        axis=1
    )

    epistemic = matrix.std(
        axis=1
    )

    aleatoric = (
        binary_entropy(
            matrix
        )
        .mean(axis=1)
    )

    predictive_entropy = (
        binary_entropy(
            ensemble
        )
    )

    mutual_information = np.maximum(
        predictive_entropy
        - aleatoric,
        0,
    )

    return {
        "epistemic_uncertainty": epistemic,
        "aleatoric_uncertainty": aleatoric,
        "predictive_entropy":
            predictive_entropy,
        "mutual_information":
            mutual_information,
    }


# ==============================================================================
# METRICS
# ==============================================================================

def calculate_metrics(
    y_true,
    probability,
    threshold,
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    prediction = (
        probability
        >= threshold
    ).astype(int)

    return {
        "ROC-AUC": roc_auc_score(
            y_true,
            probability,
        ),
        "PR-AUC": average_precision_score(
            y_true,
            probability,
        ),
        "Accuracy": accuracy_score(
            y_true,
            prediction,
        ),
        "Balanced Accuracy":
            balanced_accuracy_score(
                y_true,
                prediction,
            ),
        "Sensitivity": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "Specificity": recall_score(
            y_true,
            prediction,
            pos_label=0,
            zero_division=0,
        ),
        "Precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "Brier": brier_score_loss(
            y_true,
            probability,
        ),
        "Log Loss": log_loss(
            y_true,
            np.column_stack(
                [
                    1 - probability,
                    probability,
                ]
            ),
            labels=[0, 1],
        ),
    }


# ==============================================================================
# FAIRNESS
# ==============================================================================

def create_age_groups(age):

    age = pd.to_numeric(
        age,
        errors="coerce",
    )

    return pd.cut(
        age,
        bins=[
            -np.inf,
            24.999,
            34.999,
            44.999,
            np.inf,
        ],
        labels=AGE_GROUPS,
    )


def fairness_audit(
    data,
    probability,
    threshold,
):

    if (
        "Mother's age (years)"
        not in data.columns
    ):
        return pd.DataFrame()

    age = pd.to_numeric(
        data[
            "Mother's age (years)"
        ],
        errors="coerce",
    )

    groups = create_age_groups(
        age
    )

    predictions = (
        np.asarray(
            probability
        )
        >= threshold
    ).astype(int)

    target = normalize_binary_target(
        data[TARGET_COLUMN]
    )

    records = []

    for group in AGE_GROUPS:

        group_mask = (
            groups.astype(str)
            == group
        ).to_numpy()

        if group_mask.sum() == 0:
            continue

        selection_rate = (
            predictions[
                group_mask
            ].mean()
        )

        valid_truth = (
            group_mask
            & target.notna().to_numpy()
        )

        y_group = (
            target.to_numpy()[
                valid_truth
            ]
            .astype(int)
        )

        prediction_group = (
            predictions[
                valid_truth
            ]
        )

        if len(y_group) == 0:

            fpr = np.nan

        else:

            negatives = (
                y_group == 0
            )

            if negatives.sum() == 0:

                fpr = np.nan

            else:

                fpr = (
                    (
                        (
                            prediction_group
                            == 1
                        )
                        & negatives
                    ).sum()
                    / negatives.sum()
                )

        records.append(
            {
                "Age group": group,
                "N": int(
                    group_mask.sum()
                ),
                "Selection rate":
                    selection_rate,
                "False-positive rate":
                    fpr,
            }
        )

    return pd.DataFrame(
        records
    )


# ==============================================================================
# PSI
# ==============================================================================

def calculate_psi(
    reference,
    current,
    bins,
):

    reference = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    current = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if (
        len(reference) == 0
        or len(current) == 0
    ):
        return np.nan

    reference_counts = (
        pd.cut(
            reference,
            bins=bins,
            include_lowest=True,
        )
        .value_counts(
            sort=False
        )
    )

    current_counts = (
        pd.cut(
            current,
            bins=bins,
            include_lowest=True,
        )
        .value_counts(
            sort=False
        )
    )

    reference_prop = (
        reference_counts
        / len(reference)
    ).clip(
        lower=1e-6
    )

    current_prop = (
        current_counts
        / len(current)
    ).clip(
        lower=1e-6
    )

    return float(
        (
            (
                current_prop
                - reference_prop
            )
            * np.log(
                current_prop
                / reference_prop
            )
        ).sum()
    )


# ==============================================================================
# FEATURE IMPORTANCE
# ==============================================================================

def aggregate_feature_importance(
    models,
    preprocessor,
    original_columns,
):

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

    except Exception:

        return pd.DataFrame()

    all_importance = []

    for model_name in EXPECTED_MODEL_ORDER:

        model = models[
            model_name
        ]

        if hasattr(
            model,
            "feature_importances_",
        ):

            values = (
                model
                .feature_importances_
            )

        elif hasattr(
            model,
            "coef_",
        ):

            values = np.abs(
                model.coef_[0]
            )

        else:

            continue

        all_importance.append(
            pd.DataFrame(
                {
                    "feature":
                        feature_names,
                    "importance":
                        values,
                    "model":
                        model_name,
                }
            )
        )

    if not all_importance:

        return pd.DataFrame()

    combined = pd.concat(
        all_importance,
        ignore_index=True,
    )

    def original_variable(
        feature
    ):

        if feature.startswith(
            "numeric__"
        ):

            return feature.replace(
                "numeric__",
                "",
                1,
            )

        if feature.startswith(
            "categorical__"
        ):

            clean = feature.replace(
                "categorical__",
                "",
                1,
            )

            matches = [
                column
                for column
                in original_columns
                if clean.startswith(
                    column
                )
            ]

            if matches:

                return max(
                    matches,
                    key=len,
                )

            return clean

        return feature

    combined[
        "original_variable"
    ] = combined[
        "feature"
    ].map(
        original_variable
    )

    return (
        combined
        .groupby(
            "original_variable",
            as_index=False,
        )[
            "importance"
        ]
        .mean()
        .sort_values(
            "importance",
            ascending=False,
        )
    )


# ==============================================================================
# FILE UPLOAD
# ==============================================================================

uploaded_file = st.file_uploader(
    "Upload CBGS-compatible Excel dataset",
    type=[
        "xlsx",
        "xls",
    ],
)

if uploaded_file is None:

    st.markdown(
        """
        ### Getting started

        Upload an Excel dataset containing the required GDM outcome and
        antepartum predictor variables.

        The application will validate the dataset and then execute the
        complete uncertainty-aware modelling pipeline.
        """
    )

    st.stop()


# ==============================================================================
# READ EXCEL
# ==============================================================================

try:

    file_data = uploaded_file.read()

    df = pd.read_excel(
        io.BytesIO(
            file_data
        )
    )

    df = clean_dataframe(
        df
    )

except Exception as exc:

    st.error(
        "The uploaded Excel file could not be read."
    )

    st.exception(
        exc
    )

    st.stop()


# ==============================================================================
# DATA VALIDATION
# ==============================================================================

st.subheader(
    "1. Dataset validation"
)

available_predictors = [
    column
    for column in APPROVED_PREDICTORS
    if column in df.columns
]

validation_columns = st.columns(
    4
)

with validation_columns[0]:

    st.metric(
        "Rows",
        f"{len(df):,}",
    )

with validation_columns[1]:

    st.metric(
        "Columns",
        f"{df.shape[1]:,}",
    )

with validation_columns[2]:

    st.metric(
        "GDM target",
        (
            "Found"
            if TARGET_COLUMN
            in df.columns
            else "Missing"
        ),
    )

with validation_columns[3]:

    st.metric(
        "Predictors",
        f"{len(available_predictors)}/"
        f"{len(APPROVED_PREDICTORS)}",
    )


if TARGET_COLUMN not in df.columns:

    st.error(
        f"Required target `{TARGET_COLUMN}` "
        "was not found."
    )

    st.stop()


missing_predictors = [
    column
    for column in APPROVED_PREDICTORS
    if column not in df.columns
]

if missing_predictors:

    st.error(
        "Required predictor columns are missing:"
    )

    for column in missing_predictors:
        st.write(
            f"- {column}"
        )

    st.stop()


# ==============================================================================
# TARGET
# ==============================================================================

target = normalize_binary_target(
    df[TARGET_COLUMN]
)

labeled_mask = target.notna()

X_labeled = df.loc[
    labeled_mask,
    APPROVED_PREDICTORS,
].copy()

y_labeled = target.loc[
    labeled_mask
].astype(int)

st.write(
    f"Rows with known GDM outcome: "
    f"**{len(y_labeled):,}**"
)

target_counts = (
    y_labeled
    .value_counts()
    .sort_index()
)

target_table = pd.DataFrame(
    {
        "Outcome": [
            "No GDM",
            "GDM",
        ],
        "Count": [
            int(
                target_counts.get(
                    0,
                    0,
                )
            ),
            int(
                target_counts.get(
                    1,
                    0,
                )
            ),
        ],
    }
)

st.dataframe(
    target_table,
    width="stretch",
    hide_index=True,
)

if y_labeled.nunique() < 2:

    st.error(
        "Both GDM classes are required."
    )

    st.stop()


# ==============================================================================
# SPLIT: 60 / 15 / 25
# ==============================================================================

st.subheader(
    "2. Leakage-aware data split"
)

try:

    (
        X_development,
        X_test,
        y_development,
        y_test,
    ) = train_test_split(
        X_labeled,
        y_labeled,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_labeled,
    )

    (
        X_train,
        X_calibration,
        y_train,
        y_calibration,
    ) = train_test_split(
        X_development,
        y_development,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )

except Exception as exc:

    st.error(
        "The stratified split failed."
    )

    st.exception(
        exc
    )

    st.stop()


split_table = pd.DataFrame(
    {
        "Partition": [
            "Training",
            "Calibration",
            "Test",
        ],
        "Rows": [
            len(X_train),
            len(X_calibration),
            len(X_test),
        ],
        "GDM": [
            int(y_train.sum()),
            int(y_calibration.sum()),
            int(y_test.sum()),
        ],
        "No GDM": [
            int((y_train == 0).sum()),
            int((y_calibration == 0).sum()),
            int((y_test == 0).sum()),
        ],
    }
)

st.dataframe(
    split_table,
    width="stretch",
    hide_index=True,
)

st.caption(
    "The test partition remains untouched. "
    "Only the training partition is balanced."
)


# ==============================================================================
# TRAIN MODELS
# ==============================================================================

st.subheader(
    "3. Model training"
)

with st.spinner(
    "Training the three-model ensemble..."
):

    try:

        (
            preprocessor,
            fitted_models,
            model_probabilities,
            rose_information,
        ) = train_models(
            X_train,
            y_train,
            X_calibration,
            X_test,
        )

    except Exception as exc:

        st.error(
            "Model training failed."
        )

        st.exception(
            exc
        )

        st.stop()


# ==============================================================================
# ROSE SUMMARY
# ==============================================================================

st.markdown(
    "### Training-only ROSE balancing"
)

rose_columns = st.columns(
    4
)

with rose_columns[0]:

    st.metric(
        "Original GDM",
        rose_information[
            "original_minority"
        ],
    )

with rose_columns[1]:

    st.metric(
        "Original No GDM",
        rose_information[
            "original_majority"
        ],
    )

with rose_columns[2]:

    st.metric(
        "Synthetic GDM",
        rose_information[
            "synthetic_minority"
        ],
    )

with rose_columns[3]:

    st.metric(
        "Final training rows",
        (
            rose_information[
                "final_minority"
            ]
            +
            rose_information[
                "final_majority"
            ]
        ),
    )


# ==============================================================================
# ENSEMBLE PROBABILITIES
# ==============================================================================

calibration_ensemble_probability = (
    ensemble_probability(
        {
            model_name:
            model_probabilities[
                model_name
            ]["calibration"]
            for model_name
            in EXPECTED_MODEL_ORDER
        }
    )
)

test_ensemble_probability = (
    ensemble_probability(
        {
            model_name:
            model_probabilities[
                model_name
            ]["test"]
            for model_name
            in EXPECTED_MODEL_ORDER
        }
    )
)


# ==============================================================================
# THRESHOLD
# ==============================================================================

threshold, threshold_table = (
    select_threshold(
        y_calibration,
        calibration_ensemble_probability,
    )
)

st.subheader(
    "4. Calibration-derived decision threshold"
)

st.metric(
    "Selected threshold",
    f"{threshold:.3f}",
)

st.caption(
    "The threshold is selected using the calibration partition only. "
    "The test partition is not used to choose the threshold."
)


# ==============================================================================
# TEST PERFORMANCE
# ==============================================================================

test_metrics = calculate_metrics(
    y_test,
    test_ensemble_probability,
    threshold,
)

st.subheader(
    "5. Test-set performance"
)


# IMPORTANT:
# Two-column responsive layout instead of st.columns(5).
# This prevents values such as 0.607 from being displayed as 0....
# on mobile screens.

display_metrics = [
    (
        "ROC-AUC",
        test_metrics["ROC-AUC"],
    ),
    (
        "PR-AUC",
        test_metrics["PR-AUC"],
    ),
    (
        "Sensitivity",
        test_metrics["Sensitivity"],
    ),
    (
        "Specificity",
        test_metrics["Specificity"],
    ),
    (
        "F1",
        test_metrics["F1"],
    ),
]

for row_start in range(
    0,
    len(display_metrics),
    2,
):

    row_metrics = display_metrics[
        row_start:
        row_start + 2
    ]

    columns = st.columns(
        len(row_metrics)
    )

    for column, (
        metric_name,
        metric_value,
    ) in zip(
        columns,
        row_metrics,
    ):

        with column:

            if pd.isna(
                metric_value
            ):

                st.metric(
                    metric_name,
                    "NA",
                )

            else:

                st.metric(
                    metric_name,
                    f"{metric_value:.3f}",
                )


metric_table = pd.DataFrame(
    {
        "Metric":
            list(
                test_metrics.keys()
            ),
        "Value": [
            round(
                value,
                4,
            )
            for value
            in test_metrics.values()
        ],
    }
)

st.dataframe(
    metric_table,
    width="stretch",
    hide_index=True,
)


# ==============================================================================
# INDIVIDUAL MODEL PERFORMANCE
# ==============================================================================

st.markdown(
    "### Individual model performance"
)

individual_records = []

for model_name in EXPECTED_MODEL_ORDER:

    metrics = calculate_metrics(
        y_test,
        model_probabilities[
            model_name
        ]["test"],
        threshold,
    )

    individual_records.append(
        {
            "Model": model_name,
            **metrics,
        }
    )

individual_table = pd.DataFrame(
    individual_records
)

st.dataframe(
    individual_table.round(4),
    width="stretch",
    hide_index=True,
)


# ==============================================================================
# MODEL PERFORMANCE GRAPH
# ==============================================================================

st.markdown(
    "### Model performance comparison"
)

performance_df = (
    individual_table[
        [
            "Model",
            "Accuracy",
            "Balanced Accuracy",
            "Sensitivity",
            "Specificity",
            "Precision",
            "F1",
        ]
    ]
    .melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score",
    )
)

performance_chart = (
    alt.Chart(
        performance_df
    )
    .mark_bar()
    .encode(
        x=alt.X(
            "Model:N",
            title="Model",
            sort=EXPECTED_MODEL_ORDER,
        ),
        y=alt.Y(
            "Score:Q",
            title="Score",
            scale=alt.Scale(
                domain=[0, 1]
            ),
        ),
        color=alt.Color(
            "Metric:N",
            title="Metric",
        ),
        xOffset="Metric:N",
        tooltip=[
            "Model:N",
            "Metric:N",
            alt.Tooltip(
                "Score:Q",
                format=".3f",
            ),
        ],
    )
    .properties(
        height=420,
    )
)

st.altair_chart(
    performance_chart,
    width="stretch",
)


# ==============================================================================
# ROC CURVES
# ==============================================================================

st.markdown(
    "### ROC curves"
)

roc_records = []

roc_probability_sets = {
    "Random Forest":
        model_probabilities[
            "Random Forest"
        ]["test"],

    "XGBoost":
        model_probabilities[
            "XGBoost"
        ]["test"],

    "Logistic Regression":
        model_probabilities[
            "Logistic Regression"
        ]["test"],

    "Ensemble":
        test_ensemble_probability,
}

for model_name, probability in (
    roc_probability_sets.items()
):

    fpr, tpr, _ = roc_curve(
        y_test,
        probability,
    )

    for x_value, y_value in zip(
        fpr,
        tpr,
    ):

        roc_records.append(
            {
                "Model": model_name,
                "False Positive Rate":
                    x_value,
                "True Positive Rate":
                    y_value,
            }
        )

roc_df = pd.DataFrame(
    roc_records
)

roc_chart = (
    alt.Chart(
        roc_df
    )
    .mark_line(
        strokeWidth=3
    )
    .encode(
        x=alt.X(
            "False Positive Rate:Q",
            title="False Positive Rate",
            scale=alt.Scale(
                domain=[0, 1]
            ),
        ),
        y=alt.Y(
            "True Positive Rate:Q",
            title="True Positive Rate",
            scale=alt.Scale(
                domain=[0, 1]
            ),
        ),
        color=alt.Color(
            "Model:N",
            title="Model",
        ),
        tooltip=[
            "Model:N",
            alt.Tooltip(
                "False Positive Rate:Q",
                format=".3f",
            ),
            alt.Tooltip(
                "True Positive Rate:Q",
                format=".3f",
            ),
        ],
    )
    .properties(
        height=450,
    )
)

roc_reference = (
    alt.Chart(
        pd.DataFrame(
            {
                "x": [0, 1],
                "y": [0, 1],
            }
        )
    )
    .mark_line(
        strokeDash=[
            5,
            5,
        ],
        opacity=0.5,
    )
    .encode(
        x="x:Q",
        y="y:Q",
    )
)

st.altair_chart(
    roc_chart
    + roc_reference,
    width="stretch",
)


# ==============================================================================
# PRECISION-RECALL CURVES
# ==============================================================================

st.markdown(
    "### Precision–Recall curves"
)

pr_records = []

for model_name, probability in (
    roc_probability_sets.items()
):

    precision_values, recall_values, _ = (
        precision_recall_curve(
            y_test,
            probability,
        )
    )

    for (
        precision_value,
        recall_value,
    ) in zip(
        precision_values,
        recall_values,
    ):

        pr_records.append(
            {
                "Model": model_name,
                "Recall":
                    recall_value,
                "Precision":
                    precision_value,
            }
        )

pr_df = pd.DataFrame(
    pr_records
)

pr_chart = (
    alt.Chart(
        pr_df
    )
    .mark_line(
        strokeWidth=3
    )
    .encode(
        x=alt.X(
            "Recall:Q",
            title="Recall / Sensitivity",
            scale=alt.Scale(
                domain=[0, 1]
            ),
        ),
        y=alt.Y(
            "Precision:Q",
            title="Precision",
            scale=alt.Scale(
                domain=[0, 1]
            ),
        ),
        color=alt.Color(
            "Model:N",
            title="Model",
        ),
        tooltip=[
            "Model:N",
            alt.Tooltip(
                "Recall:Q",
                format=".3f",
            ),
            alt.Tooltip(
                "Precision:Q",
                format=".3f",
            ),
        ],
    )
    .properties(
        height=450,
    )
)

st.altair_chart(
    pr_chart,
    width="stretch",
)


# ==============================================================================
# THRESHOLD GRAPH
# ==============================================================================

st.markdown(
    "### Threshold sensitivity analysis"
)

threshold_plot_df = (
    threshold_table
    .melt(
        id_vars="threshold",
        value_vars=[
            "sensitivity",
            "specificity",
            "balanced_accuracy",
        ],
        var_name="Metric",
        value_name="Score",
    )
)

threshold_plot_df[
    "Metric"
] = (
    threshold_plot_df[
        "Metric"
    ]
    .str.replace(
        "_",
        " ",
        regex=False,
    )
    .str.title()
)

threshold_chart = (
    alt.Chart(
        threshold_plot_df
    )
    .mark_line(
        strokeWidth=3
    )
    .encode(
        x=alt.X(
            "threshold:Q",
            title="Decision threshold",
            scale=alt.Scale(
                domain=[0, 1]
            ),
        ),
        y=alt.Y(
            "Score:Q",
            title="Score",
            scale=alt.Scale(
                domain=[0, 1]
            ),
        ),
        color=alt.Color(
            "Metric:N",
            title="Metric",
        ),
        tooltip=[
            alt.Tooltip(
                "threshold:Q",
                format=".3f",
            ),
            "Metric:N",
            alt.Tooltip(
                "Score:Q",
                format=".3f",
            ),
        ],
    )
    .properties(
        height=420,
    )
)

threshold_rule = (
    alt.Chart(
        pd.DataFrame(
            {
                "threshold": [
                    threshold
                ]
            }
        )
    )
    .mark_rule(
        strokeDash=[
            6,
            4,
        ]
    )
    .encode(
        x="threshold:Q"
    )
)

st.altair_chart(
    threshold_chart
    + threshold_rule,
    width="stretch",
)


# ==============================================================================
# CONFUSION MATRIX
# ==============================================================================

test_predictions = (
    test_ensemble_probability
    >= threshold
).astype(int)

tn, fp, fn, tp = (
    confusion_matrix(
        y_test,
        test_predictions,
        labels=[
            0,
            1,
        ],
    )
    .ravel()
)

st.markdown(
    "### Confusion matrix"
)

cm_table = pd.DataFrame(
    {
        "": [
            "Actual No GDM",
            "Actual GDM",
        ],
        "Predicted No GDM": [
            tn,
            fn,
        ],
        "Predicted GDM": [
            fp,
            tp,
        ],
    }
)

st.dataframe(
    cm_table,
    width="stretch",
    hide_index=True,
)

cm_plot_df = pd.DataFrame(
    {
        "Actual": [
            "No GDM",
            "No GDM",
            "GDM",
            "GDM",
        ],
        "Predicted": [
            "No GDM",
            "GDM",
            "No GDM",
            "GDM",
        ],
        "Count": [
            tn,
            fp,
            fn,
            tp,
        ],
    }
)

cm_chart = (
    alt.Chart(
        cm_plot_df
    )
    .mark_rect()
    .encode(
        x=alt.X(
            "Predicted:N",
            title="Predicted",
        ),
        y=alt.Y(
            "Actual:N",
            title="Actual",
        ),
        color=alt.Color(
            "Count:Q",
            title="Count",
        ),
        tooltip=[
            "Actual:N",
            "Predicted:N",
            "Count:Q",
        ],
    )
    .properties(
        height=250,
    )
)

st.altair_chart(
    cm_chart,
    width="stretch",
)


# ==============================================================================
# CONFORMAL PREDICTION
# ==============================================================================

st.subheader(
    "6. 90% conformal prediction"
)

(
    conformal_sets,
    conformal_q,
) = conformal_prediction(
    calibration_ensemble_probability,
    y_calibration,
    test_ensemble_probability,
    confidence=CONFORMAL_CONFIDENCE,
)

coverage_values = []

for true_label, prediction_set in zip(
    y_test,
    conformal_sets,
):

    if true_label == 1:

        coverage_values.append(
            "GDM"
            in prediction_set
        )

    else:

        coverage_values.append(
            "No GDM"
            in prediction_set
        )

coverage = np.mean(
    coverage_values
)

conformal_columns = st.columns(
    2
)

with conformal_columns[0]:

    st.metric(
        "Nonconformity quantile",
        f"{conformal_q:.4f}",
    )

with conformal_columns[1]:

    st.metric(
        "Observed test coverage",
        f"{coverage:.1%}",
    )

st.caption(
    "The 90% conformal procedure provides prediction sets under the "
    "exchangeability assumptions of split conformal inference. "
    "It does not establish clinical safety or diagnostic validity."
)


# ==============================================================================
# UNCERTAINTY
# ==============================================================================

st.subheader(
    "7. Ensemble uncertainty"
)

uncertainty = calculate_uncertainty(
    {
        model_name:
        model_probabilities[
            model_name
        ]["test"]
        for model_name
        in EXPECTED_MODEL_ORDER
    }
)

uncertainty_table = pd.DataFrame(
    {
        "Measure": [
            "Epistemic uncertainty",
            "Aleatoric uncertainty",
            "Predictive entropy",
            "Mutual information",
        ],
        "Mean": [
            np.mean(
                uncertainty[
                    "epistemic_uncertainty"
                ]
            ),
            np.mean(
                uncertainty[
                    "aleatoric_uncertainty"
                ]
            ),
            np.mean(
                uncertainty[
                    "predictive_entropy"
                ]
            ),
            np.mean(
                uncertainty[
                    "mutual_information"
                ]
            ),
        ],
    }
)

st.dataframe(
    uncertainty_table.round(4),
    width="stretch",
    hide_index=True,
)

uncertainty_df = pd.DataFrame(
    {
        "Epistemic":
            uncertainty[
                "epistemic_uncertainty"
            ],
        "Aleatoric":
            uncertainty[
                "aleatoric_uncertainty"
            ],
        "Predictive entropy":
            uncertainty[
                "predictive_entropy"
            ],
        "Mutual information":
            uncertainty[
                "mutual_information"
            ],
    }
).melt(
    var_name="Measure",
    value_name="Value",
)

uncertainty_chart = (
    alt.Chart(
        uncertainty_df
    )
    .mark_boxplot()
    .encode(
        x=alt.X(
            "Measure:N",
            title="Uncertainty measure",
        ),
        y=alt.Y(
            "Value:Q",
            title="Value",
        ),
        tooltip=[
            "Measure:N",
            alt.Tooltip(
                "Value:Q",
                format=".4f",
            ),
        ],
    )
    .properties(
        height=420,
    )
)

st.altair_chart(
    uncertainty_chart,
    width="stretch",
)


# ==============================================================================
# FAIRNESS
# ==============================================================================

st.subheader(
    "8. Algorithmic fairness audit"
)

fairness_data = df.loc[
    X_test.index
].copy()

fairness = fairness_audit(
    fairness_data,
    test_ensemble_probability,
    threshold,
)

if fairness.empty:

    st.info(
        "Age information was not available "
        "for the fairness audit."
    )

else:

    st.dataframe(
        fairness.round(4),
        width="stretch",
        hide_index=True,
    )

    valid_fpr = (
        fairness[
            "False-positive rate"
        ]
        .dropna()
    )

    if len(valid_fpr) >= 2:

        fpr_disparity = (
            valid_fpr.max()
            - valid_fpr.min()
        )

        st.metric(
            "Age-group FPR disparity",
            f"{fpr_disparity:.3f}",
        )

    fairness_plot_df = (
        fairness
        .melt(
            id_vars=[
                "Age group",
                "N",
            ],
            value_vars=[
                "Selection rate",
                "False-positive rate",
            ],
            var_name="Metric",
            value_name="Rate",
        )
    )

    fairness_chart = (
        alt.Chart(
            fairness_plot_df
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "Age group:N",
                title="Age group",
                sort=AGE_GROUPS,
            ),
            y=alt.Y(
                "Rate:Q",
                title="Rate",
                scale=alt.Scale(
                    domain=[0, 1]
                ),
            ),
            color=alt.Color(
                "Metric:N",
                title="Metric",
            ),
            xOffset="Metric:N",
            tooltip=[
                "Age group:N",
                "Metric:N",
                alt.Tooltip(
                    "Rate:Q",
                    format=".3f",
                ),
            ],
        )
        .properties(
            height=400,
        )
    )

    st.markdown(
        "### Fairness by age group"
    )

    st.altair_chart(
        fairness_chart,
        width="stretch",
    )

    st.caption(
        "This audit reports age-group selection rates and false-positive "
        "rates. It is not a complete Equalized Odds assessment."
    )


# ==============================================================================
# BMI PSI
# ==============================================================================

st.subheader(
    "9. BMI population-stability monitoring"
)

if BMI_COLUMN in X_train.columns:

    bmi_psi = calculate_psi(
        X_train[
            BMI_COLUMN
        ],
        df[
            BMI_COLUMN
        ],
        BMI_BINS,
    )

    psi_columns = st.columns(
        2
    )

    with psi_columns[0]:

        st.metric(
            "BMI PSI",
            (
                "NA"
                if pd.isna(
                    bmi_psi
                )
                else f"{bmi_psi:.4f}"
            ),
        )

    with psi_columns[1]:

        if pd.isna(
            bmi_psi
        ):

            st.info(
                "BMI PSI could not be calculated."
            )

        elif bmi_psi < 0.10:

            st.success(
                "Minimal population shift detected."
            )

        elif bmi_psi < PSI_THRESHOLD:

            st.warning(
                "Moderate population shift detected."
            )

        else:

            st.error(
                "Substantial population shift detected."
            )

    bmi_reference = pd.to_numeric(
        X_train[
            BMI_COLUMN
        ],
        errors="coerce",
    ).dropna()

    bmi_current = pd.to_numeric(
        df[
            BMI_COLUMN
        ],
        errors="coerce",
    ).dropna()

    bmi_distribution = pd.DataFrame(
        {
            "Population": (
                ["Training"]
                * len(
                    bmi_reference
                )
                +
                ["Uploaded"]
                * len(
                    bmi_current
                )
            ),
            "BMI": (
                bmi_reference.tolist()
                +
                bmi_current.tolist()
            ),
        }
    )

    bmi_chart = (
        alt.Chart(
            bmi_distribution
        )
        .transform_density(
            "BMI",
            as_=[
                "BMI",
                "density",
            ],
            groupby=[
                "Population"
            ],
        )
        .mark_line(
            strokeWidth=3
        )
        .encode(
            x=alt.X(
                "BMI:Q",
                title="Pre-pregnancy BMI",
            ),
            y=alt.Y(
                "density:Q",
                title="Density",
            ),
            color=alt.Color(
                "Population:N",
                title="Population",
            ),
            tooltip=[
                "Population:N",
                alt.Tooltip(
                    "BMI:Q",
                    format=".2f",
                ),
                alt.Tooltip(
                    "density:Q",
                    format=".4f",
                ),
            ],
        )
        .properties(
            height=400,
        )
    )

    st.markdown(
        "### BMI distribution comparison"
    )

    st.altair_chart(
        bmi_chart,
        width="stretch",
    )

    st.caption(
        f"PSI monitoring threshold: {PSI_THRESHOLD:.2f}. "
        "PSI does not modify the conformal threshold."
    )

else:

    st.info(
        "BMI column was not available "
        "for PSI monitoring."
    )


# ==============================================================================
# EXPLAINABILITY
# ==============================================================================

st.subheader(
    "10. Model explainability"
)

feature_importance = (
    aggregate_feature_importance(
        fitted_models,
        preprocessor,
        APPROVED_PREDICTORS,
    )
)

if feature_importance.empty:

    st.info(
        "Feature importance could not be calculated."
    )

else:

    st.dataframe(
        feature_importance
        .head(20)
        .round(5),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "### Top predictive feature weights"
    )

    feature_plot_df = (
        feature_importance
        .head(15)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    feature_chart = (
        alt.Chart(
            feature_plot_df
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "importance:Q",
                title="Mean model importance",
            ),
            y=alt.Y(
                "original_variable:N",
                title="Predictor",
                sort=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "original_variable:N",
                    title="Predictor",
                ),
                alt.Tooltip(
                    "importance:Q",
                    format=".5f",
                    title="Importance",
                ),
            ],
        )
        .properties(
            height=500,
        )
    )

    st.altair_chart(
        feature_chart,
        width="stretch",
    )

    st.caption(
        "Feature importance describes predictive model behavior. "
        "It is not evidence of causal effects."
    )


# ==============================================================================
# DATASET-LEVEL INFERENCE
# ==============================================================================

st.subheader(
    "11. Dataset-level predictions"
)

X_all = df[
    APPROVED_PREDICTORS
].copy()

X_all_processed = (
    preprocessor.transform(
        X_all
    )
)

all_model_probabilities = {}

for model_name in EXPECTED_MODEL_ORDER:

    all_model_probabilities[
        model_name
    ] = (
        fitted_models[
            model_name
        ]
        .predict_proba(
            X_all_processed
        )[:, 1]
    )

all_ensemble_probability = (
    ensemble_probability(
        all_model_probabilities
    )
)

all_uncertainty = (
    calculate_uncertainty(
        all_model_probabilities
    )
)

all_prediction = (
    all_ensemble_probability
    >= threshold
).astype(int)

all_prediction_label = np.where(
    all_prediction == 1,
    "GDM risk",
    "Lower predicted GDM risk",
)

(
    all_conformal_sets,
    _,
) = conformal_prediction(
    calibration_ensemble_probability,
    y_calibration,
    all_ensemble_probability,
    confidence=CONFORMAL_CONFIDENCE,
)

output_df = df.copy()

output_df[
    "SafeTriage_GDM_probability"
] = all_ensemble_probability

output_df[
    "SafeTriage_risk_classification"
] = all_prediction_label

output_df[
    "SafeTriage_decision_threshold"
] = threshold

output_df[
    "SafeTriage_conformal_prediction_set"
] = all_conformal_sets

output_df[
    "SafeTriage_epistemic_uncertainty"
] = (
    all_uncertainty[
        "epistemic_uncertainty"
    ]
)

output_df[
    "SafeTriage_aleatoric_uncertainty"
] = (
    all_uncertainty[
        "aleatoric_uncertainty"
    ]
)

output_df[
    "SafeTriage_predictive_entropy"
] = (
    all_uncertainty[
        "predictive_entropy"
    ]
)

output_df[
    "SafeTriage_mutual_information"
] = (
    all_uncertainty[
        "mutual_information"
    ]
)

for model_name in EXPECTED_MODEL_ORDER:

    safe_name = (
        model_name
        .replace(
            " ",
            "_",
        )
        .replace(
            "-",
            "_",
        )
    )

    output_df[
        f"SafeTriage_{safe_name}_probability"
    ] = (
        all_model_probabilities[
            model_name
        ]
    )


preview_columns = [
    "SafeTriage_GDM_probability",
    "SafeTriage_risk_classification",
    "SafeTriage_conformal_prediction_set",
    "SafeTriage_epistemic_uncertainty",
    "SafeTriage_aleatoric_uncertainty",
    "SafeTriage_predictive_entropy",
    "SafeTriage_mutual_information",
]

st.dataframe(
    output_df[
        preview_columns
    ]
    .head(100)
    .round(4),
    width="stretch",
)


# ==============================================================================
# CSV EXPORT
# ==============================================================================

st.subheader(
    "12. Export results"
)

csv_bytes = (
    output_df
    .to_csv(
        index=False
    )
    .encode(
        "utf-8"
    )
)

st.download_button(
    label="Download prediction results (CSV)",
    data=csv_bytes,
    file_name="safetriage_gdm_predictions.csv",
    mime="text/csv",
)


# ==============================================================================
# METHODOLOGY SUMMARY
# ==============================================================================

st.subheader(
    "Methodology summary"
)

st.markdown(
    """
### Locked research architecture

**Data partitioning**

- 60% training
- 15% calibration
- 25% untouched test

**Missing-data handling**

MICE-style iterative imputation is fitted within the training workflow and
then applied to calibration, test, and inference data.

**Class imbalance**

ROSE-style smoothed minority oversampling is applied **only to the training
partition**.

No class weighting is used.

No SMOTETomek is used.

**Predictive models**

1. Random Forest
2. XGBoost
3. Logistic Regression

The ensemble is:

`P(GDM) = [P_RF(GDM) + P_XGB(GDM) + P_LR(GDM)] / 3`

**Threshold**

The decision threshold is selected from the calibration partition using
balanced accuracy. The test partition is not used to choose the threshold.

**Conformal prediction**

A separate calibration partition is used to construct 90% conformal
prediction sets.

**Uncertainty**

The application reports:

- epistemic uncertainty;
- aleatoric uncertainty;
- predictive entropy;
- mutual information.

**Fairness**

Age-group selection rates and false-positive-rate disparities are monitored.

**Distribution shift**

Maternal pre-pregnancy BMI is monitored using Population Stability Index (PSI).

**Explainability**

Model-based feature importance is aggregated across the three models.

These explainability results describe predictive model behavior and should
not be interpreted as causal effects.
"""
)


# ==============================================================================
# FINAL DISCLAIMER
# ==============================================================================

st.divider()

st.caption(
    "SafeTriage-GDM is a research prototype. It has not been clinically "
    "validated, externally validated, prospectively evaluated, or approved "
    "as a medical device. Model predictions, uncertainty estimates, "
    "conformal prediction sets, fairness metrics, and drift statistics "
    "should be interpreted only within the research context."
)

st.caption(
    "Author: Ikechukwu Okechi Kamalu"
)
