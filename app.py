# ==============================================================================
# SafeTriage-GDM
# Uncertainty-Quantified Clinical Triage System for Gestational Diabetes
# Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds
#
# Research Prototype — NOT a medical device
#
# Methodology:
#   1. 60/15/25 stratified train/calibration/test split
#   2. MICE-style iterative imputation
#   3. Training-only ROSE-style smoothed minority oversampling
#   4. Random Forest + XGBoost + Logistic Regression
#   5. Equal-weight probability ensemble
#   6. Calibration-set threshold selection
#   7. 90% conformal prediction
#   8. Uncertainty quantification
#   9. Fairness auditing
#  10. BMI population-stability monitoring
#  11. Model-based feature importance / explainability
#
# Author: Ikechukwu Okechi Kamalu
# ==============================================================================

import io
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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

RANDOM_STATE = 42

TEST_SIZE = 0.25
CALIBRATION_SIZE_WITHIN_DEVELOPMENT = 0.20

CONFORMAL_CONFIDENCE = 0.90

PSI_THRESHOLD = 0.20

BMI_COLUMN = "Mother's pre-pregnancy BMI (kg/m2)"

# --------------------------------------------------------------------------
# Leakage-aware antepartum predictor schema.
#
# These variables are intentionally restricted to information that can
# reasonably be available before delivery.
# --------------------------------------------------------------------------

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

EXPECTED_MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
]

EXPECTED_MODEL_KEYS = set(EXPECTED_MODEL_ORDER)

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
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# CUSTOM CSS
# ==============================================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.2rem;
    }

    .research-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background-color: #f0f2f6;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .metric-note {
        font-size: 0.85rem;
        color: #666;
    }

    .section-note {
        padding: 0.8rem 1rem;
        background-color: #f7f7f7;
        border-radius: 0.5rem;
        margin: 0.5rem 0 1rem 0;
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
        "ROSE is applied only to the training partition. "
        "Calibration and test data retain their natural class distribution."
    )

    st.caption(
        "No class weighting, SMOTETomek, or post-hoc sigmoid calibration "
        "is used."
    )


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic dataframe cleanup."""

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Convert blank strings to missing values.
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


def normalize_binary_target(series: pd.Series) -> pd.Series:
    """
    Convert the GDM target to:
        0 = No
        1 = Yes
        NaN = unavailable/unknown
    """

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64",
    )

    for idx, value in series.items():

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
            result.loc[idx] = 1.0

        elif text in {
            "no",
            "n",
            "0",
            "false",
            "negative",
        }:
            result.loc[idx] = 0.0

    return result


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric where possible."""

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ==============================================================================
# ROSE-STYLE BALANCING
# ==============================================================================


def rose_style_balance(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    noise_fraction: float = 0.10,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Python implementation of ROSE-style training-only balancing.

    This is intentionally described as ROSE-style rather than an exact
    reimplementation of the R ROSE package.

    Numeric minority observations receive Gaussian perturbations based on
    feature-specific minority-class standard deviations.

    Categorical minority observations are sampled from observed minority
    values.

    The minority class is expanded until approximate class balance is reached.

    IMPORTANT:
        This function must only be applied to the training partition.
    """

    rng = np.random.default_rng(random_state)

    X = X.copy()
    y = pd.Series(y).copy()

    y_numeric = pd.to_numeric(y, errors="coerce")

    minority_label = 1
    majority_label = 0

    minority_mask = y_numeric == minority_label
    majority_mask = y_numeric == majority_label

    X_minority = X.loc[minority_mask].copy()
    X_majority = X.loc[majority_mask].copy()

    y_minority = y_numeric.loc[minority_mask].copy()
    y_majority = y_numeric.loc[majority_mask].copy()

    minority_count = len(X_minority)
    majority_count = len(X_majority)

    if minority_count == 0 or majority_count == 0:
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

    synthetic_rows = []

    numeric_columns = X_minority.select_dtypes(
        include=[np.number]
    ).columns.tolist()

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

        if pd.isna(std) or std == 0:
            std = 0.0

        numeric_std[column] = float(std)

    for _ in range(synthetic_needed):

        base_index = rng.integers(
            0,
            len(X_minority),
        )

        base_row = X_minority.iloc[
            base_index
        ].copy()

        for column in numeric_columns:

            value = pd.to_numeric(
                base_row[column],
                errors="coerce",
            )

            if pd.isna(value):
                continue

            std = numeric_std[column]

            noise_sd = noise_fraction * std

            if noise_sd > 0:
                value = value + rng.normal(
                    loc=0.0,
                    scale=noise_sd,
                )

            base_row[column] = value

        for column in categorical_columns:

            values = X_minority[column].dropna()

            if len(values) > 0:

                base_row[column] = rng.choice(
                    values.to_numpy()
                )

        synthetic_rows.append(base_row)

    X_synthetic = pd.DataFrame(
        synthetic_rows,
        columns=X.columns,
    )

    y_synthetic = pd.Series(
        np.ones(
            len(X_synthetic),
            dtype=float,
        ),
        index=X_synthetic.index,
        name=y.name,
    )

    X_balanced = pd.concat(
        [
            X_majority,
            X_minority,
            X_synthetic,
        ],
        axis=0,
        ignore_index=True,
    )

    y_balanced = pd.concat(
        [
            y_majority.reset_index(drop=True),
            y_minority.reset_index(drop=True),
            y_synthetic.reset_index(drop=True),
        ],
        axis=0,
        ignore_index=True,
    )

    shuffle_indices = rng.permutation(
        len(X_balanced)
    )

    X_balanced = X_balanced.iloc[
        shuffle_indices
    ].reset_index(drop=True)

    y_balanced = y_balanced.iloc[
        shuffle_indices
    ].reset_index(drop=True)

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
# PREPROCESSING
# ==============================================================================


def make_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:

    numeric_columns = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

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
# MODEL DEFINITIONS
# ==============================================================================


def create_models() -> Dict[str, object]:

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
# MODEL TRAINING
# ==============================================================================


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_calibration: pd.DataFrame,
    X_test: pd.DataFrame,
):

    X_train_balanced, y_train_balanced, rose_information = (
        rose_style_balance(
            X_train,
            y_train,
            random_state=RANDOM_STATE,
            noise_fraction=0.10,
        )
    )

    preprocessor = make_preprocessor(
        X_train_balanced
    )

    X_train_processed = preprocessor.fit_transform(
        X_train_balanced
    )

    X_calibration_processed = (
        preprocessor.transform(
            X_calibration
        )
    )

    X_test_processed = preprocessor.transform(
        X_test
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

        fitted_models[model_name] = model

        probabilities[
            model_name
        ] = {
            "calibration": model.predict_proba(
                X_calibration_processed
            )[:, 1],

            "test": model.predict_proba(
                X_test_processed
            )[:, 1],
        }

    return (
        preprocessor,
        fitted_models,
        probabilities,
        rose_information,
        X_train_processed,
        X_calibration_processed,
        X_test_processed,
    )


# ==============================================================================
# ENSEMBLE
# ==============================================================================


def ensemble_probability(
    probabilities: Dict[str, np.ndarray],
) -> np.ndarray:

    return np.mean(
        np.column_stack(
            [
                probabilities[
                    model_name
                ]
                for model_name in EXPECTED_MODEL_ORDER
            ]
        ),
        axis=1,
    )


# ==============================================================================
# THRESHOLD SELECTION
# ==============================================================================


def select_threshold(
    y_true: pd.Series,
    probability: np.ndarray,
) -> Tuple[float, pd.DataFrame]:

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
            probability >= threshold
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
            sensitivity + specificity
        ) / 2

        records.append(
            {
                "threshold": threshold,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
            }
        )

    threshold_table = pd.DataFrame(
        records
    )

    best_row = threshold_table.loc[
        threshold_table[
            "balanced_accuracy"
        ].idxmax()
    ]

    return (
        float(
            best_row["threshold"]
        ),
        threshold_table,
    )


# ==============================================================================
# CONFORMAL PREDICTION
# ==============================================================================


def conformal_prediction(
    calibration_probability: np.ndarray,
    calibration_y: pd.Series,
    test_probability: np.ndarray,
    confidence: float = 0.90,
):
    """
    Split-conformal style prediction sets for binary classification.

    Nonconformity score:
        s_i = 1 - P(Y_i | X_i)

    Prediction set contains a class when its corresponding
    nonconformity score is <= q.
    """

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
            (n + 1) * confidence
        ) / n,
    )

    q = np.quantile(
        scores,
        q_level,
        method="higher",
    )

    test_probability = np.asarray(
        test_probability
    )

    prediction_sets = []

    for probability in test_probability:

        classes = []

        if 1.0 - probability <= q:
            classes.append("GDM")

        if probability <= q:
            classes.append("No GDM")

        if len(classes) == 0:
            classes = ["Uncertain"]

        prediction_sets.append(
            "{" + ", ".join(classes) + "}"
        )

    return (
        np.asarray(prediction_sets),
        float(q),
        scores,
    )


# ==============================================================================
# UNCERTAINTY
# ==============================================================================


def binary_entropy(
    probabilities: np.ndarray,
) -> np.ndarray:

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0 - 1e-12,
    )

    return -(
        probabilities * np.log2(probabilities)
        +
        (1.0 - probabilities)
        * np.log2(1.0 - probabilities)
    )


def calculate_uncertainty(
    model_probabilities: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:

    matrix = np.column_stack(
        [
            model_probabilities[
                model_name
            ]
            for model_name in EXPECTED_MODEL_ORDER
        ]
    )

    predictive_probability = matrix.mean(
        axis=1
    )

    epistemic = matrix.std(
        axis=1,
        ddof=0,
    )

    aleatoric = binary_entropy(
        matrix
    ).mean(
        axis=1
    )

    predictive_entropy = binary_entropy(
        predictive_probability
    )

    mutual_information = np.maximum(
        predictive_entropy - aleatoric,
        0.0,
    )

    return {
        "epistemic_uncertainty": epistemic,
        "aleatoric_uncertainty": aleatoric,
        "predictive_entropy": predictive_entropy,
        "mutual_information": mutual_information,
    }


# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================


def calculate_metrics(
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> Dict[str, float]:

    y_true = np.asarray(
        y_true
    ).astype(int)

    prediction = (
        probability >= threshold
    ).astype(int)

    metrics = {}

    try:
        metrics["ROC-AUC"] = roc_auc_score(
            y_true,
            probability,
        )
    except Exception:
        metrics["ROC-AUC"] = np.nan

    try:
        metrics["PR-AUC"] = average_precision_score(
            y_true,
            probability,
        )
    except Exception:
        metrics["PR-AUC"] = np.nan

    metrics["Accuracy"] = accuracy_score(
        y_true,
        prediction,
    )

    metrics["Balanced Accuracy"] = (
        balanced_accuracy_score(
            y_true,
            prediction,
        )
    )

    metrics["Sensitivity"] = recall_score(
        y_true,
        prediction,
        zero_division=0,
    )

    metrics["Specificity"] = recall_score(
        y_true,
        prediction,
        pos_label=0,
        zero_division=0,
    )

    metrics["Precision"] = precision_score(
        y_true,
        prediction,
        zero_division=0,
    )

    metrics["F1"] = f1_score(
        y_true,
        prediction,
        zero_division=0,
    )

    metrics["Brier"] = brier_score_loss(
        y_true,
        probability,
    )

    metrics["Log Loss"] = log_loss(
        y_true,
        np.column_stack(
            [
                1.0 - probability,
                probability,
            ]
        ),
        labels=[0, 1],
    )

    return metrics


# ==============================================================================
# FAIRNESS
# ==============================================================================


def age_group(
    age: pd.Series,
) -> pd.Series:

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
        right=True,
    )


def fairness_audit(
    df: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
) -> pd.DataFrame:

    if "Mother's age (years)" not in df.columns:
        return pd.DataFrame()

    age = pd.to_numeric(
        df["Mother's age (years)"],
        errors="coerce",
    )

    groups = age_group(age)

    predictions = (
        probability >= threshold
    ).astype(int)

    target = normalize_binary_target(
        df[TARGET_COLUMN]
    )

    records = []

    for group in AGE_GROUPS:

        mask_age = (
            groups.astype(str) == group
        )

        if mask_age.sum() == 0:
            continue

        selection_rate = predictions[
            mask_age
        ].mean()

        ground_truth_mask = (
            mask_age
            & target.notna()
        )

        y_group = target[
            ground_truth_mask
        ].astype(int)

        p_group = predictions[
            ground_truth_mask
        ]

        if len(y_group) > 0:

            fpr = (
                ((p_group == 1) & (y_group == 0)).sum()
                /
                max(
                    1,
                    (y_group == 0).sum(),
                )
            )

        else:
            fpr = np.nan

        records.append(
            {
                "Age group": group,
                "N": int(mask_age.sum()),
                "Selection rate": selection_rate,
                "False-positive rate": fpr,
            }
        )

    return pd.DataFrame(
        records
    )


# ==============================================================================
# POPULATION STABILITY INDEX
# ==============================================================================


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: List[float],
) -> float:

    reference = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    current = pd.to_numeric(
        current,
        errors="coerce",
    ).dropna()

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    reference_counts = pd.cut(
        reference,
        bins=bins,
        include_lowest=True,
    ).value_counts(
        sort=False
    )

    current_counts = pd.cut(
        current,
        bins=bins,
        include_lowest=True,
    ).value_counts(
        sort=False
    )

    reference_proportions = (
        reference_counts / len(reference)
    )

    current_proportions = (
        current_counts / len(current)
    )

    epsilon = 1e-6

    reference_proportions = (
        reference_proportions
        .clip(lower=epsilon)
    )

    current_proportions = (
        current_proportions
        .clip(lower=epsilon)
    )

    psi = (
        (
            current_proportions
            - reference_proportions
        )
        *
        np.log(
            current_proportions
            / reference_proportions
        )
    ).sum()

    return float(
        psi
    )


# ==============================================================================
# EXPLAINABILITY
# ==============================================================================


def aggregate_feature_importance(
    fitted_models: Dict[str, object],
    preprocessor: ColumnTransformer,
    original_columns: List[str],
) -> pd.DataFrame:

    try:
        feature_names = (
            preprocessor.get_feature_names_out()
        )
    except Exception:
        return pd.DataFrame()

    importance_tables = []

    for model_name in EXPECTED_MODEL_ORDER:

        model = fitted_models[
            model_name
        ]

        if hasattr(
            model,
            "feature_importances_",
        ):

            values = model.feature_importances_

        elif hasattr(
            model,
            "coef_",
        ):

            values = np.abs(
                model.coef_[0]
            )

        else:
            continue

        importance_tables.append(
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": values,
                }
            )
        )

    if not importance_tables:
        return pd.DataFrame()

    combined = pd.concat(
        importance_tables,
        ignore_index=True,
    )

    # Map transformed names back to source variables.
    def map_feature(name):

        clean_name = name

        if clean_name.startswith(
            "numeric__"
        ):
            return clean_name.replace(
                "numeric__",
                "",
                1,
            )

        if clean_name.startswith(
            "categorical__"
        ):
            clean_name = clean_name.replace(
                "categorical__",
                "",
                1,
            )

            # Find the longest matching original variable.
            matches = [
                column
                for column in original_columns
                if clean_name.startswith(
                    column
                )
            ]

            if matches:
                return max(
                    matches,
                    key=len,
                )

        return clean_name

    combined["original_variable"] = (
        combined["feature"]
        .map(map_feature)
    )

    result = (
        combined
        .groupby(
            "original_variable",
            as_index=False,
        )["importance"]
        .mean()
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    return result


# ==============================================================================
# VISUALIZATION HELPERS
# ==============================================================================


def style_axis(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.20, linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_outcome_distribution(y):
    counts = pd.Series(y).map({0: "No GDM", 1: "GDM"}).value_counts()
    counts = counts.reindex(["No GDM", "GDM"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values)
    ax.bar_label(bars, fmt="%d", padding=3)
    style_axis(ax, "Known GDM outcome distribution", "Outcome", "Number of observations")
    ax.set_ylim(0, max(counts.values) * 1.15 if len(counts) else 1)
    fig.tight_layout()
    return fig


def plot_split_distribution(y_train, y_calibration, y_test):
    parts = [("Training", y_train), ("Calibration", y_calibration), ("Test", y_test)]
    no_gdm = [int((y == 0).sum()) for _, y in parts]
    gdm = [int((y == 1).sum()) for _, y in parts]
    x = np.arange(len(parts))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - width / 2, no_gdm, width, label="No GDM")
    b2 = ax.bar(x + width / 2, gdm, width, label="GDM")
    ax.bar_label(b1, fmt="%d", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%d", padding=2, fontsize=8)
    ax.set_xticks(x, [name for name, _ in parts])
    style_axis(ax, "Outcome counts by data partition", "Partition", "Number of observations")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_threshold_selection(threshold_table, selected_threshold):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(threshold_table["threshold"], threshold_table["sensitivity"], label="Sensitivity")
    ax.plot(threshold_table["threshold"], threshold_table["specificity"], label="Specificity")
    ax.plot(threshold_table["threshold"], threshold_table["balanced_accuracy"], label="Balanced accuracy", linewidth=2)
    ax.axvline(selected_threshold, linestyle="--", linewidth=1.5, label=f"Selected threshold = {selected_threshold:.3f}")
    style_axis(ax, "Calibration-derived threshold selection", "Decision threshold", "Metric")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_roc_curves(y_true, model_probabilities, ensemble_probability):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in EXPECTED_MODEL_ORDER:
        fpr, tpr, _ = roc_curve(y_true, model_probabilities[model_name])
        auc_value = roc_auc_score(y_true, model_probabilities[model_name])
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")
    fpr, tpr, _ = roc_curve(y_true, ensemble_probability)
    ensemble_auc = roc_auc_score(y_true, ensemble_probability)
    ax.plot(fpr, tpr, linewidth=2.5, label=f"Ensemble (AUC={ensemble_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
    style_axis(ax, "ROC curves on the untouched test set", "False-positive rate", "True-positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_pr_curves(y_true, model_probabilities, ensemble_probability):
    prevalence = np.mean(np.asarray(y_true) == 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in EXPECTED_MODEL_ORDER:
        precision, recall, _ = precision_recall_curve(y_true, model_probabilities[model_name])
        ap = average_precision_score(y_true, model_probabilities[model_name])
        ax.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")
    precision, recall, _ = precision_recall_curve(y_true, ensemble_probability)
    ap = average_precision_score(y_true, ensemble_probability)
    ax.plot(recall, precision, linewidth=2.5, label=f"Ensemble (AP={ap:.3f})")
    ax.axhline(prevalence, linestyle=":", linewidth=1, label=f"Prevalence baseline={prevalence:.3f}")
    style_axis(ax, "Precision–recall curves on the untouched test set", "Recall", "Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, predictions):
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(cm)
    ax.set_xticks([0, 1], ["No GDM", "GDM"])
    ax.set_yticks([0, 1], ["No GDM", "GDM"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Ensemble confusion matrix", fontsize=12, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_probability_distribution(y_true, probability, threshold):
    y_true = np.asarray(y_true).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(probability[y_true == 0], bins=20, alpha=0.60, label="Observed No GDM", density=True)
    ax.hist(probability[y_true == 1], bins=20, alpha=0.60, label="Observed GDM", density=True)
    ax.axvline(threshold, linestyle="--", linewidth=1.5, label=f"Threshold={threshold:.3f}")
    style_axis(ax, "Ensemble predicted GDM probability", "Predicted P(GDM)", "Density")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_conformal_distribution(conformal_sets):
    counts = pd.Series(conformal_sets).value_counts()
    order = ["{No GDM}", "{GDM, No GDM}", "{GDM}", "{Uncertain}"]
    counts = counts.reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index, counts.values)
    ax.bar_label(bars, fmt="%d", padding=3)
    style_axis(ax, "90% conformal prediction sets", "Prediction set", "Number of observations")
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    return fig


def plot_uncertainty_distributions(uncertainty, conformal_sets):
    labels = np.asarray(conformal_sets)
    metrics = [
        ("epistemic_uncertainty", "Epistemic uncertainty"),
        ("aleatoric_uncertainty", "Aleatoric uncertainty"),
        ("predictive_entropy", "Predictive entropy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, title) in zip(axes, metrics):
        for label in ["{No GDM}", "{GDM, No GDM}", "{GDM}"]:
            values = uncertainty[key][labels == label]
            if len(values):
                ax.hist(values, bins=15, alpha=0.45, label=label)
        style_axis(ax, title, "Value", "Count")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def plot_fairness_rates(fairness):
    if fairness.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(fairness))
    width = 0.65
    selection = fairness["Selection rate"]
    b1 = axes[0].bar(x, selection.fillna(0), width)
    axes[0].bar_label(b1, labels=[f"{v:.2f}" if pd.notna(v) else "NA" for v in selection], padding=2, fontsize=8)
    axes[0].set_xticks(x, fairness["Age group"])
    axes[0].set_ylim(0, 1)
    style_axis(axes[0], "Selection rate by age group", "Age group", "Selection rate")
    fpr = fairness["False-positive rate"]
    b2 = axes[1].bar(x, fpr.fillna(0), width)
    axes[1].bar_label(b2, labels=[f"{v:.2f}" if pd.notna(v) else "NA" for v in fpr], padding=2, fontsize=8)
    axes[1].set_xticks(x, fairness["Age group"])
    axes[1].set_ylim(0, 1)
    style_axis(axes[1], "False-positive rate by age group", "Age group", "False-positive rate")
    fig.tight_layout()
    return fig


def plot_bmi_distribution(reference, current):
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if len(reference) == 0 or len(current) == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(reference, bins=20, alpha=0.55, label="Training reference BMI", density=True)
    ax.hist(current, bins=20, alpha=0.55, label="Current uploaded BMI", density=True)
    style_axis(ax, "BMI distribution: training reference vs current data", "Pre-pregnancy BMI (kg/m²)", "Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_importance, top_n=15):
    if feature_importance.empty:
        return None
    data = feature_importance.head(top_n).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(data["original_variable"], data["importance"])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    style_axis(ax, f"Top {min(top_n, len(data))} aggregated model features", "Mean model importance", "")
    fig.tight_layout()
    return fig


# ==============================================================================
# DATA UPLOAD
# ==============================================================================

uploaded_file = st.file_uploader(
    "Upload CBGS-compatible Excel dataset",
    type=["xlsx", "xls"],
)

if uploaded_file is None:

    st.markdown(
        """
        ### Getting started

        Upload an Excel dataset containing the required GDM target and
        antepartum predictor variables.

        The application will then:

        - validate the dataset;
        - identify rows with known GDM status;
        - perform the leakage-aware 60/15/25 split;
        - apply MICE-style imputation;
        - apply ROSE-style balancing **only to training data**;
        - train the three specified models;
        - generate ensemble predictions;
        - determine the decision threshold from calibration data;
        - evaluate the untouched test set;
        - calculate conformal prediction sets;
        - audit age-group fairness;
        - calculate BMI population stability;
        - provide model-based feature importance.
        """
    )

    st.stop()


# ==============================================================================
# READ DATA
# ==============================================================================

try:

    file_bytes = uploaded_file.read()

    df = pd.read_excel(
        io.BytesIO(file_bytes)
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

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        f"{df.shape[0]:,}",
    )

with col2:
    st.metric(
        "Columns",
        f"{df.shape[1]:,}",
    )

with col3:
    st.metric(
        "GDM column",
        "Found"
        if TARGET_COLUMN in df.columns
        else "Missing",
    )

with col4:
    available_predictors = [
        column
        for column in APPROVED_PREDICTORS
        if column in df.columns
    ]

    st.metric(
        "Predictors found",
        f"{len(available_predictors)}/{len(APPROVED_PREDICTORS)}",
    )


if TARGET_COLUMN not in df.columns:

    st.error(
        f"Required target column `{TARGET_COLUMN}` was not found."
    )

    st.stop()


missing_predictors = [
    column
    for column in APPROVED_PREDICTORS
    if column not in df.columns
]

if missing_predictors:

    st.error(
        "Required antepartum predictor columns are missing:"
    )

    st.write(
        missing_predictors
    )

    st.stop()


# ==============================================================================
# TARGET PREPARATION
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
    f"Rows with usable GDM outcome: **{len(y_labeled):,}**"
)

target_counts = (
    y_labeled
    .value_counts()
    .sort_index()
)

target_summary = pd.DataFrame(
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
    target_summary,
    width="stretch",
    hide_index=True,
)

with st.expander("Visualize outcome distribution", expanded=True):
    st.pyplot(plot_outcome_distribution(y_labeled), use_container_width=True)


if y_labeled.nunique() < 2:

    st.error(
        "The dataset does not contain both GDM classes among rows with "
        "known outcomes."
    )

    st.stop()


# ==============================================================================
# DATA SPLITTING
# ==============================================================================

st.subheader(
    "2. Leakage-aware data split"
)

try:

    X_development, X_test, y_development, y_test = (
        train_test_split(
            X_labeled,
            y_labeled,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_labeled,
        )
    )

    X_train, X_calibration, y_train, y_calibration = (
        train_test_split(
            X_development,
            y_development,
            test_size=CALIBRATION_SIZE_WITHIN_DEVELOPMENT,
            random_state=RANDOM_STATE,
            stratify=y_development,
        )
    )

except Exception as exc:

    st.error(
        "The stratified split could not be created."
    )

    st.exception(
        exc
    )

    st.stop()


split_summary = pd.DataFrame(
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
    split_summary,
    width="stretch",
    hide_index=True,
)

st.caption(
    "The test partition remains untouched and naturally imbalanced. "
    "ROSE-style balancing is applied only to training data."
)

with st.expander("Visualize partition class distributions", expanded=False):
    st.pyplot(plot_split_distribution(y_train, y_calibration, y_test), use_container_width=True)


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

st.subheader(
    "3. Model training"
)

with st.spinner(
    "Training Random Forest, XGBoost, and Logistic Regression..."
):

    try:

        (
            preprocessor,
            fitted_models,
            model_probabilities,
            rose_information,
            X_train_processed,
            X_calibration_processed,
            X_test_processed,
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


if set(fitted_models.keys()) != EXPECTED_MODEL_KEYS:

    st.error(
        "The model architecture does not match the required three-model "
        "architecture."
    )

    st.stop()


# ==============================================================================
# ROSE INFORMATION
# ==============================================================================

st.markdown(
    "### Training-only ROSE balancing"
)

rose_col1, rose_col2, rose_col3, rose_col4 = st.columns(4)

with rose_col1:
    st.metric(
        "Original GDM",
        rose_information[
            "original_minority"
        ],
    )

with rose_col2:
    st.metric(
        "Original No GDM",
        rose_information[
            "original_majority"
        ],
    )

with rose_col3:
    st.metric(
        "Synthetic GDM",
        rose_information[
            "synthetic_minority"
        ],
    )

with rose_col4:
    st.metric(
        "Final training rows",
        rose_information[
            "final_minority"
        ]
        + rose_information[
            "final_majority"
        ],
    )

st.caption(
    "ROSE-style synthetic observations are generated exclusively from "
    "the training partition. Calibration and test observations are not "
    "oversampled."
)


# ==============================================================================
# CALIBRATION ENSEMBLE
# ==============================================================================

calibration_ensemble_probability = ensemble_probability(
    {
        model_name: model_probabilities[
            model_name
        ]["calibration"]
        for model_name in EXPECTED_MODEL_ORDER
    }
)

test_ensemble_probability = ensemble_probability(
    {
        model_name: model_probabilities[
            model_name
        ]["test"]
        for model_name in EXPECTED_MODEL_ORDER
    }
)


# ==============================================================================
# THRESHOLD
# ==============================================================================

threshold, threshold_table = select_threshold(
    y_calibration,
    calibration_ensemble_probability,
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

with st.expander("Visualize threshold selection", expanded=True):
    st.pyplot(plot_threshold_selection(threshold_table, threshold), use_container_width=True)


# ==============================================================================
# TEST METRICS
# ==============================================================================

test_metrics = calculate_metrics(
    y_test,
    test_ensemble_probability,
    threshold,
)

st.subheader(
    "5. Test-set performance"
)

metric_columns = st.columns(5)

display_metrics = [
    ("ROC-AUC", test_metrics["ROC-AUC"]),
    ("PR-AUC", test_metrics["PR-AUC"]),
    ("Sensitivity", test_metrics["Sensitivity"]),
    ("Specificity", test_metrics["Specificity"]),
    ("F1", test_metrics["F1"]),
]

for column, (
    metric_name,
    metric_value,
) in zip(
    metric_columns,
    display_metrics,
):

    with column:

        if pd.isna(metric_value):

            st.metric(
                metric_name,
                "NA",
            )

        elif metric_name in {
            "ROC-AUC",
            "PR-AUC",
            "Sensitivity",
            "Specificity",
            "F1",
        }:

            st.metric(
                metric_name,
                f"{metric_value:.3f}",
            )

        else:

            st.metric(
                metric_name,
                f"{metric_value:.3f}",
            )


metric_table = pd.DataFrame(
    {
        "Metric": list(
            test_metrics.keys()
        ),
        "Value": [
            (
                "NA"
                if pd.isna(value)
                else round(
                    value,
                    4,
                )
            )
            for value in test_metrics.values()
        ],
    }
)

st.dataframe(
    metric_table,
    width="stretch",
    hide_index=True,
)


# ==============================================================================
# MODEL-LEVEL PERFORMANCE
# ==============================================================================

st.markdown(
    "### Individual model performance"
)

individual_records = []

for model_name in EXPECTED_MODEL_ORDER:

    model_metrics = calculate_metrics(
        y_test,
        model_probabilities[
            model_name
        ]["test"],
        threshold,
    )

    individual_records.append(
        {
            "Model": model_name,
            **model_metrics,
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

with st.expander("ROC and precision–recall curves", expanded=True):
    curve_probabilities = {
        name: model_probabilities[name]["test"]
        for name in EXPECTED_MODEL_ORDER
    }
    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        st.pyplot(plot_roc_curves(y_test, curve_probabilities, test_ensemble_probability), use_container_width=True)
    with curve_col2:
        st.pyplot(plot_pr_curves(y_test, curve_probabilities, test_ensemble_probability), use_container_width=True)


# ==============================================================================
# CONFUSION MATRIX
# ==============================================================================

test_predictions = (
    test_ensemble_probability >= threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
).ravel()

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

st.pyplot(plot_confusion_matrix(y_test, test_predictions), use_container_width=True)

with st.expander("Predicted probability distribution", expanded=False):
    st.pyplot(plot_probability_distribution(y_test, test_ensemble_probability, threshold), use_container_width=True)


# ==============================================================================
# CONFORMAL PREDICTION
# ==============================================================================

st.subheader(
    "6. 90% conformal prediction"
)

conformal_sets, conformal_q, calibration_scores = (
    conformal_prediction(
        calibration_ensemble_probability,
        y_calibration,
        test_ensemble_probability,
        confidence=CONFORMAL_CONFIDENCE,
    )
)

test_conformal_coverage = np.mean(
    [
        (
            ("GDM" in prediction_set)
            if true_label == 1
            else ("No GDM" in prediction_set)
        )
        for true_label, prediction_set in zip(
            y_test,
            conformal_sets,
        )
    ]
)

st.metric(
    "Conformal nonconformity quantile",
    f"{conformal_q:.4f}",
)

st.metric(
    "Observed test-set coverage",
    f"{test_conformal_coverage:.1%}",
)

st.caption(
    "The 90% conformal procedure provides prediction sets under the "
    "exchangeability assumptions of split conformal inference. It does "
    "not establish clinical safety or diagnostic validity."
)

with st.expander("Visualize conformal prediction sets", expanded=True):
    st.pyplot(plot_conformal_distribution(conformal_sets), use_container_width=True)

with st.expander("How to interpret conformal prediction sets", expanded=True):
    st.markdown(
        """
        **Conformal prediction set:** `{GDM, No GDM}` indicates that both
        outcomes remain plausible at the **90% conformal confidence level**.
        This does **not** mean the patient simultaneously has and does not
        have GDM. Rather, it indicates that the model does not have
        sufficient evidence to exclude either outcome and is therefore
        expressing **prediction uncertainty**.

        **Prediction-set guide:**

        - **`{GDM}`** → GDM is the only outcome retained by the conformal procedure.
        - **`{No GDM}`** → No GDM is the only outcome retained by the conformal procedure.
        - **`{GDM, No GDM}`** → Both outcomes remain plausible; prediction is uncertain.

        **Note:** Conformal prediction is an uncertainty assessment, not a diagnosis.
        """
    )


# ==============================================================================
# UNCERTAINTY
# ==============================================================================

st.subheader(
    "7. Ensemble uncertainty"
)

uncertainty = calculate_uncertainty(
    {
        model_name: model_probabilities[
            model_name
        ]["test"]
        for model_name in EXPECTED_MODEL_ORDER
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

st.caption(
    "Epistemic uncertainty is estimated from disagreement among the "
    "three model probabilities. Aleatoric uncertainty is represented "
    "using Bernoulli entropy. These are research-oriented uncertainty "
    "measures and are not clinical confidence scores."
)

with st.expander("Visualize uncertainty distributions", expanded=True):
    st.pyplot(plot_uncertainty_distributions(uncertainty, conformal_sets), use_container_width=True)


# ==============================================================================
# FAIRNESS
# ==============================================================================

st.subheader(
    "8. Algorithmic fairness audit"
)

fairness = fairness_audit(
    df.loc[
        X_test.index
    ],
    test_ensemble_probability,
    threshold,
)

if fairness.empty:

    st.info(
        "Age information was not available for the fairness audit."
    )

else:

    st.dataframe(
        fairness.round(4),
        width="stretch",
        hide_index=True,
    )

    valid_fpr = fairness[
        "False-positive rate"
    ].dropna()

    if len(valid_fpr) >= 2:

        fpr_disparity = (
            valid_fpr.max()
            - valid_fpr.min()
        )

        st.metric(
            "Age-group FPR disparity",
            f"{fpr_disparity:.3f}",
        )

    st.caption(
        "This audit reports age-group selection rates and false-positive "
        "rates. It is not a complete Equalized Odds assessment."
    )

    with st.expander("Visualize fairness audit", expanded=True):
        fairness_fig = plot_fairness_rates(fairness)
        if fairness_fig is not None:
            st.pyplot(fairness_fig, use_container_width=True)


# ==============================================================================
# BMI PSI
# ==============================================================================

st.subheader(
    "9. BMI population-stability monitoring"
)

if BMI_COLUMN in X_train.columns:

    bmi_psi = calculate_psi(
        X_train[BMI_COLUMN],
        df[BMI_COLUMN],
        BMI_BINS,
    )

    psi_col1, psi_col2 = st.columns(2)

    with psi_col1:

        st.metric(
            "BMI PSI",
            (
                "NA"
                if pd.isna(bmi_psi)
                else f"{bmi_psi:.4f}"
            ),
        )

    with psi_col2:

        if pd.isna(bmi_psi):

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

    st.caption(
        f"PSI threshold used for monitoring: {PSI_THRESHOLD:.2f}. "
        "PSI is a monitoring statistic and does not modify the conformal "
        "prediction threshold."
    )

    with st.expander("Visualize BMI population shift", expanded=True):
        bmi_fig = plot_bmi_distribution(X_train[BMI_COLUMN], df[BMI_COLUMN])
        if bmi_fig is not None:
            st.pyplot(bmi_fig, use_container_width=True)

else:

    st.info(
        "BMI column was not available for PSI monitoring."
    )


# ==============================================================================
# EXPLAINABILITY
# ==============================================================================

st.subheader(
    "10. Model explainability"
)

feature_importance = aggregate_feature_importance(
    fitted_models,
    preprocessor,
    APPROVED_PREDICTORS,
)

if feature_importance.empty:

    st.info(
        "Feature importance could not be calculated."
    )

else:

    st.dataframe(
        feature_importance.head(20).round(5),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Feature importance represents model association/contribution "
        "within the fitted predictive system. It is not causal evidence."
    )

    with st.expander("Visualize model-based feature importance", expanded=True):
        importance_fig = plot_feature_importance(feature_importance, top_n=15)
        if importance_fig is not None:
            st.pyplot(importance_fig, use_container_width=True)


# ==============================================================================
# FULL PREDICTION OUTPUT
# ==============================================================================

st.subheader(
    "11. Dataset-level predictions"
)

# Predict on every uploaded row with the fitted preprocessing/model system.
X_all = df[
    APPROVED_PREDICTORS
].copy()

X_all_processed = preprocessor.transform(
    X_all
)

all_model_probabilities = {}

for model_name in EXPECTED_MODEL_ORDER:

    all_model_probabilities[
        model_name
    ] = fitted_models[
        model_name
    ].predict_proba(
        X_all_processed
    )[:, 1]

all_ensemble_probability = ensemble_probability(
    all_model_probabilities
)

all_uncertainty = calculate_uncertainty(
    all_model_probabilities
)

all_prediction = (
    all_ensemble_probability >= threshold
).astype(int)

all_prediction_label = np.where(
    all_prediction == 1,
    "GDM risk",
    "Lower predicted GDM risk",
)

all_conformal_sets, _, _ = conformal_prediction(
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
] = all_uncertainty[
    "epistemic_uncertainty"
]

output_df[
    "SafeTriage_aleatoric_uncertainty"
] = all_uncertainty[
    "aleatoric_uncertainty"
]

output_df[
    "SafeTriage_predictive_entropy"
] = all_uncertainty[
    "predictive_entropy"
]

output_df[
    "SafeTriage_mutual_information"
] = all_uncertainty[
    "mutual_information"
]

for model_name in EXPECTED_MODEL_ORDER:

    safe_name = (
        model_name
        .replace(" ", "_")
        .replace("-", "_")
    )

    output_df[
        f"SafeTriage_{safe_name}_probability"
    ] = all_model_probabilities[
        model_name
    ]


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
    ].head(100).round(4),
    width="stretch",
)


# ==============================================================================
# DOWNLOAD
# ==============================================================================

st.subheader(
    "12. Export results"
)

csv_bytes = output_df.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    label="Download prediction results (CSV)",
    data=csv_bytes,
    file_name="safetriage_gdm_predictions.csv",
    mime="text/csv",
)


# ==============================================================================
# ==============================================================================
# METHODOLOGY SUMMARY
# ==============================================================================

st.subheader("Methodology summary")

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
