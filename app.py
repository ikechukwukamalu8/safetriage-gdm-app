# ==============================================================================
# SafeTriage-GDM
# Uncertainty-Quantified Clinical Triage System for Gestational Diabetes
# Mellitus (GDM) Risk with Algorithmic Fairness Auditing &
# Conformal Safety Bounds
#
# Standalone Research Prototype
#
# Required ensemble:
#   1. Random Forest
#   2. XGBoost
#   3. Logistic Regression
#
# Data split:
#   60% Training
#   15% Calibration
#   25% Untouched Test
#
# No synthetic labels are generated.
# ==============================================================================

import os
import io
import pickle
import hashlib
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.exceptions import ConvergenceWarning

from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier


# ------------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ------------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*cv='prefit'.*")

st.set_page_config(
    page_title="SafeTriage-GDM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "SafeTriage-GDM"

APP_SUBTITLE = (
    "Uncertainty-Quantified Clinical Triage System for "
    "Gestational Diabetes Mellitus (GDM) Risk with "
    "Algorithmic Fairness Auditing & Conformal Safety Bounds"
)

APP_VERSION = "Research Prototype"

ARTIFACT_PATH = "safetriage_gdm_mixed_pipeline.pkl"

TARGET_COLUMN = "Gestational diabetes?"
AGE_COLUMN = "Age"
ID_COLUMN = "Dummy Study Number"

RISK_THRESHOLD = 0.50
CONFORMAL_LEVEL = 0.95
DRIFT_LIMIT = 0.20

EXPECTED_MODELS = {
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
}

EXPECTED_MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
]


# ------------------------------------------------------------------------------
# PAGE STYLING
# ------------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.0rem;
    }

    .research-box {
        padding: 1rem;
        border-radius: 0.7rem;
        border: 1px solid rgba(128,128,128,0.25);
        background-color: rgba(128,128,128,0.05);
        margin-bottom: 1rem;
    }

    .metric-note {
        font-size: 0.82rem;
        color: #666;
    }

    .uncertainty-card {
        padding: 0.8rem;
        border-radius: 0.6rem;
        border: 1px solid rgba(128,128,128,0.2);
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------------------

st.markdown(
    f'<div class="main-title">🛡️ {APP_TITLE}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="subtitle">{APP_SUBTITLE}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="research-box">
    <strong>Research prototype:</strong> SafeTriage-GDM is intended for
    uncertainty-aware GDM risk triage, conformal safety assessment,
    population-shift monitoring, and algorithmic fairness auditing.
    It is not a medical device and must not be used as a substitute for
    professional medical diagnosis, treatment, or clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------------------------

def dataset_hash(dataframe: pd.DataFrame) -> str:
    """Generate a deterministic hash for the uploaded dataset."""
    raw = pd.util.hash_pandas_object(
        dataframe,
        index=True,
    ).values.tobytes()

    return hashlib.sha256(raw).hexdigest()


def safe_float(value, default=np.nan) -> float:
    """Convert a value to float safely."""
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass

    return default


def binary_entropy(probabilities):
    """
    Bernoulli entropy in nats.

    H(p) = -p log(p) - (1-p) log(1-p)
    """
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, 1e-12, 1 - 1e-12)

    return -(
        p * np.log(p)
        + (1.0 - p) * np.log(1.0 - p)
    )


def normalize_binary_target(series: pd.Series) -> pd.Series:
    """
    Parse common binary GDM encodings.

    Returns:
        0, 1, or NaN.
    """
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")

        unique = set(
            numeric.dropna().unique().tolist()
        )

        if unique.issubset({0, 1}):
            return numeric.astype(float)

        return numeric.where(
            numeric.isin([0, 1]),
            np.nan,
        )

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    positive = {
        "1",
        "yes",
        "y",
        "true",
        "positive",
        "gdm",
        "diabetes",
        "gestational diabetes",
        "present",
    }

    negative = {
        "0",
        "no",
        "n",
        "false",
        "negative",
        "non-gdm",
        "no gdm",
        "absent",
        "none",
    }

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    result.loc[text.isin(positive)] = 1.0
    result.loc[text.isin(negative)] = 0.0

    return result


def detect_target_column(dataframe: pd.DataFrame) -> Optional[str]:
    """Find the GDM outcome column."""
    if TARGET_COLUMN in dataframe.columns:
        return TARGET_COLUMN

    candidates = [
        "GDM",
        "gdm",
        "GDM Risk",
        "GDM Outcome",
        "GDM outcome",
        "Gestational Diabetes",
        "Gestational diabetes",
        "Gestational diabetes mellitus",
        "Outcome",
        "outcome",
    ]

    lower_map = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]

    return None


def detect_age_column(dataframe: pd.DataFrame) -> Optional[str]:
    """Find the age column."""
    if AGE_COLUMN in dataframe.columns:
        return AGE_COLUMN

    candidates = [
        "age",
        "maternal age",
        "mother age",
        "age_years",
    ]

    lower_map = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]

    return None


# ------------------------------------------------------------------------------
# DATA ENCODING
# ------------------------------------------------------------------------------

def encode_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert categorical variables to one-hot encoded numerical variables.
    """
    encoded = pd.get_dummies(
        dataframe,
        drop_first=True,
        dummy_na=False,
    )

    return encoded.astype(np.float64)


def align_to_schema(
    encoded_dataframe: pd.DataFrame,
    feature_schema: List[str],
) -> pd.DataFrame:
    """
    Force inference data to exactly match training feature schema.
    """
    aligned = encoded_dataframe.reindex(
        columns=feature_schema,
        fill_value=0.0,
    )

    return aligned.astype(np.float64)


# ------------------------------------------------------------------------------
# MODEL CONSTRUCTION
# ------------------------------------------------------------------------------

def create_three_model_ensemble() -> Dict[str, object]:
    """
    Create the mandatory three-model ensemble.

    XGBoost is deliberately NOT optional.
    """

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    models = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "Logistic Regression": logistic_model,
    }

    if set(models.keys()) != EXPECTED_MODELS:
        raise RuntimeError(
            "SafeTriage-GDM requires exactly three models: "
            "Random Forest, XGBoost, and Logistic Regression."
        )

    return models


# ------------------------------------------------------------------------------
# MODEL PROBABILITY
# ------------------------------------------------------------------------------

def get_probability_from_model(
    model,
    X: np.ndarray,
) -> np.ndarray:
    """Return P(GDM=1)."""

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise RuntimeError(
            "A binary probability output was expected."
        )

    return probabilities[:, 1].astype(float)


def ensemble_probabilities(
    calibrated_models: Dict[str, object],
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce probabilities for all three models and their mean.

    Returns:
        model_probabilities: shape (3, n)
        consensus_probability: shape (n,)
    """

    model_probabilities = []

    for model_name in EXPECTED_MODEL_ORDER:

        if model_name not in calibrated_models:
            raise RuntimeError(
                f"Required model '{model_name}' is missing."
            )

        probability = get_probability_from_model(
            calibrated_models[model_name],
            X,
        )

        model_probabilities.append(probability)

    model_probabilities = np.vstack(
        model_probabilities
    )

    if model_probabilities.shape[0] != 3:
        raise RuntimeError(
            "SafeTriage-GDM requires exactly three ensemble "
            "probability streams."
        )

    consensus_probability = np.mean(
        model_probabilities,
        axis=0,
    )

    return (
        model_probabilities,
        consensus_probability,
    )


# ------------------------------------------------------------------------------
# UNCERTAINTY
# ------------------------------------------------------------------------------

def calculate_uncertainty(
    model_probabilities: np.ndarray,
    consensus_probability: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Estimate uncertainty components.

    Epistemic uncertainty:
        Model disagreement, represented primarily by the standard
        deviation of the three model probabilities.

    Aleatoric uncertainty:
        Expected Bernoulli entropy across the three models.

    Predictive entropy:
        Entropy of the ensemble predictive probability.

    Mutual information:
        Predictive entropy - expected entropy.
        This is a useful epistemic-uncertainty measure for an
        ensemble of probabilistic predictors.
    """

    model_probabilities = np.asarray(
        model_probabilities,
        dtype=float,
    )

    consensus_probability = np.asarray(
        consensus_probability,
        dtype=float,
    )

    epistemic_std = np.std(
        model_probabilities,
        axis=0,
        ddof=0,
    )

    model_entropies = binary_entropy(
        model_probabilities
    )

    aleatoric_entropy = np.mean(
        model_entropies,
        axis=0,
    )

    predictive_entropy = binary_entropy(
        consensus_probability
    )

    mutual_information = np.maximum(
        predictive_entropy - aleatoric_entropy,
        0.0,
    )

    return {
        "epistemic_std": epistemic_std,
        "aleatoric_entropy": aleatoric_entropy,
        "predictive_entropy": predictive_entropy,
        "mutual_information": mutual_information,
    }


# ------------------------------------------------------------------------------
# CONFORMAL PREDICTION
# ------------------------------------------------------------------------------

def calculate_conformal_threshold(
    y_cal: np.ndarray,
    calibration_probability: np.ndarray,
    confidence_level: float = CONFORMAL_LEVEL,
) -> float:
    """
    Split-conformal threshold based on calibration nonconformity:

        score_i = 1 - P(correct class | x_i)

    The finite-sample quantile uses the standard conformal
    quantile-index correction.
    """

    y_cal = np.asarray(y_cal).astype(int)
    calibration_probability = np.asarray(
        calibration_probability,
        dtype=float,
    )

    correct_class_probability = np.where(
        y_cal == 1,
        calibration_probability,
        1.0 - calibration_probability,
    )

    scores = 1.0 - correct_class_probability

    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        raise RuntimeError(
            "Unable to calculate conformal threshold: "
            "no valid calibration scores."
        )

    n = len(scores)

    quantile_level = min(
        1.0,
        np.ceil(
            (n + 1) * confidence_level
        ) / n,
    )

    q_threshold = float(
        np.quantile(
            scores,
            quantile_level,
            method="higher",
        )
    )

    return float(
        np.clip(q_threshold, 0.0, 1.0)
    )


def conformal_prediction_sets(
    probability: np.ndarray,
    q_threshold: float,
) -> List[str]:
    """
    Construct binary conformal prediction sets.

    Class 0 is included when:
        1 - P(class 0) <= q

    Class 1 is included when:
        1 - P(class 1) <= q

    Possible sets:
        {Healthy}
        {GDM High Risk}
        {Healthy, GDM High Risk}
    """

    probability = np.asarray(
        probability,
        dtype=float,
    )

    sets = []

    for p in probability:

        include_healthy = (
            1.0 - p
        ) <= q_threshold

        include_gdm = (
            p
        ) <= q_threshold

        if include_healthy and include_gdm:
            label = "Healthy + GDM High Risk"
        elif include_healthy:
            label = "Healthy"
        elif include_gdm:
            label = "GDM High Risk"
        else:
            label = "Healthy + GDM High Risk"

        sets.append(label)

    return sets


def conformal_coverage(
    y_true: np.ndarray,
    probability: np.ndarray,
    q_threshold: float,
) -> float:
    """
    Empirical coverage of conformal prediction sets.
    """

    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)

    include_healthy = (
        1.0 - probability
    ) <= q_threshold

    include_gdm = (
        probability
    ) <= q_threshold

    covered = np.where(
        y_true == 0,
        include_healthy,
        include_gdm,
    )

    return float(
        np.mean(covered)
    )


# ------------------------------------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------------------------------------

def calculate_performance_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float = RISK_THRESHOLD,
) -> Dict:
    """
    Calculate predictive performance on a labeled dataset.
    """

    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(
        probability,
        dtype=float,
    )

    prediction = (
        probability >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    metrics = {
        "ROC-AUC": np.nan,
        "PR-AUC": np.nan,
        "Brier Score": np.nan,
        "Log Loss": np.nan,
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Sensitivity": sensitivity,
        "Recall": sensitivity,
        "Specificity": specificity,
        "F1 Score": np.nan,
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
        "N": len(y_true),
    }

    if len(np.unique(y_true)) == 2:

        metrics["ROC-AUC"] = roc_auc_score(
            y_true,
            probability,
        )

        metrics["PR-AUC"] = average_precision_score(
            y_true,
            probability,
        )

    metrics["Brier Score"] = brier_score_loss(
        y_true,
        probability,
    )

    clipped_probability = np.clip(
        probability,
        1e-7,
        1 - 1e-7,
    )

    metrics["Log Loss"] = log_loss(
        y_true,
        clipped_probability,
        labels=[0, 1],
    )

    metrics["Accuracy"] = accuracy_score(
        y_true,
        prediction,
    )

    metrics["Precision"] = precision_score(
        y_true,
        prediction,
        zero_division=0,
    )

    metrics["F1 Score"] = f1_score(
        y_true,
        prediction,
        zero_division=0,
    )

    return metrics


# ------------------------------------------------------------------------------
# FAIRNESS
# ------------------------------------------------------------------------------

def calculate_fairness_metrics(
    dataframe: pd.DataFrame,
    consensus_probability: np.ndarray,
) -> Dict:
    """
    Fairness analysis.

    Demographic parity:
        computed on all inference rows having valid age.

    FPR disparity:
        computed only on rows with both valid age and valid ground truth.

    This avoids the previous 970-row vs 666-prediction mask mismatch.
    """

    result = {
        "younger_positive_rate": np.nan,
        "older_positive_rate": np.nan,
        "demographic_parity_difference": np.nan,
        "younger_fpr": np.nan,
        "older_fpr": np.nan,
        "fpr_disparity": np.nan,
        "fairness_sample_size": 0,
    }

    if dataframe is None:
        return result

    probability = np.asarray(
        consensus_probability,
        dtype=float,
    )

    if len(dataframe) != len(probability):
        raise ValueError(
            "Fairness calculation alignment error: "
            f"dataframe contains {len(dataframe)} rows, "
            f"but predictions contain {len(probability)} rows."
        )

    age_col = detect_age_column(dataframe)

    if age_col is None:
        return result

    age_series = pd.to_numeric(
        dataframe[age_col],
        errors="coerce",
    )

    positive_flags = (
        probability >= RISK_THRESHOLD
    ).astype(int)

    younger_mask = (
        (age_series < 35)
        & age_series.notna()
    ).to_numpy()

    older_mask = (
        (age_series >= 35)
        & age_series.notna()
    ).to_numpy()

    if younger_mask.sum() > 0:
        result["younger_positive_rate"] = float(
            np.mean(
                positive_flags[younger_mask]
            )
        )

    if older_mask.sum() > 0:
        result["older_positive_rate"] = float(
            np.mean(
                positive_flags[older_mask]
            )
        )

    if (
        not np.isnan(
            result["younger_positive_rate"]
        )
        and not np.isnan(
            result["older_positive_rate"]
        )
    ):
        result[
            "demographic_parity_difference"
        ] = abs(
            result["younger_positive_rate"]
            - result["older_positive_rate"]
        )

    target_col = detect_target_column(
        dataframe
    )

    if target_col is None:
        return result

    y_true = normalize_binary_target(
        dataframe[target_col]
    )

    labeled_mask = (
        y_true.notna()
        & age_series.notna()
    ).to_numpy()

    result["fairness_sample_size"] = int(
        labeled_mask.sum()
    )

    if labeled_mask.sum() == 0:
        return result

    eval_predictions = positive_flags[
        labeled_mask
    ]

    eval_y_true = y_true.to_numpy()[
        labeled_mask
    ].astype(int)

    eval_age = age_series.to_numpy()[
        labeled_mask
    ]

    younger_eval = eval_age < 35
    older_eval = eval_age >= 35

    def calculate_fpr(
        group_y_true,
        group_y_pred,
    ):
        negatives = group_y_true == 0

        if negatives.sum() == 0:
            return np.nan

        false_positives = np.sum(
            group_y_pred[negatives] == 1
        )

        return float(
            false_positives
            / negatives.sum()
        )

    if younger_eval.sum() > 0:
        result["younger_fpr"] = calculate_fpr(
            eval_y_true[younger_eval],
            eval_predictions[younger_eval],
        )

    if older_eval.sum() > 0:
        result["older_fpr"] = calculate_fpr(
            eval_y_true[older_eval],
            eval_predictions[older_eval],
        )

    if (
        not np.isnan(result["younger_fpr"])
        and not np.isnan(result["older_fpr"])
    ):
        result["fpr_disparity"] = abs(
            result["younger_fpr"]
            - result["older_fpr"]
        )

    return result


# ------------------------------------------------------------------------------
# BMI DRIFT / PSI
# ------------------------------------------------------------------------------

BMI_BINS = [
    -np.inf,
    18.5,
    24.9,
    29.9,
    39.9,
    np.inf,
]

BMI_LABELS = [
    "Underweight",
    "Normal",
    "Overweight",
    "Obesity I",
    "Obesity II+",
]


def bmi_distribution(
    dataframe: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    """Calculate BMI-bin distribution."""

    bmi_candidates = [
        "BMI",
        "bmi",
        "Body Mass Index",
        "body mass index",
    ]

    bmi_col = None

    for column in bmi_candidates:
        if column in dataframe.columns:
            bmi_col = column
            break

    if bmi_col is None:
        return None

    bmi = pd.to_numeric(
        dataframe[bmi_col],
        errors="coerce",
    )

    valid = bmi.dropna()

    if len(valid) == 0:
        return None

    categories = pd.cut(
        valid,
        bins=BMI_BINS,
        labels=BMI_LABELS,
        include_lowest=True,
    )

    counts = (
        categories.value_counts(
            sort=False
        )
    )

    proportions = (
        counts / counts.sum()
    )

    return {
        str(label): float(
            proportions.get(label, 0.0)
        )
        for label in BMI_LABELS
    }


def population_stability_index(
    baseline: Dict[str, float],
    current: Dict[str, float],
) -> float:
    """Calculate PSI."""

    epsilon = 1e-6

    categories = sorted(
        set(baseline.keys())
        | set(current.keys())
    )

    psi = 0.0

    for category in categories:

        expected = max(
            baseline.get(category, 0.0),
            epsilon,
        )

        actual = max(
            current.get(category, 0.0),
            epsilon,
        )

        psi += (
            actual - expected
        ) * np.log(
            actual / expected
        )

    return float(psi)


# ------------------------------------------------------------------------------
# XAI
# ------------------------------------------------------------------------------

def calculate_feature_importance(
    models: Dict[str, object],
    feature_schema: List[str],
) -> pd.DataFrame:
    """
    Global feature attribution summary.

    RF and XGBoost provide tree-based feature importances.
    Logistic Regression provides coefficients.

    The final table reports each component separately and an
    aggregate normalized attribution score.
    """

    rows = []

    for index, feature in enumerate(
        feature_schema
    ):

        rf_importance = 0.0
        xgb_importance = 0.0
        logistic_abs = 0.0

        rf_model = models.get(
            "Random Forest"
        )

        xgb_model = models.get(
            "XGBoost"
        )

        lr_model = models.get(
            "Logistic Regression"
        )

        if rf_model is not None:
            try:
                base_rf = (
                    rf_model
                    .estimator
                    if hasattr(
                        rf_model,
                        "estimator",
                    )
                    else rf_model
                )

                if hasattr(
                    base_rf,
                    "feature_importances_",
                ):
                    rf_importance = float(
                        base_rf.feature_importances_[
                            index
                        ]
                    )
            except Exception:
                rf_importance = 0.0

        if xgb_model is not None:
            try:
                base_xgb = (
                    xgb_model
                    .estimator
                    if hasattr(
                        xgb_model,
                        "estimator",
                    )
                    else xgb_model
                )

                if hasattr(
                    base_xgb,
                    "feature_importances_",
                ):
                    xgb_importance = float(
                        base_xgb.feature_importances_[
                            index
                        ]
                    )
            except Exception:
                xgb_importance = 0.0

        if lr_model is not None:
            try:
                if hasattr(
                    lr_model,
                    "estimator",
                ):
                    base_lr = lr_model.estimator
                else:
                    base_lr = lr_model

                if hasattr(
                    base_lr,
                    "coef_",
                ):
                    logistic_abs = abs(
                        float(
                            base_lr.coef_[0][
                                index
                            ]
                        )
                    )
            except Exception:
                logistic_abs = 0.0

        rows.append(
            {
                "Feature": feature,
                "Random Forest": rf_importance,
                "XGBoost": xgb_importance,
                "Logistic Regression |Abs Coefficient|":
                    logistic_abs,
            }
        )

    xai_df = pd.DataFrame(rows)

    if xai_df.empty:
        return xai_df

    for column in [
        "Random Forest",
        "XGBoost",
        "Logistic Regression |Abs Coefficient|",
    ]:
        total = xai_df[column].sum()

        if total > 0:
            xai_df[
                f"{column} Normalized"
            ] = (
                xai_df[column] / total
            )
        else:
            xai_df[
                f"{column} Normalized"
            ] = 0.0

    xai_df[
        "Attribution Weight Score"
    ] = (
        xai_df[
            "Random Forest Normalized"
        ]
        + xai_df[
            "XGBoost Normalized"
        ]
        + xai_df[
            "Logistic Regression |Abs Coefficient| Normalized"
        ]
    ) / 3.0

    return xai_df.sort_values(
        "Attribution Weight Score",
        ascending=False,
    ).reset_index(drop=True)


# ------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------

def train_pipeline(
    raw_dataframe: pd.DataFrame,
) -> Dict:
    """
    Train the complete SafeTriage-GDM pipeline.

    Data:
        60% training
        15% calibration
        25% untouched test

    No information from calibration/test is used to fit the
    preprocessing pipeline.
    """

    target_col = detect_target_column(
        raw_dataframe
    )

    if target_col is None:
        raise ValueError(
            "No GDM outcome column was found. "
            "A labeled dataset containing valid GDM outcomes "
            "is required to train the pipeline."
        )

    working = raw_dataframe.copy()

    working[TARGET_COLUMN] = (
        normalize_binary_target(
            working[target_col]
        )
    )

    labeled_mask = (
        working[TARGET_COLUMN].notna()
    )

    labeled = working.loc[
        labeled_mask
    ].copy()

    if len(labeled) < 20:
        raise ValueError(
            "Too few labeled observations to train "
            "the SafeTriage-GDM pipeline."
        )

    y = labeled[
        TARGET_COLUMN
    ].astype(int).to_numpy()

    if len(np.unique(y)) != 2:
        raise ValueError(
            "The labeled GDM outcome must contain both "
            "classes: 0 and 1."
        )

    X_raw = labeled.drop(
        columns=[
            TARGET_COLUMN,
            target_col,
            ID_COLUMN,
        ],
        errors="ignore",
    )

    # --------------------------------------------------------------------------
    # 60 / 15 / 25 SPLIT
    # --------------------------------------------------------------------------

    X_train_temp, X_test, y_train_temp, y_test = (
        train_test_split(
            X_raw,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
    )

    X_train, X_cal, y_train, y_cal = (
        train_test_split(
            X_train_temp,
            y_train_temp,
            test_size=0.25,
            random_state=42,
            stratify=y_train_temp,
        )
    )

    # --------------------------------------------------------------------------
    # ENCODING
    # --------------------------------------------------------------------------

    X_train_encoded = encode_dataframe(
        X_train
    )

    feature_schema = list(
        X_train_encoded.columns
    )

    X_cal_encoded = align_to_schema(
        encode_dataframe(X_cal),
        feature_schema,
    )

    X_test_encoded = align_to_schema(
        encode_dataframe(X_test),
        feature_schema,
    )

    # --------------------------------------------------------------------------
    # IMPUTATION
    # --------------------------------------------------------------------------

    imputer = IterativeImputer(
        random_state=42,
        max_iter=10,
        initial_strategy="median",
    )

    X_train_imputed = imputer.fit_transform(
        X_train_encoded
    )

    X_cal_imputed = imputer.transform(
        X_cal_encoded
    )

    X_test_imputed = imputer.transform(
        X_test_encoded
    )

    # --------------------------------------------------------------------------
    # SCALING
    # --------------------------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_imputed
    )

    X_cal_scaled = scaler.transform(
        X_cal_imputed
    )

    X_test_scaled = scaler.transform(
        X_test_imputed
    )

    # --------------------------------------------------------------------------
    # SMOTETOMEK ONLY ON TRAINING DATA
    # --------------------------------------------------------------------------

    smote_tomek = SMOTETomek(
        random_state=42
    )

    X_train_balanced, y_train_balanced = (
        smote_tomek.fit_resample(
            X_train_scaled,
            y_train,
        )
    )

    # --------------------------------------------------------------------------
    # THREE REQUIRED MODELS
    # --------------------------------------------------------------------------

    models = create_three_model_ensemble()

    for model_name in EXPECTED_MODEL_ORDER:

        model = models[model_name]

        model.fit(
            X_train_balanced,
            y_train_balanced,
        )

    # --------------------------------------------------------------------------
    # CALIBRATION
    #
    # The base models are trained only on the training partition.
    # The calibration partition is then used to fit probability
    # calibration.
    # --------------------------------------------------------------------------

    calibrated_models = {}

    for model_name in EXPECTED_MODEL_ORDER:

        model = models[model_name]

        calibrated_model = CalibratedClassifierCV(
            model,
            method="sigmoid",
            cv="prefit",
        )

        calibrated_model.fit(
            X_cal_scaled,
            y_cal,
        )

        calibrated_models[
            model_name
        ] = calibrated_model

    # --------------------------------------------------------------------------
    # CALIBRATION PERFORMANCE
    # --------------------------------------------------------------------------

    cal_model_probabilities, cal_probability = (
        ensemble_probabilities(
            calibrated_models,
            X_cal_scaled,
        )
    )

    # --------------------------------------------------------------------------
    # TEST PERFORMANCE
    # --------------------------------------------------------------------------

    test_model_probabilities, test_probability = (
        ensemble_probabilities(
            calibrated_models,
            X_test_scaled,
        )
    )

    performance = calculate_performance_metrics(
        y_test,
        test_probability,
    )

    per_model_performance = {}

    for index, model_name in enumerate(
        EXPECTED_MODEL_ORDER
    ):

        per_model_performance[
            model_name
        ] = calculate_performance_metrics(
            y_test,
            test_model_probabilities[index],
        )

    # --------------------------------------------------------------------------
    # ROC DATA
    # --------------------------------------------------------------------------

    if len(np.unique(y_test)) == 2:

        fpr, tpr, roc_thresholds = roc_curve(
            y_test,
            test_probability,
        )

        precision, recall, pr_thresholds = (
            precision_recall_curve(
                y_test,
                test_probability,
            )
        )

    else:
        fpr = np.array([])
        tpr = np.array([])
        roc_thresholds = np.array([])
        precision = np.array([])
        recall = np.array([])
        pr_thresholds = np.array([])

    # --------------------------------------------------------------------------
    # CALIBRATION CURVE
    # --------------------------------------------------------------------------

    if len(np.unique(y_test)) == 2:

        calibration_fraction, calibration_mean = (
            calibration_curve(
                y_test,
                test_probability,
                n_bins=10,
                strategy="quantile",
            )
        )

    else:
        calibration_fraction = np.array([])
        calibration_mean = np.array([])

    # --------------------------------------------------------------------------
    # CONFORMAL THRESHOLD
    # --------------------------------------------------------------------------

    q_threshold = calculate_conformal_threshold(
        y_cal,
        cal_probability,
        CONFORMAL_LEVEL,
    )

    calibration_coverage = conformal_coverage(
        y_cal,
        cal_probability,
        q_threshold,
    )

    test_coverage = conformal_coverage(
        y_test,
        test_probability,
        q_threshold,
    )

    # --------------------------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------------------------

    calibration_uncertainty = (
        calculate_uncertainty(
            cal_model_probabilities,
            cal_probability,
        )
    )

    test_uncertainty = (
        calculate_uncertainty(
            test_model_probabilities,
            test_probability,
        )
    )

    # --------------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------------------------

    feature_importance = (
        calculate_feature_importance(
            calibrated_models,
            feature_schema,
        )
    )

    # --------------------------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------------------------

    test_predictions = (
        test_probability >= RISK_THRESHOLD
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )

    # --------------------------------------------------------------------------
    # BMI BASELINE
    # --------------------------------------------------------------------------

    baseline_bmi_distribution = (
        bmi_distribution(
            raw_dataframe
        )
    )

    # --------------------------------------------------------------------------
    # ARTIFACT
    # --------------------------------------------------------------------------

    artifact = {
        "artifact_version": "SafeTriage-GDM-2026.10",
        "architecture": {
            "models": EXPECTED_MODEL_ORDER,
            "ensemble": "Unweighted mean of calibrated model probabilities",
            "training_fraction": 0.60,
            "calibration_fraction": 0.15,
            "test_fraction": 0.25,
            "conformal_level": CONFORMAL_LEVEL,
            "risk_threshold": RISK_THRESHOLD,
        },
        "imputer": imputer,
        "scaler": scaler,
        "models": calibrated_models,
        "q_threshold": q_threshold,
        "feature_schema": feature_schema,
        "feature_importance": feature_importance,
        "performance": performance,
        "per_model_performance": per_model_performance,
        "confusion_matrix": cm,
        "roc_data": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresholds,
        },
        "pr_data": {
            "precision": precision,
            "recall": recall,
            "thresholds": pr_thresholds,
        },
        "calibration_data": {
            "fraction_of_positives":
                calibration_fraction,
            "mean_predicted_value":
                calibration_mean,
        },
        "uncertainty_test": test_uncertainty,
        "uncertainty_calibration":
            calibration_uncertainty,
        "conformal": {
            "calibration_coverage":
                calibration_coverage,
            "test_coverage":
                test_coverage,
        },
        "training_rows": len(X_train),
        "calibration_rows": len(X_cal),
        "test_rows": len(X_test),
        "labeled_rows": len(labeled),
        "baseline_bmi_distribution":
            baseline_bmi_distribution,
        "dataset_hash":
            dataset_hash(raw_dataframe),
    }

    with open(
        ARTIFACT_PATH,
        "wb",
    ) as file:
        pickle.dump(
            artifact,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return artifact


# ------------------------------------------------------------------------------
# ARTIFACT VALIDATION
# ------------------------------------------------------------------------------

def validate_artifact(
    artifact: Dict,
) -> bool:
    """
    Strictly validate the saved artifact.

    A legacy or incomplete artifact is rejected rather than
    silently changing the model architecture.
    """

    if not isinstance(
        artifact,
        dict,
    ):
        return False

    if "models" not in artifact:
        return False

    models = artifact["models"]

    if not isinstance(
        models,
        dict,
    ):
        return False

    if set(models.keys()) != EXPECTED_MODELS:
        return False

    if list(models.keys()) != EXPECTED_MODEL_ORDER:
        # Dict order should be preserved by the saved artifact.
        # Reconstructing the dictionary below is safe, but a malformed
        # artifact should still be rejected.
        return False

    required_keys = [
        "imputer",
        "scaler",
        "models",
        "q_threshold",
        "feature_schema",
        "feature_importance",
        "performance",
        "per_model_performance",
        "confusion_matrix",
        "roc_data",
        "pr_data",
        "calibration_data",
        "uncertainty_test",
        "uncertainty_calibration",
        "conformal",
        "training_rows",
        "calibration_rows",
        "test_rows",
        "labeled_rows",
    ]

    for key in required_keys:
        if key not in artifact:
            return False

    if len(
        artifact["feature_schema"]
    ) == 0:
        return False

    return True


def load_artifact() -> Tuple[
    Optional[Dict],
    Optional[str],
]:
    """
    Safely load artifact.

    Returns:
        artifact, error_message
    """

    if not os.path.exists(
        ARTIFACT_PATH
    ):
        return None, None

    try:
        with open(
            ARTIFACT_PATH,
            "rb",
        ) as file:
            artifact = pickle.load(file)

    except Exception as exc:

        return (
            None,
            (
                "The existing SafeTriage-GDM artifact is invalid "
                f"or corrupted: {type(exc).__name__}: {exc}"
            ),
        )

    if not validate_artifact(
        artifact
    ):
        return (
            None,
            (
                "The existing SafeTriage-GDM artifact failed "
                "architecture validation."
            ),
        )

    return artifact, None


# ------------------------------------------------------------------------------
# INFERENCE PREPROCESSING
# ------------------------------------------------------------------------------

def prepare_inference_matrix(
    raw_dataframe: pd.DataFrame,
    artifact: Dict,
) -> np.ndarray:
    """Prepare all uploaded rows for inference."""

    inference_raw = raw_dataframe.drop(
        columns=[
            TARGET_COLUMN,
            ID_COLUMN,
        ],
        errors="ignore",
    ).copy()

    encoded = encode_dataframe(
        inference_raw
    )

    aligned = align_to_schema(
        encoded,
        artifact["feature_schema"],
    )

    imputed = artifact[
        "imputer"
    ].transform(
        aligned
    )

    scaled = artifact[
        "scaler"
    ].transform(
        imputed
    )

    return scaled


# ------------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------------

st.sidebar.header(
    "⚙️ Administrative Control Unit"
)

force_retrain = st.sidebar.button(
    "🔄 Force Pipeline Retraining",
    width="stretch",
)

st.sidebar.markdown(
    "### Architecture"
)

st.sidebar.markdown(
    """
    • Random Forest  
    • XGBoost  
    • Logistic Regression
    """
)

st.sidebar.markdown(
    "### Data split"
)

st.sidebar.markdown(
    """
    • 60% Training  
    • 15% Calibration  
    • 25% Untouched Test
    """
)

st.sidebar.markdown(
    "### Conformal level"
)

st.sidebar.write(
    "95% nominal level"
)

st.sidebar.markdown(
    "### Risk threshold"
)

st.sidebar.write(
    "0.50"
)

st.sidebar.markdown(
    "---"
)

st.sidebar.caption(
    "SafeTriage-GDM is a standalone research prototype."
)


# ------------------------------------------------------------------------------
# DATA INPUT
# ------------------------------------------------------------------------------

st.header("📁 Data Input")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel dataset",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:

    st.info(
        "Upload the dataset to initialise SafeTriage-GDM."
    )

    st.stop()


# ------------------------------------------------------------------------------
# READ DATA
# ------------------------------------------------------------------------------

try:

    file_name = uploaded_file.name.lower()

    if file_name.endswith(
        ".csv"
    ):

        raw_dataframe = pd.read_csv(
            uploaded_file
        )

    else:

        raw_dataframe = pd.read_excel(
            uploaded_file
        )

except Exception as exc:

    st.error(
        f"Unable to read the uploaded dataset: {exc}"
    )

    st.stop()


raw_dataframe = raw_dataframe.reset_index(
    drop=True
)

st.success(
    f"Loaded {len(raw_dataframe):,} rows and "
    f"{len(raw_dataframe.columns):,} columns."
)


# ------------------------------------------------------------------------------
# TARGET PREPARATION
# ------------------------------------------------------------------------------

detected_target = detect_target_column(
    raw_dataframe
)

if detected_target is not None:

    processed_dataframe = (
        raw_dataframe.copy()
    )

    processed_dataframe[
        TARGET_COLUMN
    ] = normalize_binary_target(
        raw_dataframe[
            detected_target
        ]
    )

else:

    processed_dataframe = (
        raw_dataframe.copy()
    )

    processed_dataframe[
        TARGET_COLUMN
    ] = np.nan


labeled_mask = (
    processed_dataframe[
        TARGET_COLUMN
    ].notna()
)

labeled_count = int(
    labeled_mask.sum()
)

unlabeled_count = (
    len(processed_dataframe)
    - labeled_count
)


# ------------------------------------------------------------------------------
# ARTIFACT MANAGEMENT
# ------------------------------------------------------------------------------

artifact, artifact_error = (
    load_artifact()
)

needs_training = (
    force_retrain
    or artifact is None
)

if artifact_error is not None:

    st.warning(
        artifact_error
    )

    st.info(
        "A new SafeTriage-GDM artifact will be "
        "created from the currently uploaded labeled dataset."
    )


if needs_training:

    if labeled_count == 0:

        st.error(
            "Pipeline training requires valid GDM outcomes. "
            "The uploaded dataset contains no valid labeled GDM outcomes."
        )

        st.stop()

    labeled_values = (
        processed_dataframe.loc[
            labeled_mask,
            TARGET_COLUMN,
        ]
        .astype(int)
        .unique()
    )

    if len(labeled_values) != 2:

        st.error(
            "Pipeline training requires both GDM outcome classes "
            "(0 and 1). No synthetic labels are generated."
        )

        st.stop()

    with st.spinner(
        "Training the three-model SafeTriage-GDM pipeline..."
    ):

        try:

            artifact = train_pipeline(
                processed_dataframe
            )

        except Exception as exc:

            st.error(
                "Pipeline training failed."
            )

            st.exception(exc)

            st.stop()

    st.success(
        "SafeTriage-GDM pipeline trained successfully."
    )


# ------------------------------------------------------------------------------
# STRICT ARTIFACT CHECK
# ------------------------------------------------------------------------------

if artifact is None:

    st.error(
        "No valid SafeTriage-GDM artifact is available."
    )

    st.stop()


if not validate_artifact(
    artifact
):

    st.error(
        "The SafeTriage-GDM artifact failed strict "
        "architecture validation."
    )

    st.stop()


# ------------------------------------------------------------------------------
# DISPLAY DATASET SUMMARY
# ------------------------------------------------------------------------------

summary_col1, summary_col2, summary_col3 = (
    st.columns(3)
)

with summary_col1:

    st.metric(
        "Uploaded rows",
        f"{len(raw_dataframe):,}",
    )

with summary_col2:

    st.metric(
        "Labeled rows",
        f"{labeled_count:,}",
    )

with summary_col3:

    st.metric(
        "Unlabeled rows",
        f"{unlabeled_count:,}",
    )


if unlabeled_count > 0:

    st.info(
        f"{unlabeled_count:,} rows have no valid GDM outcome. "
        "They remain in the inference population but are excluded "
        "from supervised performance and ground-truth fairness metrics."
    )


# ------------------------------------------------------------------------------
# INFERENCE OVER ALL UPLOADED ROWS
# ------------------------------------------------------------------------------

try:

    X_inference_scaled = (
        prepare_inference_matrix(
            processed_dataframe,
            artifact,
        )
    )

    (
        inference_model_probabilities,
        consensus_prob,
    ) = ensemble_probabilities(
        artifact["models"],
        X_inference_scaled,
    )

except Exception as exc:

    st.error(
        "Inference failed."
    )

    st.exception(exc)

    st.stop()


# ------------------------------------------------------------------------------
# UNCERTAINTY
# ------------------------------------------------------------------------------

inference_uncertainty = (
    calculate_uncertainty(
        inference_model_probabilities,
        consensus_prob,
    )
)


epistemic_uncertainty = (
    inference_uncertainty[
        "epistemic_std"
    ]
)

aleatoric_uncertainty = (
    inference_uncertainty[
        "aleatoric_entropy"
    ]
)

predictive_entropy = (
    inference_uncertainty[
        "predictive_entropy"
    ]
)

mutual_information = (
    inference_uncertainty[
        "mutual_information"
    ]
)


# ------------------------------------------------------------------------------
# RISK CLASSIFICATION
# ------------------------------------------------------------------------------

risk_flags = (
    consensus_prob >= RISK_THRESHOLD
).astype(int)

risk_labels = np.where(
    risk_flags == 1,
    "High Risk",
    "Lower Risk",
)


# ------------------------------------------------------------------------------
# CONFORMAL SETS
# ------------------------------------------------------------------------------

q_threshold = float(
    artifact["q_threshold"]
)

conformal_sets = (
    conformal_prediction_sets(
        consensus_prob,
        q_threshold,
    )
)


# ------------------------------------------------------------------------------
# RESULTS DASHBOARD
# ------------------------------------------------------------------------------

results_dashboard = pd.DataFrame(
    {
        "GDM Risk Probability":
            consensus_prob,
        "Risk Classification":
            risk_labels,
        "Conformal Prediction Set":
            conformal_sets,
        "Epistemic Uncertainty":
            epistemic_uncertainty,
        "Aleatoric Uncertainty":
            aleatoric_uncertainty,
        "Predictive Entropy":
            predictive_entropy,
        "Mutual Information":
            mutual_information,
        "Model Positive Flag":
            risk_flags,
    }
)


final_output_view = pd.concat(
    [
        raw_dataframe.reset_index(
            drop=True
        ),
        results_dashboard.reset_index(
            drop=True
        ),
    ],
    axis=1,
)


# ------------------------------------------------------------------------------
# FAIRNESS
# ------------------------------------------------------------------------------

try:

    fairness = (
        calculate_fairness_metrics(
            processed_dataframe,
            consensus_prob,
        )
    )

except Exception as exc:

    fairness = {
        "younger_positive_rate": np.nan,
        "older_positive_rate": np.nan,
        "demographic_parity_difference":
            np.nan,
        "younger_fpr": np.nan,
        "older_fpr": np.nan,
        "fpr_disparity": np.nan,
        "fairness_sample_size": 0,
    }

    st.warning(
        f"Fairness calculation was unavailable: {exc}"
    )


# ------------------------------------------------------------------------------
# DRIFT
# ------------------------------------------------------------------------------

baseline_bmi = artifact.get(
    "baseline_bmi_distribution"
)

current_bmi = bmi_distribution(
    raw_dataframe
)

if (
    baseline_bmi is not None
    and current_bmi is not None
):

    psi_value = (
        population_stability_index(
            baseline_bmi,
            current_bmi,
        )
    )

else:

    psi_value = np.nan


# ------------------------------------------------------------------------------
# OPTIONAL CONFORMAL RUNTIME HEURISTIC
# ------------------------------------------------------------------------------

runtime_q = q_threshold

mean_predictive_entropy = float(
    np.mean(
        predictive_entropy
    )
)

if (
    np.isfinite(psi_value)
    and psi_value >= DRIFT_LIMIT
    and mean_predictive_entropy >= 0.70
):

    runtime_q = float(
        np.clip(
            q_threshold * 0.90,
            0.0,
            1.0,
        )
    )

    runtime_conformal_sets = (
        conformal_prediction_sets(
            consensus_prob,
            runtime_q,
        )
    )

else:

    runtime_conformal_sets = (
        conformal_sets
    )


# ------------------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------------------

(
    triage_tab,
    performance_tab,
    uncertainty_tab,
    fairness_tab,
    xai_tab,
) = st.tabs(
    [
        "🩺 Patient Triage Console",
        "📈 Predictive Performance",
        "🎲 Uncertainty & Conformal Safety",
        "⚖️ Fairness & Population Shift",
        "🔍 XAI Attribution",
    ]
)


# ==============================================================================
# TRIAGE TAB
# ==============================================================================

with triage_tab:

    st.subheader(
        "Patient Triage Console"
    )

    st.markdown(
        """
        SafeTriage-GDM provides an estimated probability of GDM risk
        together with a conformal prediction set representing predictive
        uncertainty.

        **Risk estimation** — the ensemble probability of GDM.

        **Triage classification** — the operational lower-risk/high-risk
        classification at the 0.50 threshold.

        **Conformal safety bounds** — the uncertainty-aware prediction set
        used to expose ambiguous cases.
        """
    )

    risk_col1, risk_col2, risk_col3, risk_col4 = (
        st.columns(4)
    )

    with risk_col1:

        st.metric(
            "Mean GDM Risk",
            f"{np.mean(consensus_prob):.2%}",
        )

    with risk_col2:

        st.metric(
            "High-Risk Flags",
            f"{int(risk_flags.sum()):,}",
        )

    with risk_col3:

        st.metric(
            "Mean Epistemic",
            f"{np.mean(epistemic_uncertainty):.4f}",
        )

    with risk_col4:

        st.metric(
            "Mean Aleatoric",
            f"{np.mean(aleatoric_uncertainty):.4f}",
        )

    st.markdown(
        "### Inference Results"
    )

    display_df = final_output_view.copy()

    if "GDM Risk Probability" in display_df.columns:

        display_df[
            "GDM Risk Probability"
        ] = display_df[
            "GDM Risk Probability"
        ].map(
            lambda x: f"{x:.2%}"
            if pd.notna(x)
            else ""
        )

    for column in [
        "Epistemic Uncertainty",
        "Aleatoric Uncertainty",
        "Predictive Entropy",
        "Mutual Information",
    ]:

        if column in display_df.columns:

            display_df[column] = (
                display_df[column].map(
                    lambda x: f"{x:.4f}"
                    if pd.notna(x)
                    else ""
                )
            )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "### Risk Probability Distribution"
    )

    histogram_data = pd.DataFrame(
        {
            "GDM Risk Probability":
                consensus_prob
        }
    )

    st.bar_chart(
        histogram_data,
        x=None,
        y="GDM Risk Probability",
        width="stretch",
    )

    st.markdown(
        "### Conformal Safety Interpretation"
    )

    st.info(
        f"""
        The fitted conformal threshold is
        **q = {q_threshold:.4f}** at the nominal
        **{CONFORMAL_LEVEL:.0%}** level.

        Cases receiving **Healthy + GDM High Risk** represent
        uncertainty/ambiguity under the conformal rule and should
        not be treated as confidently classified cases.

        Conformal prediction is an uncertainty quantification mechanism;
        it does not constitute a clinical diagnosis or treatment
        recommendation.
        """
    )


# ==============================================================================
# PERFORMANCE TAB
# ==============================================================================

with performance_tab:

    st.subheader(
        "📈 Predictive Performance"
    )

    st.markdown(
        """
        These performance measures describe the behaviour of the trained
        SafeTriage-GDM model on its **untouched held-out test set**.

        They are therefore distinct from predictions generated for the
        currently uploaded inference population.
        """
    )

    performance = artifact[
        "performance"
    ]

    # --------------------------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------------------------

    p1, p2, p3, p4 = st.columns(4)

    with p1:

        auc_value = performance[
            "ROC-AUC"
        ]

        st.metric(
            "ROC-AUC",
            (
                f"{auc_value:.4f}"
                if np.isfinite(auc_value)
                else "N/A"
            ),
        )

    with p2:

        st.metric(
            "PR-AUC",
            f"{performance['PR-AUC']:.4f}",
        )

    with p3:

        st.metric(
            "Brier Score",
            f"{performance['Brier Score']:.4f}",
        )

    with p4:

        st.metric(
            "Log Loss",
            f"{performance['Log Loss']:.4f}",
        )

    p5, p6, p7, p8 = st.columns(4)

    with p5:

        st.metric(
            "Sensitivity",
            f"{performance['Sensitivity']:.2%}",
        )

    with p6:

        st.metric(
            "Specificity",
            f"{performance['Specificity']:.2%}",
        )

    with p7:

        st.metric(
            "Precision",
            f"{performance['Precision']:.2%}",
        )

    with p8:

        st.metric(
            "F1 Score",
            f"{performance['F1 Score']:.4f}",
        )

    # --------------------------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------------------------

    st.markdown(
        "### Three-Model Performance Comparison"
    )

    comparison_rows = []

    for model_name in EXPECTED_MODEL_ORDER:

        model_metrics = artifact[
            "per_model_performance"
        ][model_name]

        comparison_rows.append(
            {
                "Model":
                    model_name,
                "ROC-AUC":
                    model_metrics["ROC-AUC"],
                "PR-AUC":
                    model_metrics["PR-AUC"],
                "Brier Score":
                    model_metrics["Brier Score"],
                "Log Loss":
                    model_metrics["Log Loss"],
                "Sensitivity":
                    model_metrics["Sensitivity"],
                "Specificity":
                    model_metrics["Specificity"],
                "F1":
                    model_metrics["F1 Score"],
            }
        )

    comparison_rows.append(
        {
            "Model":
                "Three-Model Ensemble",
            "ROC-AUC":
                performance["ROC-AUC"],
            "PR-AUC":
                performance["PR-AUC"],
            "Brier Score":
                performance["Brier Score"],
            "Log Loss":
                performance["Log Loss"],
            "Sensitivity":
                performance["Sensitivity"],
            "Specificity":
                performance["Specificity"],
            "F1":
                performance["F1 Score"],
        }
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------------------------

    st.markdown(
        "### ROC Curve"
    )

    roc_data = artifact[
        "roc_data"
    ]

    if len(roc_data["fpr"]) > 0:

        roc_df = pd.DataFrame(
            {
                "False Positive Rate":
                    roc_data["fpr"],
                "True Positive Rate":
                    roc_data["tpr"],
            }
        )

        st.line_chart(
            roc_df,
            x="False Positive Rate",
            y="True Positive Rate",
            width="stretch",
        )

    else:

        st.warning(
            "ROC curve unavailable because the test set does not "
            "contain both outcome classes."
        )

    # --------------------------------------------------------------------------
    # PRECISION-RECALL
    # --------------------------------------------------------------------------

    st.markdown(
        "### Precision–Recall Curve"
    )

    pr_data = artifact[
        "pr_data"
    ]

    if len(pr_data["precision"]) > 0:

        pr_df = pd.DataFrame(
            {
                "Recall":
                    pr_data["recall"],
                "Precision":
                    pr_data["precision"],
            }
        )

        st.line_chart(
            pr_df,
            x="Recall",
            y="Precision",
            width="stretch",
        )

    # --------------------------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------------------------

    st.markdown(
        "### Probability Calibration"
    )

    calibration_data = artifact[
        "calibration_data"
    ]

    if len(
        calibration_data[
            "fraction_of_positives"
        ]
    ) > 0:

        calibration_df = pd.DataFrame(
            {
                "Mean Predicted Probability":
                    calibration_data[
                        "mean_predicted_value"
                    ],
                "Observed Positive Fraction":
                    calibration_data[
                        "fraction_of_positives"
                    ],
            }
        )

        st.line_chart(
            calibration_df,
            x="Mean Predicted Probability",
            y="Observed Positive Fraction",
            width="stretch",
        )

    # --------------------------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------------------------

    st.markdown(
        "### Confusion Matrix — Untouched Test Set"
    )

    cm = np.asarray(
        artifact["confusion_matrix"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Lower Risk",
            "Actual GDM",
        ],
        columns=[
            "Predicted Lower Risk",
            "Predicted GDM",
        ],
    )

    st.dataframe(
        cm_df,
        width="stretch",
    )

    # --------------------------------------------------------------------------
    # TEST SET INFO
    # --------------------------------------------------------------------------

    st.markdown(
        "### Evaluation Population"
    )

    e1, e2, e3 = st.columns(3)

    with e1:

        st.metric(
            "Training Rows",
            f"{artifact['training_rows']:,}",
        )

    with e2:

        st.metric(
            "Calibration Rows",
            f"{artifact['calibration_rows']:,}",
        )

    with e3:

        st.metric(
            "Untouched Test Rows",
            f"{artifact['test_rows']:,}",
        )

    st.caption(
        "The test set remains untouched during preprocessing fitting, "
        "model training, SMOTETomek resampling, and calibration."
    )


# ==============================================================================
# UNCERTAINTY TAB
# ==============================================================================

with uncertainty_tab:

    st.subheader(
        "🎲 Uncertainty & Conformal Safety"
    )

    st.markdown(
        """
        SafeTriage-GDM separates uncertainty into complementary components.

        **Epistemic uncertainty** represents uncertainty associated with
        disagreement between the three predictive models. Here it is
        operationalized using the standard deviation of their predicted
        GDM probabilities.

        **Aleatoric uncertainty** represents irreducible predictive
        uncertainty and is operationalized as the expected Bernoulli
        entropy of the three model probabilities.

        **Predictive entropy** is the entropy of the final ensemble
        probability.

        **Mutual information** is the difference between predictive
        entropy and expected model entropy and provides an additional
        epistemic-uncertainty measure.
        """
    )

    u1, u2, u3, u4 = st.columns(4)

    with u1:

        st.metric(
            "Mean Epistemic Uncertainty",
            f"{np.mean(epistemic_uncertainty):.4f}",
        )

    with u2:

        st.metric(
            "Mean Aleatoric Uncertainty",
            f"{np.mean(aleatoric_uncertainty):.4f}",
        )

    with u3:

        st.metric(
            "Mean Predictive Entropy",
            f"{np.mean(predictive_entropy):.4f}",
        )

    with u4:

        st.metric(
            "Mean Mutual Information",
            f"{np.mean(mutual_information):.4f}",
        )

    # --------------------------------------------------------------------------
    # UNCERTAINTY TABLE
    # --------------------------------------------------------------------------

    uncertainty_summary = pd.DataFrame(
        {
            "Measure": [
                "Epistemic uncertainty",
                "Aleatoric uncertainty",
                "Predictive entropy",
                "Mutual information",
            ],
            "Mean": [
                np.mean(
                    epistemic_uncertainty
                ),
                np.mean(
                    aleatoric_uncertainty
                ),
                np.mean(
                    predictive_entropy
                ),
                np.mean(
                    mutual_information
                ),
            ],
            "Median": [
                np.median(
                    epistemic_uncertainty
                ),
                np.median(
                    aleatoric_uncertainty
                ),
                np.median(
                    predictive_entropy
                ),
                np.median(
                    mutual_information
                ),
            ],
            "Maximum": [
                np.max(
                    epistemic_uncertainty
                ),
                np.max(
                    aleatoric_uncertainty
                ),
                np.max(
                    predictive_entropy
                ),
                np.max(
                    mutual_information
                ),
            ],
        }
    )

    st.dataframe(
        uncertainty_summary,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------------------------
    # MODEL DISAGREEMENT
    # --------------------------------------------------------------------------

    st.markdown(
        "### Model Probability Disagreement"
    )

    disagreement_df = pd.DataFrame(
        {
            "Random Forest":
                inference_model_probabilities[0],
            "XGBoost":
                inference_model_probabilities[1],
            "Logistic Regression":
                inference_model_probabilities[2],
            "Ensemble":
                consensus_prob,
            "Epistemic SD":
                epistemic_uncertainty,
            "Mutual Information":
                mutual_information,
        }
    )

    disagreement_display = (
        disagreement_df.copy()
    )

    for column in disagreement_display.columns:

        disagreement_display[
            column
        ] = disagreement_display[
            column
        ].round(5)

    st.dataframe(
        disagreement_display,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------------------------
    # CONFORMAL
    # --------------------------------------------------------------------------

    st.markdown(
        "### Conformal Safety Assessment"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Nominal Level",
            f"{CONFORMAL_LEVEL:.0%}",
        )

    with c2:

        st.metric(
            "Conformal q",
            f"{q_threshold:.4f}",
        )

    with c3:

        st.metric(
            "Test Empirical Coverage",
            f"{artifact['conformal']['test_coverage']:.2%}",
        )

    st.markdown(
        f"""
        **Calibration empirical coverage:**
        {artifact['conformal']['calibration_coverage']:.2%}

        **Untouched test empirical coverage:**
        {artifact['conformal']['test_coverage']:.2%}

        These are empirical coverage measurements for the implemented
        conformal procedure. The runtime drift adjustment below is a
        research heuristic and should **not** be interpreted as a formal
        guarantee of 95% coverage under arbitrary population shift.
        """
    )

    conformal_counts = (
        pd.Series(
            runtime_conformal_sets
        )
        .value_counts()
        .rename_axis(
            "Prediction Set"
        )
        .reset_index(
            name="Count"
        )
    )

    st.dataframe(
        conformal_counts,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------------------------
    # DRIFT ADJUSTMENT
    # --------------------------------------------------------------------------

    if np.isfinite(psi_value):

        if psi_value >= DRIFT_LIMIT:

            st.warning(
                f"Population-shift PSI is {psi_value:.4f}, "
                f"which exceeds the configured {DRIFT_LIMIT:.2f} "
                "monitoring threshold."
            )

        else:

            st.success(
                f"Population-shift PSI is {psi_value:.4f}, "
                "below the configured monitoring threshold."
            )

        if runtime_q != q_threshold:

            st.warning(
                f"A research heuristic reduced the runtime conformal "
                f"threshold from {q_threshold:.4f} to "
                f"{runtime_q:.4f} because drift and predictive "
                "entropy were both elevated."
            )

    else:

        st.info(
            "BMI-based population-shift monitoring is unavailable "
            "because a usable BMI field was not found."
        )


# ==============================================================================
# FAIRNESS TAB
# ==============================================================================

with fairness_tab:

    st.subheader(
        "⚖️ Algorithmic Fairness & Population Shift"
    )

    st.markdown(
        """
        Fairness analysis is performed on the same inference population
        used by the ensemble.

        **Demographic parity difference** compares the positive-risk
        classification rate between the age groups below 35 and 35 or older.

        **FPR disparity** compares false-positive rates between those groups
        using only rows with valid ground-truth GDM outcomes.

        FPR disparity alone is **not equivalent to full equalized odds**,
        because equalized odds requires both FPR and TPR parity.
        """
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        value = fairness[
            "demographic_parity_difference"
        ]

        st.metric(
            "Demographic Parity Difference",
            (
                f"{value:.4f}"
                if np.isfinite(value)
                else "N/A"
            ),
        )

    with f2:

        value = fairness[
            "fpr_disparity"
        ]

        st.metric(
            "FPR Disparity",
            (
                f"{value:.4f}"
                if np.isfinite(value)
                else "N/A"
            ),
        )

    with f3:

        st.metric(
            "Labeled Fairness Sample",
            f"{fairness['fairness_sample_size']:,}",
        )

    fairness_table = pd.DataFrame(
        {
            "Metric": [
                "Positive rate — Age < 35",
                "Positive rate — Age ≥ 35",
                "False-positive rate — Age < 35",
                "False-positive rate — Age ≥ 35",
            ],
            "Value": [
                fairness[
                    "younger_positive_rate"
                ],
                fairness[
                    "older_positive_rate"
                ],
                fairness[
                    "younger_fpr"
                ],
                fairness[
                    "older_fpr"
                ],
            ],
        }
    )

    fairness_table[
        "Value"
    ] = fairness_table[
        "Value"
    ].map(
        lambda x:
            f"{x:.4f}"
            if pd.notna(x)
            else "N/A"
    )

    st.dataframe(
        fairness_table,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------------------------
    # BMI SHIFT
    # --------------------------------------------------------------------------

    st.markdown(
        "### BMI Population Shift"

    )

    if (
        baseline_bmi is not None
        and current_bmi is not None
    ):

        drift_table = pd.DataFrame(
            {
                "BMI Category":
                    BMI_LABELS,
                "Baseline":
                    [
                        baseline_bmi.get(
                            category,
                            0.0,
                        )
                        for category in BMI_LABELS
                    ],
                "Current":
                    [
                        current_bmi.get(
                            category,
                            0.0,
                        )
                        for category in BMI_LABELS
                    ],
            }
        )

        drift_table[
            "Baseline"
        ] = drift_table[
            "Baseline"
        ].map(
            lambda x: f"{x:.2%}"
        )

        drift_table[
            "Current"
        ] = drift_table[
            "Current"
        ].map(
            lambda x: f"{x:.2%}"
        )

        st.dataframe(
            drift_table,
            width="stretch",
            hide_index=True,
        )

        st.metric(
            "BMI Population Stability Index",
            f"{psi_value:.4f}",
        )

        st.caption(
            "PSI < 0.10 is commonly interpreted as little shift; "
            "0.10–0.20 as moderate shift; and >0.20 as potentially "
            "substantial shift. These are monitoring heuristics, not "
            "clinical thresholds."
        )

    else:

        st.info(
            "BMI population-shift analysis is unavailable."
        )


# ==============================================================================
# XAI TAB
# ==============================================================================

with xai_tab:

    st.subheader(
        "🔍 XAI Attribution"
    )

    st.markdown(
        """
        The attribution analysis summarizes global feature importance
        across the three components of SafeTriage-GDM.

        Random Forest and XGBoost contribute tree-based feature
        importance, while Logistic Regression contributes the absolute
        magnitude of its standardized coefficient.

        These values describe predictive contribution within the research
        system. They should **not** be interpreted as causal effects or as
        evidence that a specific characteristic independently causes GDM.
        """
    )

    xai_df = artifact[
        "feature_importance"
    ]

    if (
        xai_df is not None
        and not xai_df.empty
    ):

        display_xai = xai_df.copy()

        display_xai[
            "Attribution Weight Score"
        ] = display_xai[
            "Attribution Weight Score"
        ].round(5)

        for column in [
            "Random Forest",
            "XGBoost",
            "Logistic Regression |Abs Coefficient|",
        ]:

            if column in display_xai.columns:

                display_xai[
                    column
                ] = display_xai[
                    column
                ].round(6)

        st.dataframe(
            display_xai,
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "Feature attribution information is unavailable."
        )


# ==============================================================================
# FOOTER / TECHNICAL STATUS
# ==============================================================================

st.markdown(
    "---"
)

status_col1, status_col2, status_col3 = (
    st.columns(3)
)

with status_col1:

    st.caption(
        "SafeTriage-GDM"
    )

with status_col2:

    st.caption(
        "Three-model uncertainty-aware ensemble"
    )

with status_col3:

    st.caption(
        f"Artifact: {APP_VERSION}"
    )

st.caption(
    "Research prototype — not for clinical diagnosis, treatment, "
    "or autonomous clinical decision-making."
)
