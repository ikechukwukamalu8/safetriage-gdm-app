# ==============================================================================
# SafeTriage-GDM
# Uncertainty-Quantified Clinical Triage System for Gestational Diabetes
# Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds
#
# Standalone Research Prototype
# ==============================================================================

import os
import io
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator

from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier


# ==============================================================================
# CONFIGURATION
# ==============================================================================

APP_TITLE = "SafeTriage-GDM"

APP_SUBTITLE = (
    "Uncertainty-Quantified Clinical Triage System for "
    "Gestational Diabetes Mellitus (GDM) Risk with "
    "Algorithmic Fairness Auditing & Conformal Safety Bounds"
)

APP_VERSION = "Research Prototype"

ARTIFACT_PATH = "safetriage_gdm_mixed_pipeline.pkl"

TARGET_COLUMN = "Gestational diabetes?"
ID_COLUMN = "Dummy Study Number"
AGE_COLUMN = "Age"
BMI_COLUMN = "BMI"

RANDOM_STATE = 42

RISK_THRESHOLD = 0.50

CONFORMAL_ALPHA = 0.05

DRIFT_THRESHOLD = 0.20

EXPECTED_MODELS = {
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
}


# ==============================================================================
# STREAMLIT PAGE
# ==============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
)


# ==============================================================================
# GLOBAL STYLE
# ==============================================================================

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .sub-header {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }

    .risk-card {
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 0.75rem;
    }

    .disclaimer {
        padding: 0.9rem;
        border-radius: 0.6rem;
        background-color: rgba(255, 193, 7, 0.10);
        border: 1px solid rgba(255, 193, 7, 0.35);
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# HEADER
# ==============================================================================

st.markdown(
    f'<div class="main-header">🛡️ {APP_TITLE}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="sub-header">{APP_SUBTITLE}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer">
    <strong>Research prototype:</strong>
    SafeTriage-GDM is intended for uncertainty-aware GDM risk triage,
    conformal safety assessment, population-shift monitoring, and
    algorithmic fairness auditing. It is not a medical device and must
    not be used as a substitute for professional medical diagnosis,
    treatment, or clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ==============================================================================
# REPRODUCIBILITY / WARNINGS
# ==============================================================================

warnings.filterwarnings(
    "ignore",
    message=".*does not have valid feature names.*",
)


# ==============================================================================
# TARGET PARSING
# ==============================================================================

def parse_gdm_target(series):
    """
    Convert common GDM outcome representations to binary 0/1.

    Accepted examples include:
        0 / 1
        Yes / No
        Y / N
        True / False
        Positive / Negative
        GDM / No GDM
    """

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")

        unique_values = set(
            numeric.dropna().unique().tolist()
        )

        if unique_values.issubset({0, 1}):
            return numeric.astype(float)

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    positive_values = {
        "1",
        "yes",
        "y",
        "true",
        "positive",
        "pos",
        "gdm",
        "gestational diabetes",
        "gestational diabetes mellitus",
    }

    negative_values = {
        "0",
        "no",
        "n",
        "false",
        "negative",
        "neg",
        "normal",
        "no gdm",
        "no gestational diabetes",
        "healthy",
    }

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    result[text.isin(positive_values)] = 1.0
    result[text.isin(negative_values)] = 0.0

    numeric = pd.to_numeric(series, errors="coerce")

    result[numeric.eq(1)] = 1.0
    result[numeric.eq(0)] = 0.0

    return result


# ==============================================================================
# DATA ENCODING
# ==============================================================================

def encode_dataframe(df):
    """
    One-hot encode categorical variables.

    The resulting dataframe is explicitly converted to float64 so that
    downstream sklearn transformations receive a consistent numeric matrix.
    """

    encoded_df = pd.get_dummies(
        df,
        drop_first=True,
        dummy_na=False,
    )

    return encoded_df.astype(np.float64)


def align_to_schema(encoded_df, feature_schema):
    """
    Force inference data to exactly match the training feature schema.
    Missing training columns are filled with zero.
    Extra inference-only columns are removed.
    """

    aligned_df = encoded_df.reindex(
        columns=feature_schema,
        fill_value=0.0,
    )

    return aligned_df.astype(np.float64)


# ==============================================================================
# PROBABILITY HELPERS
# ==============================================================================

def get_probability_from_model(model, X):
    """
    Safely extract P(GDM=1).
    """

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]

        return probabilities.ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)

        return 1.0 / (
            1.0 + np.exp(-np.asarray(scores))
        )

    raise RuntimeError(
        f"Model {type(model).__name__} does not expose "
        "predict_proba or decision_function."
    )


# ==============================================================================
# ENTROPY / UNCERTAINTY
# ==============================================================================

def binary_entropy(probabilities):
    """
    Binary Shannon entropy in nats.
    """

    p = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-12,
        1.0 - 1e-12,
    )

    return -(
        p * np.log(p)
        + (1.0 - p) * np.log(1.0 - p)
    )


def calculate_uncertainty_decomposition(model_probabilities):
    """
    Estimate uncertainty from the three-model probabilistic ensemble.

    Inputs:
        model_probabilities:
            shape = (3, n_samples)

    Returns:
        ensemble_probability
        predictive_entropy
        aleatoric_uncertainty
        epistemic_uncertainty
        model_disagreement_std

    Definitions:

        Predictive entropy:
            H(mean p)

        Aleatoric uncertainty estimate:
            mean(H(p_m))

        Epistemic uncertainty estimate:
            H(mean p) - mean(H(p_m))

        Model disagreement:
            standard deviation of model probabilities

    The epistemic quantity is a model-disagreement / mutual-information-
    style proxy based on the finite three-model ensemble. It should not
    be interpreted as a complete Bayesian posterior uncertainty estimate.
    """

    probabilities = np.asarray(
        model_probabilities,
        dtype=float,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "Model probability matrix must be two-dimensional."
        )

    if probabilities.shape[0] != 3:
        raise RuntimeError(
            "SafeTriage-GDM uncertainty decomposition requires exactly "
            "three model probability vectors."
        )

    ensemble_probability = np.mean(
        probabilities,
        axis=0,
    )

    predictive_entropy = binary_entropy(
        ensemble_probability
    )

    individual_entropies = binary_entropy(
        probabilities
    )

    aleatoric_uncertainty = np.mean(
        individual_entropies,
        axis=0,
    )

    epistemic_uncertainty = np.maximum(
        predictive_entropy - aleatoric_uncertainty,
        0.0,
    )

    model_disagreement_std = np.std(
        probabilities,
        axis=0,
    )

    return {
        "ensemble_probability": ensemble_probability,
        "predictive_entropy": predictive_entropy,
        "aleatoric_uncertainty": aleatoric_uncertainty,
        "epistemic_uncertainty": epistemic_uncertainty,
        "model_disagreement_std": model_disagreement_std,
    }


# ==============================================================================
# CONFORMAL PREDICTION
# ==============================================================================

def calculate_conformal_quantile(
    calibration_probabilities,
    calibration_labels,
    alpha=CONFORMAL_ALPHA,
):
    """
    Calculate the split-conformal quantile using calibration data only.

    Nonconformity score:

        1 - probability assigned to the true class
    """

    probabilities = np.asarray(
        calibration_probabilities,
        dtype=float,
    )

    labels = np.asarray(
        calibration_labels,
        dtype=int,
    )

    true_class_probability = np.where(
        labels == 1,
        probabilities,
        1.0 - probabilities,
    )

    nonconformity_scores = (
        1.0 - true_class_probability
    )

    try:
        q_threshold = np.quantile(
            nonconformity_scores,
            1.0 - alpha,
            method="higher",
        )
    except TypeError:
        q_threshold = np.percentile(
            nonconformity_scores,
            100.0 * (1.0 - alpha),
        )

    return float(q_threshold)


def generate_conformal_sets(
    probabilities,
    q_threshold,
):
    """
    Generate binary conformal prediction sets.

    A class is included when its nonconformity score is <= q.

    Output examples:
        {Healthy}
        {GDM High Risk}
        {Healthy, GDM High Risk}
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    healthy_score = 1.0 - probabilities
    gdm_score = probabilities

    include_healthy = (
        healthy_score <= q_threshold
    )

    include_gdm = (
        gdm_score <= q_threshold
    )

    prediction_sets = []

    for healthy, gdm in zip(
        include_healthy,
        include_gdm,
    ):
        if healthy and gdm:
            prediction_sets.append(
                "Healthy + GDM High Risk"
            )
        elif healthy:
            prediction_sets.append(
                "Healthy"
            )
        elif gdm:
            prediction_sets.append(
                "GDM High Risk"
            )
        else:
            # For a binary prediction problem this means both classes
            # are excluded under the selected threshold. We expose both
            # classes rather than presenting an empty safety set.
            prediction_sets.append(
                "Healthy + GDM High Risk"
            )

    return np.asarray(
        prediction_sets,
        dtype=object,
    )


def conformal_coverage(
    probabilities,
    labels,
    q_threshold,
):
    """
    Empirical coverage on a labeled dataset.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    healthy_score = 1.0 - probabilities
    gdm_score = probabilities

    true_class_score = np.where(
        labels == 1,
        gdm_score,
        healthy_score,
    )

    coverage = np.mean(
        true_class_score <= q_threshold
    )

    return float(coverage)


# ==============================================================================
# BMI DRIFT / PSI
# ==============================================================================

BMI_BINS = [
    -np.inf,
    18.5,
    24.9,
    29.9,
    39.9,
    np.inf,
]


def calculate_bmi_distribution(bmi_series):
    """
    Calculate normalized BMI-bin distribution.
    """

    numeric_bmi = pd.to_numeric(
        bmi_series,
        errors="coerce",
    )

    valid_bmi = numeric_bmi.dropna()

    if len(valid_bmi) == 0:
        return None

    counts = pd.cut(
        valid_bmi,
        bins=BMI_BINS,
        include_lowest=True,
    ).value_counts(
        sort=False
    )

    probabilities = (
        counts / counts.sum()
    ).astype(float)

    return probabilities.to_numpy()


def calculate_psi(
    expected_distribution,
    actual_distribution,
):
    """
    Population Stability Index.

    PSI < 0.10   : little/no change
    PSI 0.10-0.20: moderate change
    PSI >= 0.20  : substantial shift

    These are monitoring heuristics, not clinical thresholds.
    """

    if (
        expected_distribution is None
        or actual_distribution is None
    ):
        return np.nan

    expected = np.asarray(
        expected_distribution,
        dtype=float,
    )

    actual = np.asarray(
        actual_distribution,
        dtype=float,
    )

    if expected.shape != actual.shape:
        return np.nan

    epsilon = 1e-6

    expected = np.clip(
        expected,
        epsilon,
        None,
    )

    actual = np.clip(
        actual,
        epsilon,
        None,
    )

    expected = expected / expected.sum()
    actual = actual / actual.sum()

    psi = np.sum(
        (actual - expected)
        * np.log(actual / expected)
    )

    return float(psi)


# ==============================================================================
# FAIRNESS
# ==============================================================================

def calculate_fairness_metrics(
    dataframe,
    consensus_prob,
):
    """
    Fairness metrics use predictions aligned to the entire uploaded
    inference population.

    Demographic parity:
        all rows with valid age.

    FPR disparity:
        only rows with both valid age and valid GDM ground truth.

    FPR disparity is NOT full equalized-odds disparity.
    """

    result = {
        "younger_positive_rate": np.nan,
        "older_positive_rate": np.nan,
        "demographic_parity_difference": np.nan,
        "younger_fpr": np.nan,
        "older_fpr": np.nan,
        "fpr_disparity": np.nan,
        "fairness_sample_size": 0,
        "younger_sample_size": 0,
        "older_sample_size": 0,
    }

    if dataframe is None:
        return result

    probabilities = np.asarray(
        consensus_prob,
        dtype=float,
    )

    if len(dataframe) != len(probabilities):
        raise ValueError(
            "Fairness calculation alignment error: "
            f"dataframe contains {len(dataframe)} rows, "
            f"but predictions contain {len(probabilities)} rows."
        )

    if AGE_COLUMN not in dataframe.columns:
        return result

    age_series = pd.to_numeric(
        dataframe[AGE_COLUMN],
        errors="coerce",
    )

    positive_flags = (
        probabilities >= RISK_THRESHOLD
    ).astype(int)

    younger_mask = (
        (age_series < 35)
        & age_series.notna()
    ).to_numpy()

    older_mask = (
        (age_series >= 35)
        & age_series.notna()
    ).to_numpy()

    result["younger_sample_size"] = int(
        younger_mask.sum()
    )

    result["older_sample_size"] = int(
        older_mask.sum()
    )

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
        result["demographic_parity_difference"] = abs(
            result["younger_positive_rate"]
            - result["older_positive_rate"]
        )

    if TARGET_COLUMN not in dataframe.columns:
        return result

    y_true = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
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
    ]

    eval_age = age_series.to_numpy()[
        labeled_mask
    ]

    younger_eval = eval_age < 35
    older_eval = eval_age >= 35

    def calculate_fpr(
        y_true_group,
        y_pred_group,
    ):
        negatives = (
            y_true_group == 0
        )

        if negatives.sum() == 0:
            return np.nan

        false_positives = np.sum(
            y_pred_group[negatives] == 1
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


# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================

def calculate_binary_metrics(
    y_true,
    probabilities,
    threshold=RISK_THRESHOLD,
):
    """
    Calculate predictive performance metrics.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
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
        "ROC-AUC": (
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            )
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "PR-AUC": (
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            )
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "Accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "Sensitivity": float(
            sensitivity
        ),
        "Specificity": float(
            specificity
        ),
        "Precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "F1-score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "Brier Score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "Log Loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
        "Confusion Matrix": cm,
    }

    return metrics


# ==============================================================================
# XAI
# ==============================================================================

def calculate_feature_importance(
    models,
    feature_schema,
):
    """
    Build a unified global feature-attribution table across all three
    ensemble models.

    Tree-based importance:
        RF feature_importances_
        XGBoost feature_importances_

    Logistic Regression:
        absolute standardized coefficient magnitude

    Each model's contribution is normalized independently before the
    three-model average is calculated.
    """

    feature_count = len(
        feature_schema
    )

    per_model = {}

    for model_name, model in models.items():

        estimator = model

        if hasattr(
            model,
            "estimator",
        ):
            estimator = model.estimator

        if hasattr(
            estimator,
            "feature_importances_",
        ):
            values = np.asarray(
                estimator.feature_importances_,
                dtype=float,
            )

        elif hasattr(
            estimator,
            "coef_",
        ):
            coefficients = np.asarray(
                estimator.coef_,
                dtype=float,
            )

            if coefficients.ndim == 2:
                values = np.abs(
                    coefficients[0]
                )
            else:
                values = np.abs(
                    coefficients
                )

        else:
            continue

        if len(values) != feature_count:
            continue

        total = values.sum()

        if total > 0:
            values = values / total

        per_model[model_name] = values

    if not per_model:
        return pd.DataFrame(
            columns=[
                "Feature",
                "Attribution Weight Score",
            ]
        )

    model_arrays = list(
        per_model.values()
    )

    average_importance = np.mean(
        np.vstack(model_arrays),
        axis=0,
    )

    result = pd.DataFrame(
        {
            "Feature": feature_schema,
            "Attribution Weight Score": average_importance,
        }
    )

    result = result.sort_values(
        "Attribution Weight Score",
        ascending=False,
    ).reset_index(drop=True)

    return result


# ==============================================================================
# MODEL CONSTRUCTION
# ==============================================================================

def build_models():
    """
    Construct the mandatory three-model ensemble.

    XGBoost is intentionally mandatory.
    There is NO two-model fallback.
    """

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
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


# ==============================================================================
# MODEL CALIBRATION
# ==============================================================================

def calibrate_models(
    models,
    X_cal,
    y_cal,
):
    """
    Calibrate each already-trained base model using the dedicated
    calibration set.

    FrozenEstimator is used with modern scikit-learn versions to make
    the prefit nature of the model explicit.
    """

    calibrated_models = {}

    for model_name, model in models.items():

        calibrated_model = CalibratedClassifierCV(
            FrozenEstimator(model),
            method="sigmoid",
        )

        calibrated_model.fit(
            X_cal,
            y_cal,
        )

        calibrated_models[
            model_name
        ] = calibrated_model

    if (
        set(calibrated_models.keys())
        != EXPECTED_MODELS
    ):
        raise RuntimeError(
            "Calibration did not produce the required three-model ensemble."
        )

    return calibrated_models


# ==============================================================================
# PREDICTION MATRIX
# ==============================================================================

def get_model_probability_matrix(
    models,
    X,
):
    """
    Return probability matrix with exactly three rows.
    """

    probabilities = []

    for model_name in [
        "Random Forest",
        "XGBoost",
        "Logistic Regression",
    ]:

        if model_name not in models:
            raise RuntimeError(
                f"Required ensemble model missing: {model_name}"
            )

        probability = get_probability_from_model(
            models[model_name],
            X,
        )

        probabilities.append(
            np.asarray(
                probability,
                dtype=float,
            )
        )

    matrix = np.vstack(
        probabilities
    )

    if matrix.shape[0] != 3:
        raise RuntimeError(
            "SafeTriage-GDM requires exactly three ensemble probability vectors."
        )

    return matrix


# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================

def train_pipeline(
    raw_dataframe,
):
    """
    Complete training pipeline.

    Population separation:

        ALL uploaded rows
            -> inference

        ONLY rows with valid GDM outcome
            -> training / calibration / test

    Split:

        60% training
        15% calibration
        25% untouched test
    """

    dataframe = raw_dataframe.copy()

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Training requires the target column "
            f"'{TARGET_COLUMN}'."
        )

    dataframe[
        TARGET_COLUMN
    ] = parse_gdm_target(
        dataframe[TARGET_COLUMN]
    )

    labeled_mask = (
        dataframe[TARGET_COLUMN]
        .notna()
    )

    labeled_dataframe = dataframe.loc[
        labeled_mask
    ].copy()

    if len(labeled_dataframe) < 20:
        raise ValueError(
            "At least 20 labeled observations are required "
            "for model training."
        )

    y = labeled_dataframe[
        TARGET_COLUMN
    ].astype(int).to_numpy()

    unique_classes = np.unique(y)

    if not np.array_equal(
        unique_classes,
        np.array([0, 1]),
    ):
        raise ValueError(
            "Training data must contain both GDM classes: "
            "0 and 1."
        )

    X_raw = labeled_dataframe.drop(
        columns=[
            TARGET_COLUMN,
            ID_COLUMN,
        ],
        errors="ignore",
    ).copy()

    # --------------------------------------------------------------------------
    # Encode
    # --------------------------------------------------------------------------

    X_encoded = encode_dataframe(
        X_raw
    )

    feature_schema = list(
        X_encoded.columns
    )

    if len(feature_schema) == 0:
        raise ValueError(
            "No usable predictor features remain after removing "
            "the target and ID columns."
        )

    # --------------------------------------------------------------------------
    # 60 / 15 / 25 split
    # --------------------------------------------------------------------------

    (
        X_train_temp,
        X_test,
        y_train_temp,
        y_test,
    ) = train_test_split(
        X_encoded,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    (
        X_train,
        X_cal,
        y_train,
        y_cal,
    ) = train_test_split(
        X_train_temp,
        y_train_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_temp,
    )

    # --------------------------------------------------------------------------
    # Imputation: TRAIN ONLY
    # --------------------------------------------------------------------------

    imputer = IterativeImputer(
        random_state=RANDOM_STATE,
        max_iter=10,
    )

    X_train_imputed = imputer.fit_transform(
        X_train
    )

    X_cal_imputed = imputer.transform(
        X_cal
    )

    X_test_imputed = imputer.transform(
        X_test
    )

    # --------------------------------------------------------------------------
    # Scaling: TRAIN ONLY
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
    # SMOTETomek: TRAIN ONLY
    # --------------------------------------------------------------------------

    sampler = SMOTETomek(
        random_state=RANDOM_STATE
    )

    X_train_balanced, y_train_balanced = (
        sampler.fit_resample(
            X_train_scaled,
            y_train,
        )
    )

    # --------------------------------------------------------------------------
    # Train exactly three base models
    # --------------------------------------------------------------------------

    models = build_models()

    for model_name, model in models.items():

        model.fit(
            X_train_balanced,
            y_train_balanced,
        )

    # --------------------------------------------------------------------------
    # Calibrate exactly three models
    # --------------------------------------------------------------------------

    calibrated_models = calibrate_models(
        models,
        X_cal_scaled,
        y_cal,
    )

    # --------------------------------------------------------------------------
    # Calibration probabilities
    # --------------------------------------------------------------------------

    calibration_model_probabilities = (
        get_model_probability_matrix(
            calibrated_models,
            X_cal_scaled,
        )
    )

    calibration_uncertainty = (
        calculate_uncertainty_decomposition(
            calibration_model_probabilities
        )
    )

    calibration_probabilities = (
        calibration_uncertainty[
            "ensemble_probability"
        ]
    )

    # --------------------------------------------------------------------------
    # Conformal threshold
    # --------------------------------------------------------------------------

    q_threshold = calculate_conformal_quantile(
        calibration_probabilities,
        y_cal,
        alpha=CONFORMAL_ALPHA,
    )

    # --------------------------------------------------------------------------
    # Untouched TEST predictions
    # --------------------------------------------------------------------------

    test_model_probabilities = (
        get_model_probability_matrix(
            calibrated_models,
            X_test_scaled,
        )
    )

    test_uncertainty = (
        calculate_uncertainty_decomposition(
            test_model_probabilities
        )
    )

    test_ensemble_probability = (
        test_uncertainty[
            "ensemble_probability"
        ]
    )

    # --------------------------------------------------------------------------
    # Test metrics for all three models
    # --------------------------------------------------------------------------

    test_metrics = {}

    model_order = [
        "Random Forest",
        "XGBoost",
        "Logistic Regression",
    ]

    for index, model_name in enumerate(
        model_order
    ):

        test_metrics[
            model_name
        ] = calculate_binary_metrics(
            y_test,
            test_model_probabilities[index],
        )

    test_metrics[
        "3-Model Ensemble"
    ] = calculate_binary_metrics(
        y_test,
        test_ensemble_probability,
    )

    # --------------------------------------------------------------------------
    # Conformal test coverage
    # --------------------------------------------------------------------------

    test_conformal_coverage = (
        conformal_coverage(
            test_ensemble_probability,
            y_test,
            q_threshold,
        )
    )

    test_prediction_sets = (
        generate_conformal_sets(
            test_ensemble_probability,
            q_threshold,
        )
    )

    test_set_sizes = np.array(
        [
            2
            if " + " in prediction_set
            else 1
            for prediction_set
            in test_prediction_sets
        ]
    )

    # --------------------------------------------------------------------------
    # Calibration curve
    # --------------------------------------------------------------------------

    calibration_fraction_pos, calibration_mean_pred = (
        calibration_curve(
            y_test,
            test_ensemble_probability,
            n_bins=10,
            strategy="quantile",
        )
    )

    # --------------------------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------------------------

    fpr, tpr, roc_thresholds = roc_curve(
        y_test,
        test_ensemble_probability,
    )

    # --------------------------------------------------------------------------
    # Feature importance / XAI
    # --------------------------------------------------------------------------

    feature_importance_df = (
        calculate_feature_importance(
            calibrated_models,
            feature_schema,
        )
    )

    # --------------------------------------------------------------------------
    # Baseline BMI distribution
    #
    # Use the training population only to establish the reference
    # distribution.
    # --------------------------------------------------------------------------

    baseline_bmi_distribution = None

    if BMI_COLUMN in labeled_dataframe.columns:

        train_original_indices = (
            X_train.index
        )

        baseline_bmi_distribution = (
            calculate_bmi_distribution(
                dataframe.loc[
                    train_original_indices,
                    BMI_COLUMN,
                ]
            )
        )

    # --------------------------------------------------------------------------
    # Artifact
    # --------------------------------------------------------------------------

    artifact = {
        "artifact_version": 2,
        "app_title": APP_TITLE,
        "app_version": APP_VERSION,

        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,

        "imputer": imputer,
        "scaler": scaler,

        "models": calibrated_models,

        "base_models": models,

        "feature_schema": feature_schema,

        "q_threshold": q_threshold,
        "conformal_alpha": CONFORMAL_ALPHA,

        "feature_importance": feature_importance_df,

        "test_metrics": test_metrics,

        "test_conformal_coverage": (
            test_conformal_coverage
        ),

        "test_mean_conformal_set_size": (
            float(
                np.mean(test_set_sizes)
            )
        ),

        "test_singleton_rate": (
            float(
                np.mean(test_set_sizes == 1)
            )
        ),

        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresholds,
        },

        "calibration_curve": {
            "fraction_of_positives": (
                calibration_fraction_pos
            ),
            "mean_predicted_value": (
                calibration_mean_pred
            ),
        },

        "feature_importance": (
            feature_importance_df
        ),

        "training_rows": int(
            len(X_train)
        ),

        "calibration_rows": int(
            len(X_cal)
        ),

        "test_rows": int(
            len(X_test)
        ),

        "total_labeled_rows": int(
            len(labeled_dataframe)
        ),

        "baseline_bmi_distribution": (
            baseline_bmi_distribution
        ),

        "train_positive_rate": float(
            np.mean(y_train)
        ),

        "calibration_positive_rate": float(
            np.mean(y_cal)
        ),

        "test_positive_rate": float(
            np.mean(y_test)
        ),
    }

    return artifact


# ==============================================================================
# ARTIFACT VALIDATION
# ==============================================================================

def validate_artifact(artifact):
    """
    Verify that the stored artifact belongs to the intended
    SafeTriage-GDM architecture.
    """

    required_keys = {
        "imputer",
        "scaler",
        "models",
        "feature_schema",
        "q_threshold",
        "test_metrics",
    }

    missing_keys = (
        required_keys
        - set(artifact.keys())
    )

    if missing_keys:
        raise RuntimeError(
            "The saved SafeTriage-GDM artifact is incomplete. "
            f"Missing: {sorted(missing_keys)}"
        )

    model_names = set(
        artifact["models"].keys()
    )

    if model_names != EXPECTED_MODELS:
        raise RuntimeError(
            "The saved artifact does not contain exactly the required "
            "three-model ensemble. "
            f"Found: {sorted(model_names)}; "
            f"Expected: {sorted(EXPECTED_MODELS)}."
        )

    if len(
        artifact["feature_schema"]
    ) == 0:
        raise RuntimeError(
            "The saved artifact contains an empty feature schema."
        )

    if not np.isfinite(
        artifact["q_threshold"]
    ):
        raise RuntimeError(
            "The saved conformal q-threshold is invalid."
        )

    return True


# ==============================================================================
# LOAD / SAVE ARTIFACT
# ==============================================================================

def save_artifact(artifact):
    with open(
        ARTIFACT_PATH,
        "wb",
    ) as file:
        pickle.dump(
            artifact,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_artifact():
    with open(
        ARTIFACT_PATH,
        "rb",
    ) as file:
        artifact = pickle.load(file)

    validate_artifact(
        artifact
    )

    return artifact


# ==============================================================================
# FILE LOADING
# ==============================================================================

def load_uploaded_dataframe(uploaded_file):
    """
    Load CSV or Excel input.
    """

    filename = (
        uploaded_file.name.lower()
    )

    if filename.endswith(
        ".csv"
    ):
        dataframe = pd.read_csv(
            uploaded_file
        )

    elif filename.endswith(
        (".xlsx", ".xls")
    ):
        dataframe = pd.read_excel(
            uploaded_file
        )

    else:
        raise ValueError(
            "Unsupported file type. "
            "Please upload CSV or Excel data."
        )

    if dataframe.empty:
        raise ValueError(
            "The uploaded dataset is empty."
        )

    dataframe = dataframe.copy()

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


# ==============================================================================
# SIDEBAR
# ==============================================================================

st.sidebar.header(
    "⚙️ Administrative Control Unit"
)

force_retrain = st.sidebar.button(
    "🔄 Force Pipeline Retraining",
    width="stretch",
)

st.sidebar.markdown(
    "---"
)

st.sidebar.markdown(
    f"""
    **Architecture**

    • Random Forest  
    • XGBoost  
    • Logistic Regression  

    **Data split**

    • 60% Training  
    • 15% Calibration  
    • 25% Untouched Test  

    **Conformal level**

    • {100 * (1 - CONFORMAL_ALPHA):.0f}% nominal level

    **Risk threshold**

    • {RISK_THRESHOLD:.2f}
    """
)

st.sidebar.markdown(
    "---"
)

st.sidebar.caption(
    "SafeTriage-GDM is a standalone research prototype."
)


# ==============================================================================
# DATA UPLOAD
# ==============================================================================

st.header(
    "📁 Data Input"
)

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel dataset",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
)

if uploaded_file is None:

    st.info(
        "Upload a dataset to begin inference, evaluation, "
        "or pipeline training."
    )

    st.stop()


# ==============================================================================
# LOAD DATA
# ==============================================================================

try:

    raw_dataframe = load_uploaded_dataframe(
        uploaded_file
    )

except Exception as error:

    st.error(
        f"Unable to read the uploaded dataset: {error}"
    )

    st.stop()


st.success(
    f"Loaded {len(raw_dataframe):,} rows "
    f"and {len(raw_dataframe.columns):,} columns."
)


# ==============================================================================
# PREPARE TARGET
# ==============================================================================

processed_dataframe = (
    raw_dataframe.copy()
)

if TARGET_COLUMN in processed_dataframe.columns:

    processed_dataframe[
        TARGET_COLUMN
    ] = parse_gdm_target(
        processed_dataframe[
            TARGET_COLUMN
        ]
    )

else:

    processed_dataframe[
        TARGET_COLUMN
    ] = np.nan


# ==============================================================================
# ADMINISTRATIVE TRAINING
# ==============================================================================

should_train = (
    force_retrain
    or not os.path.exists(
        ARTIFACT_PATH
    )
)


if should_train:

    labeled_count = int(
        processed_dataframe[
            TARGET_COLUMN
        ].notna().sum()
    )

    if labeled_count == 0:

        if force_retrain:

            st.error(
                "Force Retraining requested, but the uploaded dataset "
                f"contains no valid '{TARGET_COLUMN}' outcomes. "
                "Training cannot proceed without real labeled outcomes."
            )

            st.stop()

        else:

            st.error(
                "No trained SafeTriage-GDM artifact exists and the "
                "uploaded dataset contains no valid GDM outcomes. "
                "Training cannot proceed without real labeled outcomes."
            )

            st.stop()

    labeled_values = (
        processed_dataframe.loc[
            processed_dataframe[
                TARGET_COLUMN
            ].notna(),
            TARGET_COLUMN,
        ]
        .astype(int)
        .unique()
    )

    if len(labeled_values) < 2:

        st.error(
            "Training requires both GDM classes (0 and 1). "
            "The uploaded labeled dataset does not contain both classes."
        )

        st.stop()

    with st.spinner(
        "Training SafeTriage-GDM using the required "
        "three-model architecture..."
    ):

        try:

            artifact = train_pipeline(
                processed_dataframe
            )

            save_artifact(
                artifact
            )

        except Exception as error:

            st.error(
                "Pipeline training failed."
            )

            st.exception(
                error
            )

            st.stop()

    st.success(
        "SafeTriage-GDM pipeline trained and saved successfully."
    )

else:

    try:

        artifact = load_artifact()

    except Exception as error:

        st.error(
            "The existing SafeTriage-GDM artifact could not be loaded "
            "or failed architecture validation."
        )

        st.exception(
            error
        )

        st.info(
            "Use 'Force Pipeline Retraining' after supplying a dataset "
            "containing valid GDM outcomes."
        )

        st.stop()


# ==============================================================================
# ARTIFACT COMPONENTS
# ==============================================================================

imputer = artifact[
    "imputer"
]

scaler = artifact[
    "scaler"
]

calibrated_models = artifact[
    "models"
]

feature_schema = artifact[
    "feature_schema"
]

q_threshold = float(
    artifact[
        "q_threshold"
    ]
)

baseline_bmi_distribution = artifact.get(
    "baseline_bmi_distribution"
)

test_metrics = artifact.get(
    "test_metrics",
    {},
)


# ==============================================================================
# VERIFY EXACT THREE MODELS
# ==============================================================================

if set(
    calibrated_models.keys()
) != EXPECTED_MODELS:

    st.error(
        "Fatal architecture error: SafeTriage-GDM must contain "
        "Random Forest, XGBoost, and Logistic Regression."
    )

    st.stop()


# ==============================================================================
# INFERENCE POPULATION
#
# IMPORTANT:
# ALL uploaded rows are inference rows.
# The presence or absence of a target does not remove a row from inference.
# ==============================================================================

X_inference_raw = (
    processed_dataframe.drop(
        columns=[
            TARGET_COLUMN,
            ID_COLUMN,
        ],
        errors="ignore",
    ).copy()
)

try:

    X_inference_encoded = (
        encode_dataframe(
            X_inference_raw
        )
    )

    X_inference_aligned = (
        align_to_schema(
            X_inference_encoded,
            feature_schema,
        )
    )

    X_inference_imputed = (
        imputer.transform(
            X_inference_aligned
        )
    )

    X_inference_scaled = (
        scaler.transform(
            X_inference_imputed
        )
    )

except Exception as error:

    st.error(
        "Inference preprocessing failed."
    )

    st.exception(
        error
    )

    st.stop()


# ==============================================================================
# THREE-MODEL INFERENCE
# ==============================================================================

try:

    model_probabilities = (
        get_model_probability_matrix(
            calibrated_models,
            X_inference_scaled,
        )
    )

except Exception as error:

    st.error(
        "The three-model ensemble could not generate predictions."
    )

    st.exception(
        error
    )

    st.stop()


if model_probabilities.shape[0] != 3:

    st.error(
        "SafeTriage-GDM requires exactly three ensemble models."
    )

    st.stop()


# ==============================================================================
# UNCERTAINTY DECOMPOSITION
# ==============================================================================

uncertainty = (
    calculate_uncertainty_decomposition(
        model_probabilities
    )
)

consensus_prob = uncertainty[
    "ensemble_probability"
]

predictive_entropy = uncertainty[
    "predictive_entropy"
]

aleatoric_uncertainty = uncertainty[
    "aleatoric_uncertainty"
]

epistemic_uncertainty = uncertainty[
    "epistemic_uncertainty"
]

model_disagreement_std = uncertainty[
    "model_disagreement_std"
]


# ==============================================================================
# RISK CLASSIFICATION
# ==============================================================================

risk_flags = (
    consensus_prob >= RISK_THRESHOLD
).astype(int)

risk_labels = np.where(
    risk_flags == 1,
    "High Risk",
    "Lower Risk",
)


# ==============================================================================
# CONFORMAL PREDICTION SETS
# ==============================================================================

conformal_sets = (
    generate_conformal_sets(
        consensus_prob,
        q_threshold,
    )
)


conformal_set_sizes = np.array(
    [
        2
        if " + " in prediction_set
        else 1
        for prediction_set
        in conformal_sets
    ]
)


# ==============================================================================
# POPULATION DRIFT
# ==============================================================================

current_bmi_distribution = None
bmi_psi = np.nan

if BMI_COLUMN in processed_dataframe.columns:

    current_bmi_distribution = (
        calculate_bmi_distribution(
            processed_dataframe[
                BMI_COLUMN
            ]
        )
    )

    bmi_psi = calculate_psi(
        baseline_bmi_distribution,
        current_bmi_distribution,
    )


mean_predictive_entropy = float(
    np.nanmean(
        predictive_entropy
    )
)

mean_epistemic_uncertainty = float(
    np.nanmean(
        epistemic_uncertainty
    )
)


# ==============================================================================
# RUNTIME CONFORMAL SAFETY ADAPTATION
# ==============================================================================
#
# IMPORTANT:
# This is a monitoring heuristic.
# It does NOT preserve a formal 95% conformal coverage guarantee under
# distribution shift.
# ==============================================================================

adapted_q_threshold = q_threshold

drift_adaptation_applied = False

if (
    np.isfinite(bmi_psi)
    and bmi_psi >= DRIFT_THRESHOLD
    and mean_predictive_entropy > 0.70
):

    adapted_q_threshold = (
        q_threshold * 0.90
    )

    drift_adaptation_applied = True

    conformal_sets = (
        generate_conformal_sets(
            consensus_prob,
            adapted_q_threshold,
        )
    )

    conformal_set_sizes = np.array(
        [
            2
            if " + " in prediction_set
            else 1
            for prediction_set
            in conformal_sets
        ]
    )


# ==============================================================================
# RESULTS DASHBOARD
# ==============================================================================

results_dashboard = pd.DataFrame(
    {
        "GDM Risk Probability": (
            consensus_prob
        ),

        "Risk Classification": (
            risk_labels
        ),

        "Conformal Prediction Set": (
            conformal_sets
        ),

        "Model Positive Flag": (
            risk_flags
        ),

        "RF Probability": (
            model_probabilities[0]
        ),

        "XGBoost Probability": (
            model_probabilities[1]
        ),

        "Logistic Probability": (
            model_probabilities[2]
        ),

        "Epistemic Uncertainty": (
            epistemic_uncertainty
        ),

        "Aleatoric Uncertainty": (
            aleatoric_uncertainty
        ),

        "Predictive Entropy": (
            predictive_entropy
        ),

        "Model Disagreement SD": (
            model_disagreement_std
        ),
    }
)


# ==============================================================================
# FINAL OUTPUT
# ==============================================================================

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


# ==============================================================================
# FAIRNESS
# ==============================================================================

try:

    fairness_metrics = (
        calculate_fairness_metrics(
            processed_dataframe,
            consensus_prob,
        )
    )

except Exception as error:

    st.error(
        "Fairness calculation failed."
    )

    st.exception(
        error
    )

    st.stop()


# ==============================================================================
# TABS
# ==============================================================================

(
    triage_tab,
    xai_tab,
    performance_tab,
) = st.tabs(
    [
        "🩺 Patient Triage Console",
        "🔎 XAI Attribution",
        "📊 Performance, Uncertainty & Fairness",
    ]
)


# ==============================================================================
# TAB 1 — TRIAGE
# ==============================================================================

with triage_tab:

    st.header(
        "Patient Triage Console"
    )

    st.markdown(
        """
        SafeTriage-GDM provides an estimated probability of GDM risk
        together with a conformal prediction set representing predictive
        uncertainty.

        The system separates:

        **Risk estimation** — the ensemble probability of GDM.

        **Triage classification** — the operational lower-risk/high-risk
        classification.

        **Conformal safety bounds** — the uncertainty-aware prediction
        set used to expose ambiguous cases.

        **Model uncertainty** — disagreement and entropy derived from
        the three-model ensemble.
        """
    )

    st.divider()

    # --------------------------------------------------------------------------
    # Population summary
    # --------------------------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Uploaded Rows",
            f"{len(raw_dataframe):,}",
        )

    with col2:
        st.metric(
            "High-Risk Flags",
            f"{int(risk_flags.sum()):,}",
        )

    with col3:
        st.metric(
            "Ambiguous Conformal Sets",
            f"{int(np.sum(conformal_set_sizes == 2)):,}",
        )

    with col4:
        st.metric(
            "Mean GDM Probability",
            f"{100 * np.mean(consensus_prob):.1f}%",
        )

    st.divider()

    # --------------------------------------------------------------------------
    # Individual patient selection
    # --------------------------------------------------------------------------

    st.subheader(
        "Individual Case Inspection"
    )

    selected_index = st.number_input(
        "Select uploaded row",
        min_value=0,
        max_value=max(
            len(final_output_view) - 1,
            0,
        ),
        value=0,
        step=1,
    )

    selected_probability = float(
        consensus_prob[selected_index]
    )

    selected_epistemic = float(
        epistemic_uncertainty[
            selected_index
        ]
    )

    selected_aleatoric = float(
        aleatoric_uncertainty[
            selected_index
        ]
    )

    selected_entropy = float(
        predictive_entropy[
            selected_index
        ]
    )

    selected_set = conformal_sets[
        selected_index
    ]

    selected_risk = risk_labels[
        selected_index
    ]

    st.subheader(
        f"Case {selected_index}"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "GDM Risk Probability",
            f"{100 * selected_probability:.1f}%",
        )

        st.metric(
            "Risk Classification",
            selected_risk,
        )

    with c2:

        st.metric(
            "Conformal Prediction Set",
            selected_set,
        )

        st.metric(
            "Epistemic Uncertainty",
            f"{selected_epistemic:.4f}",
        )

    with c3:

        st.metric(
            "Aleatoric Uncertainty",
            f"{selected_aleatoric:.4f}",
        )

        st.metric(
            "Predictive Entropy",
            f"{selected_entropy:.4f}",
        )

    st.subheader(
        "Three-Model Probability Profile"
    )

    selected_model_table = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "XGBoost",
                "Logistic Regression",
                "3-Model Ensemble",
            ],
            "GDM Probability": [
                model_probabilities[
                    0,
                    selected_index,
                ],
                model_probabilities[
                    1,
                    selected_index,
                ],
                model_probabilities[
                    2,
                    selected_index,
                ],
                selected_probability,
            ],
        }
    )

    selected_model_table[
        "GDM Probability"
    ] = selected_model_table[
        "GDM Probability"
    ].map(
        lambda value: f"{100 * value:.2f}%"
    )

    st.dataframe(
        selected_model_table,
        width="stretch",
        hide_index=True,
    )

    st.subheader(
        "Uploaded Patient Data"
    )

    patient_data = (
        raw_dataframe.iloc[
            [selected_index]
        ].T
    )

    patient_data.columns = [
        "Value"
    ]

    st.dataframe(
        patient_data,
        width="stretch",
    )

    st.divider()

    st.subheader(
        "Full Prediction Dashboard"
    )

    display_dashboard = final_output_view.copy()

    probability_columns = [
        "GDM Risk Probability",
        "RF Probability",
        "XGBoost Probability",
        "Logistic Probability",
        "Epistemic Uncertainty",
        "Aleatoric Uncertainty",
        "Predictive Entropy",
        "Model Disagreement SD",
    ]

    for column in probability_columns:

        if column in display_dashboard.columns:

            display_dashboard[
                column
            ] = display_dashboard[
                column
            ].astype(float).round(4)

    st.dataframe(
        display_dashboard,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------------------------
    # Download
    # --------------------------------------------------------------------------

    csv_output = (
        final_output_view.to_csv(
            index=False
        ).encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Prediction Results",
        data=csv_output,
        file_name="safetriage_gdm_predictions.csv",
        mime="text/csv",
        width="stretch",
    )


# ==============================================================================
# TAB 2 — XAI
# ==============================================================================

with xai_tab:

    st.header(
        "🔎 XAI Attribution"
    )

    st.markdown(
        """
        The attribution analysis summarizes global feature importance
        across the three trained SafeTriage-GDM components.

        Random Forest and XGBoost contribute tree-based feature
        importance, while Logistic Regression contributes the absolute
        magnitude of its fitted coefficients.

        Each model's importance values are normalized before the
        three-model average is calculated.

        These values describe model behaviour within the research
        system. They should **not** be interpreted as causal effects
        or as evidence that a particular characteristic independently
        causes GDM.
        """
    )

    st.divider()

    xai_df = artifact.get(
        "feature_importance"
    )

    if (
        xai_df is None
        or xai_df.empty
    ):

        st.warning(
            "Feature attribution information is unavailable."
        )

    else:

        st.subheader(
            "Global Three-Model Attribution"
        )

        xai_display = (
            xai_df.copy()
        )

        xai_display[
            "Attribution Weight Score"
        ] = xai_display[
            "Attribution Weight Score"
        ].astype(float)

        xai_display[
            "Attribution Weight Score"
        ] = xai_display[
            "Attribution Weight Score"
        ].map(
            lambda value: f"{100 * value:.2f}%"
        )

        st.dataframe(
            xai_display,
            width="stretch",
            hide_index=True,
        )

        st.info(
            "Attribution weights are relative global model-importance "
            "scores, not causal effects."
        )


# ==============================================================================
# TAB 3 — PERFORMANCE, UNCERTAINTY & FAIRNESS
# ==============================================================================

with performance_tab:

    st.header(
        "📊 Performance, Uncertainty & Fairness"
    )

    st.markdown(
        """
        Predictive performance below is calculated from the model's
        **untouched held-out test set** created during training.

        These metrics are therefore distinct from predictions generated
        for the currently uploaded inference population.
        """
    )

    # ==========================================================================
    # PERFORMANCE
    # ==========================================================================

    st.subheader(
        "1. Predictive Performance"
    )

    if test_metrics:

        metric_rows = []

        for model_name in [
            "Random Forest",
            "XGBoost",
            "Logistic Regression",
            "3-Model Ensemble",
        ]:

            if model_name not in test_metrics:
                continue

            model_metrics = (
                test_metrics[
                    model_name
                ]
            )

            metric_rows.append(
                {
                    "Model": model_name,
                    "ROC-AUC": model_metrics.get(
                        "ROC-AUC",
                        np.nan,
                    ),
                    "PR-AUC": model_metrics.get(
                        "PR-AUC",
                        np.nan,
                    ),
                    "Accuracy": model_metrics.get(
                        "Accuracy",
                        np.nan,
                    ),
                    "Sensitivity": model_metrics.get(
                        "Sensitivity",
                        np.nan,
                    ),
                    "Specificity": model_metrics.get(
                        "Specificity",
                        np.nan,
                    ),
                    "Precision": model_metrics.get(
                        "Precision",
                        np.nan,
                    ),
                    "F1-score": model_metrics.get(
                        "F1-score",
                        np.nan,
                    ),
                    "Brier Score": model_metrics.get(
                        "Brier Score",
                        np.nan,
                    ),
                    "Log Loss": model_metrics.get(
                        "Log Loss",
                        np.nan,
                    ),
                }
            )

        metrics_df = pd.DataFrame(
            metric_rows
        )

        st.dataframe(
            metrics_df.style.format(
                {
                    "ROC-AUC": "{:.4f}",
                    "PR-AUC": "{:.4f}",
                    "Accuracy": "{:.4f}",
                    "Sensitivity": "{:.4f}",
                    "Specificity": "{:.4f}",
                    "Precision": "{:.4f}",
                    "F1-score": "{:.4f}",
                    "Brier Score": "{:.4f}",
                    "Log Loss": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    else:

        st.warning(
            "Test-set performance metrics are unavailable."
        )

    # ==========================================================================
    # ENSEMBLE PERFORMANCE SUMMARY
    # ==========================================================================

    if (
        "3-Model Ensemble"
        in test_metrics
    ):

        ensemble_metrics = test_metrics[
            "3-Model Ensemble"
        ]

        st.subheader(
            "3-Model Ensemble Summary"
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric(
                "ROC-AUC",
                f"{ensemble_metrics['ROC-AUC']:.4f}",
            )

        with c2:
            st.metric(
                "PR-AUC",
                f"{ensemble_metrics['PR-AUC']:.4f}",
            )

        with c3:
            st.metric(
                "Brier Score",
                f"{ensemble_metrics['Brier Score']:.4f}",
            )

        with c4:
            st.metric(
                "Sensitivity",
                f"{ensemble_metrics['Sensitivity']:.4f}",
            )

        with c5:
            st.metric(
                "Specificity",
                f"{ensemble_metrics['Specificity']:.4f}",
            )

    # ==========================================================================
    # CONFUSION MATRIX
    # ==========================================================================

    st.subheader(
        "Confusion Matrix — Untouched Test Set"
    )

    if (
        "3-Model Ensemble"
        in test_metrics
    ):

        cm = test_metrics[
            "3-Model Ensemble"
        ][
            "Confusion Matrix"
        ]

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual Healthy",
                "Actual GDM",
            ],
            columns=[
                "Predicted Healthy",
                "Predicted GDM",
            ],
        )

        st.dataframe(
            cm_df,
            width="stretch",
        )

    # ==========================================================================
    # ROC CURVE DATA
    # ==========================================================================

    st.subheader(
        "ROC Curve — Untouched Test Set"
    )

    roc_data = artifact.get(
        "roc_curve"
    )

    if roc_data is not None:

        roc_df = pd.DataFrame(
            {
                "False Positive Rate": (
                    roc_data["fpr"]
                ),
                "True Positive Rate": (
                    roc_data["tpr"]
                ),
                "Threshold": (
                    roc_data["thresholds"]
                ),
            }
        )

        st.line_chart(
            roc_df.set_index(
                "False Positive Rate"
            )[
                "True Positive Rate"
            ],
            width="stretch",
        )

    # ==========================================================================
    # CALIBRATION CURVE
    # ==========================================================================

    st.subheader(
        "Probability Calibration"
    )

    calibration_data = artifact.get(
        "calibration_curve"
    )

    if calibration_data is not None:

        calibration_df = pd.DataFrame(
            {
                "Mean Predicted Probability": (
                    calibration_data[
                        "mean_predicted_value"
                    ]
                ),
                "Observed Positive Fraction": (
                    calibration_data[
                        "fraction_of_positives"
                    ]
                ),
            }
        )

        st.line_chart(
            calibration_df.set_index(
                "Mean Predicted Probability"
            ),
            width="stretch",
        )

        st.caption(
            "A well-calibrated model should produce observed event "
            "frequencies that approximately track predicted probabilities."
        )

    # ==========================================================================
    # UNCERTAINTY
    # ==========================================================================

    st.divider()

    st.subheader(
        "2. Uncertainty Analysis"
    )

    st.markdown(
        """
        SafeTriage-GDM separates uncertainty into several related
        quantities.

        **Epistemic uncertainty:** a model-disagreement / mutual-information-
        style proxy calculated from the three-model ensemble.

        **Aleatoric uncertainty:** expected predictive entropy across the
        individual ensemble probabilities.

        **Predictive entropy:** entropy of the ensemble-averaged GDM
        probability.

        These quantities should be interpreted as uncertainty estimates
        within the finite three-model research ensemble, not as exact
        Bayesian posterior quantities.
        """
    )

    u1, u2, u3, u4 = st.columns(4)

    with u1:
        st.metric(
            "Mean Epistemic",
            f"{mean_epistemic_uncertainty:.4f}",
        )

    with u2:
        st.metric(
            "Mean Aleatoric",
            f"{float(np.nanmean(aleatoric_uncertainty)):.4f}",
        )

    with u3:
        st.metric(
            "Mean Predictive Entropy",
            f"{mean_predictive_entropy:.4f}",
        )

    with u4:
        st.metric(
            "Mean Model Disagreement",
            f"{float(np.nanmean(model_disagreement_std)):.4f}",
        )

    st.subheader(
        "Uncertainty Distribution"
    )

    uncertainty_distribution = pd.DataFrame(
        {
            "Epistemic Uncertainty": (
                epistemic_uncertainty
            ),
            "Aleatoric Uncertainty": (
                aleatoric_uncertainty
            ),
            "Predictive Entropy": (
                predictive_entropy
            ),
        }
    )

    st.line_chart(
        uncertainty_distribution,
        width="stretch",
    )

    st.subheader(
        "Risk Probability vs Uncertainty"
    )

    risk_uncertainty_df = pd.DataFrame(
        {
            "GDM Risk Probability": (
                consensus_prob
            ),
            "Epistemic Uncertainty": (
                epistemic_uncertainty
            ),
            "Aleatoric Uncertainty": (
                aleatoric_uncertainty
            ),
            "Predictive Entropy": (
                predictive_entropy
            ),
        }
    )

    st.dataframe(
        risk_uncertainty_df.describe().T,
        width="stretch",
    )

    # ==========================================================================
    # CONFORMAL SAFETY
    # ==========================================================================

    st.divider()

    st.subheader(
        "3. Conformal Safety Bounds"
    )

    cc1, cc2, cc3, cc4 = st.columns(4)

    with cc1:
        st.metric(
            "Calibration q-threshold",
            f"{q_threshold:.4f}",
        )

    with cc2:
        st.metric(
            "Effective Runtime q-threshold",
            f"{adapted_q_threshold:.4f}",
        )

    with cc3:
        st.metric(
            "Test Coverage",
            (
                f"{100 * artifact.get('test_conformal_coverage', np.nan):.2f}%"
            ),
        )

    with cc4:
        st.metric(
            "Test Singleton Rate",
            (
                f"{100 * artifact.get('test_singleton_rate', np.nan):.2f}%"
            ),
        )

    st.write(
        "Mean test conformal prediction-set size:",
        f"{artifact.get('test_mean_conformal_set_size', np.nan):.3f}",
    )

    st.write(
        "Current uploaded-population ambiguous-set rate:",
        f"{100 * np.mean(conformal_set_sizes == 2):.2f}%",
    )

    if drift_adaptation_applied:

        st.warning(
            "A runtime conformal-threshold adaptation was triggered "
            "because the BMI drift monitor exceeded the monitoring "
            "threshold and predictive entropy was high. This adaptation "
            "is a safety-monitoring heuristic and does NOT preserve a "
            "formal conformal coverage guarantee under distribution shift."
        )

    else:

        st.info(
            "No runtime conformal-threshold adaptation was triggered."
        )

    # ==========================================================================
    # DRIFT
    # ==========================================================================

    st.subheader(
        "4. Population-Shift Monitoring"
    )

    d1, d2, d3 = st.columns(3)

    with d1:

        if np.isfinite(bmi_psi):

            st.metric(
                "BMI PSI",
                f"{bmi_psi:.4f}",
            )

        else:

            st.metric(
                "BMI PSI",
                "Unavailable",
            )

    with d2:

        st.metric(
            "Drift Threshold",
            f"{DRIFT_THRESHOLD:.2f}",
        )

    with d3:

        if (
            np.isfinite(bmi_psi)
            and bmi_psi >= DRIFT_THRESHOLD
        ):

            st.metric(
                "Drift Status",
                "Elevated",
            )

        else:

            st.metric(
                "Drift Status",
                "Within Monitor",
            )

    st.caption(
        "BMI PSI is a population-shift monitoring statistic. "
        "It is not a clinical risk score."
    )

    # ==========================================================================
    # FAIRNESS
    # ==========================================================================

    st.divider()

    st.subheader(
        "5. Algorithmic Fairness Audit"
    )

    st.markdown(
        """
        Demographic parity compares the positive prediction rates of
        the age groups <35 and ≥35 years.

        FPR disparity compares false-positive rates between the same
        groups using only observations with valid GDM ground truth.

        **FPR disparity is not equivalent to full equalized odds**, which
        requires comparison of both false-positive and true-positive rates.
        """
    )

    f1, f2, f3, f4 = st.columns(4)

    with f1:

        value = fairness_metrics[
            "demographic_parity_difference"
        ]

        st.metric(
            "Demographic Parity Difference",
            (
                "Unavailable"
                if np.isnan(value)
                else f"{value:.4f}"
            ),
        )

    with f2:

        value = fairness_metrics[
            "fpr_disparity"
        ]

        st.metric(
            "FPR Disparity",
            (
                "Unavailable"
                if np.isnan(value)
                else f"{value:.4f}"
            ),
        )

    with f3:

        st.metric(
            "Age <35 Positive Rate",
            (
                "Unavailable"
                if np.isnan(
                    fairness_metrics[
                        "younger_positive_rate"
                    ]
                )
                else (
                    f"{100 * fairness_metrics['younger_positive_rate']:.2f}%"
                )
            ),
        )

    with f4:

        st.metric(
            "Age ≥35 Positive Rate",
            (
                "Unavailable"
                if np.isnan(
                    fairness_metrics[
                        "older_positive_rate"
                    ]
                )
                else (
                    f"{100 * fairness_metrics['older_positive_rate']:.2f}%"
                )
            ),
        )

    fairness_table = pd.DataFrame(
        {
            "Metric": [
                "Age <35 sample size",
                "Age ≥35 sample size",
                "Fairness labeled sample size",
                "Age <35 FPR",
                "Age ≥35 FPR",
                "FPR disparity",
            ],
            "Value": [
                fairness_metrics[
                    "younger_sample_size"
                ],
                fairness_metrics[
                    "older_sample_size"
                ],
                fairness_metrics[
                    "fairness_sample_size"
                ],
                fairness_metrics[
                    "younger_fpr"
                ],
                fairness_metrics[
                    "older_fpr"
                ],
                fairness_metrics[
                    "fpr_disparity"
                ],
            ],
        }
    )

    st.dataframe(
        fairness_table.style.format(
            {
                "Value": lambda value: (
                    f"{value:.4f}"
                    if isinstance(
                        value,
                        (float, np.floating),
                    )
                    and np.isfinite(value)
                    else str(value)
                )
            }
        ),
        width="stretch",
        hide_index=True,
    )


# ==============================================================================
# FOOTER
# ==============================================================================

st.divider()

st.caption(
    f"{APP_TITLE} — {APP_VERSION} | "
    "Three-model ensemble: Random Forest + XGBoost + Logistic Regression | "
    "No synthetic outcomes are generated."
)
