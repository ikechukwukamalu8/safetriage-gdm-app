# ==============================================================================
# SafeTriage-GDM
# Uncertainty-Quantified Clinical Triage System for Gestational Diabetes
# Mellitus (GDM) Risk with Algorithmic Fairness Auditing &
# Conformal Safety Bounds
#
# Standalone research prototype
# ==============================================================================

import io
import hashlib
import warnings
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier


# ==============================================================================
# 1. APPLICATION CONFIGURATION
# ==============================================================================

APP_TITLE = "SafeTriage-GDM"

APP_SUBTITLE = (
    "Uncertainty-Quantified Clinical Triage System for "
    "Gestational Diabetes Mellitus (GDM) Risk with "
    "Algorithmic Fairness Auditing & Conformal Safety Bounds"
)

APP_VERSION = "Research Prototype"

RESEARCH_DISCLAIMER = (
    "SafeTriage-GDM is a research prototype for uncertainty-aware GDM risk "
    "triage, conformal safety assessment, population-shift monitoring, and "
    "algorithmic fairness auditing. It is not a medical device and must not "
    "be used as a substitute for professional medical diagnosis, treatment, "
    "or clinical decision-making."
)

TARGET_COLUMN = "Gestational diabetes?"
ID_COLUMN = "Dummy Study Number"

BMI_COLUMN = "Mother's pre-pregnancy BMI (kg/m2)"
AGE_COLUMN = "Mother's age (years)"

ARTIFACT_PATH = Path("safetriage_gdm_pipeline.joblib")

CONFORMAL_LEVEL = 0.95
CONFORMAL_ALPHA = 1.0 - CONFORMAL_LEVEL

RISK_THRESHOLD = 0.50

PSI_WARNING_THRESHOLD = 0.20

RANDOM_STATE = 42


# ==============================================================================
# 2. STRICT ANTEPARTUM PREDICTOR SET
# ==============================================================================

# These variables are deliberately restricted to information that can
# reasonably be available during pregnancy and before delivery.
#
# Post-delivery variables such as infant birth weight, infant BMI,
# gestational age at birth, skinfold measurements, etc. are excluded
# to prevent temporal/data leakage.

PREDICTOR_COLUMNS = [
    "Evidence of maternal anaemia?",
    "Do we have data related to multiple micronutrient supplementation?",
    "Did the mother supplement with multiple micronutrients during pregnancy?",
    "Relative to the start of pregnancy, when did multiple micronutrient supplementation start?",
    "Relative to the start of pregnancy, when did multiple micronutrient supplementation stop?",
    "Did the mothers just supplement with multiple micronutrients during pregnancy and nothing else?",
    "For how many weeks were multiple micronutrients taken?",
    "sex of the baby",
    "Mother's pre-pregnancy BMI (kg/m2)",
    "Mother's height (cm)",
    "Mother's weight before pregnancy (kg)",
    "Mother's pregnancy weight gain (kg)",
    "Mother's age (years)",
    "Did the mother smoke during pregnancy?",
    "Twin pregnancy?",
    "Parity",
]


# Explicitly excluded variables are documented for transparency.
POST_DELIVERY_OR_LEAKAGE_COLUMNS = [
    "Premature birth?",
    "Low birth weight baby?",
    "Age of baby (days) when the birth assessments were done",
    "Baby's BMI at birth (kg/m2)",
    "Baby birth weight (kg)",
    "Baby's head circumference at birth (cm)",
    "Baby's flank skinfolds thickness at birth",
    "Baby's gestational age at birth (weeks)",
    "Baby's length at birth (cm)",
    "Baby's ponderal index at birth (kg/m3)",
    "Baby's quadriceps skinfolds thickness at birth",
    "Baby's subscapular skinfolds thickness at birth",
    "Baby's triceps skinfolds thickness at birth",
    "Was the baby born small for gestational age?",
]


# ==============================================================================
# 3. MODEL ARCHITECTURE
# ==============================================================================

EXPECTED_MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
]

EXPECTED_MODEL_KEYS = set(EXPECTED_MODEL_ORDER)


# ==============================================================================
# 4. PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
)


# ==============================================================================
# 5. UTILITY FUNCTIONS
# ==============================================================================

def normalize_binary_target(series):
    """
    Convert common binary GDM encodings to 0/1.

    Unknown or missing values become NaN.
    """
    positive_values = {
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

    negative_values = {
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

    result = []

    for value in series:
        if pd.isna(value):
            result.append(np.nan)
            continue

        if isinstance(value, (int, float, np.integer, np.floating)):
            if float(value) == 1:
                result.append(1)
                continue

            if float(value) == 0:
                result.append(0)
                continue

        text = str(value).strip().lower()

        if text in positive_values:
            result.append(1)
        elif text in negative_values:
            result.append(0)
        else:
            result.append(np.nan)

    return pd.Series(result, index=series.index, dtype="float64")


def find_column(df, candidates):
    """
    Case-insensitive exact column matching.
    """
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = str(candidate).strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def find_bmi_column(df):
    """
    Robust BMI detection.

    The supplied CBGS dataset uses:
        Mother's pre-pregnancy BMI (kg/m2)
    """
    candidates = [
        "Mother's pre-pregnancy BMI (kg/m2)",
        "Mother's pre-pregnancy BMI (kg/m²)",
        "Pre-pregnancy BMI",
        "Prepregnancy BMI",
        "Pre-pregnancy Body Mass Index",
        "Prepregnancy Body Mass Index",
        "BMI",
        "Body Mass Index",
        "body mass index",
    ]

    exact = find_column(df, candidates)

    if exact is not None:
        return exact

    # Conservative fallback based on normalized text.
    for column in df.columns:
        normalized = (
            str(column)
            .strip()
            .lower()
            .replace("²", "2")
        )

        if "bmi" in normalized and "mother" in normalized:
            return column

        if "body mass index" in normalized and "mother" in normalized:
            return column

    return None


def encode_dataframe(df):
    """
    One-hot encode categorical predictors.

    The resulting dataframe is numeric and suitable for imputation/scaling.
    """
    encoded = pd.get_dummies(
        df,
        drop_first=True,
        dummy_na=False,
    )

    return encoded.astype(np.float64)


def align_to_schema(encoded_df, feature_schema):
    """
    Align inference data to the exact training feature schema.
    """
    aligned = encoded_df.reindex(
        columns=feature_schema,
        fill_value=0.0,
    )

    return aligned.astype(np.float64)


def get_positive_probability(model, X):
    """
    Return P(GDM) for a binary classifier.
    """
    probabilities = model.predict_proba(X)

    if probabilities.ndim == 2:
        return probabilities[:, 1]

    return probabilities.ravel()


def binary_entropy(probabilities):
    """
    Bernoulli entropy in nats.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0 - 1e-12,
    )

    return -(
        probabilities * np.log(probabilities)
        + (1.0 - probabilities)
        * np.log(1.0 - probabilities)
    )


def calculate_uncertainty(model_probabilities):
    """
    Decompose ensemble predictive uncertainty.

    Epistemic uncertainty:
        Standard deviation across model probabilities.

    Aleatoric proxy:
        Mean Bernoulli entropy across models.

    Predictive entropy:
        Entropy of ensemble mean probability.

    Mutual information:
        Predictive entropy - expected entropy.
    """
    probabilities = np.asarray(
        model_probabilities,
        dtype=float,
    )

    consensus = np.mean(
        probabilities,
        axis=0,
    )

    epistemic_std = np.std(
        probabilities,
        axis=0,
        ddof=0,
    )

    individual_entropy = binary_entropy(
        probabilities
    )

    aleatoric_entropy = np.mean(
        individual_entropy,
        axis=0,
    )

    predictive_entropy = binary_entropy(
        consensus
    )

    mutual_information = np.maximum(
        predictive_entropy - aleatoric_entropy,
        0.0,
    )

    return {
        "consensus_probability": consensus,
        "epistemic_std": epistemic_std,
        "aleatoric_entropy": aleatoric_entropy,
        "predictive_entropy": predictive_entropy,
        "mutual_information": mutual_information,
    }


# ==============================================================================
# 6. CONFORMAL PREDICTION
# ==============================================================================

def calculate_conformal_threshold(
    calibration_probabilities,
    calibration_labels,
    confidence_level=CONFORMAL_LEVEL,
):
    """
    Split-conformal nonconformity threshold.

    Score:
        1 - probability assigned to the true class.

    The finite-sample quantile uses the conservative higher quantile.
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

    scores = 1.0 - true_class_probability

    n = len(scores)

    if n == 0:
        raise ValueError(
            "The calibration set is empty."
        )

    quantile_level = min(
        1.0,
        np.ceil(
            (n + 1) * confidence_level
        ) / n,
    )

    q_threshold = np.quantile(
        scores,
        quantile_level,
        method="higher",
    )

    return float(
        np.clip(q_threshold, 0.0, 1.0)
    )


def conformal_prediction_sets(
    probabilities,
    q_threshold,
):
    """
    Construct binary conformal prediction sets.

    Class 0 is included when:
        1 - P(GDM) <= q

    Class 1 is included when:
        P(GDM) <= q
    """
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    include_no_gdm = (
        1.0 - probabilities
    ) <= q_threshold

    include_gdm = (
        probabilities
    ) <= q_threshold

    return include_no_gdm, include_gdm


def conformal_coverage(
    probabilities,
    labels,
    q_threshold,
):
    """
    Empirical test-set coverage.
    """
    include_no_gdm, include_gdm = (
        conformal_prediction_sets(
            probabilities,
            q_threshold,
        )
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    covered = np.where(
        labels == 1,
        include_gdm,
        include_no_gdm,
    )

    return float(
        np.mean(covered)
    )


# ==============================================================================
# 7. MODEL PERFORMANCE
# ==============================================================================

def calculate_binary_metrics(
    y_true,
    probabilities,
    threshold=RISK_THRESHOLD,
):
    """
    Calculate threshold-independent and threshold-dependent metrics.
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

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "ROC-AUC": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "PR-AUC": float(
            average_precision_score(
                y_true,
                probabilities,
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
        "F1 Score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def calculate_roc_data(y_true, probabilities):
    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities,
    )

    return pd.DataFrame(
        {
            "False Positive Rate": fpr,
            "True Positive Rate": tpr,
            "Threshold": thresholds,
        }
    )


def calculate_pr_data(y_true, probabilities):
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    return pd.DataFrame(
        {
            "Recall": recall,
            "Precision": precision,
        }
    )


def calculate_calibration_data(
    y_true,
    probabilities,
    n_bins=8,
):
    fraction_positive
