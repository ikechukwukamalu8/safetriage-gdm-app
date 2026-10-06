# ==============================================================================
# SafeTriage-GDM
# Uncertainty-Quantified Clinical Triage System for Gestational Diabetes
# Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds
#
# Standalone research prototype
# Environment: Streamlit
# ==============================================================================

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# Optional scientific imports
# ------------------------------------------------------------------------------

try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
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
        precision_recall_curve,
    )
    from sklearn.calibration import calibration_curve
except Exception as exc:
    st.error("A required scikit-learn component could not be imported.")
    st.exception(exc)
    st.stop()

try:
    from xgboost import XGBClassifier
except Exception as exc:
    st.error("XGBoost could not be imported.")
    st.exception(exc)
    st.stop()


# ==============================================================================
# 1. APPLICATION CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="SafeTriage-GDM",
    page_icon="🩺",
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

DATASET_NAME = "dataCBGS_dataset.xlsx"

ARTIFACT_PATH = Path("safetriage_gdm_pipeline.joblib")

RANDOM_STATE = 42

TRAIN_FRACTION = 0.60
CALIBRATION_FRACTION = 0.15
TEST_FRACTION = 0.25

CONFORMAL_CONFIDENCE = 0.90
CONFORMAL_Q_THRESHOLD = 0.20

BMI_PSI_THRESHOLD = 0.20

RISK_THRESHOLD = 0.50

MIN_FAIRNESS_GROUP_SIZE = 10

TARGET_COLUMN = "Gestational diabetes?"

BMI_COLUMN = "Mother's pre-pregnancy BMI (kg/m2)"
AGE_COLUMN = "Mother's age (years)"

# ------------------------------------------------------------------------------

EXPECTED_MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
]

EXPECTED_MODEL_KEYS = set(EXPECTED_MODEL_ORDER)

# ==============================================================================
# 2. ANTEPARTUM PREDICTOR SCHEMA
# ==============================================================================

# These variables are deliberately restricted to maternal/pregnancy information.
#
# Baby-at-birth measurements and other post-delivery outcomes are excluded.
#
# The application therefore does NOT use:
#   - baby's birth weight
#   - baby's BMI at birth
#   - baby's gestational age at birth
#   - premature birth
#   - low birth weight
#   - small-for-gestational-age status
#   - baby sex
#   - etc.
#
# NOTE:
# This is an antepartum research model. Variables such as pregnancy weight gain
# and pregnancy micronutrient exposure are only valid if they are available at
# the intended prediction time. The application explicitly documents this
# assumption rather than silently treating post-index variables as baseline.

ANTEpartum_PREDICTORS = [
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
    "Mother's pregnancy weight gain (kg)",
    "Mother's age (years)",
    "Did the mother smoke during pregnancy?",
    "Twin pregnancy?",
    "Parity",
]

# Exact known post-delivery/leakage variables in this dataset.
POST_DELIVERY_COLUMNS = [
    "sex of the baby",
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

# Outcome-adjacent maternal variables are intentionally excluded from the
# baseline predictor list because their timing can be later in pregnancy.
TIMING_SENSITIVE_EXCLUDED = [
    "Gestational hypertension?",
    "Evidence of pre-eclampsia?",
]

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


# ==============================================================================
# 3. GENERAL UTILITIES
# ==============================================================================

def safe_float(value: Any) -> float:
    """Convert a value to float safely."""
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def dataset_hash(df: pd.DataFrame) -> str:
    """Create a deterministic hash for artifact/data compatibility."""
    try:
        payload = pd.util.hash_pandas_object(
            df,
            index=True,
        ).values.tobytes()

        return hashlib.sha256(payload).hexdigest()

    except Exception:
        return hashlib.sha256(
            df.to_csv(index=True).encode("utf-8", errors="ignore")
        ).hexdigest()


def normalize_target(value: Any) -> Optional[int]:
    """
    Convert common binary target representations to:
        0 = No
        1 = Yes
    """
    if pd.isna(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return 1
        if value == 0:
            return 0

    text = str(value).strip().lower()

    positive = {
        "yes",
        "y",
        "true",
        "1",
        "positive",
        "present",
        "diabetic",
    }

    negative = {
        "no",
        "n",
        "false",
        "0",
        "negative",
        "absent",
        "non-diabetic",
        "non diabetic",
    }

    if text in positive:
        return 1

    if text in negative:
        return 0

    return None


def prepare_target(df: pd.DataFrame) -> pd.Series:
    """Parse the GDM target."""
    return df[TARGET_COLUMN].apply(normalize_target).astype("Int64")


def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert mixed-type clinical data into numeric one-hot encoded features.
    """
    encoded = pd.get_dummies(
        df,
        drop_first=True,
        dummy_na=False,
    )

    return encoded.astype(np.float64)


def align_to_schema(
    encoded_df: pd.DataFrame,
    feature_schema: List[str],
) -> pd.DataFrame:
    """
    Align inference data to the exact training feature schema.
    """
    return (
        encoded_df
        .reindex(columns=feature_schema, fill_value=0.0)
        .astype(np.float64)
    )


def bernoulli_entropy(probability: float) -> float:
    """Binary entropy in nats."""
    p = float(np.clip(probability, 1e-12, 1 - 1e-12))

    return float(
        -(p * math.log(p) + (1 - p) * math.log(1 - p))
    )


def format_pct(value: Any) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"

    return f"{100 * float(value):.1f}%"


def format_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"

    try:
        if not np.isfinite(value):
            return "N/A"

        return f"{float(value):.{digits}f}"

    except Exception:
        return "N/A"


# ==============================================================================
# 4. TARGET / DATA VALIDATION
# ==============================================================================

def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate the minimum schema required by the application."""
    errors = []

    if TARGET_COLUMN not in df.columns:
        errors.append(
            f"Required target column is missing: `{TARGET_COLUMN}`"
        )

    available_predictors = [
        col for col in ANTEpartum_PREDICTORS
        if col in df.columns
    ]

    if len(available_predictors) < 5:
        errors.append(
            "Fewer than five approved antepartum predictors are available."
        )

    if BMI_COLUMN not in df.columns:
        errors.append(
            f"BMI column is missing: `{BMI_COLUMN}`"
        )

    if AGE_COLUMN not in df.columns:
        errors.append(
            f"Age column is missing: `{AGE_COLUMN}`"
        )

    return len(errors) == 0, errors


def get_available_predictors(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in ANTEpartum_PREDICTORS
        if col in df.columns
    ]


# ==============================================================================
# 5. DATA SPLITTING
# ==============================================================================

def split_data(
    X_raw: pd.DataFrame,
    y: pd.Series,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Exact 60/15/25 train/calibration/test split.

    First:
        75% development + 25% untouched test

    Second:
        80% train + 20% calibration of the 75% development set

    Result:
        60% train
        15% calibration
        25% test
    """

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_temp,
        y_train_temp,
        test_size=CALIBRATION_FRACTION
        / (TRAIN_FRACTION + CALIBRATION_FRACTION),
        random_state=RANDOM_STATE,
        stratify=y_train_temp,
    )

    return (
        X_train,
        X_cal,
        X_test,
        y_train,
        y_cal,
        y_test,
    )


# ==============================================================================
# 6. MODEL TRAINING
# ==============================================================================

def build_models(y_train: pd.Series) -> Dict[str, Any]:
    """
    Build exactly three required models.

    No SMOTETomek.
    No post-hoc sigmoid calibration.

    Class weighting is used to address class imbalance.
    """

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())

    if positive_count == 0:
        raise ValueError("Training set contains no positive GDM cases.")

    if negative_count == 0:
        raise ValueError("Training set contains no negative GDM cases.")

    scale_pos_weight = negative_count / positive_count

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
            min_samples_leaf=3,
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
            scale_pos_weight=scale_pos_weight,
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    return models


def fit_pipeline(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Train the complete SafeTriage-GDM pipeline.
    """

    available_predictors = get_available_predictors(df)

    if len(available_predictors) < 5:
        raise ValueError(
            "Insufficient approved predictors available in the dataset."
        )

    target = prepare_target(df)

    labeled_mask = target.notna()

    labeled_df = df.loc[labeled_mask].copy()
    y = target.loc[labeled_mask].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The labeled dataset must contain both GDM-positive and "
            "GDM-negative observations."
        )

    X_raw = labeled_df[available_predictors].copy()

    (
        X_train_raw,
        X_cal_raw,
        X_test_raw,
        y_train,
        y_cal,
        y_test,
    ) = split_data(
        X_raw,
        y,
    )

    # --------------------------------------------------------------------------
    # Encoding
    # --------------------------------------------------------------------------

    X_train_encoded = encode_dataframe(X_train_raw)

    feature_schema = list(X_train_encoded.columns)

    X_cal_encoded = align_to_schema(
        encode_dataframe(X_cal_raw),
        feature_schema,
    )

    X_test_encoded = align_to_schema(
        encode_dataframe(X_test_raw),
        feature_schema,
    )

    # --------------------------------------------------------------------------
    # Imputation
    # --------------------------------------------------------------------------

    imputer = IterativeImputer(
        random_state=RANDOM_STATE,
        max_iter=15,
        initial_strategy="median",
        skip_complete=True,
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
    # Scaling
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
    # Models
    # --------------------------------------------------------------------------

    models = build_models(y_train)

    for name in EXPECTED_MODEL_ORDER:
        models[name].fit(
            X_train_scaled,
            y_train,
        )

    if set(models.keys()) != EXPECTED_MODEL_KEYS:
        raise RuntimeError(
            "Model architecture validation failed. "
            "Exactly Random Forest, XGBoost and Logistic Regression "
            "are required."
        )

    # --------------------------------------------------------------------------
    # Calibration predictions
    # --------------------------------------------------------------------------

    calibration_probabilities = {
        name: model.predict_proba(X_cal_scaled)[:, 1]
        for name, model in models.items()
    }

    calibration_ensemble = np.mean(
        np.column_stack(
            [
                calibration_probabilities[name]
                for name in EXPECTED_MODEL_ORDER
            ]
        ),
        axis=1,
    )

    # --------------------------------------------------------------------------
    # Conformal calibration
    # --------------------------------------------------------------------------

    true_class_probability = np.where(
        y_cal.to_numpy() == 1,
        calibration_ensemble,
        1.0 - calibration_ensemble,
    )

    nonconformity_scores = 1.0 - true_class_probability

    n_calibration = len(nonconformity_scores)

    q_level = min(
        1.0,
        np.ceil(
            (n_calibration + 1) * CONFORMAL_CONFIDENCE
        ) / n_calibration,
    )

    try:
        conformal_q = float(
            np.quantile(
                nonconformity_scores,
                q_level,
                method="higher",
            )
        )
    except TypeError:
        conformal_q = float(
            np.quantile(
                nonconformity_scores,
                q_level,
                interpolation="higher",
            )
        )

    # --------------------------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------------------------

    test_probabilities = {
        name: model.predict_proba(X_test_scaled)[:, 1]
        for name, model in models.items()
    }

    test_probability_matrix = np.column_stack(
        [
            test_probabilities[name]
            for name in EXPECTED_MODEL_ORDER
        ]
    )

    ensemble_test_probability = test_probability_matrix.mean(
        axis=1
    )

    # --------------------------------------------------------------------------
    # Artifact
    # --------------------------------------------------------------------------

    artifact = {
        "artifact_version": "2.0",
        "random_state": RANDOM_STATE,
        "target_column": TARGET_COLUMN,
        "predictors": available_predictors,
        "feature_schema": feature_schema,
        "models": models,
        "imputer": imputer,
        "scaler": scaler,
        "conformal_q": conformal_q,
        "conformal_confidence": CONFORMAL_CONFIDENCE,
        "conformal_q_level": q_level,
        "calibration_scores": nonconformity_scores,
        "train_size": len(y_train),
        "calibration_size": len(y_cal),
        "test_size": len(y_test),
        "dataset_hash": dataset_hash(df),
        "training_distribution": {
            "negative": int((y_train == 0).sum()),
            "positive": int((y_train == 1).sum()),
        },
        "model_order": EXPECTED_MODEL_ORDER,
    }

    return {
        "artifact": artifact,
        "X_test_raw": X_test_raw,
        "y_test": y_test,
        "test_probabilities": test_probabilities,
        "ensemble_test_probability": ensemble_test_probability,
        "X_train_raw": X_train_raw,
        "y_train": y_train,
        "X_cal_raw": X_cal_raw,
        "y_cal": y_cal,
        "available_predictors": available_predictors,
    }


# ==============================================================================
# 7. ARTIFACT LOADING / SAVING
# ==============================================================================

def save_artifact(artifact: Dict[str, Any]) -> None:
    """
    Save and immediately validate the joblib artifact.
    """
    joblib.dump(
        artifact,
        ARTIFACT_PATH,
        compress=3,
    )

    # Immediate round-trip validation.
    loaded = joblib.load(ARTIFACT_PATH)

    if set(loaded.get("models", {}).keys()) != EXPECTED_MODEL_KEYS:
        raise RuntimeError(
            "Saved artifact failed model architecture validation."
        )


def artifact_is_valid_for_dataset(
    artifact: Any,
    df: pd.DataFrame,
) -> bool:
    """Validate artifact structure and dataset compatibility."""

    if not isinstance(artifact, dict):
        return False

    if artifact.get("artifact_version") != "2.0":
        return False

    if artifact.get("target_column") != TARGET_COLUMN:
        return False

    models = artifact.get("models")

    if not isinstance(models, dict):
        return False

    if set(models.keys()) != EXPECTED_MODEL_KEYS:
        return False

    if artifact.get("dataset_hash") != dataset_hash(df):
        return False

    if not artifact.get("feature_schema"):
        return False

    if not artifact.get("predictors"):
        return False

    return True


def load_or_train(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    # --------------------------------------------------------------------------
    # Attempt artifact reuse
    # --------------------------------------------------------------------------

    if ARTIFACT_PATH.exists():

        try:
            artifact = joblib.load(
                ARTIFACT_PATH
            )

            if artifact_is_valid_for_dataset(
                artifact,
                df,
            ):
                return {
                    "artifact": artifact,
                    "reused_artifact": True,
                }

        except Exception:
            # Invalid/corrupt artifacts are ignored and rebuilt.
            pass

    # --------------------------------------------------------------------------
    # Train a new pipeline
    # --------------------------------------------------------------------------

    result = fit_pipeline(df)

    artifact = result["artifact"]

    save_artifact(artifact)

    return {
        **result,
        "reused_artifact": False,
    }


# ==============================================================================
# 8. INFERENCE
# ==============================================================================

def transform_for_inference(
    df: pd.DataFrame,
    artifact: Dict[str, Any],
) -> np.ndarray:

    predictors = artifact["predictors"]
    feature_schema = artifact["feature_schema"]

    X = df[predictors].copy()

    encoded = encode_dataframe(X)

    aligned = align_to_schema(
        encoded,
        feature_schema,
    )

    imputed = artifact["imputer"].transform(
        aligned
    )

    scaled = artifact["scaler"].transform(
        imputed
    )

    return scaled


def predict_population(
    df: pd.DataFrame,
    artifact: Dict[str, Any],
) -> pd.DataFrame:

    X_scaled = transform_for_inference(
        df,
        artifact,
    )

    model_probabilities = {}

    for name in EXPECTED_MODEL_ORDER:
        model = artifact["models"][name]

        model_probabilities[name] = (
            model.predict_proba(X_scaled)[:, 1]
        )

    probability_matrix = np.column_stack(
        [
            model_probabilities[name]
            for name in EXPECTED_MODEL_ORDER
        ]
    )

    ensemble_probability = probability_matrix.mean(
        axis=1
    )

    epistemic_std = probability_matrix.std(
        axis=1,
        ddof=0,
    )

    aleatoric_entropy = np.mean(
        [
            np.array(
                [
                    bernoulli_entropy(p)
                    for p in model_probabilities[name]
                ]
            )
            for name in EXPECTED_MODEL_ORDER
        ],
        axis=0,
    )

    predictive_entropy = np.array(
        [
            bernoulli_entropy(p)
            for p in ensemble_probability
        ]
    )

    mutual_information = np.maximum(
        predictive_entropy - aleatoric_entropy,
        0.0,
    )

    conformal_q = float(
        artifact["conformal_q"]
    )

    # A prediction is considered conformally "safe/decisive" when the
    # nonconformity of the predicted class does not exceed q.
    predicted_class = (
        ensemble_probability >= RISK_THRESHOLD
    ).astype(int)

    predicted_class_probability = np.where(
        predicted_class == 1,
        ensemble_probability,
        1.0 - ensemble_probability,
    )

    nonconformity = (
        1.0 - predicted_class_probability
    )

    conformal_valid = (
        nonconformity <= conformal_q
    )

    result = pd.DataFrame(
        {
            "P_RF": model_probabilities["Random Forest"],
            "P_XGB": model_probabilities["XGBoost"],
            "P_LR": model_probabilities["Logistic Regression"],
            "P_GDM": ensemble_probability,
            "Predicted_Class": predicted_class,
            "Epistemic_STD": epistemic_std,
            "Aleatoric_Entropy": aleatoric_entropy,
            "Predictive_Entropy": predictive_entropy,
            "Mutual_Information": mutual_information,
            "Conformal_Nonconformity": nonconformity,
            "Conformal_Safe": conformal_valid,
        },
        index=df.index,
    )

    return result


# ==============================================================================
# 9. PERFORMANCE METRICS
# ==============================================================================

def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = RISK_THRESHOLD,
) -> Dict[str, Any]:

    y_true_np = np.asarray(y_true).astype(int)

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_np,
        predictions,
        labels=[0, 1],
    ).ravel()

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

    try:
        roc_auc = roc_auc_score(
            y_true_np,
            probabilities,
        )
    except Exception:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_true_np,
            probabilities,
        )
    except Exception:
        pr_auc = np.nan

    try:
        brier = brier_score_loss(
            y_true_np,
            probabilities,
        )
    except Exception:
        brier = np.nan

    try:
        ll = log_loss(
            y_true_np,
            probabilities,
            labels=[0, 1],
        )
    except Exception:
        ll = np.nan

    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Brier Score": brier,
        "Log Loss": ll,
        "Accuracy": accuracy_score(
            y_true_np,
            predictions,
        ),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision_score(
            y_true_np,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true_np,
            predictions,
            zero_division=0,
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def build_performance_table(
    y_true: pd.Series,
    probability_dict: Dict[str, np.ndarray],
) -> pd.DataFrame:

    rows = []

    for name in EXPECTED_MODEL_ORDER:

        metrics = calculate_metrics(
            y_true,
            probability_dict[name],
        )

        rows.append(
            {
                "Model": name,
                "ROC-AUC": metrics["ROC-AUC"],
                "PR-AUC": metrics["PR-AUC"],
                "Brier": metrics["Brier Score"],
                "Log Loss": metrics["Log Loss"],
                "Sensitivity": metrics["Sensitivity"],
                "Specificity": metrics["Specificity"],
                "Precision": metrics["Precision"],
                "F1": metrics["F1"],
            }
        )

    ensemble_probability = np.mean(
        np.column_stack(
            [
                probability_dict[name]
                for name in EXPECTED_MODEL_ORDER
            ]
        ),
        axis=1,
    )

    metrics = calculate_metrics(
        y_true,
        ensemble_probability,
    )

    rows.append(
        {
            "Model": "Ensemble",
            "ROC-AUC": metrics["ROC-AUC"],
            "PR-AUC": metrics["PR-AUC"],
            "Brier": metrics["Brier Score"],
            "Log Loss": metrics["Log Loss"],
            "Sensitivity": metrics["Sensitivity"],
            "Specificity": metrics["Specificity"],
            "Precision": metrics["Precision"],
            "F1": metrics["F1"],
        }
    )

    return pd.DataFrame(rows)


# ==============================================================================
# 10. CONFORMAL TEST COVERAGE
# ==============================================================================

def conformal_test_summary(
    y_true: pd.Series,
    probabilities: np.ndarray,
    q: float,
) -> Dict[str, Any]:

    y_np = np.asarray(
        y_true
    ).astype(int)

    p = np.asarray(
        probabilities
    )

    true_class_probability = np.where(
        y_np == 1,
        p,
        1.0 - p,
    )

    nonconformity = (
        1.0 - true_class_probability
    )

    covered = (
        nonconformity <= q
    )

    return {
        "coverage": float(
            np.mean(covered)
        ),
        "target_confidence": CONFORMAL_CONFIDENCE,
        "q": float(q),
        "n": len(y_np),
        "covered": int(
            np.sum(covered)
        ),
    }


# ==============================================================================
# 11. FAIRNESS
# ==============================================================================

def age_group(age: Any) -> Optional[str]:

    value = safe_float(age)

    if not np.isfinite(value):
        return None

    if value < 25:
        return "<25"

    if value < 35:
        return "25–34"

    if value < 45:
        return "35–44"

    return "45+"


def fairness_analysis(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    working = pd.DataFrame(
        {
            "Age": df[AGE_COLUMN].apply(
                safe_float
            ),
            "Prediction": predictions[
                "Predicted_Class"
            ].astype(float),
        },
        index=df.index,
    )

    working["Age_Group"] = working["Age"].apply(
        age_group
    )

    # --------------------------------------------------------------------------
    # Selection-rate / demographic-parity style analysis
    # --------------------------------------------------------------------------

    selection_rows = []

    for group, group_df in working.dropna(
        subset=["Age_Group"]
    ).groupby("Age_Group"):

        n = len(group_df)

        if n < MIN_FAIRNESS_GROUP_SIZE:
            selection_rate = np.nan
        else:
            selection_rate = group_df[
                "Prediction"
            ].mean()

        selection_rows.append(
            {
                "Age Group": group,
                "N": n,
                "Selection Rate": selection_rate,
            }
        )

    selection_table = pd.DataFrame(
        selection_rows
    )

    # --------------------------------------------------------------------------
    # FPR analysis requires ground truth.
    # --------------------------------------------------------------------------

    if TARGET_COLUMN not in df.columns:
        return (
            selection_table,
            pd.DataFrame(),
        )

    ground_truth = df[TARGET_COLUMN].apply(
        normalize_target
    )

    fpr_working = working.copy()

    fpr_working["Truth"] = ground_truth

    fpr_rows = []

    for group, group_df in fpr_working.dropna(
        subset=["Age_Group", "Truth"]
    ).groupby("Age_Group"):

        n = len(group_df)

        negatives = group_df[
            group_df["Truth"] == 0
        ]

        if (
            n < MIN_FAIRNESS_GROUP_SIZE
            or len(negatives) == 0
        ):
            fpr = np.nan
        else:
            fpr = (
                (
                    negatives["Prediction"] == 1
                ).sum()
                / len(negatives)
            )

        fpr_rows.append(
            {
                "Age Group": group,
                "N with Truth": n,
                "Negative Cases": len(negatives),
                "FPR": fpr,
            }
        )

    fpr_table = pd.DataFrame(
        fpr_rows
    )

    return (
        selection_table,
        fpr_table,
    )


# ==============================================================================
# 12. BMI POPULATION SHIFT / PSI
# ==============================================================================

def bmi_category(
    bmi: Any,
) -> Optional[str]:

    value = safe_float(bmi)

    if not np.isfinite(value):
        return None

    category = pd.cut(
        [value],
        bins=BMI_BINS,
        labels=BMI_LABELS,
        include_lowest=True,
    )

    return str(category[0])


def calculate_bmi_psi(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Tuple[float, pd.DataFrame]:

    baseline = baseline_df[
        BMI_COLUMN
    ].apply(safe_float)

    current = current_df[
        BMI_COLUMN
    ].apply(safe_float)

    baseline = baseline[
        np.isfinite(baseline)
    ]

    current = current[
        np.isfinite(current)
    ]

    if len(baseline) == 0 or len(current) == 0:
        return np.nan, pd.DataFrame()

    baseline_categories = pd.cut(
        baseline,
        bins=BMI_BINS,
        labels=BMI_LABELS,
        include_lowest=True,
    )

    current_categories = pd.cut(
        current,
        bins=BMI_BINS,
        labels=BMI_LABELS,
        include_lowest=True,
    )

    baseline_counts = (
        baseline_categories
        .value_counts()
        .reindex(BMI_LABELS, fill_value=0)
    )

    current_counts = (
        current_categories
        .value_counts()
        .reindex(BMI_LABELS, fill_value=0)
    )

    epsilon = 1e-6

    expected = (
        baseline_counts / baseline_counts.sum()
    ).astype(float)

    actual = (
        current_counts / current_counts.sum()
    ).astype(float)

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

    psi_components = (
        (actual - expected)
        * np.log(actual / expected)
    )

    psi = float(
        psi_components.sum()
    )

    table = pd.DataFrame(
        {
            "BMI Group": BMI_LABELS,
            "Baseline Proportion": expected.values,
            "Current Proportion": actual.values,
            "PSI Component": psi_components.values,
        }
    )

    return psi, table


# ==============================================================================
# 13. XAI FEATURE CONTRIBUTIONS
# ==============================================================================

def aggregate_feature_contributions(
    artifact: Dict[str, Any],
) -> pd.DataFrame:
    """
    Estimate global feature contribution across RF, XGB and LR.

    RF/XGB:
        feature_importances_

    Logistic regression:
        absolute standardized coefficient magnitude

    One-hot encoded features are aggregated back to their parent
    clinical variable.
    """

    schema = artifact["feature_schema"]

    model_contributions = []

    # --------------------------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------------------------

    rf = artifact["models"]["Random Forest"]

    rf_values = np.asarray(
        rf.feature_importances_,
        dtype=float,
    )

    if rf_values.sum() > 0:
        rf_values = rf_values / rf_values.sum()

    model_contributions.append(
        pd.Series(
            rf_values,
            index=schema,
            name="Random Forest",
        )
    )

    # --------------------------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------------------------

    xgb = artifact["models"]["XGBoost"]

    xgb_values = np.asarray(
        xgb.feature_importances_,
        dtype=float,
    )

    if xgb_values.sum() > 0:
        xgb_values = xgb_values / xgb_values.sum()

    model_contributions.append(
        pd.Series(
            xgb_values,
            index=schema,
            name="XGBoost",
        )
    )

    # --------------------------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------------------------

    lr = artifact["models"]["Logistic Regression"]

    lr_values = np.abs(
        np.asarray(
            lr.coef_[0],
            dtype=float,
        )
    )

    if lr_values.sum() > 0:
        lr_values = lr_values / lr_values.sum()

    model_contributions.append(
        pd.Series(
            lr_values,
            index=schema,
            name="Logistic Regression",
        )
    )

    raw_table = pd.concat(
        model_contributions,
        axis=1,
    ).fillna(0)

    raw_table["Mean Contribution"] = (
        raw_table.mean(axis=1)
    )

    # --------------------------------------------------------------------------
    # Aggregate one-hot columns back to original predictor.
    # --------------------------------------------------------------------------

    predictors = artifact["predictors"]

    rows = []

    for predictor in predictors:

        matching = [
            column
            for column in schema
            if (
                column == predictor
                or column.startswith(
                    predictor + "_"
                )
            )
        ]

        if not matching:
            continue

        subset = raw_table.loc[
            matching
        ]

        rows.append(
            {
                "Feature": predictor,
                "Random Forest": subset[
                    "Random Forest"
                ].sum(),
                "XGBoost": subset[
                    "XGBoost"
                ].sum(),
                "Logistic Regression": subset[
                    "Logistic Regression"
                ].sum(),
                "Mean Contribution": subset[
                    "Mean Contribution"
                ].sum(),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result = result.sort_values(
        "Mean Contribution",
        ascending=False,
    ).reset_index(drop=True)

    return result


# ==============================================================================
# 14. SESSION STATE
# ==============================================================================

if "pipeline_bundle" not in st.session_state:
    st.session_state.pipeline_bundle = None

if "population_predictions" not in st.session_state:
    st.session_state.population_predictions = None

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "dataset_source" not in st.session_state:
    st.session_state.dataset_source = None


# ==============================================================================
# 15. HEADER
# ==============================================================================

st.title(APP_TITLE)

st.markdown(
    f"### {APP_SUBTITLE}"
)

st.caption(
    f"{APP_VERSION} • Three-model ensemble • "
    f"60/15/25 train/calibration/test design"
)

st.warning(
    "SafeTriage-GDM is a research prototype for uncertainty-aware GDM "
    "risk triage, conformal safety assessment, population-shift monitoring, "
    "and algorithmic fairness auditing. It is not a medical device and must "
    "not be used as a substitute for professional medical diagnosis, "
    "treatment, or clinical decision-making."
)


# ==============================================================================
# 16. SIDEBAR
# ==============================================================================

with st.sidebar:

    st.header("System Configuration")

    st.write(
        "Required model architecture:"
    )

    for model_name in EXPECTED_MODEL_ORDER:
        st.write(f"✓ {model_name}")

    st.divider()

    st.write(
        f"Risk threshold: **{RISK_THRESHOLD:.2f}**"
    )

    st.write(
        f"Conformal confidence: "
        f"**{CONFORMAL_CONFIDENCE:.0%}**"
    )

    st.write(
        f"BMI PSI threshold: "
        f"**{BMI_PSI_THRESHOLD:.2f}**"
    )

    st.divider()

    st.caption(
        "The operational threshold is a research setting, "
        "not a clinical diagnostic threshold."
    )


# ==============================================================================
# 17. DATA UPLOAD
# ==============================================================================

st.header("Data Input")

uploaded_file = st.file_uploader(
    "Upload the CBGS dataset",
    type=["xlsx", "xls"],
    help=(
        "Expected dataset containing the column "
        "`Gestational diabetes?`."
    ),
)

# Automatically use repository dataset if present.
default_dataset_path = Path(DATASET_NAME)

if uploaded_file is not None:

    try:
        df = pd.read_excel(
            uploaded_file
        )

        st.session_state.dataset_source = (
            "Uploaded file"
        )

    except Exception as exc:

        st.error(
            "The uploaded Excel file could not be read."
        )

        st.exception(exc)

        st.stop()

elif default_dataset_path.exists():

    try:
        df = pd.read_excel(
            default_dataset_path
        )

        st.session_state.dataset_source = (
            f"Repository file: {DATASET_NAME}"
        )

    except Exception as exc:

        st.error(
            f"`{DATASET_NAME}` exists but could not be read."
        )

        st.exception(exc)

        st.stop()

else:

    st.info(
        "Upload `dataCBGS_dataset.xlsx` to initialise "
        "the SafeTriage-GDM research pipeline."
    )

    st.stop()


# ==============================================================================
# 18. VALIDATE DATASET
# ==============================================================================

valid_dataset, validation_errors = validate_dataset(
    df
)

if not valid_dataset:

    st.error(
        "The supplied dataset does not satisfy the minimum "
        "SafeTriage-GDM schema."
    )

    for error in validation_errors:
        st.error(error)

    st.stop()

st.session_state.dataset = df


# ==============================================================================
# 19. DATASET SUMMARY
# ==============================================================================

target_series = prepare_target(df)

n_total = len(df)
n_labeled = int(
    target_series.notna().sum()
)
n_unlabeled = int(
    target_series.isna().sum()
)

n_positive = int(
    (target_series == 1).sum()
)

n_negative = int(
    (target_series == 0).sum()
)

available_predictors = get_available_predictors(
    df
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Uploaded Rows",
        f"{n_total:,}",
    )

with col2:
    st.metric(
        "Labeled Rows",
        f"{n_labeled:,}",
    )

with col3:
    st.metric(
        "GDM Positive",
        f"{n_positive:,}",
    )

with col4:
    st.metric(
        "Unlabeled Rows",
        f"{n_unlabeled:,}",
    )

st.caption(
    "All uploaded rows remain available for inference. "
    "Only rows with a valid GDM ground truth are used for supervised "
    "training and test evaluation."
)


# ==============================================================================
# 20. INITIALISE / TRAIN PIPELINE
# ==============================================================================

if st.session_state.pipeline_bundle is None:

    with st.spinner(
        "Initialising the SafeTriage-GDM three-model pipeline..."
    ):

        try:

            bundle = load_or_train(
                df
            )

            st.session_state.pipeline_bundle = (
                bundle
            )

        except Exception as exc:

            st.error(
                "SafeTriage-GDM could not initialise the model pipeline."
            )

            st.exception(exc)

            st.info(
                "This diagnostic output is intentionally displayed instead "
                "of allowing the application to fail silently."
            )

            st.stop()


bundle = st.session_state.pipeline_bundle

artifact = bundle["artifact"]

reused_artifact = bundle.get(
    "reused_artifact",
    False,
)

if reused_artifact:

    st.success(
        "Verified SafeTriage-GDM model artifact loaded."
    )

else:

    st.success(
        "SafeTriage-GDM model trained and artifact validated."
    )


# ==============================================================================
# 21. RUN POPULATION PREDICTIONS
# ==============================================================================

if (
    st.session_state.population_predictions
    is None
):

    try:

        predictions = predict_population(
            df,
            artifact,
        )

        st.session_state.population_predictions = (
            predictions
        )

    except Exception as exc:

        st.error(
            "Population inference failed."
        )

        st.exception(exc)

        st.stop()


predictions = (
    st.session_state.population_predictions
)


# ==============================================================================
# 22. TABS
# ==============================================================================

(
    tab_triage,
    tab_performance,
    tab_uncertainty,
    tab_fairness,
    tab_xai,
) = st.tabs(
    [
        "🩺 Patient Triage Console",
        "📈 Predictive Performance",
        "🛡️ Uncertainty & Conformal Safety",
        "⚖️ Fairness & Population Shift",
        "🔎 XAI Attribution",
    ]
)


# ==============================================================================
# TAB 1 — PATIENT TRIAGE CONSOLE
# ==============================================================================

with tab_triage:

    st.header(
        "Patient Triage Console"
    )

    st.write(
        "Select an observation from the uploaded dataset to inspect its "
        "ensemble GDM risk, model agreement, uncertainty, and conformal "
        "safety status."
    )

    patient_index = st.selectbox(
        "Select patient / observation",
        options=list(df.index),
        format_func=lambda x: f"Observation {x + 1}",
    )

    patient_row = df.loc[
        [patient_index]
    ]

    patient_prediction = predictions.loc[
        patient_index
    ]

    probability = float(
        patient_prediction["P_GDM"]
    )

    predicted_class = int(
        patient_prediction["Predicted_Class"]
    )

    conformal_safe = bool(
        patient_prediction["Conformal_Safe"]
    )

    if predicted_class == 1:
        st.error(
            f"Research risk classification: elevated "
            f"({probability:.1%} ensemble probability)"
        )
    else:
        st.success(
            f"Research risk classification: lower "
            f"({probability:.1%} ensemble probability)"
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Ensemble P(GDM)",
            f"{probability:.1%}",
        )

    with col2:
        st.metric(
            "RF",
            f"{patient_prediction['P_RF']:.1%}",
        )

    with col3:
        st.metric(
            "XGBoost",
            f"{patient_prediction['P_XGB']:.1%}",
        )

    with col4:
        st.metric(
            "Logistic Regression",
            f"{patient_prediction['P_LR']:.1%}",
        )

    st.subheader(
        "Uncertainty"
    )

    u1, u2, u3 = st.columns(3)

    with u1:
        st.metric(
            "Model-disagreement SD",
            format_metric(
                patient_prediction[
                    "Epistemic_STD"
                ]
            ),
        )

    with u2:
        st.metric(
            "Predictive Entropy",
            format_metric(
                patient_prediction[
                    "Predictive_Entropy"
                ]
            ),
        )

    with u3:
        st.metric(
            "Mutual Information",
            format_metric(
                patient_prediction[
                    "Mutual_Information"
                ]
            ),
        )

    st.subheader(
        "Conformal Safety"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Nonconformity",
            format_metric(
                patient_prediction[
                    "Conformal_Nonconformity"
                ]
            ),
        )

    with c2:

        if conformal_safe:
            st.success(
                "Predicted class is inside the conformal safety bound."
            )
        else:
            st.warning(
                "Prediction does not satisfy the conformal safety bound."
            )

    st.subheader(
        "Patient Features"
    )

    display_patient = patient_row[
        available_predictors
    ].T

    display_patient.columns = [
        "Value"
    ]

    st.dataframe(
        display_patient,
        width="stretch",
    )

    st.caption(
        "Feature contribution and uncertainty measures are research "
        "analytics and do not establish causality or diagnosis."
    )


# ==============================================================================
# TAB 2 — PREDICTIVE PERFORMANCE
# ==============================================================================

with tab_performance:

    st.header(
        "Predictive Performance"
    )

    y_test = bundle["y_test"]

    test_probability_dict = bundle[
        "test_probabilities"
    ]

    performance_table = build_performance_table(
        y_test,
        test_probability_dict,
    )

    st.subheader(
        "Untouched Test-Set Performance"
    )

    st.dataframe(
        performance_table.round(4),
        width="stretch",
        hide_index=True,
    )

    ensemble_test_probability = bundle[
        "ensemble_test_probability"
    ]

    ensemble_metrics = calculate_metrics(
        y_test,
        ensemble_test_probability,
    )

    st.subheader(
        "Ensemble Metrics"
    )

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "ROC-AUC",
        format_metric(
            ensemble_metrics["ROC-AUC"]
        ),
    )

    metric_columns[1].metric(
        "PR-AUC",
        format_metric(
            ensemble_metrics["PR-AUC"]
        ),
    )

    metric_columns[2].metric(
        "Brier",
        format_metric(
            ensemble_metrics["Brier Score"]
        ),
    )

    metric_columns[3].metric(
        "Sensitivity",
        format_pct(
            ensemble_metrics["Sensitivity"]
        ),
    )

    metric_columns[4].metric(
        "Specificity",
        format_pct(
            ensemble_metrics["Specificity"]
        ),
    )

    st.subheader(
        "Confusion Matrix"
    )

    cm = np.array(
        [
            [
                ensemble_metrics["TN"],
                ensemble_metrics["FP"],
            ],
            [
                ensemble_metrics["FN"],
                ensemble_metrics["TP"],
            ],
        ]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual No",
            "Actual Yes",
        ],
        columns=[
            "Predicted No",
            "Predicted Yes",
        ],
    )

    st.dataframe(
        cm_df,
        width="stretch",
    )

    # --------------------------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------------------------

    st.subheader(
        "ROC Curve"
    )

    try:

        fpr, tpr, _ = roc_curve(
            np.asarray(y_test).astype(int),
            ensemble_test_probability,
        )

        roc_df = pd.DataFrame(
            {
                "False Positive Rate": fpr,
                "True Positive Rate": tpr,
            }
        )

        st.line_chart(
            roc_df,
            x="False Positive Rate",
            y="True Positive Rate",
            width="stretch",
        )

    except Exception:
        st.info(
            "ROC curve could not be calculated for this test split."
        )

    # --------------------------------------------------------------------------
    # Precision-recall curve
    # --------------------------------------------------------------------------

    st.subheader(
        "Precision–Recall Curve"
    )

    try:

        precision, recall, _ = precision_recall_curve(
            np.asarray(y_test).astype(int),
            ensemble_test_probability,
        )

        pr_df = pd.DataFrame(
            {
                "Recall": recall,
                "Precision": precision,
            }
        )

        st.line_chart(
            pr_df,
            x="Recall",
            y="Precision",
            width="stretch",
        )

    except Exception:
        st.info(
            "Precision–recall curve could not be calculated."
        )

    # --------------------------------------------------------------------------
    # Calibration curve
    # --------------------------------------------------------------------------

    st.subheader(
        "Descriptive Calibration Curve"
    )

    try:

        fraction_positive, mean_predicted = calibration_curve(
            np.asarray(y_test).astype(int),
            ensemble_test_probability,
            n_bins=5,
            strategy="quantile",
        )

        calibration_df = pd.DataFrame(
            {
                "Mean Predicted Probability": mean_predicted,
                "Observed Positive Fraction": fraction_positive,
            }
        )

        st.line_chart(
            calibration_df,
            x="Mean Predicted Probability",
            y="Observed Positive Fraction",
            width="stretch",
        )

    except Exception:
        st.info(
            "Calibration curve could not be calculated."
        )

    st.info(
        "The test set is untouched during model fitting. The displayed "
        "probabilities are ensemble predictive scores; the calibration "
        "curve is descriptive and does not imply guaranteed probability "
        "calibration."
    )


# ==============================================================================
# TAB 3 — UNCERTAINTY & CONFORMAL SAFETY
# ==============================================================================

with tab_uncertainty:

    st.header(
        "Uncertainty & Conformal Safety"
    )

    q = float(
        artifact["conformal_q"]
    )

    conformal_test = conformal_test_summary(
        bundle["y_test"],
        bundle["ensemble_test_probability"],
        q,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Conformal q",
            format_metric(q),
        )

    with col2:
        st.metric(
            "Target Confidence",
            f"{CONFORMAL_CONFIDENCE:.0%}",
        )

    with col3:
        st.metric(
            "Observed Test Coverage",
            f"{conformal_test['coverage']:.1%}",
        )

    st.subheader(
        "Conformal Calibration"
    )

    st.write(
        f"Calibration sample size: "
        f"**{artifact['calibration_size']}**"
    )

    st.write(
        f"Finite-sample quantile level: "
        f"**{artifact['conformal_q_level']:.4f}**"
    )

    st.write(
        "The conformal threshold is estimated from the dedicated "
        "calibration partition and is not altered by BMI drift or "
        "prediction-time uncertainty."
    )

    st.subheader(
        "Uncertainty Decomposition"
    )

    uncertainty_df = predictions[
        [
            "P_GDM",
            "Epistemic_STD",
            "Aleatoric_Entropy",
            "Predictive_Entropy",
            "Mutual_Information",
            "Conformal_Nonconformity",
        ]
    ].copy()

    st.dataframe(
        uncertainty_df.head(100).round(4),
        width="stretch",
    )

    st.subheader(
        "Interpretation"
    )

    st.markdown(
        """
- **Epistemic STD:** disagreement among the three deterministic models,
  used here as a model-disagreement proxy.
- **Aleatoric entropy:** average Bernoulli uncertainty across the models.
- **Predictive entropy:** entropy of the ensemble probability.
- **Mutual information:** predictive entropy minus mean model entropy,
  truncated at zero.
- **Conformal nonconformity:** `1 − probability of the predicted class`.
- **Conformal safety:** whether the predicted class satisfies the calibrated
  conformal bound.
"""
    )

    st.warning(
        "These quantities quantify model uncertainty; they do not establish "
        "clinical certainty or diagnostic validity."
    )


# ==============================================================================
# TAB 4 — FAIRNESS & POPULATION SHIFT
# ==============================================================================

with tab_fairness:

    st.header(
        "Fairness & Population Shift"
    )

    st.subheader(
        "Age-Stratified Selection Rates"
    )

    selection_table, fpr_table = fairness_analysis(
        df,
        predictions,
    )

    if selection_table.empty:

        st.info(
            "No valid age groups are available for fairness analysis."
        )

    else:

        st.dataframe(
            selection_table.round(4),
            width="stretch",
            hide_index=True,
        )

        valid_selection = selection_table[
            "Selection Rate"
        ].dropna()

        if len(valid_selection) >= 2:

            selection_gap = (
                valid_selection.max()
                - valid_selection.min()
            )

            st.metric(
                "Selection-rate disparity",
                format_pct(
                    selection_gap
                ),
            )

    st.subheader(
        "False Positive Rate Disparity"
    )

    if fpr_table.empty:

        st.info(
            "Ground-truth data are unavailable for FPR analysis."
        )

    else:

        st.dataframe(
            fpr_table.round(4),
            width="stretch",
            hide_index=True,
        )

        valid_fpr = fpr_table[
            "FPR"
        ].dropna()

        if len(valid_fpr) >= 2:

            fpr_disparity = (
                valid_fpr.max()
                - valid_fpr.min()
            )

            st.metric(
                "FPR disparity",
                format_pct(
                    fpr_disparity
                ),
            )

            st.caption(
                "FPR disparity is the maximum minus minimum group FPR. "
                "It is not labelled Equalized Odds Disparity because "
                "Equalized Odds also requires comparison of true-positive rates."
            )

    st.divider()

    st.subheader(
        "BMI Population Shift"
    )

    psi, bmi_table = calculate_bmi_psi(
        bundle["X_train_raw"],
        df,
    )

    if np.isfinite(psi):

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "BMI PSI",
                format_metric(psi),
            )

        with col2:

            if psi >= BMI_PSI_THRESHOLD:

                st.error(
                    "Population shift flag: BMI distribution exceeds "
                    "the predefined PSI threshold."
                )

            else:

                st.success(
                    "No substantial BMI population shift detected "
                    "under the predefined PSI threshold."
                )

        st.dataframe(
            bmi_table.round(4),
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "BMI population shift could not be calculated because "
            "insufficient valid BMI values were available."
        )

    st.caption(
        "BMI PSI is a monitoring signal only. It does not alter the "
        "conformal q value, model probabilities, or predictions."
    )


# ==============================================================================
# TAB 5 — XAI
# ==============================================================================

with tab_xai:

    st.header(
        "XAI Attribution"
    )

    st.write(
        "The figure below estimates global feature contribution across "
        "the Random Forest, XGBoost and Logistic Regression models."
    )

    contribution_table = (
        aggregate_feature_contributions(
            artifact
        )
    )

    if contribution_table.empty:

        st.warning(
            "Feature contribution analysis could not be generated."
        )

    else:

        top_n = st.slider(
            "Number of features to display",
            min_value=5,
            max_value=min(
                15,
                len(contribution_table),
            ),
            value=min(
                10,
                len(contribution_table),
            ),
        )

        plot_df = contribution_table.head(
            top_n
        ).copy()

        plot_df = plot_df.sort_values(
            "Mean Contribution",
            ascending=True,
        )

        st.subheader(
            "Global Feature Contribution"
        )

        st.bar_chart(
            plot_df.set_index(
                "Feature"
            )[
                "Mean Contribution"
            ],
            width="stretch",
        )

        st.subheader(
            "Model-Specific Contributions"
        )

        st.dataframe(
            contribution_table.head(
                top_n
            ).round(4),
            width="stretch",
            hide_index=True,
        )

        st.info(
            "These are model-based global contribution measures. "
            "They are not causal effects and should not be interpreted "
            "as evidence that changing a feature will change GDM risk."
        )


# ==============================================================================
# 23. MODEL / PIPELINE INFORMATION
# ==============================================================================

with st.expander(
    "Pipeline Diagnostics & Reproducibility"
):

    st.write(
        "**Dataset source:**",
        st.session_state.dataset_source,
    )

    st.write(
        "**Rows:**",
        n_total,
    )

    st.write(
        "**Labeled rows:**",
        n_labeled,
    )

    st.write(
        "**Predictor count:**",
        len(available_predictors),
    )

    st.write(
        "**Training rows:**",
        artifact["train_size"],
    )

    st.write(
        "**Calibration rows:**",
        artifact["calibration_size"],
    )

    st.write(
        "**Test rows:**",
        artifact["test_size"],
    )

    st.write(
        "**Model architecture:**",
        ", ".join(EXPECTED_MODEL_ORDER),
    )

    st.write(
        "**Artifact:**",
        str(ARTIFACT_PATH),
    )

    st.write(
        "**Artifact version:**",
        artifact["artifact_version"],
    )

    st.write(
        "**Dataset hash:**",
        artifact["dataset_hash"],
    )

    st.subheader(
        "Approved Antepartum Predictors"
    )

    for predictor in artifact["predictors"]:
        st.write(f"- {predictor}")

    st.subheader(
        "Explicitly Excluded Post-Delivery Variables"
    )

    for variable in POST_DELIVERY_COLUMNS:
        if variable in df.columns:
            st.write(f"- {variable}")

    st.subheader(
        "Timing-Sensitive Variables Excluded from Baseline Model"
    )

    for variable in TIMING_SENSITIVE_EXCLUDED:
        if variable in df.columns:
            st.write(f"- {variable}")

    st.caption(
        "The predictor whitelist is deliberately explicit so that "
        "post-delivery variables cannot silently enter the model."
    )


# ==============================================================================
# 24. FOOTER
# ==============================================================================

st.divider()

st.caption(
    "SafeTriage-GDM • Research Prototype • "
    "Uncertainty Quantification • Conformal Safety • "
    "Fairness Auditing • Population Shift Monitoring • XAI"
)

st.caption(
    "Not for clinical diagnosis, treatment, or autonomous medical decision-making."
)
