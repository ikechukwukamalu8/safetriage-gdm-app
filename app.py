# ==============================================================================
# SafeTriage-GDM
# Uncertainty-Quantified Clinical Triage System for Gestational Diabetes
# Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal Safety Bounds
#
# Standalone Research Prototype
# ==============================================================================

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ==============================================================================
# 1. REQUIRED MACHINE-LEARNING IMPORTS
# ==============================================================================

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
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
    )
    from sklearn.calibration import calibration_curve

except Exception as exc:
    st.error(
        "A required scikit-learn component could not be imported."
    )
    st.exception(exc)
    st.stop()


try:
    from xgboost import XGBClassifier

except Exception as exc:
    st.error(
        "XGBoost could not be imported."
    )
    st.exception(exc)
    st.stop()


# ==============================================================================
# 2. STREAMLIT CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="SafeTriage-GDM",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 3. APPLICATION CONSTANTS
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

AGE_COLUMN = "Mother's age (years)"

RANDOM_STATE = 42

TRAIN_FRACTION = 0.60
CALIBRATION_FRACTION = 0.15
TEST_FRACTION = 0.25

CONFORMAL_CONFIDENCE = 0.90

RISK_THRESHOLD = 0.50

BMI_PSI_THRESHOLD = 0.20

MIN_FAIRNESS_GROUP_SIZE = 10


# ==============================================================================
# 4. REQUIRED MODEL ARCHITECTURE
# ==============================================================================

EXPECTED_MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
]

EXPECTED_MODEL_KEYS = set(
    EXPECTED_MODEL_ORDER
)


# ==============================================================================
# 5. APPROVED ANTEPARTUM PREDICTORS
# ==============================================================================

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


# ==============================================================================
# 6. EXPLICITLY EXCLUDED POST-DELIVERY VARIABLES
# ==============================================================================

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


# Outcome-adjacent variables are excluded from the baseline predictor schema.
TIMING_SENSITIVE_EXCLUDED = [
    "Gestational hypertension?",
    "Evidence of pre-eclampsia?",
]


# ==============================================================================
# 7. BMI BINS
# ==============================================================================

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
# 8. GENERAL UTILITIES
# ==============================================================================

def safe_float(value: Any) -> float:
    """Safely convert a value to float."""

    try:
        if pd.isna(value):
            return np.nan

        return float(value)

    except Exception:
        return np.nan


def dataframe_signature(df: pd.DataFrame) -> str:
    """
    Create a deterministic signature for the current dataset.

    This ensures that uploading a different Excel file with the same
    columns triggers a new model-training run.
    """

    try:
        hashed = pd.util.hash_pandas_object(
            df,
            index=True,
        ).values.tobytes()

        return hashlib.sha256(
            hashed
        ).hexdigest()

    except Exception:
        return hashlib.sha256(
            df.to_csv(
                index=True
            ).encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()


def normalize_target(
    value: Any,
) -> Optional[int]:
    """
    Convert common binary target representations to:
        0 = No
        1 = Yes
    """

    if pd.isna(value):
        return None

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return int(value)

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):

        if value == 1:
            return 1

        if value == 0:
            return 0

    text = str(
        value
    ).strip().lower()

    positive_values = {
        "yes",
        "y",
        "true",
        "1",
        "positive",
        "present",
        "diabetic",
    }

    negative_values = {
        "no",
        "n",
        "false",
        "0",
        "negative",
        "absent",
        "non-diabetic",
        "non diabetic",
    }

    if text in positive_values:
        return 1

    if text in negative_values:
        return 0

    return None


def prepare_target(
    df: pd.DataFrame,
) -> pd.Series:

    return (
        df[TARGET_COLUMN]
        .apply(normalize_target)
        .astype("Int64")
    )


def encode_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert mixed-type variables to numeric one-hot encoded variables.
    """

    encoded = pd.get_dummies(
        df,
        drop_first=True,
        dummy_na=False,
    )

    return encoded.astype(
        np.float64
    )


def align_to_schema(
    encoded_df: pd.DataFrame,
    feature_schema: List[str],
) -> pd.DataFrame:

    return (
        encoded_df
        .reindex(
            columns=feature_schema,
            fill_value=0.0,
        )
        .astype(
            np.float64
        )
    )


def bernoulli_entropy(
    probability: float,
) -> float:
    """Binary entropy in nats."""

    p = float(
        np.clip(
            probability,
            1e-12,
            1 - 1e-12,
        )
    )

    return float(
        -(
            p * math.log(p)
            + (1 - p)
            * math.log(1 - p)
        )
    )


def format_metric(
    value: Any,
    digits: int = 3,
) -> str:

    try:

        if (
            value is None
            or not np.isfinite(value)
        ):
            return "N/A"

        return (
            f"{float(value):.{digits}f}"
        )

    except Exception:
        return "N/A"


def format_pct(
    value: Any,
) -> str:

    try:

        if (
            value is None
            or not np.isfinite(value)
        ):
            return "N/A"

        return (
            f"{100 * float(value):.1f}%"
        )

    except Exception:
        return "N/A"


# ==============================================================================
# 9. DATASET VALIDATION
# ==============================================================================

def validate_dataset(
    df: pd.DataFrame,
) -> Tuple[bool, List[str]]:

    errors = []

    if TARGET_COLUMN not in df.columns:

        errors.append(
            f"Required target column is missing: "
            f"`{TARGET_COLUMN}`"
        )

    available_predictors = [
        column
        for column
        in ANTEpartum_PREDICTORS
        if column in df.columns
    ]

    if len(available_predictors) < 5:

        errors.append(
            "Fewer than five approved antepartum predictors "
            "are available in the uploaded dataset."
        )

    if BMI_COLUMN not in df.columns:

        errors.append(
            f"Required BMI column is missing: "
            f"`{BMI_COLUMN}`"
        )

    if AGE_COLUMN not in df.columns:

        errors.append(
            f"Required maternal age column is missing: "
            f"`{AGE_COLUMN}`"
        )

    return (
        len(errors) == 0,
        errors,
    )


def get_available_predictors(
    df: pd.DataFrame,
) -> List[str]:

    return [
        column
        for column
        in ANTEpartum_PREDICTORS
        if column in df.columns
    ]


# ==============================================================================
# 10. DATA SPLITTING
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
    """

    (
        X_train_temp,
        X_test,
        y_train_temp,
        y_test,
    ) = train_test_split(
        X_raw,
        y,
        test_size=TEST_FRACTION,
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
        test_size=0.20,
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
# 11. MODEL CONSTRUCTION
# ==============================================================================

def build_models(
    y_train: pd.Series,
) -> Dict[str, Any]:

    positive_count = int(
        (y_train == 1).sum()
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    if positive_count == 0:

        raise ValueError(
            "The training set contains no positive GDM cases."
        )

    if negative_count == 0:

        raise ValueError(
            "The training set contains no negative GDM cases."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    models = {

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
                min_samples_leaf=3,
            ),

        "XGBoost":
            XGBClassifier(
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

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
    }

    return models


# ==============================================================================
# 12. TRAIN COMPLETE PIPELINE
# ==============================================================================

def train_pipeline(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Train the complete SafeTriage-GDM pipeline.
    """

    predictors = get_available_predictors(
        df
    )

    if len(predictors) < 5:

        raise ValueError(
            "Insufficient approved predictors are available."
        )

    target = prepare_target(
        df
    )

    labeled_mask = (
        target.notna()
    )

    labeled_df = df.loc[
        labeled_mask
    ].copy()

    y = (
        target.loc[
            labeled_mask
        ]
        .astype(int)
    )

    if y.nunique() < 2:

        raise ValueError(
            "The labeled dataset must contain both GDM-positive "
            "and GDM-negative observations."
        )

    X_raw = labeled_df[
        predictors
    ].copy()

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
    # One-hot encoding
    # --------------------------------------------------------------------------

    X_train_encoded = encode_dataframe(
        X_train_raw
    )

    feature_schema = list(
        X_train_encoded.columns
    )

    X_cal_encoded = align_to_schema(
        encode_dataframe(
            X_cal_raw
        ),
        feature_schema,
    )

    X_test_encoded = align_to_schema(
        encode_dataframe(
            X_test_raw
        ),
        feature_schema,
    )

    # --------------------------------------------------------------------------
    # Iterative imputation
    # --------------------------------------------------------------------------

    imputer = IterativeImputer(
        random_state=RANDOM_STATE,
        max_iter=15,
        initial_strategy="median",
        skip_complete=True,
    )

    X_train_imputed = (
        imputer.fit_transform(
            X_train_encoded
        )
    )

    X_cal_imputed = (
        imputer.transform(
            X_cal_encoded
        )
    )

    X_test_imputed = (
        imputer.transform(
            X_test_encoded
        )
    )

    # --------------------------------------------------------------------------
    # Standardisation
    # --------------------------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train_imputed
        )
    )

    X_cal_scaled = (
        scaler.transform(
            X_cal_imputed
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test_imputed
        )
    )

    # --------------------------------------------------------------------------
    # Train exactly three models
    # --------------------------------------------------------------------------

    models = build_models(
        y_train
    )

    for model_name in EXPECTED_MODEL_ORDER:

        models[
            model_name
        ].fit(
            X_train_scaled,
            y_train,
        )

    if (
        set(models.keys())
        != EXPECTED_MODEL_KEYS
    ):

        raise RuntimeError(
            "Model architecture validation failed."
        )

    # --------------------------------------------------------------------------
    # Calibration probabilities
    # --------------------------------------------------------------------------

    calibration_probability_matrix = (
        np.column_stack(
            [
                models[name]
                .predict_proba(
                    X_cal_scaled
                )[:, 1]

                for name
                in EXPECTED_MODEL_ORDER
            ]
        )
    )

    calibration_ensemble = (
        calibration_probability_matrix
        .mean(axis=1)
    )

    # --------------------------------------------------------------------------
    # Conformal calibration
    # --------------------------------------------------------------------------

    y_cal_array = (
        np.asarray(
            y_cal
        ).astype(int)
    )

    true_class_probability = np.where(
        y_cal_array == 1,
        calibration_ensemble,
        1.0 - calibration_ensemble,
    )

    nonconformity_scores = (
        1.0
        - true_class_probability
    )

    n_calibration = len(
        nonconformity_scores
    )

    q_level = min(
        1.0,
        np.ceil(
            (
                n_calibration + 1
            )
            * CONFORMAL_CONFIDENCE
        )
        / n_calibration,
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
    # Test probabilities
    # --------------------------------------------------------------------------

    test_probability_dict = {

        name:
            models[name]
            .predict_proba(
                X_test_scaled
            )[:, 1]

        for name
        in EXPECTED_MODEL_ORDER
    }

    test_probability_matrix = (
        np.column_stack(
            [
                test_probability_dict[name]
                for name
                in EXPECTED_MODEL_ORDER
            ]
        )
    )

    ensemble_test_probability = (
        test_probability_matrix
        .mean(axis=1)
    )

    return {

        "models":
            models,

        "imputer":
            imputer,

        "scaler":
            scaler,

        "feature_schema":
            feature_schema,

        "predictors":
            predictors,

        "conformal_q":
            conformal_q,

        "conformal_confidence":
            CONFORMAL_CONFIDENCE,

        "conformal_q_level":
            q_level,

        "calibration_scores":
            nonconformity_scores,

        "train_size":
            len(y_train),

        "calibration_size":
            len(y_cal),

        "test_size":
            len(y_test),

        "X_train_raw":
            X_train_raw,

        "X_cal_raw":
            X_cal_raw,

        "X_test_raw":
            X_test_raw,

        "y_train":
            y_train,

        "y_cal":
            y_cal,

        "y_test":
            y_test,

        "test_probabilities":
            test_probability_dict,

        "ensemble_test_probability":
            ensemble_test_probability,
    }


# ==============================================================================
# 13. INFERENCE
# ==============================================================================

def transform_for_inference(
    df: pd.DataFrame,
    pipeline: Dict[str, Any],
) -> np.ndarray:

    predictors = pipeline[
        "predictors"
    ]

    feature_schema = pipeline[
        "feature_schema"
    ]

    X = df[
        predictors
    ].copy()

    encoded = encode_dataframe(
        X
    )

    aligned = align_to_schema(
        encoded,
        feature_schema,
    )

    imputed = pipeline[
        "imputer"
    ].transform(
        aligned
    )

    scaled = pipeline[
        "scaler"
    ].transform(
        imputed
    )

    return scaled


def predict_population(
    df: pd.DataFrame,
    pipeline: Dict[str, Any],
) -> pd.DataFrame:

    X_scaled = transform_for_inference(
        df,
        pipeline,
    )

    model_probabilities = {}

    for model_name in EXPECTED_MODEL_ORDER:

        model = pipeline[
            "models"
        ][model_name]

        model_probabilities[
            model_name
        ] = (
            model.predict_proba(
                X_scaled
            )[:, 1]
        )

    probability_matrix = (
        np.column_stack(
            [
                model_probabilities[name]
                for name
                in EXPECTED_MODEL_ORDER
            ]
        )
    )

    ensemble_probability = (
        probability_matrix
        .mean(axis=1)
    )

    epistemic_std = (
        probability_matrix
        .std(
            axis=1,
            ddof=0,
        )
    )

    aleatoric_entropy = np.mean(
        [
            np.array(
                [
                    bernoulli_entropy(
                        p
                    )
                    for p
                    in model_probabilities[
                        model_name
                    ]
                ]
            )

            for model_name
            in EXPECTED_MODEL_ORDER
        ],
        axis=0,
    )

    predictive_entropy = np.array(
        [
            bernoulli_entropy(
                p
            )
            for p
            in ensemble_probability
        ]
    )

    mutual_information = np.maximum(
        predictive_entropy
        - aleatoric_entropy,
        0.0,
    )

    predicted_class = (
        ensemble_probability
        >= RISK_THRESHOLD
    ).astype(int)

    predicted_class_probability = np.where(
        predicted_class == 1,
        ensemble_probability,
        1.0 - ensemble_probability,
    )

    conformal_nonconformity = (
        1.0
        - predicted_class_probability
    )

    conformal_q = float(
        pipeline[
            "conformal_q"
        ]
    )

    conformal_safe = (
        conformal_nonconformity
        <= conformal_q
    )

    return pd.DataFrame(
        {

            "P_RF":
                model_probabilities[
                    "Random Forest"
                ],

            "P_XGB":
                model_probabilities[
                    "XGBoost"
                ],

            "P_LR":
                model_probabilities[
                    "Logistic Regression"
                ],

            "P_GDM":
                ensemble_probability,

            "Predicted_Class":
                predicted_class,

            "Epistemic_STD":
                epistemic_std,

            "Aleatoric_Entropy":
                aleatoric_entropy,

            "Predictive_Entropy":
                predictive_entropy,

            "Mutual_Information":
                mutual_information,

            "Conformal_Nonconformity":
                conformal_nonconformity,

            "Conformal_Safe":
                conformal_safe,
        },
        index=df.index,
    )


# ==============================================================================
# 14. PERFORMANCE METRICS
# ==============================================================================

def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = RISK_THRESHOLD,
) -> Dict[str, Any]:

    y_true_array = (
        np.asarray(
            y_true
        ).astype(int)
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true_array,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

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
            y_true_array,
            probabilities,
        )

    except Exception:

        roc_auc = np.nan

    try:

        pr_auc = average_precision_score(
            y_true_array,
            probabilities,
        )

    except Exception:

        pr_auc = np.nan

    try:

        brier = brier_score_loss(
            y_true_array,
            probabilities,
        )

    except Exception:

        brier = np.nan

    try:

        logloss = log_loss(
            y_true_array,
            probabilities,
            labels=[0, 1],
        )

    except Exception:

        logloss = np.nan

    return {

        "ROC-AUC":
            roc_auc,

        "PR-AUC":
            pr_auc,

        "Brier Score":
            brier,

        "Log Loss":
            logloss,

        "Accuracy":
            accuracy_score(
                y_true_array,
                predictions,
            ),

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "Precision":
            precision_score(
                y_true_array,
                predictions,
                zero_division=0,
            ),

        "F1":
            f1_score(
                y_true_array,
                predictions,
                zero_division=0,
            ),

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp,
    }


def build_performance_table(
    y_true: pd.Series,
    probability_dict: Dict[str, np.ndarray],
) -> pd.DataFrame:

    rows = []

    for model_name in EXPECTED_MODEL_ORDER:

        metrics = calculate_metrics(
            y_true,
            probability_dict[
                model_name
            ],
        )

        rows.append(
            {

                "Model":
                    model_name,

                "ROC-AUC":
                    metrics["ROC-AUC"],

                "PR-AUC":
                    metrics["PR-AUC"],

                "Brier":
                    metrics["Brier Score"],

                "Log Loss":
                    metrics["Log Loss"],

                "Sensitivity":
                    metrics["Sensitivity"],

                "Specificity":
                    metrics["Specificity"],

                "Precision":
                    metrics["Precision"],

                "F1":
                    metrics["F1"],
            }
        )

    ensemble_probability = (
        np.column_stack(
            [
                probability_dict[name]
                for name
                in EXPECTED_MODEL_ORDER
            ]
        )
        .mean(axis=1)
    )

    metrics = calculate_metrics(
        y_true,
        ensemble_probability,
    )

    rows.append(
        {

            "Model":
                "Ensemble",

            "ROC-AUC":
                metrics["ROC-AUC"],

            "PR-AUC":
                metrics["PR-AUC"],

            "Brier":
                metrics["Brier Score"],

            "Log Loss":
                metrics["Log Loss"],

            "Sensitivity":
                metrics["Sensitivity"],

            "Specificity":
                metrics["Specificity"],

            "Precision":
                metrics["Precision"],

            "F1":
                metrics["F1"],
        }
    )

    return pd.DataFrame(
        rows
    )


# ==============================================================================
# 15. CONFORMAL TEST COVERAGE
# ==============================================================================

def conformal_test_summary(
    y_true: pd.Series,
    probabilities: np.ndarray,
    q: float,
) -> Dict[str, Any]:

    y_array = (
        np.asarray(
            y_true
        ).astype(int)
    )

    p = np.asarray(
        probabilities,
        dtype=float,
    )

    true_class_probability = np.where(
        y_array == 1,
        p,
        1.0 - p,
    )

    nonconformity = (
        1.0
        - true_class_probability
    )

    covered = (
        nonconformity
        <= q
    )

    return {

        "coverage":
            float(
                np.mean(
                    covered
                )
            ),

        "target_confidence":
            CONFORMAL_CONFIDENCE,

        "q":
            float(q),

        "n":
            len(y_array),

        "covered":
            int(
                covered.sum()
            ),
    }


# ==============================================================================
# 16. FAIRNESS ANALYSIS
# ==============================================================================

def age_group(
    age: Any,
) -> Optional[str]:

    value = safe_float(
        age
    )

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
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    working = pd.DataFrame(
        {

            "Age":
                df[
                    AGE_COLUMN
                ].apply(
                    safe_float
                ),

            "Prediction":
                predictions[
                    "Predicted_Class"
                ].astype(float),
        },
        index=df.index,
    )

    working[
        "Age_Group"
    ] = working[
        "Age"
    ].apply(
        age_group
    )

    # --------------------------------------------------------------------------
    # Selection rate
    # --------------------------------------------------------------------------

    selection_rows = []

    for group, group_df in (
        working
        .dropna(
            subset=[
                "Age_Group"
            ]
        )
        .groupby(
            "Age_Group"
        )
    ):

        n = len(
            group_df
        )

        if (
            n
            < MIN_FAIRNESS_GROUP_SIZE
        ):

            selection_rate = np.nan

        else:

            selection_rate = (
                group_df[
                    "Prediction"
                ].mean()
            )

        selection_rows.append(
            {

                "Age Group":
                    group,

                "N":
                    n,

                "Selection Rate":
                    selection_rate,
            }
        )

    selection_table = (
        pd.DataFrame(
            selection_rows
        )
    )

    # --------------------------------------------------------------------------
    # FPR
    # --------------------------------------------------------------------------

    ground_truth = (
        df[
            TARGET_COLUMN
        ].apply(
            normalize_target
        )
    )

    fpr_working = (
        working.copy()
    )

    fpr_working[
        "Truth"
    ] = ground_truth

    fpr_rows = []

    for group, group_df in (
        fpr_working
        .dropna(
            subset=[
                "Age_Group",
                "Truth",
            ]
        )
        .groupby(
            "Age_Group"
        )
    ):

        n = len(
            group_df
        )

        negatives = (
            group_df[
                group_df[
                    "Truth"
                ] == 0
            ]
        )

        if (
            n
            < MIN_FAIRNESS_GROUP_SIZE
            or len(negatives) == 0
        ):

            fpr = np.nan

        else:

            fpr = (
                (
                    negatives[
                        "Prediction"
                    ] == 1
                ).sum()
                / len(negatives)
            )

        fpr_rows.append(
            {

                "Age Group":
                    group,

                "N with Truth":
                    n,

                "Negative Cases":
                    len(negatives),

                "FPR":
                    fpr,
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
# 17. BMI PSI
# ==============================================================================

def calculate_bmi_psi(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Tuple[
    float,
    pd.DataFrame,
]:

    baseline = (
        baseline_df[
            BMI_COLUMN
        ].apply(
            safe_float
        )
    )

    current = (
        current_df[
            BMI_COLUMN
        ].apply(
            safe_float
        )
    )

    baseline = baseline[
        np.isfinite(
            baseline
        )
    ]

    current = current[
        np.isfinite(
            current
        )
    ]

    if (
        len(baseline) == 0
        or len(current) == 0
    ):

        return (
            np.nan,
            pd.DataFrame(),
        )

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
        .reindex(
            BMI_LABELS,
            fill_value=0,
        )
    )

    current_counts = (
        current_categories
        .value_counts()
        .reindex(
            BMI_LABELS,
            fill_value=0,
        )
    )

    epsilon = 1e-6

    expected = (
        baseline_counts
        / baseline_counts.sum()
    ).astype(float)

    actual = (
        current_counts
        / current_counts.sum()
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
        actual - expected
    ) * np.log(
        actual / expected
    )

    psi = float(
        psi_components.sum()
    )

    table = pd.DataFrame(
        {

            "BMI Group":
                BMI_LABELS,

            "Baseline Proportion":
                expected.values,

            "Current Proportion":
                actual.values,

            "PSI Component":
                psi_components.values,
        }
    )

    return (
        psi,
        table,
    )


# ==============================================================================
# 18. XAI FEATURE CONTRIBUTIONS
# ==============================================================================

def aggregate_feature_contributions(
    pipeline: Dict[str, Any],
) -> pd.DataFrame:

    schema = pipeline[
        "feature_schema"
    ]

    model_contributions = []

    # --------------------------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------------------------

    rf = pipeline[
        "models"
    ][
        "Random Forest"
    ]

    rf_values = np.asarray(
        rf.feature_importances_,
        dtype=float,
    )

    if rf_values.sum() > 0:

        rf_values = (
            rf_values
            / rf_values.sum()
        )

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

    xgb = pipeline[
        "models"
    ][
        "XGBoost"
    ]

    xgb_values = np.asarray(
        xgb.feature_importances_,
        dtype=float,
    )

    if xgb_values.sum() > 0:

        xgb_values = (
            xgb_values
            / xgb_values.sum()
        )

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

    lr = pipeline[
        "models"
    ][
        "Logistic Regression"
    ]

    lr_values = np.abs(
        np.asarray(
            lr.coef_[0],
            dtype=float,
        )
    )

    if lr_values.sum() > 0:

        lr_values = (
            lr_values
            / lr_values.sum()
        )

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

    raw_table[
        "Mean Contribution"
    ] = raw_table.mean(
        axis=1
    )

    predictors = pipeline[
        "predictors"
    ]

    rows = []

    for predictor in predictors:

        matching_columns = [
            column

            for column
            in schema

            if (
                column == predictor
                or column.startswith(
                    predictor + "_"
                )
            )
        ]

        if not matching_columns:
            continue

        subset = raw_table.loc[
            matching_columns
        ]

        rows.append(
            {

                "Feature":
                    predictor,

                "Random Forest":
                    subset[
                        "Random Forest"
                    ].sum(),

                "XGBoost":
                    subset[
                        "XGBoost"
                    ].sum(),

                "Logistic Regression":
                    subset[
                        "Logistic Regression"
                    ].sum(),

                "Mean Contribution":
                    subset[
                        "Mean Contribution"
                    ].sum(),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    return (
        result
        .sort_values(
            "Mean Contribution",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ==============================================================================
# 19. SESSION STATE
# ==============================================================================

if "pipeline" not in st.session_state:

    st.session_state.pipeline = None

if "predictions" not in st.session_state:

    st.session_state.predictions = None

if "data_signature" not in st.session_state:

    st.session_state.data_signature = None


# ==============================================================================
# 20. APPLICATION HEADER
# ==============================================================================

st.title(
    APP_TITLE
)

st.markdown(
    f"### {APP_SUBTITLE}"
)

st.caption(
    f"{APP_VERSION} • "
    "Random Forest + XGBoost + Logistic Regression • "
    "60/15/25 train/calibration/test design"
)

st.warning(
    "SafeTriage-GDM is a research prototype for uncertainty-aware GDM "
    "risk triage, conformal safety assessment, population-shift monitoring, "
    "and algorithmic fairness auditing. It is not a medical device and must "
    "not be used as a substitute for professional medical diagnosis, "
    "treatment, or clinical decision-making."
)


# ==============================================================================
# 21. SIDEBAR
# ==============================================================================

with st.sidebar:

    st.header(
        "System Configuration"
    )

    st.write(
        "Required model architecture:"
    )

    for model_name in EXPECTED_MODEL_ORDER:

        st.write(
            f"✓ {model_name}"
        )

    st.divider()

    st.write(
        f"Research risk threshold: "
        f"**{RISK_THRESHOLD:.2f}**"
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
        "These thresholds are research settings and are not "
        "clinically validated diagnostic thresholds."
    )


# ==============================================================================
# 22. EXCEL UPLOAD
# ==============================================================================

st.header(
    "Data Input"
)

st.info(
    "Upload a compatible research dataset in Excel format. "
    "Do not upload names, medical record numbers, addresses, or other "
    "directly identifiable patient information."
)

uploaded_file = st.file_uploader(
    "Upload your research dataset",
    type=[
        "xlsx",
        "xls",
    ],
    help=(
        "Only Excel files (.xlsx or .xls) are accepted."
    ),
)


# ==============================================================================
# 23. READ DATA
# ==============================================================================

DEFAULT_DATASET = (
    "dataCBGS_dataset.xlsx"
)

if uploaded_file is not None:

    try:

        df = pd.read_excel(
            uploaded_file
        )

        data_source = (
            "User-uploaded Excel file"
        )

    except Exception as exc:

        st.error(
            "The uploaded Excel file could not be read."
        )

        st.exception(
            exc
        )

        st.stop()

else:

    try:

        df = pd.read_excel(
            DEFAULT_DATASET
        )

        data_source = (
            f"Repository dataset: "
            f"{DEFAULT_DATASET}"
        )

        st.info(
            "No file has been uploaded. "
            "The repository's example CBGS dataset is being used."
        )

    except FileNotFoundError:

        st.info(
            "Upload an Excel dataset to initialise the application."
        )

        st.stop()

    except Exception as exc:

        st.error(
            "The repository dataset could not be read."
        )

        st.exception(
            exc
        )

        st.stop()


# ==============================================================================
# 24. VALIDATE DATASET
# ==============================================================================

valid_dataset, validation_errors = (
    validate_dataset(
        df
    )
)

if not valid_dataset:

    st.error(
        "The supplied dataset does not satisfy "
        "the SafeTriage-GDM input requirements."
    )

    for error in validation_errors:

        st.error(
            error
        )

    st.stop()


# ==============================================================================
# 25. DATASET SUMMARY
# ==============================================================================

available_predictors = (
    get_available_predictors(
        df
    )
)

target = prepare_target(
    df
)

n_total = len(
    df
)

n_labeled = int(
    target.notna().sum()
)

n_unlabeled = int(
    target.isna().sum()
)

n_positive = int(
    (target == 1).sum()
)

n_negative = int(
    (target == 0).sum()
)

st.write(
    f"**Data source:** {data_source}"
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
    "Only observations with valid GDM ground truth are used "
    "for supervised training and evaluation."
)


# ==============================================================================
# 26. DETECT DATASET CHANGES
# ==============================================================================

current_signature = (
    dataframe_signature(
        df
    )
)

if (
    st.session_state.data_signature
    != current_signature
):

    st.session_state.pipeline = None

    st.session_state.predictions = None

    st.session_state.data_signature = (
        current_signature
    )


# ==============================================================================
# 27. TRAIN PIPELINE
# ==============================================================================

if st.session_state.pipeline is None:

    with st.spinner(
        "Training SafeTriage-GDM on the supplied dataset..."
    ):

        try:

            pipeline = train_pipeline(
                df
            )

            st.session_state.pipeline = (
                pipeline
            )

        except Exception as exc:

            st.error(
                "SafeTriage-GDM could not train on this dataset."
            )

            st.exception(
                exc
            )

            st.info(
                "Please verify that the Excel file contains "
                "the required GDM target and compatible predictor fields."
            )

            st.stop()


pipeline = (
    st.session_state.pipeline
)


# ==============================================================================
# 28. GENERATE POPULATION PREDICTIONS
# ==============================================================================

if st.session_state.predictions is None:

    try:

        predictions = predict_population(
            df,
            pipeline,
        )

        st.session_state.predictions = (
            predictions
        )

    except Exception as exc:

        st.error(
            "Population inference failed."
        )

        st.exception(
            exc
        )

        st.stop()


predictions = (
    st.session_state.predictions
)


st.success(
    "SafeTriage-GDM pipeline successfully initialised "
    "for the supplied dataset."
)


# ==============================================================================
# 29. MAIN TABS
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
        "Select an observation from the supplied dataset to inspect "
        "its ensemble GDM risk, model agreement, uncertainty, and "
        "conformal safety status."
    )

    patient_index = st.selectbox(
        "Select observation",
        options=list(
            df.index
        ),
        format_func=lambda x:
            f"Observation {x + 1}",
    )

    patient_prediction = (
        predictions.loc[
            patient_index
        ]
    )

    probability = float(
        patient_prediction[
            "P_GDM"
        ]
    )

    predicted_class = int(
        patient_prediction[
            "Predicted_Class"
        ]
    )

    conformal_safe = bool(
        patient_prediction[
            "Conformal_Safe"
        ]
    )

    if predicted_class == 1:

        st.error(
            f"Research classification: elevated "
            f"({probability:.1%} ensemble probability)"
        )

    else:

        st.success(
            f"Research classification: lower "
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
            "Random Forest",
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

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model-disagreement SD",
            format_metric(
                patient_prediction[
                    "Epistemic_STD"
                ]
            ),
        )

    with col2:

        st.metric(
            "Predictive Entropy",
            format_metric(
                patient_prediction[
                    "Predictive_Entropy"
                ]
            ),
        )

    with col3:

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

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Nonconformity",
            format_metric(
                patient_prediction[
                    "Conformal_Nonconformity"
                ]
            ),
        )

    with col2:

        if conformal_safe:

            st.success(
                "Prediction satisfies the conformal safety bound."
            )

        else:

            st.warning(
                "Prediction does not satisfy the conformal safety bound."
            )

    st.subheader(
        "Observation Features"
    )

    patient_row = df.loc[
        [patient_index]
    ]

    display_patient = (
        patient_row[
            available_predictors
        ]
        .T
    )

    display_patient.columns = [
        "Value"
    ]

    st.dataframe(
        display_patient,
        width="stretch",
    )

    st.caption(
        "Risk probabilities, uncertainty measures, and conformal "
        "status are research outputs and do not constitute a diagnosis."
    )


# ==============================================================================
# TAB 2 — PREDICTIVE PERFORMANCE
# ==============================================================================

with tab_performance:

    st.header(
        "Predictive Performance"
    )

    y_test = pipeline[
        "y_test"
    ]

    test_probability_dict = (
        pipeline[
            "test_probabilities"
        ]
    )

    performance_table = (
        build_performance_table(
            y_test,
            test_probability_dict,
        )
    )

    st.subheader(
        "Untouched Test-Set Performance"
    )

    st.dataframe(
        performance_table.round(4),
        width="stretch",
        hide_index=True,
    )

    ensemble_test_probability = (
        pipeline[
            "ensemble_test_probability"
        ]
    )

    ensemble_metrics = (
        calculate_metrics(
            y_test,
            ensemble_test_probability,
        )
    )

    st.subheader(
        "Ensemble Metrics"
    )

    # --------------------------------------------------------------------------
    # MOBILE-FRIENDLY METRIC LAYOUT
    # --------------------------------------------------------------------------

    row1 = st.columns(3)

    row1[0].metric(
        "ROC-AUC",
        format_metric(
            ensemble_metrics[
                "ROC-AUC"
            ]
        ),
    )

    row1[1].metric(
        "PR-AUC",
        format_metric(
            ensemble_metrics[
                "PR-AUC"
            ]
        ),
    )

    row1[2].metric(
        "Brier Score",
        format_metric(
            ensemble_metrics[
                "Brier Score"
            ]
        ),
    )

    row2 = st.columns(3)

    row2[0].metric(
        "Sensitivity",
        format_pct(
            ensemble_metrics[
                "Sensitivity"
            ]
        ),
    )

    row2[1].metric(
        "Specificity",
        format_pct(
            ensemble_metrics[
                "Specificity"
            ]
        ),
    )

    row2[2].metric(
        "F1 Score",
        format_metric(
            ensemble_metrics[
                "F1"
            ]
        ),
    )

    st.subheader(
        "Confusion Matrix"
    )

    confusion_df = pd.DataFrame(
        [
            [
                ensemble_metrics[
                    "TN"
                ],
                ensemble_metrics[
                    "FP"
                ],
            ],
            [
                ensemble_metrics[
                    "FN"
                ],
                ensemble_metrics[
                    "TP"
                ],
            ],
        ],
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
        confusion_df,
        width="stretch",
    )

    # --------------------------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------------------------

    st.subheader(
        "ROC Curve"
    )

    try:

        fpr, tpr, _ = roc_curve(
            np.asarray(
                y_test
            ).astype(int),
            ensemble_test_probability,
        )

        roc_df = pd.DataFrame(
            {
                "False Positive Rate":
                    fpr,

                "True Positive Rate":
                    tpr,
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
            "ROC curve could not be calculated."
        )

    # --------------------------------------------------------------------------
    # PRECISION-RECALL CURVE
    # --------------------------------------------------------------------------

    st.subheader(
        "Precision–Recall Curve"
    )

    try:

        precision, recall, _ = (
            precision_recall_curve(
                np.asarray(
                    y_test
                ).astype(int),
                ensemble_test_probability,
            )
        )

        pr_df = pd.DataFrame(
            {
                "Recall":
                    recall,

                "Precision":
                    precision,
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
    # CALIBRATION CURVE
    # --------------------------------------------------------------------------

    st.subheader(
        "Descriptive Calibration Curve"
    )

    try:

        fraction_positive, mean_predicted = (
            calibration_curve(
                np.asarray(
                    y_test
                ).astype(int),
                ensemble_test_probability,
                n_bins=5,
                strategy="quantile",
            )
        )

        calibration_df = pd.DataFrame(
            {
                "Mean Predicted Probability":
                    mean_predicted,

                "Observed Positive Fraction":
                    fraction_positive,
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
        "The test set is untouched during model fitting. "
        "The calibration curve is descriptive; the displayed probabilities "
        "should not be interpreted as guaranteed calibrated clinical risks."
    )


# ==============================================================================
# TAB 3 — UNCERTAINTY & CONFORMAL SAFETY
# ==============================================================================

with tab_uncertainty:

    st.header(
        "Uncertainty & Conformal Safety"
    )

    q = float(
        pipeline[
            "conformal_q"
        ]
    )

    conformal_summary = (
        conformal_test_summary(
            pipeline[
                "y_test"
            ],
            pipeline[
                "ensemble_test_probability"
            ],
            q,
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Conformal q",
            format_metric(
                q
            ),
        )

    with col2:

        st.metric(
            "Target Confidence",
            f"{CONFORMAL_CONFIDENCE:.0%}",
        )

    with col3:

        st.metric(
            "Observed Test Coverage",
            f"{conformal_summary['coverage']:.1%}",
        )

    st.subheader(
        "Conformal Calibration"
    )

    st.write(
        f"Calibration observations: "
        f"**{pipeline['calibration_size']}**"
    )

    st.write(
        f"Finite-sample quantile level: "
        f"**{pipeline['conformal_q_level']:.4f}**"
    )

    st.write(
        "The conformal threshold is estimated using the dedicated "
        "calibration partition. Population-shift monitoring does not "
        "modify this threshold."
    )

    st.subheader(
        "Uncertainty Components"
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
    ]

    st.dataframe(
        uncertainty_df
        .head(100)
        .round(4),
        width="stretch",
    )

    st.subheader(
        "Interpretation"
    )

    st.markdown(
        """
- **Epistemic STD:** model-disagreement proxy across the three models.
- **Aleatoric entropy:** average Bernoulli uncertainty across the models.
- **Predictive entropy:** entropy of the ensemble probability.
- **Mutual information:** predictive entropy minus mean model entropy,
  truncated at zero.
- **Conformal nonconformity:** one minus the probability assigned to the
  predicted class.
- **Conformal safety:** whether the prediction satisfies the calibrated
  conformal bound.
"""
    )

    st.warning(
        "These measures quantify model uncertainty. They do not establish "
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

    (
        selection_table,
        fpr_table,
    ) = fairness_analysis(
        df,
        predictions,
    )

    if selection_table.empty:

        st.info(
            "No valid age groups are available."
        )

    else:

        st.dataframe(
            selection_table.round(4),
            width="stretch",
            hide_index=True,
        )

        valid_selection = (
            selection_table[
                "Selection Rate"
            ]
            .dropna()
        )

        if len(
            valid_selection
        ) >= 2:

            selection_disparity = (
                valid_selection.max()
                - valid_selection.min()
            )

            st.metric(
                "Selection-rate disparity",
                format_pct(
                    selection_disparity
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

        valid_fpr = (
            fpr_table[
                "FPR"
            ]
            .dropna()
        )

        if len(
            valid_fpr
        ) >= 2:

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
                "This is FPR disparity, not Equalized Odds Disparity. "
                "Equalized Odds requires both FPR and TPR comparisons."
            )

    st.divider()

    st.subheader(
        "BMI Population Shift"
    )

    psi, bmi_table = calculate_bmi_psi(
        pipeline[
            "X_train_raw"
        ],
        df,
    )

    if np.isfinite(
        psi
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "BMI PSI",
                format_metric(
                    psi
                ),
            )

        with col2:

            if (
                psi
                >= BMI_PSI_THRESHOLD
            ):

                st.error(
                    "Population-shift warning: BMI distribution "
                    "exceeds the predefined PSI threshold."
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
            "BMI PSI could not be calculated because insufficient "
            "valid BMI observations were available."
        )

    st.caption(
        "BMI PSI is a monitoring signal only. It does not alter "
        "model probabilities or the conformal q value."
    )


# ==============================================================================
# TAB 5 — XAI
# ==============================================================================

with tab_xai:

    st.header(
        "XAI Attribution"
    )

    st.write(
        "The following analysis estimates global feature contribution "
        "across Random Forest, XGBoost, and Logistic Regression."
    )

    contribution_table = (
        aggregate_feature_contributions(
            pipeline
        )
    )

    if contribution_table.empty:

        st.warning(
            "Feature contribution analysis could not be generated."
        )

    else:

        maximum_features = min(
            15,
            len(
                contribution_table
            ),
        )

        default_features = min(
            10,
            maximum_features,
        )

        top_n = st.slider(
            "Number of features to display",
            min_value=5,
            max_value=maximum_features,
            value=default_features,
        )

        plot_df = (
            contribution_table
            .head(
                top_n
            )
            .sort_values(
                "Mean Contribution",
                ascending=True,
            )
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
            contribution_table
            .head(
                top_n
            )
            .round(4),
            width="stretch",
            hide_index=True,
        )

        st.info(
            "Feature contribution is a model-explanation measure, "
            "not a causal effect. High contribution does not mean that "
            "changing the feature will necessarily change GDM risk."
        )


# ==============================================================================
# 30. PIPELINE DIAGNOSTICS
# ==============================================================================

with st.expander(
    "Pipeline Diagnostics & Reproducibility"
):

    st.write(
        "**Data source:**",
        data_source,
    )

    st.write(
        "**Dataset signature:**",
        current_signature[:16] + "...",
    )

    st.write(
        "**Total rows:**",
        n_total,
    )

    st.write(
        "**Labeled rows:**",
        n_labeled,
    )

    st.write(
        "**Training rows:**",
        pipeline[
            "train_size"
        ],
    )

    st.write(
        "**Calibration rows:**",
        pipeline[
            "calibration_size"
        ],
    )

    st.write(
        "**Test rows:**",
        pipeline[
            "test_size"
        ],
    )

    st.write(
        "**Approved predictor count:**",
        len(
            pipeline[
                "predictors"
            ]
        ),
    )

    st.write(
        "**Model architecture:**",
        ", ".join(
            EXPECTED_MODEL_ORDER
        ),
    )

    st.subheader(
        "Approved Predictors"
    )

    for predictor in pipeline[
        "predictors"
    ]:

        st.write(
            f"- {predictor}"
        )

    st.subheader(
        "Excluded Post-Delivery Variables"
    )

    for variable in POST_DELIVERY_COLUMNS:

        if variable in df.columns:

            st.write(
                f"- {variable}"
            )

    st.subheader(
        "Timing-Sensitive Variables Excluded"
    )

    for variable in TIMING_SENSITIVE_EXCLUDED:

        if variable in df.columns:

            st.write(
                f"- {variable}"
            )


# ==============================================================================
# 31. FOOTER
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
