# ==============================================================================
# SafeTriage-GDM
# Uncertainty-Quantified Clinical Triage System for Gestational Diabetes
# Mellitus (GDM) Risk with Algorithmic Fairness Auditing & Conformal
# Safety Bounds
#
# Standalone Research Prototype
# ==============================================================================

import os
import warnings
from typing import Dict, List, Optional, Tuple

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
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

from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier


warnings.filterwarnings(
    "ignore",
    message=".*Unknown solver options: iprint.*",
)

# ==============================================================================
# 1. APPLICATION CONSTANTS
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
AGE_COLUMN = "Age"
ID_COLUMN = "Dummy Study Number"

RISK_THRESHOLD = 0.50

CONFORMAL_ALPHA = 0.05

RANDOM_STATE = 42

EXPECTED_MODELS = {
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
}

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

PSI_WARNING_LIMIT = 0.20


# ==============================================================================
# 2. PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def resolve_column(
    dataframe: pd.DataFrame,
    preferred: str,
    alternatives: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Resolve a column using exact match first, then case-insensitive matching.
    """
    if preferred in dataframe.columns:
        return preferred

    alternatives = alternatives or []

    candidates = [preferred] + alternatives

    normalized = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def parse_gdm_target(series: pd.Series) -> pd.Series:
    """
    Convert common GDM target representations into 0/1/NaN.

    Unknown or missing values are intentionally mapped to NaN.
    No labels are ever generated synthetically.
    """
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")

        result = pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

        result.loc[numeric == 0] = 0.0
        result.loc[numeric == 1] = 1.0

        return result

    text = series.astype(str).str.strip().str.lower()

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
        "diabetes",
    }

    negative_values = {
        "0",
        "no",
        "n",
        "false",
        "negative",
        "neg",
        "no gdm",
        "no gestational diabetes",
        "none",
        "normal",
    }

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    result.loc[text.isin(positive_values)] = 1.0
    result.loc[text.isin(negative_values)] = 0.0

    return result


def encode_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns.
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
    Force inference data to exactly match the training feature schema.
    """
    aligned = encoded_dataframe.reindex(
        columns=feature_schema,
        fill_value=0.0,
    )

    return aligned.astype(np.float64)


def get_positive_probability(
    model,
    X: np.ndarray,
) -> np.ndarray:
    """
    Obtain P(GDM) from a fitted binary classifier.
    """
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(
            model.predict_proba(X)
        )

        if probabilities.ndim == 2:
            if probabilities.shape[1] != 2:
                raise ValueError(
                    "Expected a binary classifier with exactly two "
                    "probability columns."
                )

            return probabilities[:, 1].astype(float)

        return probabilities.ravel().astype(float)

    if hasattr(model, "decision_function"):
        decision = np.asarray(
            model.decision_function(X),
            dtype=float,
        )

        decision = np.clip(
            decision,
            -50,
            50,
        )

        return (
            1.0 /
            (1.0 + np.exp(-decision))
        )

    raise TypeError(
        f"Model {type(model).__name__} does not provide "
        "predict_proba or decision_function."
    )


def binary_entropy(probabilities) -> np.ndarray:
    """
    Binary Shannon entropy.
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


def uncertainty_decomposition(
    model_probabilities: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Decompose ensemble predictive uncertainty.

    model_probabilities:
        shape = (3, n_samples)

    Epistemic proxy:
        standard deviation of the three model probabilities.

    Aleatoric proxy:
        mean Bernoulli entropy across models.

    Predictive entropy:
        entropy of the ensemble mean probability.

    Mutual information:
        predictive entropy - expected entropy.
    """
    if model_probabilities.ndim != 2:
        raise ValueError(
            "Model probability matrix must be two-dimensional."
        )

    if model_probabilities.shape[0] != 3:
        raise ValueError(
            "SafeTriage-GDM requires exactly three model "
            "probability vectors."
        )

    consensus_probability = np.mean(
        model_probabilities,
        axis=0,
    )

    epistemic_std = np.std(
        model_probabilities,
        axis=0,
        ddof=0,
    )

    aleatoric_entropy = np.mean(
        binary_entropy(model_probabilities),
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
        "consensus_probability": consensus_probability,
        "epistemic_std": epistemic_std,
        "aleatoric_entropy": aleatoric_entropy,
        "predictive_entropy": predictive_entropy,
        "mutual_information": mutual_information,
    }


def conformal_prediction_sets(
    probabilities: np.ndarray,
    q_threshold: float,
) -> List[str]:
    """
    Binary split-conformal prediction sets.

    Class 0 nonconformity:
        p(GDM)

    Class 1 nonconformity:
        1 - p(GDM)
    """
    output = []

    for probability in probabilities:
        include_healthy = probability <= q_threshold
        include_gdm = (1.0 - probability) <= q_threshold

        if include_healthy and include_gdm:
            output.append(
                "{Healthy, GDM High Risk}"
            )
        elif include_gdm:
            output.append(
                "{GDM High Risk}"
            )
        elif include_healthy:
            output.append(
                "{Healthy}"
            )
        else:
            # Safety-preserving fallback against an empty set.
            output.append(
                "{Healthy, GDM High Risk}"
            )

    return output


def calculate_conformal_threshold(
    calibration_probabilities: np.ndarray,
    y_calibration: np.ndarray,
    alpha: float,
) -> float:
    """
    Finite-sample split-conformal quantile.
    """
    calibration_probabilities = np.asarray(
        calibration_probabilities,
        dtype=float,
    )

    y_calibration = np.asarray(
        y_calibration,
        dtype=int,
    )

    scores = np.where(
        y_calibration == 1,
        1.0 - calibration_probabilities,
        calibration_probabilities,
    )

    n = len(scores)

    if n == 0:
        raise ValueError(
            "Calibration set is empty."
        )

    quantile_level = min(
        1.0,
        np.ceil(
            (n + 1) * (1.0 - alpha)
        ) / n,
    )

    try:
        return float(
            np.quantile(
                scores,
                quantile_level,
                method="higher",
            )
        )
    except TypeError:
        # Compatibility fallback for older NumPy.
        return float(
            np.quantile(
                scores,
                quantile_level,
                interpolation="higher",
            )
        )


def conformal_coverage(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    q_threshold: float,
) -> Tuple[float, float]:
    """
    Empirical conformal coverage and mean prediction-set size.
    """
    sets = []

    for probability in probabilities:
        include_healthy = probability <= q_threshold
        include_gdm = (1.0 - probability) <= q_threshold

        if include_healthy and include_gdm:
            sets.append({0, 1})
        elif include_gdm:
            sets.append({1})
        elif include_healthy:
            sets.append({0})
        else:
            sets.append({0, 1})

    y_true = np.asarray(y_true, dtype=int)

    covered = [
        int(y in prediction_set)
        for y, prediction_set in zip(y_true, sets)
    ]

    sizes = [
        len(prediction_set)
        for prediction_set in sets
    ]

    return (
        float(np.mean(covered)),
        float(np.mean(sizes)),
    )


def calculate_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict:
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
        probabilities >= RISK_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity_denominator = tn + fp

    specificity = (
        float(tn / specificity_denominator)
        if specificity_denominator > 0
        else np.nan
    )

    try:
        roc_auc = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )
    except Exception:
        roc_auc = np.nan

    try:
        pr_auc = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )
    except Exception:
        pr_auc = np.nan

    try:
        brier = float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        )
    except Exception:
        brier = np.nan

    try:
        ll = float(
            log_loss(
                y_true,
                np.column_stack(
                    [
                        1.0 - probabilities,
                        probabilities,
                    ]
                ),
                labels=[0, 1],
            )
        )
    except Exception:
        ll = np.nan

    return {
        "Accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "Sensitivity / Recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "Specificity": specificity,
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
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Brier Score": brier,
        "Log Loss": ll,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def calculate_fairness_metrics(
    dataframe: pd.DataFrame,
    consensus_probability: np.ndarray,
) -> Dict:
    """
    Fairness audit with strict row alignment.

    Demographic parity:
        evaluated over all rows with valid age.

    FPR disparity:
        evaluated only among rows having both valid age
        and valid ground truth.
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

    probabilities = np.asarray(
        consensus_probability,
        dtype=float,
    )

    if len(dataframe) != len(probabilities):
        raise ValueError(
            "Fairness calculation alignment error: "
            f"dataframe contains {len(dataframe)} rows, "
            f"but predictions contain {len(probabilities)} rows."
        )

    age_column = resolve_column(
        dataframe,
        AGE_COLUMN,
        [
            "age",
            "Maternal Age",
            "maternal age",
        ],
    )

    if age_column is None:
        return result

    age_series = pd.to_numeric(
        dataframe[age_column],
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
        result["demographic_parity_difference"] = float(
            abs(
                result["younger_positive_rate"]
                - result["older_positive_rate"]
            )
        )

    target_column = resolve_column(
        dataframe,
        TARGET_COLUMN,
        [
            "GDM",
            "GDM Status",
            "Gestational Diabetes",
            "Gestational Diabetes Mellitus",
        ],
    )

    if target_column is None:
        return result

    y_true = pd.to_numeric(
        dataframe[target_column],
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

    eval_y_true = y_true.to_numpy(
    )[labeled_mask].astype(int)

    eval_age = age_series.to_numpy(
    )[labeled_mask]

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
            false_positives / negatives.sum()
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
        result["fpr_disparity"] = float(
            abs(
                result["younger_fpr"]
                - result["older_fpr"]
            )
        )

    return result


def find_bmi_column(
    dataframe: pd.DataFrame,
) -> Optional[str]:
    return resolve_column(
        dataframe,
        "BMI",
        [
            "Body Mass Index",
            "Body Mass Index (BMI)",
            "BMI (kg/m2)",
            "BMI (kg/m²)",
            "body mass index",
        ],
    )


def calculate_bmi_distribution(
    dataframe: pd.DataFrame,
) -> Optional[np.ndarray]:
    """
    Return normalized BMI category distribution.
    """
    bmi_column = find_bmi_column(dataframe)

    if bmi_column is None:
        return None

    bmi = pd.to_numeric(
        dataframe[bmi_column],
        errors="coerce",
    ).dropna()

    if bmi.empty:
        return None

    categories = pd.cut(
        bmi,
        bins=BMI_BINS,
        labels=BMI_LABELS,
        include_lowest=True,
    )

    distribution = (
        categories
        .value_counts(
            sort=False,
            normalize=True,
        )
        .reindex(
            BMI_LABELS,
            fill_value=0.0,
        )
        .to_numpy(dtype=float)
    )

    return distribution


def calculate_psi(
    baseline_distribution: Optional[np.ndarray],
    current_distribution: Optional[np.ndarray],
) -> float:
    """
    Population Stability Index for the saved BMI distribution.
    """
    if (
        baseline_distribution is None
        or current_distribution is None
    ):
        return np.nan

    baseline = np.asarray(
        baseline_distribution,
        dtype=float,
    )

    current = np.asarray(
        current_distribution,
        dtype=float,
    )

    if len(baseline) != len(current):
        return np.nan

    epsilon = 1e-6

    baseline = np.clip(
        baseline,
        epsilon,
        None,
    )

    current = np.clip(
        current,
        epsilon,
        None,
    )

    baseline = baseline / baseline.sum()
    current = current / current.sum()

    return float(
        np.sum(
            (current - baseline)
            * np.log(current / baseline)
        )
    )


def normalize_importance(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    total = values.sum()

    if total <= 0:
        return np.zeros_like(values)

    return values / total


def build_feature_importance(
    feature_schema: List[str],
    rf_model,
    xgb_model,
    logistic_model,
) -> pd.DataFrame:
    """
    Build a genuine three-model global attribution table.

    RF:
        impurity-based feature importance.

    XGBoost:
        model feature importance.

    Logistic Regression:
        absolute standardized coefficients.
    """
    n_features = len(feature_schema)

    rf_values = np.asarray(
        rf_model.feature_importances_,
        dtype=float,
    )

    xgb_values = np.asarray(
        xgb_model.feature_importances_,
        dtype=float,
    )

    lr_values = np.abs(
        np.asarray(
            logistic_model.coef_,
            dtype=float,
        ).ravel()
    )

    if len(rf_values) != n_features:
        raise ValueError(
            "Random Forest feature importance length does not "
            "match the feature schema."
        )

    if len(xgb_values) != n_features:
        raise ValueError(
            "XGBoost feature importance length does not "
            "match the feature schema."
        )

    if len(lr_values) != n_features:
        raise ValueError(
            "Logistic Regression coefficient length does not "
            "match the feature schema."
        )

    rf_norm = normalize_importance(rf_values)
    xgb_norm = normalize_importance(xgb_values)
    lr_norm = normalize_importance(lr_values)

    aggregate = (
        rf_norm
        + xgb_norm
        + lr_norm
    ) / 3.0

    output = pd.DataFrame(
        {
            "Feature": feature_schema,
            "Random Forest": rf_norm,
            "XGBoost": xgb_norm,
            "Logistic Regression": lr_norm,
            "Attribution Weight Score": aggregate,
        }
    )

    return output.sort_values(
        "Attribution Weight Score",
        ascending=False,
    ).reset_index(drop=True)


def train_pipeline(
    labeled_dataframe: pd.DataFrame,
) -> Dict:
    """
    Train the mandatory three-model SafeTriage-GDM architecture.

    Split:
        60% training
        15% calibration
        25% untouched test
    """
    if len(labeled_dataframe) < 30:
        raise ValueError(
            "At least 30 labeled observations are required "
            "to train the research pipeline."
        )

    y = labeled_dataframe[
        TARGET_COLUMN
    ].astype(int).to_numpy()

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Training requires both GDM classes: 0 and 1."
        )

    X_raw = labeled_dataframe.drop(
        columns=[
            TARGET_COLUMN,
            ID_COLUMN,
        ],
        errors="ignore",
    ).copy()

    X_encoded = encode_dataframe(
        X_raw
    )

    feature_schema = list(
        X_encoded.columns
    )

    # --------------------------------------------------------------------------
    # EXACT 60 / 15 / 25 SPLIT
    # --------------------------------------------------------------------------

    X_train_temp, X_test, y_train_temp, y_test = (
        train_test_split(
            X_encoded,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    X_train, X_cal, y_train, y_cal = (
        train_test_split(
            X_train_temp,
            y_train_temp,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y_train_temp,
        )
    )

    # --------------------------------------------------------------------------
    # IMPUTATION — FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------------------------

    imputer = IterativeImputer(
        random_state=RANDOM_STATE,
        max_iter=20,
        initial_strategy="median",
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
    # SCALING — FIT ONLY ON TRAINING DATA
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
    # CLASS IMBALANCE HANDLING — TRAINING ONLY
    # --------------------------------------------------------------------------

    smote_tomek = SMOTETomek(
        random_state=RANDOM_STATE
    )

    X_train_resampled, y_train_resampled = (
        smote_tomek.fit_resample(
            X_train_scaled,
            y_train,
        )
    )

    # --------------------------------------------------------------------------
    # MANDATORY THREE-MODEL ARCHITECTURE
    # --------------------------------------------------------------------------

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # IMPORTANT:
    # No "iprint" option is supplied.
    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )

    base_models = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "Logistic Regression": logistic_model,
    }

    if set(base_models.keys()) != EXPECTED_MODELS:
        raise RuntimeError(
            "SafeTriage-GDM architecture integrity failure: "
            "the pipeline must contain Random Forest, XGBoost, "
            "and Logistic Regression."
        )

    # --------------------------------------------------------------------------
    # FIT ALL THREE MODELS
    # --------------------------------------------------------------------------

    for model_name, model in base_models.items():
        try:
            model.fit(
                X_train_resampled,
                y_train_resampled,
            )
        except Exception as exc:
            raise RuntimeError(
                f"{model_name} failed during training. "
                "The mandatory three-model architecture was not "
                "successfully completed."
            ) from exc

    # --------------------------------------------------------------------------
    # CALIBRATE ALL THREE MODELS ON DEDICATED CALIBRATION SET
    # --------------------------------------------------------------------------

    calibrated_models = {}

    for model_name, model in base_models.items():
        try:
            calibrated_model = CalibratedClassifierCV(
                FrozenEstimator(model),
                method="sigmoid",
            )

            calibrated_model.fit(
                X_cal_scaled,
                y_cal,
            )

            calibrated_models[
                model_name
            ] = calibrated_model

        except Exception as exc:
            raise RuntimeError(
                f"{model_name} failed during calibration."
            ) from exc

    if set(calibrated_models.keys()) != EXPECTED_MODELS:
        raise RuntimeError(
            "Calibration architecture integrity failure."
        )

    # --------------------------------------------------------------------------
    # CALIBRATION PROBABILITIES
    # --------------------------------------------------------------------------

    calibration_probability_vectors = []

    for model_name in [
        "Random Forest",
        "XGBoost",
        "Logistic Regression",
    ]:
        probability = get_positive_probability(
            calibrated_models[model_name],
            X_cal_scaled,
        )

        if len(probability) != len(y_cal):
            raise RuntimeError(
                f"{model_name} returned an incorrectly sized "
                "calibration probability vector."
            )

        calibration_probability_vectors.append(
            probability
        )

    calibration_probability_matrix = np.vstack(
        calibration_probability_vectors
    )

    calibration_uncertainty = uncertainty_decomposition(
        calibration_probability_matrix
    )

    calibration_consensus_probability = (
        calibration_uncertainty[
            "consensus_probability"
        ]
    )

    # --------------------------------------------------------------------------
    # CONFORMAL CALIBRATION
    # --------------------------------------------------------------------------

    q_threshold = calculate_conformal_threshold(
        calibration_consensus_probability,
        y_cal,
        CONFORMAL_ALPHA,
    )

    # --------------------------------------------------------------------------
    # TEST PROBABILITIES
    # --------------------------------------------------------------------------

    test_probability_vectors = []

    for model_name in [
        "Random Forest",
        "XGBoost",
        "Logistic Regression",
    ]:
        probability = get_positive_probability(
            calibrated_models[model_name],
            X_test_scaled,
        )

        if len(probability) != len(y_test):
            raise RuntimeError(
                f"{model_name} returned an incorrectly sized "
                "test probability vector."
            )

        test_probability_vectors.append(
            probability
        )

    test_probability_matrix = np.vstack(
        test_probability_vectors
    )

    test_uncertainty = uncertainty_decomposition(
        test_probability_matrix
    )

    test_consensus_probability = (
        test_uncertainty[
            "consensus_probability"
        ]
    )

    # --------------------------------------------------------------------------
    # PERFORMANCE — UNTOUCHED TEST SET
    # --------------------------------------------------------------------------

    performance = {}

    model_names = [
        "Random Forest",
        "XGBoost",
        "Logistic Regression",
    ]

    for index, model_name in enumerate(
        model_names
    ):
        performance[model_name] = calculate_binary_metrics(
            y_test,
            test_probability_matrix[index],
        )

    performance["Ensemble"] = calculate_binary_metrics(
        y_test,
        test_consensus_probability,
    )

    # --------------------------------------------------------------------------
    # ROC CURVE DATA
    # --------------------------------------------------------------------------

    try:
        fpr, tpr, _ = roc_curve(
            y_test,
            test_consensus_probability,
        )

        roc_curve_data = pd.DataFrame(
            {
                "False Positive Rate": fpr,
                "True Positive Rate": tpr,
            }
        )
    except Exception:
        roc_curve_data = pd.DataFrame()

    # --------------------------------------------------------------------------
    # CALIBRATION CURVE DATA
    # --------------------------------------------------------------------------

    try:
        calibration_true, calibration_pred = (
            calibration_curve(
                y_test,
                test_consensus_probability,
                n_bins=10,
                strategy="uniform",
            )
        )

        calibration_curve_data = pd.DataFrame(
            {
                "Mean Predicted Probability": calibration_pred,
                "Observed Frequency": calibration_true,
            }
        )
    except Exception:
        calibration_curve_data = pd.DataFrame()

    # --------------------------------------------------------------------------
    # CONFORMAL TEST COVERAGE
    # --------------------------------------------------------------------------

    test_conformal_coverage, test_mean_set_size = (
        conformal_coverage(
            test_consensus_probability,
            y_test,
            q_threshold,
        )
    )

    # --------------------------------------------------------------------------
    # XAI
    # --------------------------------------------------------------------------

    feature_importance = build_feature_importance(
        feature_schema,
        rf_model,
        xgb_model,
        logistic_model,
    )

    # --------------------------------------------------------------------------
    # BMI BASELINE
    # --------------------------------------------------------------------------

    baseline_bmi_distribution = (
        calculate_bmi_distribution(
            labeled_dataframe
        )
    )

    # --------------------------------------------------------------------------
    # ARTIFACT
    # --------------------------------------------------------------------------

    artifact = {
        "artifact_version": 2,
        "app_title": APP_TITLE,
        "app_subtitle": APP_SUBTITLE,
        "models": calibrated_models,
        "base_models": base_models,
        "imputer": imputer,
        "scaler": scaler,
        "q_threshold": q_threshold,
        "conformal_alpha": CONFORMAL_ALPHA,
        "feature_schema": feature_schema,
        "feature_importance": feature_importance,
        "performance": performance,
        "roc_curve_data": roc_curve_data,
        "calibration_curve_data": calibration_curve_data,
        "uncertainty_summary": {
            "test_mean_epistemic_std": float(
                np.mean(
                    test_uncertainty[
                        "epistemic_std"
                    ]
                )
            ),
            "test_mean_aleatoric_entropy": float(
                np.mean(
                    test_uncertainty[
                        "aleatoric_entropy"
                    ]
                )
            ),
            "test_mean_predictive_entropy": float(
                np.mean(
                    test_uncertainty[
                        "predictive_entropy"
                    ]
                )
            ),
            "test_mean_mutual_information": float(
                np.mean(
                    test_uncertainty[
                        "mutual_information"
                    ]
                )
            ),
        },
        "conformal_metrics": {
            "nominal_coverage": 1.0 - CONFORMAL_ALPHA,
            "test_empirical_coverage": test_conformal_coverage,
            "test_mean_prediction_set_size": (
                test_mean_set_size
            ),
        },
        "training_rows": int(len(y_train)),
        "calibration_rows": int(len(y_cal)),
        "test_rows": int(len(y_test)),
        "total_labeled_rows": int(len(y)),
        "baseline_bmi_distribution": (
            baseline_bmi_distribution
        ),
        "target_column": TARGET_COLUMN,
        "age_column": AGE_COLUMN,
        "id_column": ID_COLUMN,
    }

    joblib.dump(
        artifact,
        ARTIFACT_PATH,
        compress=3,
    )

    return artifact


def validate_artifact(
    artifact: Dict,
) -> None:
    """
    Validate the research artifact before use.
    """
    if not isinstance(
        artifact,
        dict,
    ):
        raise ValueError(
            "Artifact is not a dictionary."
        )

    if artifact.get(
        "artifact_version"
    ) != 2:
        raise ValueError(
            "Artifact version is incompatible."
        )

    required_keys = {
        "models",
        "base_models",
        "imputer",
        "scaler",
        "q_threshold",
        "conformal_alpha",
        "feature_schema",
        "feature_importance",
        "performance",
    }

    missing = required_keys.difference(
        artifact.keys()
    )

    if missing:
        raise ValueError(
            "Artifact is missing required fields: "
            + ", ".join(sorted(missing))
        )

    model_keys = set(
        artifact["models"].keys()
    )

    if model_keys != EXPECTED_MODELS:
        raise ValueError(
            "Artifact model architecture mismatch. "
            f"Expected {EXPECTED_MODELS}, "
            f"found {model_keys}."
        )

    base_model_keys = set(
        artifact["base_models"].keys()
    )

    if base_model_keys != EXPECTED_MODELS:
        raise ValueError(
            "Artifact base-model architecture mismatch."
        )

    feature_schema = artifact[
        "feature_schema"
    ]

    if not feature_schema:
        raise ValueError(
            "Artifact contains an empty feature schema."
        )

    q_threshold = float(
        artifact["q_threshold"]
    )

    if not (
        0.0 <= q_threshold <= 1.0
    ):
        raise ValueError(
            "Invalid conformal threshold."
        )


def load_artifact() -> Tuple[
    Optional[Dict],
    Optional[str],
]:
    """
    Safely load and validate the artifact.

    A corrupt/old artifact never crashes the application.
    """
    if not os.path.exists(
        ARTIFACT_PATH
    ):
        return (
            None,
            "Artifact does not exist.",
        )

    try:
        artifact = joblib.load(
            ARTIFACT_PATH
        )
    except Exception as exc:
        return (
            None,
            f"Artifact could not be loaded: {exc}",
        )

    try:
        validate_artifact(
            artifact
        )
    except Exception as exc:
        return (
            None,
            f"Artifact validation failed: {exc}",
        )

    return artifact, None


# ==============================================================================
# 4. SIDEBAR
# ==============================================================================

with st.sidebar:

    st.header(
        "⚙️ Administrative Control Unit"
    )

    st.markdown(
        """
### Architecture

• Random Forest  
• XGBoost  
• Logistic Regression  

### Data split

• 60% Training  
• 15% Calibration  
• 25% Untouched Test  

### Conformal level

• 95% nominal level  

### Risk threshold

• 0.50  

### System status

SafeTriage-GDM is a standalone research prototype.
"""
    )

    st.divider()

    force_retrain = st.button(
        "🔄 Force Pipeline Retraining",
        width="stretch",
    )


# ==============================================================================
# 5. HEADER
# ==============================================================================

st.title(
    f"🩺 {APP_TITLE}"
)

st.subheader(
    APP_SUBTITLE
)

st.caption(
    f"Version: {APP_VERSION}"
)

st.warning(
    "SafeTriage-GDM is a research prototype for uncertainty-aware "
    "GDM risk triage, conformal safety assessment, population-shift "
    "monitoring, and algorithmic fairness auditing. It is not a medical "
    "device and must not be used as a substitute for professional medical "
    "diagnosis, treatment, or clinical decision-making."
)


# ==============================================================================
# 6. DATA UPLOAD
# ==============================================================================

st.header(
    "1. Research Dataset"
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel data",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
)

if uploaded_file is None:
    st.info(
        "Upload a research dataset to begin."
    )
    st.stop()


# ==============================================================================
# 7. LOAD DATA
# ==============================================================================

try:
    if uploaded_file.name.lower().endswith(
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

if raw_dataframe.empty:
    st.error(
        "The uploaded dataset contains no rows."
    )
    st.stop()

raw_dataframe = raw_dataframe.reset_index(
    drop=True
)

st.success(
    f"Dataset loaded successfully: "
    f"{len(raw_dataframe):,} rows × "
    f"{len(raw_dataframe.columns):,} columns."
)


# ==============================================================================
# 8. RESOLVE TARGET COLUMN
# ==============================================================================

resolved_target = resolve_column(
    raw_dataframe,
    TARGET_COLUMN,
    [
        "GDM",
        "GDM Status",
        "Gestational Diabetes",
        "Gestational Diabetes Mellitus",
    ],
)

resolved_age = resolve_column(
    raw_dataframe,
    AGE_COLUMN,
    [
        "age",
        "Maternal Age",
        "maternal age",
    ],
)


# ==============================================================================
# 9. CREATE PROCESSING DATAFRAME
# ==============================================================================

processed_dataframe = raw_dataframe.copy()

if resolved_target is not None:

    processed_dataframe[
        TARGET_COLUMN
    ] = parse_gdm_target(
        processed_dataframe[
            resolved_target
        ]
    )

else:

    processed_dataframe[
        TARGET_COLUMN
    ] = np.nan


# ==============================================================================
# 10. TRAINING / EVALUATION POPULATION
# ==============================================================================

labeled_mask = (
    processed_dataframe[
        TARGET_COLUMN
    ].notna()
)

eval_dataframe = (
    processed_dataframe
    .loc[labeled_mask]
    .copy()
)

y_eval = (
    processed_dataframe
    .loc[labeled_mask, TARGET_COLUMN]
    .astype(int)
    .to_numpy()
)

labeled_count = len(
    eval_dataframe
)

unlabeled_count = (
    len(processed_dataframe)
    - labeled_count
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Uploaded Rows",
        f"{len(processed_dataframe):,}",
    )

with col2:
    st.metric(
        "Rows With Valid GDM Target",
        f"{labeled_count:,}",
    )

with col3:
    st.metric(
        "Rows Without Valid Target",
        f"{unlabeled_count:,}",
    )

st.caption(
    "All uploaded rows remain in the inference population. "
    "Only rows with a valid GDM target are eligible for model "
    "training, test evaluation, and ground-truth fairness metrics."
)


# ==============================================================================
# 11. ARTIFACT LOADING / RETRAINING
# ==============================================================================

artifact = None
artifact_error = None

if not force_retrain:
    artifact, artifact_error = load_artifact()

if force_retrain:

    if labeled_count == 0:
        st.error(
            "Force retraining requires labeled GDM outcomes. "
            "No synthetic labels are generated."
        )
        st.stop()

    if len(
        np.unique(y_eval)
    ) != 2:
        st.error(
            "Force retraining requires both GDM classes "
            "(0 and 1)."
        )
        st.stop()

    with st.spinner(
        "Training Random Forest + XGBoost + Logistic Regression..."
    ):
        try:
            artifact = train_pipeline(
                eval_dataframe
            )
            artifact_error = None

        except Exception as exc:
            st.error(
                "Pipeline training failed."
            )
            st.exception(exc)
            st.stop()

    st.success(
        "Pipeline retrained successfully and artifact saved."
    )

elif artifact is None:

    if labeled_count == 0:

        st.error(
            "No valid GDM target values are available and the "
            "saved model artifact is unavailable or invalid. "
            "Upload a dataset containing valid GDM labels or "
            "provide a valid trained artifact."
        )

        if artifact_error:
            st.caption(
                f"Artifact status: {artifact_error}"
            )

        st.stop()

    if len(
        np.unique(y_eval)
    ) != 2:

        st.error(
            "The uploaded dataset does not contain both GDM classes. "
            "The model cannot be retrained safely."
        )

        if artifact_error:
            st.caption(
                f"Artifact status: {artifact_error}"
            )

        st.stop()

    st.warning(
        "The existing model artifact is missing, corrupt, or incompatible. "
        "Because valid labeled data are available, SafeTriage-GDM is "
        "rebuilding the complete three-model pipeline."
    )

    if artifact_error:
        st.caption(
            f"Previous artifact status: {artifact_error}"
        )

    with st.spinner(
        "Rebuilding Random Forest + XGBoost + Logistic Regression..."
    ):
        try:
            artifact = train_pipeline(
                eval_dataframe
            )

        except Exception as exc:
            st.error(
                "Automatic pipeline rebuild failed."
            )
            st.exception(exc)
            st.stop()

    st.success(
        "A new validated model artifact has been created."
    )


# ==============================================================================
# 12. ARTIFACT INTEGRITY CHECK
# ==============================================================================

try:
    validate_artifact(
        artifact
    )
except Exception as exc:
    st.error(
        "The loaded model artifact failed the architecture "
        "integrity check."
    )
    st.exception(exc)
    st.stop()


# ==============================================================================
# 13. PREPARE ALL UPLOADED ROWS FOR INFERENCE
# ==============================================================================

X_inference_raw = (
    processed_dataframe
    .drop(
        columns=[
            TARGET_COLUMN,
            ID_COLUMN,
        ],
        errors="ignore",
    )
    .copy()
)

X_inference_encoded = encode_dataframe(
    X_inference_raw
)

X_inference_aligned = align_to_schema(
    X_inference_encoded,
    artifact["feature_schema"],
)

try:

    X_inference_imputed = (
        artifact["imputer"]
        .transform(
            X_inference_aligned
        )
    )

    X_inference_scaled = (
        artifact["scaler"]
        .transform(
            X_inference_imputed
        )
    )

except Exception as exc:

    st.error(
        "Inference preprocessing failed. "
        "The uploaded dataset may be incompatible with "
        "the trained feature schema."
    )

    st.exception(exc)

    st.stop()


# ==============================================================================
# 14. THREE-MODEL INFERENCE
# ==============================================================================

model_names = [
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
]

probability_vectors = []

for model_name in model_names:

    try:

        probability = get_positive_probability(
            artifact["models"][model_name],
            X_inference_scaled,
        )

    except Exception as exc:

        st.error(
            f"{model_name} inference failed."
        )

        st.exception(exc)

        st.stop()

    if len(probability) != len(
        processed_dataframe
    ):
        st.error(
            f"{model_name} returned "
            f"{len(probability):,} predictions for "
            f"{len(processed_dataframe):,} uploaded rows."
        )
        st.stop()

    probability_vectors.append(
        probability
    )


# Strictly enforce three-model architecture.
if len(probability_vectors) != 3:
    st.error(
        "SafeTriage-GDM requires exactly three model probability vectors."
    )
    st.stop()

model_probability_matrix = np.vstack(
    probability_vectors
)

uncertainty = uncertainty_decomposition(
    model_probability_matrix
)

consensus_prob = uncertainty[
    "consensus_probability"
]

epistemic_std = uncertainty[
    "epistemic_std"
]

aleatoric_entropy = uncertainty[
    "aleatoric_entropy"
]

predictive_entropy = uncertainty[
    "predictive_entropy"
]

mutual_information = uncertainty[
    "mutual_information"
]


# ==============================================================================
# 15. RISK CLASSIFICATION
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
# 16. CONFORMAL PREDICTION SETS
# ==============================================================================

q_threshold = float(
    artifact["q_threshold"]
)

conformal_sets = conformal_prediction_sets(
    consensus_prob,
    q_threshold,
)


# ==============================================================================
# 17. FAIRNESS AUDIT
# ==============================================================================

try:

    fairness_metrics = (
        calculate_fairness_metrics(
            processed_dataframe,
            consensus_prob,
        )
    )

except Exception as exc:

    st.error(
        "Fairness audit failed because the prediction and "
        "demographic populations were not aligned."
    )

    st.exception(exc)

    st.stop()


# ==============================================================================
# 18. BMI DRIFT MONITORING
# ==============================================================================

current_bmi_distribution = (
    calculate_bmi_distribution(
        processed_dataframe
    )
)

baseline_bmi_distribution = artifact.get(
    "baseline_bmi_distribution"
)

psi_value = calculate_psi(
    baseline_bmi_distribution,
    current_bmi_distribution,
)


# ==============================================================================
# 19. MAIN DASHBOARD
# ==============================================================================

st.header(
    "2. Patient Triage Console"
)

st.markdown(
    """
SafeTriage-GDM provides an estimated probability of GDM risk together
with a conformal prediction set representing predictive uncertainty.

The system separates:

**Risk estimation** — the ensemble probability of GDM.

**Triage classification** — the operational low-risk/high-risk
classification.

**Conformal safety bounds** — the uncertainty-aware prediction set
used to expose ambiguous cases.
"""
)


# --------------------------------------------------------------------------
# Summary metrics
# --------------------------------------------------------------------------

summary_col1, summary_col2, summary_col3, summary_col4 = (
    st.columns(4)
)

with summary_col1:
    st.metric(
        "Mean GDM Risk Probability",
        f"{np.mean(consensus_prob):.3f}",
    )

with summary_col2:
    st.metric(
        "High-Risk Classifications",
        f"{int(np.sum(risk_flags)):,}",
    )

with summary_col3:
    st.metric(
        "Ambiguous Conformal Sets",
        f"{sum('Healthy, GDM' in s for s in conformal_sets):,}",
    )

with summary_col4:
    st.metric(
        "Mean Epistemic Uncertainty",
        f"{np.mean(epistemic_std):.3f}",
    )


# ==============================================================================
# 20. MODEL PROBABILITIES
# ==============================================================================

st.subheader(
    "Three-Model Probability Decomposition"
)

model_probability_df = pd.DataFrame(
    {
        "Random Forest P(GDM)": probability_vectors[0],
        "XGBoost P(GDM)": probability_vectors[1],
        "Logistic Regression P(GDM)": probability_vectors[2],
        "Ensemble P(GDM)": consensus_prob,
    }
)

st.dataframe(
    model_probability_df.describe().T,
    width="stretch",
)


# ==============================================================================
# 21. UNCERTAINTY
# ==============================================================================

st.subheader(
    "Uncertainty Quantification"
)

st.markdown(
    """
**Epistemic uncertainty** is reported as ensemble model disagreement
(the standard deviation of the three model probabilities) together
with mutual information. This is a model-disagreement proxy, not a
Bayesian posterior variance.

**Aleatoric uncertainty** is approximated by expected Bernoulli
predictive entropy across the three calibrated models. It reflects
uncertainty inherent in the predicted outcome distribution, not a
direct measurement of clinical measurement noise.

**Predictive entropy** is the entropy of the ensemble mean probability.
"""
)

unc_col1, unc_col2, unc_col3, unc_col4 = st.columns(4)

with unc_col1:
    st.metric(
        "Epistemic — Model Disagreement",
        f"{np.mean(epistemic_std):.4f}",
    )

with unc_col2:
    st.metric(
        "Aleatoric — Expected Entropy",
        f"{np.mean(aleatoric_entropy):.4f}",
    )

with unc_col3:
    st.metric(
        "Predictive Entropy",
        f"{np.mean(predictive_entropy):.4f}",
    )

with unc_col4:
    st.metric(
        "Mutual Information",
        f"{np.mean(mutual_information):.4f}",
    )


# ==============================================================================
# 22. CONFORMAL SAFETY BOUNDS
# ==============================================================================

st.subheader(
    "Conformal Safety Bounds"
)

conformal_col1, conformal_col2, conformal_col3 = (
    st.columns(3)
)

with conformal_col1:
    st.metric(
        "Nominal Coverage",
        f"{(1.0 - CONFORMAL_ALPHA) * 100:.1f}%",
    )

with conformal_col2:
    st.metric(
        "Calibrated Nonconformity Threshold",
        f"{q_threshold:.4f}",
    )

with conformal_col3:
    st.metric(
        "Ambiguous Prediction Sets",
        f"{sum('Healthy, GDM' in s for s in conformal_sets):,}",
    )

st.info(
    "The 95% conformal level is calibrated on the dedicated calibration "
    "split. Empirical coverage is evaluated separately on the untouched "
    "test set. Formal coverage guarantees depend on the exchangeability "
    "assumptions of split conformal prediction and should not be "
    "interpreted as guaranteed under population shift."
)


# ==============================================================================
# 23. DRIFT
# ==============================================================================

st.subheader(
    "Population-Shift Monitoring"
)

if np.isnan(psi_value):

    st.info(
        "BMI drift monitoring is unavailable because a compatible BMI "
        "variable was not detected."
    )

else:

    drift_col1, drift_col2 = st.columns(2)

    with drift_col1:
        st.metric(
            "BMI PSI",
            f"{psi_value:.4f}",
        )

    with drift_col2:

        if psi_value >= PSI_WARNING_LIMIT:
            st.warning(
                "BMI population shift exceeds the configured monitoring "
                "threshold of 0.20."
            )
        else:
            st.success(
                "BMI population shift is below the configured monitoring "
                "threshold of 0.20."
            )

    st.caption(
        "PSI is used here as a population-shift monitoring statistic. "
        "It does not itself establish clinical safety or invalidate the "
        "model."
    )


# ==============================================================================
# 24. FULL INFERENCE TABLE
# ==============================================================================

st.subheader(
    "Complete Inference Population"
)

results_dashboard = pd.DataFrame(
    {
        "Random Forest P(GDM)": probability_vectors[0],
        "XGBoost P(GDM)": probability_vectors[1],
        "Logistic Regression P(GDM)": probability_vectors[2],
        "GDM Risk Probability": consensus_prob,
        "Risk Classification": risk_labels,
        "Conformal Prediction Set": conformal_sets,
        "Model Positive Flag": risk_flags,
        "Epistemic Uncertainty (Model Disagreement)": epistemic_std,
        "Aleatoric Uncertainty (Expected Entropy)": aleatoric_entropy,
        "Predictive Entropy": predictive_entropy,
        "Mutual Information": mutual_information,
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

st.dataframe(
    final_output_view,
    width="stretch",
    hide_index=True,
)


# ==============================================================================
# 25. DOWNLOAD
# ==============================================================================

download_csv = (
    final_output_view.to_csv(
        index=False
    ).encode("utf-8")
)

st.download_button(
    label="⬇️ Download Complete Triage Results",
    data=download_csv,
    file_name="safetriage_gdm_results.csv",
    mime="text/csv",
    width="stretch",
)


# ==============================================================================
# 26. TABS
# ==============================================================================

tab_xai, tab_performance, tab_fairness = st.tabs(
    [
        "🔎 XAI Attribution",
        "📈 Performance & Uncertainty",
        "⚖️ Fairness Audit",
    ]
)


# ==============================================================================
# 27. XAI TAB
# ==============================================================================

with tab_xai:

    st.header(
        "Global Feature Attribution"
    )

    st.markdown(
        """
The attribution analysis summarizes global feature importance from
the trained components of SafeTriage-GDM.

These values describe the contribution of features to model predictions
within the research system. They should not be interpreted as causal
effects or as evidence that a specific characteristic independently
causes GDM.
"""
    )

    xai_df = artifact[
        "feature_importance"
    ].copy()

    if xai_df.empty:

        st.info(
            "No feature-attribution information is available."
        )

    else:

        display_xai = xai_df.copy()

        for column in [
            "Random Forest",
            "XGBoost",
            "Logistic Regression",
            "Attribution Weight Score",
        ]:
            if column in display_xai.columns:
                display_xai[column] = (
                    display_xai[column]
                    .astype(float)
                    .round(6)
                )

        # IMPORTANT:
        # No background_gradient() is used.
        # Therefore matplotlib is NOT required.
        st.dataframe(
            display_xai,
            width="stretch",
            hide_index=True,
        )


# ==============================================================================
# 28. PERFORMANCE TAB
# ==============================================================================

with tab_performance:

    st.header(
        "Pipeline Predictive Performance"
    )

    st.markdown(
        """
These performance measures describe the behaviour of the trained
SafeTriage-GDM model on its untouched held-out test set. They are
distinct from predictions generated for the currently uploaded
inference population.
"""
    )

    performance = artifact[
        "performance"
    ]

    performance_rows = []

    for model_name in [
        "Random Forest",
        "XGBoost",
        "Logistic Regression",
        "Ensemble",
    ]:

        metrics = performance[
            model_name
        ]

        performance_rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Sensitivity / Recall": metrics[
                    "Sensitivity / Recall"
                ],
                "Specificity": metrics[
                    "Specificity"
                ],
                "Precision": metrics[
                    "Precision"
                ],
                "F1 Score": metrics[
                    "F1 Score"
                ],
                "ROC-AUC": metrics[
                    "ROC-AUC"
                ],
                "PR-AUC": metrics[
                    "PR-AUC"
                ],
                "Brier Score": metrics[
                    "Brier Score"
                ],
                "Log Loss": metrics[
                    "Log Loss"
                ],
            }
        )

    performance_table = pd.DataFrame(
        performance_rows
    )

    st.dataframe(
        performance_table,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    ensemble_metrics = performance[
        "Ensemble"
    ]

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    with metric_col1:
        st.metric(
            "ROC-AUC",
            (
                f"{ensemble_metrics['ROC-AUC']:.4f}"
                if not np.isnan(
                    ensemble_metrics["ROC-AUC"]
                )
                else "N/A"
            ),
        )

    with metric_col2:
        st.metric(
            "PR-AUC",
            (
                f"{ensemble_metrics['PR-AUC']:.4f}"
                if not np.isnan(
                    ensemble_metrics["PR-AUC"]
                )
                else "N/A"
            ),
        )

    with metric_col3:
        st.metric(
            "Brier Score",
            (
                f"{ensemble_metrics['Brier Score']:.4f}"
                if not np.isnan(
                    ensemble_metrics["Brier Score"]
                )
                else "N/A"
            ),
        )

    with metric_col4:
        st.metric(
            "Log Loss",
            (
                f"{ensemble_metrics['Log Loss']:.4f}"
                if not np.isnan(
                    ensemble_metrics["Log Loss"]
                )
                else "N/A"
            ),
        )

    # --------------------------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------------------------

    st.subheader(
        "ROC Curve — Untouched Test Set"
    )

    roc_data = artifact.get(
        "roc_curve_data"
    )

    if (
        roc_data is not None
        and not roc_data.empty
    ):

        roc_line = (
            alt.Chart(
                roc_data
            )
            .mark_line()
            .encode(
                x=alt.X(
                    "False Positive Rate",
                    title="False Positive Rate",
                ),
                y=alt.Y(
                    "True Positive Rate",
                    title="True Positive Rate",
                ),
                tooltip=[
                    "False Positive Rate",
                    "True Positive Rate",
                ],
            )
        )

        diagonal_data = pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
            }
        )

        diagonal = (
            alt.Chart(
                diagonal_data
            )
            .mark_line(
                strokeDash=[5, 5]
            )
            .encode(
                x=alt.X(
                    "x",
                    title="False Positive Rate",
                ),
                y=alt.Y(
                    "y",
                    title="True Positive Rate",
                ),
            )
        )

        st.altair_chart(
            roc_line + diagonal,
            width="stretch",
        )

    else:

        st.info(
            "ROC curve data are unavailable."
        )

    # --------------------------------------------------------------------------
    # CALIBRATION CURVE
    # --------------------------------------------------------------------------

    st.subheader(
        "Calibration / Reliability Curve"
    )

    calibration_data = artifact.get(
        "calibration_curve_data"
    )

    if (
        calibration_data is not None
        and not calibration_data.empty
    ):

        calibration_line = (
            alt.Chart(
                calibration_data
            )
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "Mean Predicted Probability",
                    title="Mean Predicted Probability",
                ),
                y=alt.Y(
                    "Observed Frequency",
                    title="Observed Frequency",
                ),
                tooltip=[
                    "Mean Predicted Probability",
                    "Observed Frequency",
                ],
            )
        )

        calibration_reference_data = pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
            }
        )

        calibration_reference = (
            alt.Chart(
                calibration_reference_data
            )
            .mark_line(
                strokeDash=[5, 5]
            )
            .encode(
                x=alt.X(
                    "x",
                    title="Mean Predicted Probability",
                ),
                y=alt.Y(
                    "y",
                    title="Observed Frequency",
                ),
            )
        )

        st.altair_chart(
            calibration_line
            + calibration_reference,
            width="stretch",
        )

    else:

        st.info(
            "Calibration curve data are unavailable."
        )

    # --------------------------------------------------------------------------
    # TEST CONFUSION MATRIX
    # --------------------------------------------------------------------------

    st.subheader(
        "Ensemble Confusion Matrix"
    )

    cm_data = pd.DataFrame(
        {
            "": [
                "Actual Healthy",
                "Actual GDM",
            ],
            "Predicted Healthy": [
                ensemble_metrics["TN"],
                ensemble_metrics["FN"],
            ],
            "Predicted GDM": [
                ensemble_metrics["FP"],
                ensemble_metrics["TP"],
            ],
        }
    )

    st.dataframe(
        cm_data,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------------------------
    # TEST UNCERTAINTY SUMMARY
    # --------------------------------------------------------------------------

    st.subheader(
        "Test-Set Uncertainty Summary"
    )

    uncertainty_summary = artifact[
        "uncertainty_summary"
    ]

    uncertainty_summary_df = pd.DataFrame(
        {
            "Uncertainty Measure": [
                "Epistemic — Model Disagreement",
                "Aleatoric — Expected Entropy",
                "Predictive Entropy",
                "Mutual Information",
            ],
            "Mean Value": [
                uncertainty_summary[
                    "test_mean_epistemic_std"
                ],
                uncertainty_summary[
                    "test_mean_aleatoric_entropy"
                ],
                uncertainty_summary[
                    "test_mean_predictive_entropy"
                ],
                uncertainty_summary[
                    "test_mean_mutual_information"
                ],
            ],
        }
    )

    st.dataframe(
        uncertainty_summary_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------------------------
    # CONFORMAL TEST PERFORMANCE
    # --------------------------------------------------------------------------

    st.subheader(
        "Conformal Test-Set Assessment"
    )

    conformal_metrics = artifact[
        "conformal_metrics"
    ]

    conformal_performance_df = pd.DataFrame(
        {
            "Measure": [
                "Nominal Coverage",
                "Empirical Test Coverage",
                "Mean Prediction-Set Size",
            ],
            "Value": [
                conformal_metrics[
                    "nominal_coverage"
                ],
                conformal_metrics[
                    "test_empirical_coverage"
                ],
                conformal_metrics[
                    "test_mean_prediction_set_size"
                ],
            ],
        }
    )

    st.dataframe(
        conformal_performance_df,
        width="stretch",
        hide_index=True,
    )


# ==============================================================================
# 29. FAIRNESS TAB
# ==============================================================================

with tab_fairness:

    st.header(
        "Algorithmic Fairness Audit"
    )

    st.markdown(
        """
The fairness audit is calculated against the current uploaded
population.

Demographic parity compares the positive prediction rate between
participants younger than 35 and participants aged 35 or older.

FPR disparity compares false-positive rates between the same age groups,
but only among records with valid ground-truth GDM outcomes.
"""
    )

    fair_col1, fair_col2, fair_col3 = st.columns(3)

    with fair_col1:

        st.metric(
            "Younger Positive Rate",
            (
                f"{fairness_metrics['younger_positive_rate']:.3f}"
                if not np.isnan(
                    fairness_metrics[
                        "younger_positive_rate"
                    ]
                )
                else "N/A"
            ),
        )

    with fair_col2:

        st.metric(
            "Older Positive Rate",
            (
                f"{fairness_metrics['older_positive_rate']:.3f}"
                if not np.isnan(
                    fairness_metrics[
                        "older_positive_rate"
                    ]
                )
                else "N/A"
            ),
        )

    with fair_col3:

        st.metric(
            "Demographic Parity Difference",
            (
                f"{fairness_metrics['demographic_parity_difference']:.3f}"
                if not np.isnan(
                    fairness_metrics[
                        "demographic_parity_difference"
                    ]
                )
                else "N/A"
            ),
        )

    st.divider()

    fpr_col1, fpr_col2, fpr_col3 = st.columns(3)

    with fpr_col1:

        st.metric(
            "Younger FPR",
            (
                f"{fairness_metrics['younger_fpr']:.3f}"
                if not np.isnan(
                    fairness_metrics[
                        "younger_fpr"
                    ]
                )
                else "N/A"
            ),
        )

    with fpr_col2:

        st.metric(
            "Older FPR",
            (
                f"{fairness_metrics['older_fpr']:.3f}"
                if not np.isnan(
                    fairness_metrics[
                        "older_fpr"
                    ]
                )
                else "N/A"
            ),
        )

    with fpr_col3:

        st.metric(
            "FPR Disparity",
            (
                f"{fairness_metrics['fpr_disparity']:.3f}"
                if not np.isnan(
                    fairness_metrics[
                        "fpr_disparity"
                    ]
                )
                else "N/A"
            ),
        )

    st.caption(
        f"Ground-truth fairness evaluation sample size: "
        f"{fairness_metrics['fairness_sample_size']:,} rows."
    )

    st.info(
        "These are auditing statistics, not clinical safety thresholds. "
        "A disparity measure should be interpreted in the context of "
        "sample size, prevalence, uncertainty, and the intended use of "
        "the model."
    )


# ==============================================================================
# 30. FOOTER
# ==============================================================================

st.divider()

st.caption(
    "SafeTriage-GDM — standalone research prototype. "
    "Not a medical device. Not for diagnosis or treatment."
)
