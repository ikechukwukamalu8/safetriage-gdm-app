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
#   5. Development-data-weighted probability ensemble
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
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
    fbeta_score,
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
# VISUALIZATION HELPERS
# ==============================================================================


def render_bar_chart(df_chart, x, y, title, y_title=None, text=None, horizontal=False):
    """Render a consistent interactive Plotly bar chart."""
    fig = px.bar(
        df_chart,
        x=x,
        y=y,
        text=text,
        title=title,
        orientation="h" if horizontal else "v",
    )
    if y_title:
        fig.update_yaxes(title=y_title)
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_model_performance_chart(individual_table):
    """Compare the three locked models across core discrimination metrics."""
    plot_df = individual_table[[
        "Model", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "F1"
    ]].copy()
    long_df = plot_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig = px.bar(
        long_df,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Individual Model Performance",
    )
    fig.update_yaxes(range=[0, 1], title="Score")
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_roc_pr_curves(y_true, probability_dict, title_prefix="Test-set"):
    """Render ROC and precision-recall curves for the locked models plus ensemble."""
    roc_fig = go.Figure()
    pr_fig = go.Figure()

    for model_name, probability in probability_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probability)
        roc_auc = roc_auc_score(y_true, probability)
        roc_fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{model_name} (AUC={roc_auc:.3f})",
            )
        )

        precision, recall, _ = precision_recall_curve(y_true, probability)
        pr_auc = average_precision_score(y_true, probability)
        pr_fig.add_trace(
            go.Scatter(
                x=recall, y=precision, mode="lines",
                name=f"{model_name} (AP={pr_auc:.3f})",
            )
        )

    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Chance", line=dict(dash="dash"),
        )
    )
    roc_fig.update_layout(
        title=f"{title_prefix} ROC Curves",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        margin=dict(l=20, r=20, t=55, b=20),
    )
    pr_fig.update_layout(
        title=f"{title_prefix} Precision–Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        margin=dict(l=20, r=20, t=55, b=20),
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(roc_fig, use_container_width=True)
    with c2:
        st.plotly_chart(pr_fig, use_container_width=True)


def render_confusion_matrix(y_true, y_pred, title="Ensemble Confusion Matrix"):
    """Render an annotated 2x2 confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted No GDM", "Predicted GDM"],
        y=["Actual No GDM", "Actual GDM"],
        title=title,
        labels={"x": "Prediction", "y": "Observed outcome", "color": "Count"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_threshold_curve(threshold_table, selected_threshold):
    """Show calibration-derived sensitivity, specificity and balanced accuracy."""
    plot_df = threshold_table.copy()
    fig = go.Figure()
    for column in ["sensitivity", "specificity", "balanced_accuracy", "F2"]:
        fig.add_trace(
            go.Scatter(
                x=plot_df["threshold"],
                y=plot_df[column],
                mode="lines",
                name=column.replace("_", " ").title(),
            )
        )
    fig.add_vline(x=selected_threshold, line_dash="dash", annotation_text=f"Selected = {selected_threshold:.3f}")
    fig.update_layout(
        title="Calibration Threshold Selection",
        xaxis_title="Decision threshold",
        yaxis_title="Score",
        xaxis=dict(range=[0.05, 0.95]), yaxis=dict(range=[0, 1]),
        margin=dict(l=20, r=20, t=55, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_conformal_chart(conformal_sets, title="Conformal Prediction Sets"):
    counts = (
        pd.Series(conformal_sets)
        .value_counts()
        .rename_axis("Conformal set")
        .reset_index(name="Count")
    )
    fig = px.bar(
        counts, x="Conformal set", y="Count", text="Count",
        title=title,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_uncertainty_chart(uncertainty_dict, title="Ensemble Uncertainty"):
    uncertainty_df = pd.DataFrame({
        "Measure": [
            "Epistemic uncertainty",
            "Aleatoric uncertainty",
            "Predictive entropy",
            "Mutual information",
        ],
        "Mean": [
            np.mean(uncertainty_dict["epistemic_uncertainty"]),
            np.mean(uncertainty_dict["aleatoric_uncertainty"]),
            np.mean(uncertainty_dict["predictive_entropy"]),
            np.mean(uncertainty_dict["mutual_information"]),
        ],
    })
    fig = px.bar(
        uncertainty_df, x="Measure", y="Mean", text="Mean",
        title=title,
    )
    fig.update_yaxes(rangemode="tozero", title="Mean value")
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_fairness_chart(fairness, title="Age-group Fairness Audit"):
    if fairness.empty:
        return
    plot_df = fairness[["Age group", "Selection rate", "False-positive rate"]].melt(
        id_vars="Age group", var_name="Measure", value_name="Rate"
    )
    fig = px.bar(
        plot_df, x="Age group", y="Rate", color="Measure",
        barmode="group", title=title,
    )
    fig.update_yaxes(range=[0, 1], title="Rate")
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_bmi_psi_chart(reference_bmi, current_bmi, bins, psi_value, title="BMI Population Stability"):
    labels = ["Underweight", "Normal", "Overweight", "Obesity I", "Obesity II+"]
    reference = pd.to_numeric(reference_bmi, errors="coerce").dropna()
    current = pd.to_numeric(current_bmi, errors="coerce").dropna()
    if reference.empty or current.empty:
        return
    ref_props = pd.cut(reference, bins=bins, include_lowest=True).value_counts(sort=False, normalize=True)
    cur_props = pd.cut(current, bins=bins, include_lowest=True).value_counts(sort=False, normalize=True)
    plot_df = pd.DataFrame({
        "BMI category": labels,
        "Reference": ref_props.to_numpy() * 100,
        "Current / uploaded": cur_props.to_numpy() * 100,
    }).melt(id_vars="BMI category", var_name="Population", value_name="Percent")
    fig = px.bar(
        plot_df, x="BMI category", y="Percent", color="Population",
        barmode="group", title=f"{title} (PSI={psi_value:.4f})",
    )
    fig.update_yaxes(title="Percent of observations")
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(feature_importance, top_n=15):
    plot_df = feature_importance.head(top_n).sort_values("importance", ascending=True)
    fig = px.bar(
        plot_df, x="importance", y="original_variable", orientation="h",
        title=f"Top {min(top_n, len(feature_importance))} Predictive Feature Contributions",
    )
    fig.update_xaxes(title="Mean model-based importance")
    fig.update_yaxes(title="Predictor")
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_probability_distribution(y_true, probability, title="Test-set GDM Probability Distribution"):
    plot_df = pd.DataFrame({
        "Observed outcome": np.where(np.asarray(y_true) == 1, "GDM", "No GDM"),
        "Predicted probability": probability,
    })
    fig = px.histogram(
        plot_df, x="Predicted probability", color="Observed outcome",
        barmode="overlay", nbins=20, opacity=0.65, title=title,
    )
    fig.update_xaxes(range=[0, 1], title="Ensemble P(GDM)")
    fig.update_yaxes(title="Number of observations")
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


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

# Strict 28-week landmark predictor schema.
# Only predictors that are defensible as known by the GDM assessment
# landmark are accepted by the application.
BASE_PREDICTORS = [
    "Mother's age (years)",
    "Mother's pre-pregnancy BMI (kg/m2)",
    "Parity",
]
MMS_PREDICTOR = "MMS started by 28 weeks"
APPROVED_PREDICTORS = BASE_PREDICTORS.copy()
MODEL_CONFIGURATION = "3-predictor"

EXPECTED_MODEL_ORDER = [
    "Random Forest",
    "XGBoost",
    "Logistic Regression",
]

EXPECTED_MODEL_KEYS = set(EXPECTED_MODEL_ORDER)

# Inner development-set tuning. Hyperparameters are selected using only the
# original training partition; calibration and test partitions remain untouched.
TUNING_FOLDS = 5
TUNING_ENABLED = True
ENSEMBLE_WEIGHT_STEP = 0.05
CALIBRATION_OBJECTIVE = "F2"
# Prevent recall-only threshold selection from collapsing into an almost-all-positive rule.
# This is a research design constraint, not a clinical performance claim.
CALIBRATION_MIN_SPECIFICITY = 0.60

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
    st.info(
        "The predictor configuration is selected explicitly above. The 3-predictor "
        "model uses maternal age, pre-pregnancy BMI, and parity. The 4-predictor "
        "model additionally uses whether Maternal Multiple Micronutrient Supplementation "
        "(MMS) started by week 28. Whole-pregnancy MMS exposure, MMS stop, duration, "
        "delivery outcomes, and newborn measurements are not used as predictors."
    )

    st.markdown(
        """
        **Locked architecture**

        1. Strict 28-week landmark-safe predictors
        2. 60/15/25 stratified split
        3. MICE-style imputation
        4. Training-only ROSE-style balancing
        5. Random Forest
        6. XGBoost
        7. Logistic Regression
        8. Development-data-weighted ensemble
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

    st.caption(
        "Interactive Plotly visualizations are generated from the existing "
        "model outputs; they do not alter the modelling pipeline."
    )

    st.divider()

    analysis_mode = st.radio(
        "Analysis mode",
        [
            "Build & Evaluate Model",
            "External Validation",
        ],
        index=0,
    )

    st.divider()

    predictor_configuration = st.radio(
        "Predictor configuration",
        [
            "3-predictor model",
            "4-predictor model",
        ],
        index=0,
        help=(
            "Choose the model specification explicitly. The 3-predictor model uses "
            "maternal age, pre-pregnancy BMI, and parity. The 4-predictor model adds "
            "Maternal Multiple Micronutrient Supplementation (MMS) started by week 28. "
            "Extra MMS columns are ignored in 3-predictor mode."
        ),
    )

    st.caption(
        "3-predictor: age + pre-pregnancy BMI + parity.  "
        "4-predictor: age + pre-pregnancy BMI + parity + MMS started by week 28."
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


def create_models(model_params: Optional[Dict[str, Dict]] = None) -> Dict[str, object]:
    """Create the fixed three-model architecture, optionally with tuned parameters."""
    params = model_params or {}

    rf_params = {
        "n_estimators": 300,
        "min_samples_leaf": 3,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    rf_params.update(params.get("Random Forest", {}))

    xgb_params = {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    xgb_params.update(params.get("XGBoost", {}))

    lr_params = {
        "max_iter": 5000,
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
    }
    lr_params.update(params.get("Logistic Regression", {}))

    return {
        "Random Forest": RandomForestClassifier(**rf_params),
        "XGBoost": XGBClassifier(**xgb_params),
        "Logistic Regression": LogisticRegression(**lr_params),
    }


def _candidate_model_parameters() -> Dict[str, List[Dict]]:
    """Small, deliberately bounded tuning spaces suitable for the research app."""
    return {
        "Random Forest": [
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
            {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 2, "max_features": "sqrt"},
            {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 3, "max_features": "sqrt"},
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt"},
            {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 5, "max_features": 1.0},
        ],
        "XGBoost": [
            {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.03, "min_child_weight": 1, "subsample": 0.9, "colsample_bytree": 0.9},
            {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.03, "min_child_weight": 3, "subsample": 0.9, "colsample_bytree": 0.9},
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 1, "subsample": 0.9, "colsample_bytree": 0.9},
            {"n_estimators": 300, "max_depth": 2, "learning_rate": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.9},
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.9},
        ],
        "Logistic Regression": [
            {"C": 0.01},
            {"C": 0.1},
            {"C": 1.0},
            {"C": 10.0},
        ],
    }


@st.cache_data(show_spinner=False)
def tune_models_on_training_data(X_train: pd.DataFrame, y_train: pd.Series):
    """Tune the three fixed model families using only the training partition.

    Each CV fold independently applies ROSE-style balancing and preprocessing to
    its fold-training data. The validation fold remains naturally distributed.
    This prevents the calibration and final test partitions from influencing
    hyperparameter selection.
    """
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(
        n_splits=TUNING_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    candidates = _candidate_model_parameters()
    tuning_rows = []
    best_params = {}

    for model_name in EXPECTED_MODEL_ORDER:
        best_key = None
        best_score = -np.inf
        best_pr = -np.inf

        for candidate_index, candidate in enumerate(candidates[model_name], start=1):
            fold_auc = []
            fold_pr = []

            for fold_number, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
                X_fold = X_train.iloc[train_idx].copy()
                y_fold = y_train.iloc[train_idx].copy()
                X_valid = X_train.iloc[valid_idx].copy()
                y_valid = y_train.iloc[valid_idx].copy()

                X_bal, y_bal, _ = rose_style_balance(
                    X_fold, y_fold,
                    random_state=RANDOM_STATE + fold_number,
                    noise_fraction=0.10,
                )
                fold_preprocessor = make_preprocessor(X_bal)
                X_bal_processed = fold_preprocessor.fit_transform(X_bal)
                X_valid_processed = fold_preprocessor.transform(X_valid)

                model = create_models({model_name: candidate})[model_name]
                model.fit(X_bal_processed, y_bal)
                probability = model.predict_proba(X_valid_processed)[:, 1]

                try:
                    fold_auc.append(roc_auc_score(y_valid, probability))
                except Exception:
                    fold_auc.append(np.nan)
                try:
                    fold_pr.append(average_precision_score(y_valid, probability))
                except Exception:
                    fold_pr.append(np.nan)

            mean_auc = float(np.nanmean(fold_auc))
            mean_pr = float(np.nanmean(fold_pr))
            tuning_rows.append({
                "Model": model_name,
                "Candidate": candidate_index,
                "CV ROC-AUC": mean_auc,
                "CV PR-AUC": mean_pr,
                "Parameters": str(candidate),
            })

            if (mean_auc > best_score) or (np.isclose(mean_auc, best_score) and mean_pr > best_pr):
                best_score = mean_auc
                best_pr = mean_pr
                best_key = candidate

        best_params[model_name] = best_key

    return best_params, pd.DataFrame(tuning_rows)


# ==============================================================================
# MODEL TRAINING
# ==============================================================================


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_calibration: pd.DataFrame,
    X_test: pd.DataFrame,
    model_params: Optional[Dict[str, Dict]] = None,
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

    models = create_models(model_params)

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
# DEVELOPMENT-DATA-WEIGHTED ENSEMBLE
# ==============================================================================


def weighted_ensemble_probability(
    probabilities: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """Combine model probabilities using weights frozen from development data."""
    matrix = np.column_stack(
        [probabilities[model_name] for model_name in EXPECTED_MODEL_ORDER]
    )
    weight_vector = np.array(
        [weights[model_name] for model_name in EXPECTED_MODEL_ORDER],
        dtype=float,
    )
    return np.average(matrix, axis=1, weights=weight_vector)


def ensemble_probability(
    probabilities: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Return weighted ensemble probabilities; equal weights are a fallback."""
    if weights is None:
        weights = {model_name: 1.0 for model_name in EXPECTED_MODEL_ORDER}
    return weighted_ensemble_probability(probabilities, weights)


@st.cache_data(show_spinner=False)
def learn_ensemble_weights_on_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Dict[str, Dict],
):
    """Learn simplex ensemble weights using OOF predictions from training data only."""
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(
        n_splits=TUNING_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE + 100,
    )
    oof = {
        model_name: np.full(len(X_train), np.nan, dtype=float)
        for model_name in EXPECTED_MODEL_ORDER
    }

    for fold_number, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_fold = X_train.iloc[train_idx].copy()
        y_fold = y_train.iloc[train_idx].copy()
        X_valid = X_train.iloc[valid_idx].copy()

        X_bal, y_bal, _ = rose_style_balance(
            X_fold, y_fold,
            random_state=RANDOM_STATE + 100 + fold_number,
            noise_fraction=0.10,
        )
        fold_preprocessor = make_preprocessor(X_bal)
        X_bal_processed = fold_preprocessor.fit_transform(X_bal)
        X_valid_processed = fold_preprocessor.transform(X_valid)

        models = create_models(model_params)
        for model_name in EXPECTED_MODEL_ORDER:
            model = clone(models[model_name])
            model.fit(X_bal_processed, y_bal)
            oof[model_name][valid_idx] = model.predict_proba(X_valid_processed)[:, 1]

    matrix = np.column_stack([oof[name] for name in EXPECTED_MODEL_ORDER])
    y_array = np.asarray(y_train).astype(int)

    best_weights = None
    best_pr = -np.inf
    best_auc = -np.inf
    records = []

    steps = int(round(1.0 / ENSEMBLE_WEIGHT_STEP))
    for i in range(steps + 1):
        for j in range(steps - i + 1):
            k = steps - i - j
            weights_vector = np.array([i, j, k], dtype=float) / steps
            if np.sum(weights_vector) == 0:
                continue
            probability = matrix @ weights_vector
            pr_auc = average_precision_score(y_array, probability)
            roc_auc = roc_auc_score(y_array, probability)
            records.append({
                "Random Forest": weights_vector[0],
                "XGBoost": weights_vector[1],
                "Logistic Regression": weights_vector[2],
                "OOF PR-AUC": pr_auc,
                "OOF ROC-AUC": roc_auc,
            })
            if (pr_auc > best_pr) or (np.isclose(pr_auc, best_pr) and roc_auc > best_auc):
                best_pr = pr_auc
                best_auc = roc_auc
                best_weights = {
                    name: float(weight)
                    for name, weight in zip(EXPECTED_MODEL_ORDER, weights_vector)
                }

    return best_weights, pd.DataFrame(records).sort_values(
        ["OOF PR-AUC", "OOF ROC-AUC"], ascending=False
    ).reset_index(drop=True)


# ==============================================================================
# THRESHOLD SELECTION
# ==============================================================================


def select_threshold(
    y_true: pd.Series,
    probability: np.ndarray,
) -> Tuple[float, pd.DataFrame]:
    """Select a recall-oriented threshold from calibration data only."""
    thresholds = np.linspace(0.05, 0.95, 181)
    records = []
    y_true = np.asarray(y_true).astype(int)

    for threshold in thresholds:
        prediction = (probability >= threshold).astype(int)
        sensitivity = recall_score(y_true, prediction, zero_division=0)
        specificity = recall_score(y_true, prediction, pos_label=0, zero_division=0)
        balanced_accuracy = (sensitivity + specificity) / 2
        f2 = fbeta_score(y_true, prediction, beta=2, zero_division=0)
        records.append({
            "threshold": threshold,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "F2": f2,
        })

    threshold_table = pd.DataFrame(records)

    # The threshold objective is recall-oriented, but it is constrained by a
    # prespecified minimum specificity. This prevents a very low threshold
    # from classifying nearly everyone as GDM simply to maximize F2.
    feasible = threshold_table[
        threshold_table["specificity"] >= CALIBRATION_MIN_SPECIFICITY
    ].copy()

    if not feasible.empty and CALIBRATION_OBJECTIVE == "F2":
        best_row = feasible.sort_values(
            ["F2", "sensitivity", "specificity", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
    elif not feasible.empty:
        best_row = feasible.sort_values(
            ["balanced_accuracy", "sensitivity", "specificity", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
    elif CALIBRATION_OBJECTIVE == "F2":
        # Defensive fallback: choose the most specific available threshold,
        # then maximize F2. This should rarely be needed because high thresholds
        # generally produce high specificity.
        best_row = threshold_table.sort_values(
            ["specificity", "F2", "sensitivity", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
    else:
        best_row = threshold_table.sort_values(
            ["specificity", "balanced_accuracy", "sensitivity", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]

    return float(best_row["threshold"]), threshold_table


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
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:

    if weights is None:
        weights = {model_name: 1.0 for model_name in EXPECTED_MODEL_ORDER}

    matrix = np.column_stack([model_probabilities[name] for name in EXPECTED_MODEL_ORDER])
    weight_vector = np.array([weights[name] for name in EXPECTED_MODEL_ORDER], dtype=float)
    weight_vector = weight_vector / weight_vector.sum()

    predictive_probability = np.average(matrix, axis=1, weights=weight_vector)
    epistemic = np.sqrt(
        np.average((matrix - predictive_probability[:, None]) ** 2, axis=1, weights=weight_vector)
    )
    aleatoric = np.average(binary_entropy(matrix), axis=1, weights=weight_vector)
    predictive_entropy = binary_entropy(predictive_probability)
    mutual_information = np.maximum(predictive_entropy - aleatoric, 0.0)

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
# FLEXIBLE DATASET SCHEMA MAPPING
# ==============================================================================

PREDICTOR_LABELS = {
    "Mother's age (years)": "Maternal age",
    "Mother's pre-pregnancy BMI (kg/m2)": "Pre-pregnancy BMI",
    "Parity": "Parity",
    "MMS started by 28 weeks": "MMS started by 28 weeks",
}

COLUMN_ALIASES = {
    "Mother's age (years)": ["maternal age", "mother age", "age years", "age"],
    "Mother's pre-pregnancy BMI (kg/m2)": ["pre pregnancy bmi", "prepregnancy bmi", "prepreg bmi", "pre pregnancy body mass index", "bmi"],
    "Parity": ["parity", "parity count", "number of previous births", "birth order"],
    "MMS started by 28 weeks": [
        "mms started by 28 weeks",
        "mms_started_by_28_weeks",
        "mmn started by 28 weeks",
        "mmn_started_by_28_weeks",
        "supplementation started by 28 weeks",
        "supplementation start by 28 weeks",
        "mms start by 28 weeks",
        "mmn start by 28 weeks",
        "maternal multiple micronutrient supplementation started by week 28",
        "multiple micronutrient supplementation started by week 28",
        # Raw CBGS start-week field is accepted only so the app can derive
        # the landmark-safe binary feature.
        "relative to the start of pregnancy when did multiple micronutrient supplementation start",
        "supplementation start timing",
        "mmn start",
        "supplementation start",
    ],
}


def normalized_column_name(value: str) -> str:
    """Normalize column names for conservative automatic matching."""
    text = str(value).strip().lower()
    for token in ["_", "-", "?", "(", ")", "/", ":", ","]:
        text = text.replace(token, " ")
    return " ".join(text.split())


# Prediction landmark: GDM assessment is based on the 75-g OGTT around week 28.
LANDMARK_WEEK = 28.0


def derive_landmark_mms(value: pd.Series) -> pd.Series:
    """Derive the 28-week MMS predictor from the raw CBGS start-week field.

    1 = supplementation started at or before week 28;
    0 = supplementation started after week 28;
    missing = start timing unavailable or non-numeric.
    """
    numeric = pd.to_numeric(value, errors="coerce")
    result = pd.Series(np.nan, index=value.index, dtype=float)
    valid = numeric.notna()
    result.loc[valid] = (numeric.loc[valid] <= LANDMARK_WEEK).astype(float)
    return result


def build_landmark_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct leakage-aware landmark predictors from raw CBGS columns."""
    out = df.copy()
    canonical = "MMS started by 28 weeks"

    if canonical in out.columns:
        out[canonical] = pd.to_numeric(out[canonical], errors="coerce")
        return out

    raw_candidates = [
        "Relative to the start of pregnancy, when did multiple micronutrient supplementation start?",
        "supplementation start timing",
        "mmn start",
        "mms start",
    ]
    normalized = {normalized_column_name(c): c for c in out.columns}
    source = None
    for candidate in raw_candidates:
        source = normalized.get(normalized_column_name(candidate))
        if source is not None:
            break

    if source is not None:
        out[canonical] = derive_landmark_mms(out[source])

    return out


def suggest_column_mapping(columns: List[str]) -> Dict[str, str]:
    """Suggest a one-to-one mapping from uploaded names to canonical variables."""
    normalized = {normalized_column_name(c): c for c in columns}
    mapping = {}
    used = set()
    for canonical in APPROVED_PREDICTORS:
        candidates = [canonical] + COLUMN_ALIASES.get(canonical, [])
        for candidate in candidates:
            source = normalized.get(normalized_column_name(candidate))
            if source is not None and source not in used:
                mapping[canonical] = source
                used.add(source)
                break
    return mapping


def schema_mapping_ui(
    df: pd.DataFrame,
    key_prefix: str,
    require_target: bool,
) -> Tuple[Dict[str, str], str]:
    """Render an editable mapping from arbitrary uploaded columns to SafeTriage variables."""
    columns = list(df.columns)
    suggestions = suggest_column_mapping(columns)
    mapping = {}
    used = set()

    st.markdown("#### Map uploaded columns to the SafeTriage-GDM schema")
    st.caption(
        "Column names and order may differ from the reference dataset. "
        "Extra columns are ignored. Each required predictor must map to one "
        "distinct uploaded column."
    )

    with st.expander("Predictor mapping", expanded=True):
        for canonical in APPROVED_PREDICTORS:
            options = ["— Not mapped —"] + [c for c in columns if c not in used]
            suggested = suggestions.get(canonical)
            default_index = options.index(suggested) if suggested in options else 0
            selected = st.selectbox(
                PREDICTOR_LABELS.get(canonical, canonical),
                options,
                index=default_index,
                key=f"{key_prefix}_{canonical}",
                help=f"SafeTriage variable: {canonical}",
            )
            if selected != "— Not mapped —":
                mapping[canonical] = selected
                used.add(selected)

    target_aliases = [
        TARGET_COLUMN,
        "gdm",
        "gdm status",
        "gdm outcome",
        "gestational diabetes",
        "gestational diabetes mellitus",
    ]
    normalized = {normalized_column_name(c): c for c in columns}
    target_suggestion = None
    for alias in target_aliases:
        candidate = normalized.get(normalized_column_name(alias))
        if candidate is not None and candidate not in used:
            target_suggestion = candidate
            break

    if require_target:
        target_options = ["— Not mapped —"] + [c for c in columns if c not in used]
        default_index = target_options.index(target_suggestion) if target_suggestion in target_options else 0
        target_source = st.selectbox(
            "GDM outcome column",
            target_options,
            index=default_index,
            key=f"{key_prefix}_target",
            help="Yes/No, Y/N, 1/0, or equivalent binary outcome encodings are supported.",
        )
    else:
        target_options = ["— No outcome / unlabeled external data —"] + [c for c in columns if c not in used]
        default_index = target_options.index(target_suggestion) if target_suggestion in target_options else 0
        target_source = st.selectbox(
            "Optional GDM outcome column",
            target_options,
            index=default_index,
            key=f"{key_prefix}_target",
            help="Map this only if the external dataset contains a known GDM outcome. Leave unmapped for unlabeled inference.",
        )

    return mapping, target_source


def derive_mms_started_by_28(value: pd.Series) -> pd.Series:
    """Derive a strict landmark-safe MMS indicator from a raw start-week field.

    Numeric gestational weeks are interpreted directly. Common trimester labels
    are supported for synthetic/external testing only; ambiguous values become
    missing rather than being guessed.
    """
    numeric = pd.to_numeric(value, errors="coerce")
    result = pd.Series(np.nan, index=value.index, dtype=float)
    numeric_mask = numeric.notna()
    result.loc[numeric_mask] = (numeric.loc[numeric_mask] <= 28).astype(float)

    text = value.astype(str).str.strip().str.lower()
    result.loc[text.str.contains("first trimester", na=False)] = 1.0
    result.loc[text.str.contains("second trimester", na=False)] = 1.0
    result.loc[text.str.contains("third trimester", na=False)] = 0.0
    result.loc[text.isin(["yes", "1", "true"])] = 1.0
    result.loc[text.isin(["no", "0", "false"])] = 0.0
    return result


def apply_schema_mapping(
    df: pd.DataFrame,
    predictor_mapping: Dict[str, str],
    target_source: str = "",
) -> pd.DataFrame:
    """Create a canonical SafeTriage dataframe using the strict 28-week schema."""
    mapped = pd.DataFrame(index=df.index)
    for canonical in APPROVED_PREDICTORS:
        source = predictor_mapping[canonical]
        if canonical == "MMS started by 28 weeks":
            normalized_source = normalized_column_name(source)
            direct_names = {
                normalized_column_name(x)
                for x in [
                    "MMS started by 28 weeks",
                    "mms_started_by_28_weeks",
                    "mmn_started_by_28_weeks",
                    "supplementation started by 28 weeks",
                ]
            }
            if normalized_source in direct_names:
                mapped[canonical] = pd.to_numeric(df[source], errors="coerce")
            else:
                mapped[canonical] = derive_mms_started_by_28(df[source])
        else:
            mapped[canonical] = df[source]
    if target_source and target_source != "— Not mapped —":
        mapped[TARGET_COLUMN] = df[target_source]
    return mapped


def read_excel_upload(uploaded_file) -> pd.DataFrame:
    """Read and clean an uploaded Excel workbook."""
    return clean_dataframe(pd.read_excel(io.BytesIO(uploaded_file.getvalue())))


def align_external_feature_types(
    X_external: pd.DataFrame,
    X_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Align external feature dtypes with the reference model's training schema.

    Numeric canonical predictors are coerced to numeric and categorical predictors
    are converted to object-compatible values. Values that cannot be converted to a
    required numeric type become missing and are subsequently handled by the frozen training imputer.
    """
    X = X_external.copy()

    for column in X.columns:
        reference_dtype = X_reference[column].dtype

        if pd.api.types.is_numeric_dtype(reference_dtype):
            if not pd.api.types.is_numeric_dtype(X[column]):
                X[column] = pd.to_numeric(X[column], errors="coerce")
            else:
                X[column] = pd.to_numeric(X[column], errors="coerce")
        else:
            # Preserve missing values while making categorical/text columns
            # compatible with the fitted categorical pipeline.
            X[column] = X[column].astype("object")

    return X


def run_external_validation_mode():
    """Develop a reference model once and evaluate it on an independent dataset."""
    global APPROVED_PREDICTORS, MODEL_CONFIGURATION

    st.subheader("External Validation")
    st.info(
        "This mode separates model development from external evaluation. "
        "The reference dataset determines preprocessing, ROSE balancing, "
        "the decision threshold, and conformal calibration. The external "
        "dataset is never used to retrain or recalibrate the frozen model."
    )

    st.markdown("### 1. Reference/development dataset")
    reference_upload = st.file_uploader(
        "Upload the reference dataset used to develop the model",
        type=["xlsx", "xls"],
        key="external_reference_upload",
    )
    if reference_upload is None:
        st.caption(
            "For a genuine external-validation study, this dataset should be "
            "independent of the external population you are evaluating."
        )
        st.stop()

    try:
        reference_raw = read_excel_upload(reference_upload)
        reference_raw = build_landmark_features(reference_raw)
    except Exception as exc:
        st.error("The reference Excel file could not be read.")
        st.exception(exc)
        st.stop()

    st.write(
        f"Reference dataset: **{reference_raw.shape[0]:,} rows × "
        f"{reference_raw.shape[1]:,} columns**"
    )

    # The researcher explicitly chooses the reference-model specification in the sidebar.
    # Extra MMS information in a 3-predictor reference dataset is intentionally ignored.
    APPROVED_PREDICTORS = (
        BASE_PREDICTORS + [MMS_PREDICTOR]
        if predictor_configuration == "4-predictor model"
        else BASE_PREDICTORS.copy()
    )
    MODEL_CONFIGURATION = (
        "4-predictor"
        if predictor_configuration == "4-predictor model"
        else "3-predictor"
    )

    reference_mapping, reference_target_source = schema_mapping_ui(
        reference_raw,
        "reference_mapping",
        require_target=True,
    )

    missing_reference = [
        c for c in APPROVED_PREDICTORS if c not in reference_mapping
    ]
    if missing_reference or reference_target_source in {"", "— Not mapped —"}:
        st.error("Complete the reference predictor mapping and GDM outcome mapping before continuing.")
        st.write("Unmapped predictors:", [PREDICTOR_LABELS[c] for c in missing_reference])
        st.stop()

    reference_df = apply_schema_mapping(
        reference_raw,
        reference_mapping,
        reference_target_source,
    )
    reference_target = normalize_binary_target(reference_df[TARGET_COLUMN])
    reference_labeled = reference_target.notna()

    if reference_labeled.sum() < 30 or reference_target.loc[reference_labeled].nunique() < 2:
        st.error("The reference dataset must contain at least 30 usable labeled rows and both GDM classes.")
        st.stop()

    st.markdown("### 2. Independent external dataset")
    external_upload = st.file_uploader(
        "Upload the independent external validation dataset",
        type=["xlsx", "xls"],
        key="external_validation_upload",
    )
    if external_upload is None:
        st.caption(
            "The external dataset may use different column names, a different "
            "column order, additional columns, and may omit the GDM outcome."
        )
        st.stop()

    try:
        external_raw = read_excel_upload(external_upload)
        external_raw = build_landmark_features(external_raw)
    except Exception as exc:
        st.error("The external Excel file could not be read.")
        st.exception(exc)
        st.stop()

    st.write(
        f"External dataset: **{external_raw.shape[0]:,} rows × "
        f"{external_raw.shape[1]:,} columns**"
    )

    external_mapping, external_target_source = schema_mapping_ui(
        external_raw,
        "external_mapping",
        require_target=False,
    )
    missing_external = [
        c for c in APPROVED_PREDICTORS if c not in external_mapping
    ]
    if missing_external:
        st.error("The external dataset is missing one or more required predictor concepts.")
        st.write("Unmapped predictors:", [PREDICTOR_LABELS[c] for c in missing_external])
        st.stop()

    if external_target_source == "— No outcome / unlabeled external data —":
        external_target_source = ""

    external_df = apply_schema_mapping(
        external_raw,
        external_mapping,
        external_target_source,
    )
    external_target = (
        normalize_binary_target(external_df[TARGET_COLUMN])
        if TARGET_COLUMN in external_df.columns
        else pd.Series(np.nan, index=external_df.index, dtype=float)
    )

    # ------------------------------------------------------------------
    # Reference model development: 60/15/25, then freeze everything.
    # ------------------------------------------------------------------
    X_reference = reference_df.loc[reference_labeled, APPROVED_PREDICTORS].copy()
    y_reference = reference_target.loc[reference_labeled].astype(int)

    try:
        X_development, X_reference_test, y_development, y_reference_test = train_test_split(
            X_reference,
            y_reference,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_reference,
        )
        X_train, X_calibration, y_train, y_calibration = train_test_split(
            X_development,
            y_development,
            test_size=CALIBRATION_SIZE_WITHIN_DEVELOPMENT,
            random_state=RANDOM_STATE,
            stratify=y_development,
        )
    except Exception as exc:
        st.error("The reference dataset could not be split into 60/15/25 stratified partitions.")
        st.exception(exc)
        st.stop()

    with st.spinner("Tuning and developing the reference model..."):
        try:
            reference_tuned_params, _ = tune_models_on_training_data(X_train, y_train)
            reference_weights, _ = learn_ensemble_weights_on_training_data(
                X_train, y_train, reference_tuned_params
            )
            (
                preprocessor,
                fitted_models,
                reference_probabilities,
                rose_information,
                _,
                _,
                _,
            ) = train_models(
                X_train,
                y_train,
                X_calibration,
                X_reference_test,
                model_params=reference_tuned_params,
            )
        except Exception as exc:
            st.error("Reference model training failed.")
            st.exception(exc)
            st.stop()

    if set(fitted_models.keys()) != EXPECTED_MODEL_KEYS:
        st.error("The frozen architecture is invalid. SafeTriage-GDM requires Random Forest, XGBoost, and Logistic Regression.")
        st.stop()

    reference_calibration_ensemble = ensemble_probability(
        {
            model_name: reference_probabilities[model_name]["calibration"]
            for model_name in EXPECTED_MODEL_ORDER
        },
        reference_weights,
    )

    frozen_threshold, _ = select_threshold(
        y_calibration,
        reference_calibration_ensemble,
    )

    # External data are transformed by the already-fitted reference preprocessor.
    # Align dtypes first because the CBGS reference dataset stores the
    # supplementation timing variables as numeric weeks relative to pregnancy
    # start. This prevents strings in an external workbook from being passed
    # into the reference IterativeImputer.
    external_features = align_external_feature_types(
        external_df[APPROVED_PREDICTORS],
        X_train,
    )

    try:
        external_processed = preprocessor.transform(external_features)
    except (TypeError, ValueError) as exc:
        st.error(
            "The external dataset could not be transformed using the frozen "
            "reference preprocessing schema. Check that numeric predictors "
            "(including supplementation timing in weeks) contain numeric "
            "values and that categorical predictors use compatible values."
        )
        with st.expander("Technical diagnostic", expanded=False):
            st.write("Reference predictor dtypes:")
            st.dataframe(
                pd.DataFrame({"Reference dtype": X_train[APPROVED_PREDICTORS].dtypes.astype(str)})
            )
            st.write("External predictor dtypes after alignment:")
            st.dataframe(
                pd.DataFrame({"External dtype": external_features.dtypes.astype(str)})
            )
            st.exception(exc)
        st.stop()
    external_probabilities = {
        model_name: fitted_models[model_name].predict_proba(external_processed)[:, 1]
        for model_name in EXPECTED_MODEL_ORDER
    }
    external_ensemble = ensemble_probability(external_probabilities, reference_weights)

    external_conformal_sets, frozen_conformal_q, _ = conformal_prediction(
        reference_calibration_ensemble,
        y_calibration,
        external_ensemble,
        confidence=CONFORMAL_CONFIDENCE,
    )

    external_predictions = (external_ensemble >= frozen_threshold).astype(int)
    external_uncertainty = calculate_uncertainty(external_probabilities, reference_weights)

    st.success(
        "Reference model frozen. The external dataset was evaluated without retraining, "
        "rebalancing, threshold optimization, or conformal recalibration."
    )

    st.subheader("3. Frozen reference model")
    ref_cols = st.columns(5)
    ref_cols[0].metric("Reference labeled", f"{len(y_reference):,}")
    ref_cols[1].metric("Training", f"{len(y_train):,}")
    ref_cols[2].metric("Calibration", f"{len(y_calibration):,}")
    ref_cols[3].metric("Synthetic GDM", f"{rose_information['synthetic_minority']:,}")
    ref_cols[4].metric("Frozen threshold", f"{frozen_threshold:.3f}")

    st.caption(
        "The reference calibration partition alone determines the decision threshold "
        "and the conformal nonconformity quantile used for the external dataset."
    )

    st.subheader("4. External validation performance")
    external_labeled = external_target.notna()
    if external_labeled.sum() >= 2 and external_target.loc[external_labeled].nunique() == 2:
        y_external = external_target.loc[external_labeled].astype(int)
        p_external = external_ensemble[external_labeled.to_numpy()]
        external_metrics = calculate_metrics(
            y_external,
            p_external,
            frozen_threshold,
        )
        metrics_table = pd.DataFrame(
            {
                "Metric": list(external_metrics.keys()),
                "Value": [
                    "NA" if pd.isna(v) else round(float(v), 4)
                    for v in external_metrics.values()
                ],
            }
        )
        st.dataframe(metrics_table, width="stretch", hide_index=True)

        external_curve_probabilities = {
            model_name: external_probabilities[model_name][external_labeled.to_numpy()]
            for model_name in EXPECTED_MODEL_ORDER
        }
        external_curve_probabilities["Ensemble"] = p_external
        render_roc_pr_curves(y_external, external_curve_probabilities, "External validation")

        tn, fp, fn, tp = confusion_matrix(
            y_external,
            (p_external >= frozen_threshold).astype(int),
            labels=[0, 1],
        ).ravel()
        st.dataframe(
            pd.DataFrame(
                {
                    "": ["Actual No GDM", "Actual GDM"],
                    "Predicted No GDM": [tn, fn],
                    "Predicted GDM": [fp, tp],
                }
            ),
            width="stretch",
            hide_index=True,
        )
        render_confusion_matrix(
            y_external,
            (p_external >= frozen_threshold).astype(int),
            "External Validation Confusion Matrix",
        )
    else:
        st.info(
            "The external dataset has no complete two-class outcome. Predictions can still be generated, "
            "but ROC-AUC, PR-AUC, sensitivity, specificity, precision, F1, Brier, and log loss cannot be evaluated."
        )

    st.subheader("5. External conformal prediction")
    st.metric("Frozen conformal quantile", f"{frozen_conformal_q:.4f}")
    conformal_counts = (
        pd.Series(external_conformal_sets)
        .value_counts()
        .rename_axis("Conformal set")
        .reset_index(name="Count")
    )
    st.dataframe(conformal_counts, width="stretch", hide_index=True)
    fig_external_conformal = px.bar(
        conformal_counts, x="Conformal set", y="Count", text="Count",
        title="External Conformal Prediction Sets",
    )
    fig_external_conformal.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig_external_conformal, use_container_width=True)
    st.caption(
        "{GDM, No GDM} means both outcomes remain plausible under the reference calibration. "
        "It is an uncertainty statement, not a diagnosis."
    )

    st.subheader("6. External uncertainty")
    uncertainty_summary = pd.DataFrame(
        {
            "Measure": [
                "Mean P(GDM)",
                "Mean epistemic uncertainty",
                "Mean aleatoric uncertainty",
                "Mean predictive entropy",
                "Mean mutual information",
            ],
            "Value": [
                np.mean(external_ensemble),
                np.mean(external_uncertainty["epistemic_uncertainty"]),
                np.mean(external_uncertainty["aleatoric_uncertainty"]),
                np.mean(external_uncertainty["predictive_entropy"]),
                np.mean(external_uncertainty["mutual_information"]),
            ],
        }
    )
    st.dataframe(uncertainty_summary.round(4), width="stretch", hide_index=True)
    render_uncertainty_chart(external_uncertainty, "External Ensemble Uncertainty")

    st.subheader("7. External fairness audit")
    fairness_input = external_df.copy()
    fairness_input[TARGET_COLUMN] = external_target
    fairness = fairness_audit(
        fairness_input,
        external_ensemble,
        frozen_threshold,
    )
    if fairness.empty:
        st.info("Age-group fairness could not be calculated from the mapped external data.")
    else:
        st.dataframe(fairness.round(4), width="stretch", hide_index=True)
        valid_fpr = fairness["False-positive rate"].dropna()
        if len(valid_fpr) >= 2:
            st.metric("External age-group FPR disparity", f"{(valid_fpr.max() - valid_fpr.min()):.4f}")
        st.caption(
            "This reports age-group selection rates and false-positive-rate disparity. "
            "It is not a complete Equalized Odds assessment."
        )
        render_fairness_chart(fairness, "External Age-group Fairness Audit")

    st.subheader("8. External BMI population-shift monitoring")
    reference_bmi = safe_numeric(X_train[BMI_COLUMN])
    external_bmi = safe_numeric(external_df[BMI_COLUMN])
    external_psi = calculate_psi(reference_bmi, external_bmi, BMI_BINS)
    if pd.isna(external_psi):
        st.info("BMI PSI could not be calculated.")
    else:
        st.metric("External BMI PSI", f"{external_psi:.4f}")
        if external_psi < 0.10:
            st.success("Minimal BMI distribution shift detected.")
        elif external_psi < PSI_THRESHOLD:
            st.warning("Moderate BMI distribution shift detected.")
        else:
            st.error(f"Substantial BMI distribution shift detected (PSI ≥ {PSI_THRESHOLD:.2f}).")
        st.caption(
            "PSI is a monitoring statistic. It does not modify the frozen model, threshold, or conformal quantile."
        )
        render_bmi_psi_chart(
            reference_bmi,
            external_bmi,
            BMI_BINS,
            external_psi,
            "External BMI Population Stability",
        )

    st.subheader("9. External predictions")
    results = external_raw.copy()
    results["SafeTriage_GDM_probability"] = external_ensemble
    results["SafeTriage_risk_classification"] = np.where(
        external_predictions == 1,
        "GDM risk",
        "Lower predicted GDM risk",
    )
    results["SafeTriage_decision_threshold"] = frozen_threshold
    results["SafeTriage_conformal_prediction_set"] = external_conformal_sets
    results["SafeTriage_conformal_q_reference"] = frozen_conformal_q
    results["SafeTriage_epistemic_uncertainty"] = external_uncertainty["epistemic_uncertainty"]
    results["SafeTriage_aleatoric_uncertainty"] = external_uncertainty["aleatoric_uncertainty"]
    results["SafeTriage_predictive_entropy"] = external_uncertainty["predictive_entropy"]
    results["SafeTriage_mutual_information"] = external_uncertainty["mutual_information"]
    for model_name in EXPECTED_MODEL_ORDER:
        safe_name = model_name.replace(" ", "_").replace("-", "_")
        results[f"SafeTriage_{safe_name}_probability"] = external_probabilities[model_name]
    results["SafeTriage_external_validation"] = True

    st.dataframe(
        results[[
            "SafeTriage_GDM_probability",
            "SafeTriage_risk_classification",
            "SafeTriage_conformal_prediction_set",
            "SafeTriage_epistemic_uncertainty",
            "SafeTriage_aleatoric_uncertainty",
        ]].head(100).round(4),
        width="stretch",
    )

    st.download_button(
        label="Download external-validation predictions (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="safetriage_gdm_external_validation_predictions.csv",
        mime="text/csv",
    )

    st.subheader("Research interpretation")
    st.markdown(
        "The external-validation mode asks whether a model developed in one dataset "
        "transports to an independent population. Differences in discrimination, "
        "threshold performance, uncertainty, fairness, or BMI distribution are "
        "transportability findings; they are not automatically evidence that the "
        "external dataset is incorrect."
    )
    st.warning(
        "SafeTriage-GDM remains a research prototype. External validation, conformal "
        "prediction, fairness auditing, uncertainty estimates, and PSI do not establish "
        "clinical validity or clinical safety."
    )


# ==============================================================================
# AUTOMATIC INPUT SCHEMA NORMALIZATION
# ==============================================================================

def auto_canonicalize_prediction_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Map common uploaded names to the explicitly selected predictor schema."""
    global APPROVED_PREDICTORS, MODEL_CONFIGURATION
    normalized = {normalized_column_name(c): c for c in df.columns}

    def find_alias(canonical: str):
        for candidate in [canonical] + COLUMN_ALIASES.get(canonical, []):
            source = normalized.get(normalized_column_name(candidate))
            if source is not None:
                return source
        return None

    base_mapping = {c: find_alias(c) for c in BASE_PREDICTORS}
    if not all(v is not None for v in base_mapping.values()):
        return df

    if predictor_configuration == "4-predictor model":
        APPROVED_PREDICTORS = BASE_PREDICTORS + [MMS_PREDICTOR]
        MODEL_CONFIGURATION = "4-predictor"
    else:
        APPROVED_PREDICTORS = BASE_PREDICTORS.copy()
        MODEL_CONFIGURATION = "3-predictor"

    predictor_mapping = dict(base_mapping)
    if MODEL_CONFIGURATION == "4-predictor":
        mms_source = find_alias(MMS_PREDICTOR)
        if mms_source is not None:
            predictor_mapping[MMS_PREDICTOR] = mms_source

    # Do not silently invent a missing fourth predictor. The validation section
    # below will stop with a clear mapping error when 4-predictor mode is selected.
    if MODEL_CONFIGURATION == "4-predictor" and MMS_PREDICTOR not in predictor_mapping:
        return df

    target_aliases = [TARGET_COLUMN, "gdm", "gdm status", "gdm outcome", "gestational diabetes", "gestational diabetes mellitus"]
    target_source = next((normalized.get(normalized_column_name(a)) for a in target_aliases if normalized.get(normalized_column_name(a)) is not None), None)
    return apply_schema_mapping(df, predictor_mapping, target_source or "")


# ==============================================================================
# ANALYSIS MODE ROUTING
# ==============================================================================

if analysis_mode == "External Validation":
    run_external_validation_mode()
    st.stop()


# ==============================================================================
# DATA UPLOAD
# ==============================================================================

uploaded_file = st.file_uploader(
    "Upload compatible Excel dataset",
    type=["xlsx", "xls"],
)

if uploaded_file is None:

    st.markdown(
        """
        ### Getting started

        Upload an Excel dataset containing the GDM target and the predictors required
        by the **selected predictor configuration** in the sidebar. The model
        configuration is selected explicitly rather than inferred from extra columns.

        The 4-predictor configuration uses **Maternal Multiple Micronutrient
        Supplementation (MMS) started by week 28**, derived from raw MMS start week
        information when necessary.

        The application will then:

        - validate the dataset;
        - identify rows with known GDM status;
        - perform the leakage-aware 60/15/25 split;
        - apply MICE-style imputation;
        - apply ROSE-style balancing **only to training data**;
        - train the three specified model families under the detected 3- or 4-predictor configuration;
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

    # Build supported landmark-derived fields and normalize common input names.
    df = build_landmark_features(df)
    df = auto_canonicalize_prediction_inputs(df)

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

st.subheader("1. Dataset validation")

# Apply the explicitly selected model specification. Extra columns are allowed
# and are ignored unless they are part of the selected predictor configuration.
APPROVED_PREDICTORS = (
    BASE_PREDICTORS + [MMS_PREDICTOR]
    if predictor_configuration == "4-predictor model"
    else BASE_PREDICTORS.copy()
)
MODEL_CONFIGURATION = (
    "4-predictor"
    if predictor_configuration == "4-predictor model"
    else "3-predictor"
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", f"{df.shape[0]:,}")
with col2:
    st.metric("Columns", f"{df.shape[1]:,}")
with col3:
    st.metric("GDM column", "Found" if TARGET_COLUMN in df.columns else "Missing")
with col4:
    st.metric("Predictor configuration", MODEL_CONFIGURATION)

if MODEL_CONFIGURATION == "4-predictor":
    st.success(
        "4-predictor model selected: maternal age + pre-pregnancy BMI + parity + "
        "Maternal Multiple Micronutrient Supplementation (MMS) started by week 28."
    )
else:
    st.info(
        "3-predictor model selected: maternal age + pre-pregnancy BMI + parity. "
        "Any MMS columns in the uploaded dataset will be ignored for this model."
    )

if TARGET_COLUMN not in df.columns:
    st.error(f"Required target column `{TARGET_COLUMN}` was not found.")
    st.stop()

missing_predictors = [c for c in APPROVED_PREDICTORS if c not in df.columns]
if missing_predictors:
    st.error("The selected predictor configuration cannot be used because required predictor information is missing.")
    st.write("Missing predictors:", [PREDICTOR_LABELS[c] for c in missing_predictors])
    if MODEL_CONFIGURATION == "4-predictor":
        st.info(
            "For the 4-predictor model, provide a usable Maternal Multiple Micronutrient "
            "Supplementation (MMS) start field or switch the sidebar selection to the "
            "3-predictor model."
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

fig_target = px.bar(
    target_summary,
    x="Outcome",
    y="Count",
    text="Count",
    title="Known GDM Outcome Distribution",
)
fig_target.update_layout(margin=dict(l=20, r=20, t=55, b=20))
st.plotly_chart(fig_target, use_container_width=True)


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

split_plot = split_summary.melt(
    id_vars="Partition",
    value_vars=["GDM", "No GDM"],
    var_name="Outcome",
    value_name="Count",
)
fig_split = px.bar(
    split_plot, x="Partition", y="Count", color="Outcome",
    barmode="stack", title="Labeled Data Distribution Across Partitions",
)
fig_split.update_layout(margin=dict(l=20, r=20, t=55, b=20))
st.plotly_chart(fig_split, use_container_width=True)

st.caption(
    "The test partition remains untouched and naturally imbalanced. "
    "ROSE-style balancing is applied only to training data."
)


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

st.subheader(
    "3. Model training"
)

if TUNING_ENABLED:

    st.info(
        "Hyperparameter tuning uses 5-fold stratified cross-validation inside "
        "the training partition only. ROSE-style balancing and preprocessing "
        "are repeated independently within each inner fold; calibration and "
        "test data are not used for tuning."
    )

    with st.spinner("Tuning Random Forest, XGBoost, and Logistic Regression on the training partition..."):
        try:
            tuned_model_params, tuning_table = tune_models_on_training_data(X_train, y_train)
        except Exception as exc:
            st.warning("Inner tuning failed; the prespecified model parameters will be used instead.")
            st.exception(exc)
            tuned_model_params = {}
            tuning_table = pd.DataFrame()
else:
    tuned_model_params = {}
    tuning_table = pd.DataFrame()

if not tuned_model_params:
    tuned_model_params = {
        name: {} for name in EXPECTED_MODEL_ORDER
    }

with st.spinner("Learning ensemble weights from out-of-fold training predictions..."):
    try:
        ensemble_weights, ensemble_weight_table = learn_ensemble_weights_on_training_data(
            X_train, y_train, tuned_model_params
        )
    except Exception as exc:
        st.warning("Development-only ensemble weighting failed; equal weights will be used.")
        st.exception(exc)
        ensemble_weights = {name: 1.0 / len(EXPECTED_MODEL_ORDER) for name in EXPECTED_MODEL_ORDER}
        ensemble_weight_table = pd.DataFrame()

if tuned_model_params:
    st.markdown("### Training-partition hyperparameter selection")
    tuning_summary = pd.DataFrame([
        {"Model": name, "Selected parameters": str(params)}
        for name, params in tuned_model_params.items()
    ])
    st.dataframe(tuning_summary, width="stretch", hide_index=True)
    if not tuning_table.empty:
        with st.expander("View inner cross-validation tuning results"):
            st.dataframe(
                tuning_table.sort_values(["Model", "CV ROC-AUC", "CV PR-AUC"], ascending=[True, False, False]).round(4),
                width="stretch",
                hide_index=True,
            )

st.markdown("### Development-data ensemble weights")
weight_table = pd.DataFrame({
    "Model": EXPECTED_MODEL_ORDER,
    "Weight": [ensemble_weights[name] for name in EXPECTED_MODEL_ORDER],
})
st.dataframe(weight_table.round(4), width="stretch", hide_index=True)
st.caption(
    "Weights are learned from out-of-fold predictions within the development/training partition only. "
    "The calibration and test partitions are not used to learn ensemble weights."
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
            model_params=tuned_model_params,
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
        model_name: model_probabilities[model_name]["calibration"]
        for model_name in EXPECTED_MODEL_ORDER
    },
    ensemble_weights,
)

test_ensemble_probability = ensemble_probability(
    {
        model_name: model_probabilities[model_name]["test"]
        for model_name in EXPECTED_MODEL_ORDER
    },
    ensemble_weights,
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
    f"The threshold is selected using the calibration partition only by maximizing F2 "
    f"subject to a prespecified minimum specificity of {CALIBRATION_MIN_SPECIFICITY:.0%}. "
    "This prioritizes sensitivity while preventing an almost-all-positive rule. "
    "The test partition is not used to choose the threshold."
)

# Diagnostic: show how the frozen ensemble threshold translates into positive
# predictions for each model. This is descriptive only and does not alter the
# threshold or use the test set for model selection.
threshold_diagnostic = pd.DataFrame({
    "Model": list(EXPECTED_MODEL_ORDER) + ["Weighted ensemble"],
    "Predicted GDM": [
        int((model_probabilities[name]["test"] >= threshold).sum())
        for name in EXPECTED_MODEL_ORDER
    ] + [int((test_ensemble_probability >= threshold).sum())],
})
threshold_diagnostic["Predicted No GDM"] = len(y_test) - threshold_diagnostic["Predicted GDM"]
st.markdown("**Test-set prediction-count diagnostic at the frozen threshold**")
st.dataframe(threshold_diagnostic, use_container_width=True, hide_index=True)
st.caption(
    "Individual-model classification metrics are shown at the same frozen threshold selected for the weighted ensemble; they are descriptive and do not use model-specific test-set threshold optimization. A zero sensitivity value means no true GDM cases were classified as GDM "
    "at the frozen threshold; the diagnostic above shows the corresponding "
    "number of positive predictions without changing the threshold."
)

render_threshold_curve(threshold_table, threshold)


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

# Research interpretation: report the frozen-test result without overstating
# discrimination or clinical performance.
st.info(
    "**Test-set interpretation:** The model demonstrates modest discrimination "
    f"(ROC-AUC {test_metrics['ROC-AUC']:.3f}). At the prespecified "
    f"calibration-derived threshold of {threshold:.3f}, sensitivity was "
    f"{test_metrics['Sensitivity']:.1%} and specificity was "
    f"{test_metrics['Specificity']:.1%} on the untouched test set. "
    "These results are exploratory and do not establish clinical diagnostic performance."
)


# ==============================================================================
# MODEL-LEVEL PERFORMANCE
# ==============================================================================

st.markdown(
    "### Individual model performance at the frozen weighted-ensemble threshold"
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

render_model_performance_chart(individual_table)

model_curve_probabilities = {
    model_name: model_probabilities[model_name]["test"]
    for model_name in EXPECTED_MODEL_ORDER
}
model_curve_probabilities["Ensemble"] = test_ensemble_probability
render_roc_pr_curves(y_test, model_curve_probabilities, "Test-set")


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

render_confusion_matrix(
    y_test,
    test_predictions,
    "Ensemble Confusion Matrix — Untouched Test Set",
)


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

conformal_counts = (
    pd.Series(conformal_sets, name="Prediction set")
    .value_counts()
    .rename_axis("Prediction set")
    .reset_index(name="Count")
)
conformal_counts["Percent"] = conformal_counts["Count"] / len(conformal_sets) * 100

st.markdown("### Conformal prediction-set composition")
st.dataframe(
    conformal_counts.round({"Percent": 1}),
    width="stretch",
    hide_index=True,
)
st.caption(
    "Coverage and prediction-set composition answer different questions: "
    "coverage asks whether the true outcome was retained, while composition "
    "shows how often the conformal procedure returned a single outcome versus "
    "an ambiguous set containing both outcomes."
)

st.caption(
    "The 90% conformal procedure provides prediction sets under the "
    "exchangeability assumptions of split conformal inference. It does "
    "not establish clinical safety or diagnostic validity."
)

render_conformal_chart(conformal_sets)

with st.expander("How to interpret conformal prediction sets"):
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
        model_name: model_probabilities[model_name]["test"]
        for model_name in EXPECTED_MODEL_ORDER
    },
    ensemble_weights,
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

render_uncertainty_chart(uncertainty)

st.caption(
    "Epistemic uncertainty is estimated from disagreement among the "
    "three model probabilities. Aleatoric uncertainty is represented "
    "using Bernoulli entropy. These are research-oriented uncertainty "
    "measures and are not clinical confidence scores."
)

render_probability_distribution(
    y_test,
    test_ensemble_probability,
)


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

    # Very small subgroups can produce unstable fairness estimates. We retain
    # their raw counts/rates for transparency but exclude groups below the
    # prespecified minimum from the headline FPR disparity.
    FAIRNESS_MIN_GROUP_N = 10
    small_groups = fairness.loc[
        fairness["N"] < FAIRNESS_MIN_GROUP_N,
        "Age group",
    ].tolist()

    if small_groups:
        st.warning(
            "Age groups with fewer than "
            f"{FAIRNESS_MIN_GROUP_N} observations are not suitable for reliable "
            "subgroup fairness conclusions. The affected group(s) are retained "
            "in the table for transparency but excluded from the headline FPR disparity."
        )

    eligible_fairness = fairness.loc[
        fairness["N"] >= FAIRNESS_MIN_GROUP_N
    ]
    valid_fpr = eligible_fairness[
        "False-positive rate"
    ].dropna()

    if len(valid_fpr) >= 2:

        fpr_disparity = (
            valid_fpr.max()
            - valid_fpr.min()
        )

        st.metric(
            "Age-group FPR disparity (eligible groups)",
            f"{fpr_disparity:.3f}",
        )
    else:
        st.info(
            "There are fewer than two age groups with sufficient sample size "
            "for a reliable FPR disparity estimate."
        )

    st.caption(
        "This audit reports age-group selection rates and false-positive "
        "rates. It is not a complete Equalized Odds assessment. "
        f"Groups with N < {FAIRNESS_MIN_GROUP_N} are treated as insufficient "
        "for reliable subgroup conclusions."
    )

    render_fairness_chart(fairness)


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

    if not pd.isna(bmi_psi):
        render_bmi_psi_chart(
            X_train[BMI_COLUMN],
            df[BMI_COLUMN],
            BMI_BINS,
            bmi_psi,
        )

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

    render_feature_importance(feature_importance, top_n=15)


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
    all_model_probabilities,
    ensemble_weights,
)

all_uncertainty = calculate_uncertainty(
    all_model_probabilities,
    ensemble_weights,
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

The ensemble is a development-data-weighted combination of the three model probabilities.
The non-negative model weights sum to 1 and are learned from out-of-fold predictions
within the training/development partition only. The calibration and test partitions
are not used to learn ensemble weights.

**Threshold**

The decision threshold is selected from the calibration partition by maximizing
F2, subject to a prespecified minimum specificity of 60%. This gives greater
weight to sensitivity while preventing threshold selection from collapsing into
an almost-all-positive rule. The test partition is not used to choose the threshold.

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
    "Research prototype — not clinically validated or approved as a medical device. "
    "Results should be interpreted only within the research context."
)

st.caption(
    "Author: Ikechukwu Okechi Kamalu"
)
