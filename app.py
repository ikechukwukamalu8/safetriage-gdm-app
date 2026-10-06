# ==============================================================================
# SafeTriage-GDM: Unified Uncertainty-Quantified Triage & Performance Platform
# Production Clinical ML Dashboard
#
# Dataset Protocol:
#   - All uploaded records are eligible for inference.
#   - Only records with known GDM outcomes are used for supervised evaluation.
#   - No synthetic/random ground-truth labels are generated.
#
# ==============================================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
)

from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="SafeTriage GDM Platform",
    layout="wide",
)

st.title("SafeTriage-GDM: Uncertainty-Quantified Triage System")
st.markdown(
    "### Production-Grade Clinical Risk Triage with Ensemble Uncertainty, "
    "Conformal Safety Sets & Population Drift Telemetry"
)
st.write("---")


# ==============================================================================
# 2. SYSTEM CONFIGURATION
# ==============================================================================

@st.cache_resource
def initialize_system_engine():
    return {
        "confidence_level": 0.95,
        "drift_limit": 0.25,
        "model_path": "safetriage_gdm_mixed_pipeline.pkl",
    }


sys_config = initialize_system_engine()


# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================

def safe_numeric_series(series):
    """
    Convert a pandas Series to numeric values.
    Invalid values become NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def encode_dataframe(df):
    """
    Convert mixed-type dataframe into a numerical one-hot encoded matrix.

    Returns
    -------
    encoded_df : pandas.DataFrame
        Numerical dataframe containing dummy variables.
    """
    encoded_df = pd.get_dummies(
        df,
        drop_first=True,
        dummy_na=False,
    )

    encoded_df = encoded_df.astype(np.float64)

    return encoded_df


def align_to_schema(encoded_df, feature_schema):
    """
    Align an encoded dataframe to the exact feature schema expected by
    the trained model.

    Missing columns are created as zeros.
    Unexpected columns are discarded.
    """
    aligned_df = encoded_df.reindex(
        columns=feature_schema,
        fill_value=0.0,
    )

    return aligned_df.astype(np.float64)


def calculate_distribution_drift(baseline_bmi, incoming_bmi):
    """
    Compute Population Stability Index (PSI) for BMI distributions.

    PSI interpretation is application-dependent. The configured threshold
    is used only as a monitoring trigger and does not itself establish
    statistical validity of a model.
    """
    try:
        baseline = pd.to_numeric(
            pd.Series(baseline_bmi),
            errors="coerce",
        ).dropna()

        incoming = pd.to_numeric(
            pd.Series(incoming_bmi),
            errors="coerce",
        ).dropna()

        if len(baseline) == 0 or len(incoming) == 0:
            return 0.0

        bins = [
            -np.inf,
            18.5,
            24.9,
            29.9,
            39.9,
            np.inf,
        ]

        baseline_counts = (
            pd.cut(
                baseline,
                bins=bins,
                include_lowest=True,
            )
            .value_counts(sort=False)
        )

        incoming_counts = (
            pd.cut(
                incoming,
                bins=bins,
                include_lowest=True,
            )
            .value_counts(sort=False)
        )

        expected_pct = (
            baseline_counts / len(baseline)
        ).values.astype(float)

        actual_pct = (
            incoming_counts / len(incoming)
        ).values.astype(float)

        expected_pct = np.where(
            expected_pct == 0,
            1e-4,
            expected_pct,
        )

        actual_pct = np.where(
            actual_pct == 0,
            1e-4,
            actual_pct,
        )

        psi_value = np.sum(
            (actual_pct - expected_pct)
            * np.log(actual_pct / expected_pct)
        )

        return float(psi_value)

    except Exception:
        return 0.0


def autonomous_unsupervised_parameter_sift(
    trained_models,
    X_new_batch,
    current_q,
    psi_score,
    sys_config,
):
    """
    Runtime uncertainty monitoring.

    If substantial population drift is detected and ensemble entropy is high,
    the active conformal threshold is adjusted.

    IMPORTANT:
    This runtime adaptation is treated as an operational heuristic. It does
    not constitute a mathematically guaranteed preservation of 95% conformal
    coverage under distribution shift.
    """

    if psi_score >= sys_config["drift_limit"]:

        model_predictions = [
            model.predict_proba(X_new_batch)[:, 1]
            for model in trained_models
        ]

        batch_consensus = np.mean(
            model_predictions,
            axis=0,
        )

        batch_entropy = -(
            batch_consensus
            * np.log2(batch_consensus + 1e-12)
            +
            (1 - batch_consensus)
            * np.log2(1 - batch_consensus + 1e-12)
        )

        mean_drift_entropy = float(
            np.mean(batch_entropy)
        )

        if mean_drift_entropy > 0.70:

            adapted_q = current_q * 0.90

            return adapted_q, True

    return current_q, False


def calculate_conformal_sets(consensus_prob, q_active):
    """
    Construct prediction sets from the active non-conformity threshold.
    """

    prediction_intervals = []

    for p in consensus_prob:

        p_set = []

        # Healthy class
        if (1.0 - p) <= q_active:
            p_set.append("Healthy")

        # GDM High Risk class
        if p <= q_active:
            p_set.append("GDM High Risk")

        # If neither class satisfies the threshold, force an ambiguous
        # two-class set requiring human review.
        if len(p_set) == 0:
            p_set = [
                "Healthy",
                "GDM High Risk",
            ]

        prediction_intervals.append(p_set)

    return prediction_intervals


def calculate_fairness_metrics(
    dataframe,
    consensus_prob,
):
    """
    Calculate fairness metrics while maintaining strict row alignment.

    Demographic parity:
        Uses all records with valid age because it only requires predictions.

    FPR disparity:
        Uses only records with known GDM outcomes because ground truth is
        required.

    This function deliberately does NOT fabricate labels.
    """

    age_col = "Mother's age (years)"
    target_col = "Gestational diabetes?"

    positive_flags = (
        np.asarray(consensus_prob) >= 0.5
    ).astype(int)

    demographic_parity_disparity = np.nan
    fpr_disparity = np.nan

    if age_col not in dataframe.columns:
        return (
            demographic_parity_disparity,
            fpr_disparity,
        )

    age_series = pd.to_numeric(
        dataframe[age_col],
        errors="coerce",
    )

    valid_age_mask = age_series.notna().to_numpy()

    if np.sum(valid_age_mask) > 0:

        younger_mask = (
            (age_series < 35)
            & age_series.notna()
        ).to_numpy()

        older_mask = (
            (age_series >= 35)
            & age_series.notna()
        ).to_numpy()

        if np.sum(younger_mask) > 0:
            rate_younger = float(
                np.mean(
                    positive_flags[younger_mask]
                )
            )
        else:
            rate_younger = np.nan

        if np.sum(older_mask) > 0:
            rate_older = float(
                np.mean(
                    positive_flags[older_mask]
                )
            )
        else:
            rate_older = np.nan

        if (
            not np.isnan(rate_younger)
            and not np.isnan(rate_older)
        ):
            demographic_parity_disparity = abs(
                rate_younger - rate_older
            )

    # ------------------------------------------------------------------
    # Ground-truth-dependent fairness analysis
    # ------------------------------------------------------------------

    if target_col not in dataframe.columns:
        return (
            demographic_parity_disparity,
            fpr_disparity,
        )

    y_true_series = dataframe[target_col]

    labeled_mask = (
        y_true_series.notna()
        & age_series.notna()
    ).to_numpy()

    if np.sum(labeled_mask) == 0:
        return (
            demographic_parity_disparity,
            fpr_disparity,
        )

    y_true = y_true_series.to_numpy(dtype=float)

    eval_predictions = positive_flags[labeled_mask]
    eval_y_true = y_true[labeled_mask]
    eval_age = age_series.to_numpy()[labeled_mask]

    younger_eval = eval_age < 35
    older_eval = eval_age >= 35

    def false_positive_rate(group_mask):

        if np.sum(group_mask) == 0:
            return np.nan

        group_true = eval_y_true[group_mask]
        group_pred = eval_predictions[group_mask]

        negatives = group_true == 0

        if np.sum(negatives) == 0:
            return np.nan

        fp = np.sum(
            (group_pred == 1)
            & negatives
        )

        tn = np.sum(
            (group_pred == 0)
            & negatives
        )

        denominator = fp + tn

        if denominator == 0:
            return np.nan

        return float(fp / denominator)

    fpr_younger = false_positive_rate(
        younger_eval
    )

    fpr_older = false_positive_rate(
        older_eval
    )

    if (
        not np.isnan(fpr_younger)
        and not np.isnan(fpr_older)
    ):
        fpr_disparity = abs(
            fpr_younger - fpr_older
        )

    return (
        demographic_parity_disparity,
        fpr_disparity,
    )


# ==============================================================================
# 4. DATA INGESTION
# ==============================================================================

st.header(
    "Arden Ingestion Protocol: CBGS Mixed-Type Registry"
)

uploaded_file = st.file_uploader(
    "Browse files or drop your clinic master spreadsheet here:",
    type=["xlsx", "csv"],
)


if uploaded_file is not None:

    # --------------------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------------------

    try:

        if uploaded_file.name.lower().endswith(".csv"):
            raw_dataframe = pd.read_csv(
                uploaded_file
            )

        else:
            raw_dataframe = pd.read_excel(
                uploaded_file
            )

    except Exception as e:

        st.error(
            f"❌ Unable to read uploaded dataset: {e}"
        )

        st.stop()

    st.success(
        f"✅ Connection Established: Loaded "
        f"'{uploaded_file.name}' containing "
        f"{len(raw_dataframe)} records into secure RAM."
    )

    if len(raw_dataframe) == 0:

        st.error(
            "❌ The uploaded dataset contains no records."
        )

        st.stop()

    # Preserve the original row identity.
    raw_dataframe = raw_dataframe.reset_index(
        drop=True
    )

    processed_df = raw_dataframe.copy()

    target_col = "Gestational diabetes?"
    id_col = "Dummy Study Number"

    # ==============================================================================
    # 5. TARGET / EVALUATION POPULATION
    # ==============================================================================

    has_ground_truth = target_col in processed_df.columns

    if has_ground_truth:

        # Convert Yes/No outcome to numeric without fabricating labels.
        processed_df[target_col] = (
            processed_df[target_col]
            .apply(
                lambda x:
                    1
                    if str(x).strip().lower() == "yes"
                    else (
                        0
                        if str(x).strip().lower() == "no"
                        else np.nan
                    )
            )
        )

        labeled_mask = (
            processed_df[target_col]
            .notna()
        )

        eval_df = (
            processed_df.loc[
                labeled_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        y_raw = (
            eval_df[target_col]
            .astype(int)
            .to_numpy()
        )

        X_eval_raw = (
            eval_df
            .drop(
                columns=[
                    target_col,
                    id_col,
                ],
                errors="ignore",
            )
            .copy()
        )

        st.info(
            f"📊 Evaluation population: "
            f"{len(eval_df)} of "
            f"{len(processed_df)} records have valid "
            f"GDM ground-truth labels."
        )

    else:

        labeled_mask = np.zeros(
            len(processed_df),
            dtype=bool,
        )

        eval_df = pd.DataFrame()

        y_raw = None

        X_eval_raw = None

        st.warning(
            "⚠️ No 'Gestational diabetes?' column was found. "
            "Supervised performance metrics will only be available "
            "from an existing trained model artifact."
        )

    # ==============================================================================
    # 6. INFERENCE POPULATION
    # ==============================================================================

    # IMPORTANT:
    # Every uploaded patient is retained for inference.
    #
    # The target column is removed from the feature matrix, but rows with
    # missing target labels are NOT removed.

    X_inference_raw = (
        processed_df
        .drop(
            columns=[
                target_col,
                id_col,
            ],
            errors="ignore",
        )
        .copy()
    )

    # Encode inference population.
    X_inference_encoded = encode_dataframe(
        X_inference_raw
    )

    # Encode labeled evaluation population if available.
    if has_ground_truth and len(X_eval_raw) > 0:

        X_eval_encoded = encode_dataframe(
            X_eval_raw
        )

    else:

        X_eval_encoded = None

    # ==============================================================================
    # 7. ADMINISTRATIVE CONTROLS
    # ==============================================================================

    st.sidebar.header(
        "⚙️ Administrative Control Unit"
    )

    force_retrain = st.sidebar.button(
        "🔄 Force Pipeline Retraining"
    )

    pipeline_loaded = False

    # These variables will be populated either from the artifact
    # or from a fresh training cycle.
    feature_names = None
    imputer = None
    scaler = None
    rf = None
    xgb = None
    lr = None
    q_threshold = None
    ensemble_importance = None
    auc_score = np.nan
    sensitivity = np.nan
    brier_score = np.nan
    tn = fp = fn = tp = 0
    baseline_bmi_vector = None

    # ==============================================================================
    # 8. LOAD SAVED MODEL
    # ==============================================================================

    if (
        os.path.exists(
            sys_config["model_path"]
        )
        and not force_retrain
    ):

        with st.spinner(
            "💾 Warm Boot: Restoring trained pipeline..."
        ):

            try:

                cached_pipeline = joblib.load(
                    sys_config["model_path"]
                )

                required_artifact_keys = [
                    "feature_schema",
                    "imputer",
                    "scaler",
                    "rf",
                    "xgb",
                    "lr",
                    "q_threshold",
                    "importance",
                    "auc",
                    "sensitivity",
                    "brier",
                    "cm",
                    "baseline_bmi",
                ]

                missing_keys = [
                    key
                    for key in required_artifact_keys
                    if key not in cached_pipeline
                ]

                if missing_keys:

                    raise ValueError(
                        "Saved model artifact is missing "
                        f"required fields: {missing_keys}"
                    )

                # Load exact training schema.
                feature_names = list(
                    cached_pipeline[
                        "feature_schema"
                    ]
                )

                # Restore model objects.
                imputer = cached_pipeline[
                    "imputer"
                ]

                scaler = cached_pipeline[
                    "scaler"
                ]

                rf = cached_pipeline[
                    "rf"
                ]

                xgb = cached_pipeline[
                    "xgb"
                ]

                lr = cached_pipeline[
                    "lr"
                ]

                q_threshold = float(
                    cached_pipeline[
                        "q_threshold"
                    ]
                )

                ensemble_importance = np.asarray(
                    cached_pipeline[
                        "importance"
                    ]
                )

                auc_score = cached_pipeline[
                    "auc"
                ]

                sensitivity = cached_pipeline[
                    "sensitivity"
                ]

                brier_score = cached_pipeline[
                    "brier"
                ]

                (
                    tn,
                    fp,
                    fn,
                    tp,
                ) = cached_pipeline["cm"]

                baseline_bmi_vector = cached_pipeline[
                    "baseline_bmi"
                ]

                pipeline_loaded = True

                st.sidebar.success(
                    "🎯 Status: Utilizing optimized saved models."
                )

            except Exception as e:

                st.sidebar.warning(
                    "⚠️ Saved model could not be loaded. "
                    f"Reason: {e}"
                )

                pipeline_loaded = False

    # ==============================================================================
    # 9. TRAINING / RETRAINING
    # ==============================================================================

    if not pipeline_loaded:

        # A new model cannot be legitimately trained without ground-truth labels.
        if (
            not has_ground_truth
            or y_raw is None
            or len(y_raw) == 0
        ):

            st.error(
                "❌ A new model cannot be trained because the uploaded "
                "dataset contains no valid 'Gestational diabetes?' labels."
            )

            st.info(
                "Please upload a labeled training dataset or remove "
                "the force-retrain request and use an existing "
                "SafeTriage-GDM model artifact."
            )

            st.stop()

        # Need both outcome classes for meaningful supervised training.
        if len(np.unique(y_raw)) < 2:

            st.error(
                "❌ Training requires both GDM-positive and "
                "GDM-negative records."
            )

            st.stop()

        with st.spinner(
            "🏗️ Cold Boot: Running MICE recovery, "
            "three-way partitioning and ensemble training..."
        ):

            # ------------------------------------------------------------------
            # A. Establish training feature schema
            # ------------------------------------------------------------------

            feature_names = (
                X_eval_encoded.columns.tolist()
            )

            # Align inference data to training schema.
            X_inference_encoded = align_to_schema(
                X_inference_encoded,
                feature_names,
            )

            # Ensure evaluation data has exactly the same schema.
            X_eval_encoded = align_to_schema(
                X_eval_encoded,
                feature_names,
            )

            X_matrix = (
                X_eval_encoded.to_numpy(
                    dtype=np.float64
                )
            )

            # ------------------------------------------------------------------
            # B. MICE IMPUTATION
            # ------------------------------------------------------------------

            imputer = IterativeImputer(
                max_iter=30,
                random_state=123,
            )

            X_imp = imputer.fit_transform(
                X_matrix
            )

            # ------------------------------------------------------------------
            # C. THREE-WAY DATA PARTITION
            #
            # 60% Training
            # 15% Calibration
            # 25% Testing
            # ------------------------------------------------------------------

            (
                X_train,
                X_temp,
                y_train,
                y_temp,
            ) = train_test_split(
                X_imp,
                y_raw,
                test_size=0.40,
                random_state=123,
                stratify=y_raw,
            )

            (
                X_cal,
                X_test,
                y_cal,
                y_test,
            ) = train_test_split(
                X_temp,
                y_temp,
                test_size=0.625,
                random_state=123,
                stratify=y_temp,
            )

            # ------------------------------------------------------------------
            # D. BASELINE BMI FOR DRIFT MONITORING
            # ------------------------------------------------------------------

            bmi_col = (
                "Mother's pre-pregnancy BMI (kg/m2)"
            )

            if bmi_col in feature_names:

                bmi_idx = feature_names.index(
                    bmi_col
                )

            else:

                bmi_idx = 0

            baseline_bmi_vector = (
                X_train[:, bmi_idx]
                .tolist()
            )

            # ------------------------------------------------------------------
            # E. STANDARDIZATION
            # ------------------------------------------------------------------

            scaler = StandardScaler()

            X_train_scl = (
                scaler.fit_transform(
                    X_train
                )
            )

            X_cal_scl = (
                scaler.transform(
                    X_cal
                )
            )

            X_test_scl = (
                scaler.transform(
                    X_test
                )
            )

            # ------------------------------------------------------------------
            # F. CLASS BALANCING
            # ------------------------------------------------------------------

            sampler = SMOTETomek(
                random_state=123
            )

            X_train_bal, y_train_bal = (
                sampler.fit_resample(
                    X_train_scl,
                    y_train,
                )
            )

            # ------------------------------------------------------------------
            # G. TRAIN ENSEMBLE
            # ------------------------------------------------------------------

            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=123,
                n_jobs=-1,
            )

            rf.fit(
                X_train_bal,
                y_train_bal,
            )

            xgb = XGBClassifier(
                n_estimators=100,
                random_state=123,
                eval_metric="logloss",
                n_jobs=-1,
            )

            xgb.fit(
                X_train_bal,
                y_train_bal,
            )

            lr = LogisticRegression(
                max_iter=1000,
                random_state=123,
            )

            lr.fit(
                X_train_bal,
                y_train_bal,
            )

            # ------------------------------------------------------------------
            # H. CONFORMAL CALIBRATION
            # ------------------------------------------------------------------

            cal_preds_rf = (
                rf.predict_proba(
                    X_cal_scl
                )[:, 1]
            )

            cal_preds_xgb = (
                xgb.predict_proba(
                    X_cal_scl
                )[:, 1]
            )

            cal_preds_lr = (
                lr.predict_proba(
                    X_cal_scl
                )[:, 1]
            )

            cal_consensus = np.mean(
                [
                    cal_preds_rf,
                    cal_preds_xgb,
                    cal_preds_lr,
                ],
                axis=0,
            )

            non_conformity_scores = np.abs(
                y_cal - cal_consensus
            )

            alpha = (
                1.0
                - sys_config[
                    "confidence_level"
                ]
            )

            n_cal = len(y_cal)

            quantile_target = (
                (1.0 - alpha)
                * (1.0 + (1.0 / n_cal))
            )

            q_threshold = float(
                np.quantile(
                    non_conformity_scores,
                    min(
                        quantile_target,
                        1.0,
                    ),
                    method="higher",
                )
            )

            # ------------------------------------------------------------------
            # I. TEST-SET PERFORMANCE
            # ------------------------------------------------------------------

            test_preds_rf = (
                rf.predict_proba(
                    X_test_scl
                )[:, 1]
            )

            test_preds_xgb = (
                xgb.predict_proba(
                    X_test_scl
                )[:, 1]
            )

            test_preds_lr = (
                lr.predict_proba(
                    X_test_scl
                )[:, 1]
            )

            test_consensus_prob = np.mean(
                [
                    test_preds_rf,
                    test_preds_xgb,
                    test_preds_lr,
                ],
                axis=0,
            )

            if len(
                np.unique(y_test)
            ) > 1:

                auc_score = roc_auc_score(
                    y_test,
                    test_consensus_prob,
                )

            else:

                auc_score = np.nan

            binary_predictions = (
                test_consensus_prob >= 0.5
            ).astype(int)

            sensitivity = recall_score(
                y_test,
                binary_predictions,
                zero_division=0,
            )

            brier_score = (
                brier_score_loss(
                    y_test,
                    test_consensus_prob,
                )
            )

            if len(
                np.unique(y_test)
            ) > 1:

                (
                    tn,
                    fp,
                    fn,
                    tp,
                ) = confusion_matrix(
                    y_test,
                    binary_predictions,
                    labels=[0, 1],
                ).ravel()

            else:

                tn = int(
                    np.sum(
                        y_test == 0
                    )
                )

                fp = 0
                fn = 0

                tp = int(
                    np.sum(
                        y_test == 1
                    )
                )

            # ------------------------------------------------------------------
            # J. ENSEMBLE FEATURE IMPORTANCE
            # ------------------------------------------------------------------

            rf_imp = (
                rf.feature_importances_
            )

            xgb_imp = (
                xgb.feature_importances_
            )

            lr_imp = np.abs(
                lr.coef_[0]
            )

            lr_sum = np.sum(
                lr_imp
            )

            if lr_sum > 0:

                lr_imp = (
                    lr_imp / lr_sum
                )

            ensemble_importance = np.mean(
                [
                    rf_imp,
                    xgb_imp,
                    lr_imp,
                ],
                axis=0,
            )

            importance_sum = np.sum(
                ensemble_importance
            )

            if importance_sum > 0:

                ensemble_importance = (
                    ensemble_importance
                    / importance_sum
                )

            # ------------------------------------------------------------------
            # K. SAVE MODEL ARTIFACT
            # ------------------------------------------------------------------

            pipeline_payload = {
                "imputer": imputer,
                "scaler": scaler,
                "rf": rf,
                "xgb": xgb,
                "lr": lr,
                "q_threshold": q_threshold,
                "importance": ensemble_importance,
                "auc": auc_score,
                "sensitivity": sensitivity,
                "brier": brier_score,
                "cm": (
                    tn,
                    fp,
                    fn,
                    tp,
                ),
                "baseline_bmi": baseline_bmi_vector,
                "feature_schema": feature_names,
            }

            joblib.dump(
                pipeline_payload,
                sys_config[
                    "model_path"
                ],
            )

            st.sidebar.success(
                "💾 Telemetry Storage Synced: "
                "Optimization artifact saved."
            )

    else:

        # ----------------------------------------------------------------------
        # Existing artifact path
        #
        # Always align the incoming dataset to the saved model schema.
        # This is critical because pd.get_dummies() can create different
        # columns for different batches.
        # ----------------------------------------------------------------------

        X_inference_encoded = align_to_schema(
            X_inference_encoded,
            feature_names,
        )

    # ==============================================================================
    # 10. PRODUCTION INFERENCE ON ALL UPLOADED RECORDS
    # ==============================================================================

    # X_inference_encoded has exactly one row per uploaded patient.
    #
    # If there are 970 uploaded records:
    #   X_inference_encoded.shape[0] == 970

    X_inference_matrix = (
        X_inference_encoded.to_numpy(
            dtype=np.float64
        )
    )

    # MICE transformation.
    X_all_imp = imputer.transform(
        X_inference_matrix
    )

    # Standardization.
    X_all_scl = scaler.transform(
        X_all_imp
    )

    # ==============================================================================
    # 11. POPULATION DRIFT TELEMETRY
    # ==============================================================================

    bmi_col = (
        "Mother's pre-pregnancy BMI (kg/m2)"
    )

    if bmi_col in X_inference_raw.columns:

        incoming_bmi_vector = (
            pd.to_numeric(
                X_inference_raw[bmi_col],
                errors="coerce",
            )
            .dropna()
            .to_numpy()
        )

    elif (
        feature_names
        and bmi_col in feature_names
    ):

        bmi_idx = feature_names.index(
            bmi_col
        )

        incoming_bmi_vector = (
            X_all_imp[:, bmi_idx]
        )

    else:

        incoming_bmi_vector = np.array([])

    calculated_psi = (
        calculate_distribution_drift(
            baseline_bmi_vector,
            incoming_bmi_vector,
        )
    )

    drift_alert = (
        calculated_psi
        >= sys_config["drift_limit"]
    )

    # ==============================================================================
    # 12. AUTONOMOUS UNCERTAINTY SIFTING
    # ==============================================================================

    q_active, sift_activated = (
        autonomous_unsupervised_parameter_sift(
            [rf, xgb, lr],
            X_all_scl,
            q_threshold,
            calculated_psi,
            sys_config,
        )
    )

    # ==============================================================================
    # 13. ENSEMBLE INFERENCE
    # ==============================================================================

    all_preds_rf = (
        rf.predict_proba(
            X_all_scl
        )[:, 1]
    )

    all_preds_xgb = (
        xgb.predict_proba(
            X_all_scl
        )[:, 1]
    )

    all_preds_lr = (
        lr.predict_proba(
            X_all_scl
        )[:, 1]
    )

    preds_matrix = np.array(
        [
            all_preds_rf,
            all_preds_xgb,
            all_preds_lr,
        ]
    )

    consensus_prob = np.mean(
        preds_matrix,
        axis=0,
    )

    epistemic_var = np.var(
        preds_matrix,
        axis=0,
    )

    aleatoric_ent = -(
        consensus_prob
        * np.log2(
            consensus_prob + 1e-12
        )
        +
        (1 - consensus_prob)
        * np.log2(
            1 - consensus_prob + 1e-12
        )
    )

    # ==============================================================================
    # 14. CONFORMAL PREDICTION SETS
    # ==============================================================================

    prediction_intervals = (
        calculate_conformal_sets(
            consensus_prob,
            q_active,
        )
    )

    # ==============================================================================
    # 15. RESULTS DASHBOARD
    # ==============================================================================

    results_dashboard = pd.DataFrame(
        {
            "Consensus_GDM_Probability": [
                f"{p * 100:.2f}%"
                for p in consensus_prob
            ],
            "Aleatoric_Data_Noise": np.round(
                aleatoric_ent,
                4,
            ),
            "Epistemic_Model_Ignorance": np.round(
                epistemic_var,
                4,
            ),
            "Conformal_Prediction_Interval":
                prediction_intervals,
        }
    )

    results_dashboard[
        "Requires_Human_Medical_Audit"
    ] = (
        results_dashboard[
            "Conformal_Prediction_Interval"
        ]
        .apply(
            lambda x:
                len(x) != 1
        )
    )

    # IMPORTANT:
    # Both dataframes now contain ALL uploaded records.
    #
    # If raw_dataframe contains 970 records,
    # results_dashboard also contains 970 records.
    #
    # Therefore concatenation is row-aligned.
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
    # 16. FAIRNESS ANALYSIS
    # ==============================================================================

    (
        demographic_parity_disparity,
        fpr_disparity,
    ) = calculate_fairness_metrics(
        processed_df,
        consensus_prob,
    )

    # ==============================================================================
    # 17. UI TABS
    # ==============================================================================

    (
        tab_triage,
        tab_xai,
        tab_performance,
    ) = st.tabs(
        [
            "🖥️ Patient Triage Console",
            "🧠 Explainable AI (XAI) Attribution",
            "📈 Performance Validation",
        ]
    )

    # ==============================================================================
    # TAB 1 — TRIAGE
    # ==============================================================================

    with tab_triage:

        st.subheader(
            "🖥 Clinical Production Triage Database Log"
        )

        st.caption(
            f"Showing {len(final_output_view)} "
            "records. Every uploaded record receives "
            "an inference score; supervised metrics use "
            "only records with known outcomes."
        )

        st.dataframe(
            final_output_view,
            use_container_width=True,
            height=500,
        )

        st.write("---")

        col_plot1, col_plot2 = st.columns(2)

        # ----------------------------------------------------------------------
        # FAIRNESS
        # ----------------------------------------------------------------------

        with col_plot1:

            st.subheader(
                "⚖️ Algorithmic Fairness Audit Console"
            )

            if not np.isnan(
                demographic_parity_disparity
            ):

                st.metric(
                    label=(
                        "Demographic Parity Disparity "
                        "(<35 vs ≥35)"
                    ),
                    value=(
                        f"{demographic_parity_disparity:.4f}"
                    ),
                    delta=(
                        "PASS"
                        if demographic_parity_disparity <= 0.10
                        else "WARNING"
                    ),
                    delta_color=(
                        "normal"
                        if demographic_parity_disparity <= 0.10
                        else "inverse"
                    ),
                )

            else:

                st.metric(
                    label=(
                        "Demographic Parity Disparity "
                        "(<35 vs ≥35)"
                    ),
                    value="N/A",
                )

            if not np.isnan(
                fpr_disparity
            ):

                st.metric(
                    label=(
                        "False-Positive-Rate Disparity "
                        "(<35 vs ≥35)"
                    ),
                    value=(
                        f"{fpr_disparity:.4f}"
                    ),
                    delta=(
                        "PASS"
                        if fpr_disparity <= 0.05
                        else "WARNING"
                    ),
                    delta_color=(
                        "normal"
                        if fpr_disparity <= 0.05
                        else "inverse"
                    ),
                )

            else:

                st.metric(
                    label=(
                        "False-Positive-Rate Disparity "
                        "(<35 vs ≥35)"
                    ),
                    value="N/A",
                )

            st.caption(
                "FPR disparity is calculated only for records "
                "with known GDM outcomes. It should not be "
                "interpreted as full equalized-odds assessment."
            )

        # ----------------------------------------------------------------------
        # DRIFT
        # ----------------------------------------------------------------------

        with col_plot2:

            st.subheader(
                "📡 Batch Population Stability Telemetry"
            )

            st.metric(
                label=(
                    "Calculated Population Stability "
                    "Index (PSI)"
                ),
                value=(
                    f"{calculated_psi:.4f}"
                ),
                delta=(
                    "🚨 SYSTEM DISTRIBUTION DRIFT ALERT"
                    if drift_alert
                    else "Clinical Metrics Profile Stable"
                ),
                delta_color=(
                    "inverse"
                    if drift_alert
                    else "normal"
                ),
            )

            if drift_alert:

                if sift_activated:

                    st.warning(
                        "⚠️ Adaptive Parameter Sift Active: "
                        "The active uncertainty threshold has "
                        "been adjusted because population drift "
                        "and elevated ensemble entropy were detected."
                    )

                else:

                    st.warning(
                        "⚠️ System Alert: Incoming batch parameters "
                        "deviate from the baseline training population. "
                        "Pipeline retraining is recommended."
                    )

    # ==============================================================================
    # TAB 2 — XAI
    # ==============================================================================

    with tab_xai:

        st.subheader(
            "🧠 Explainable AI (XAI): "
            "Global Clinical Attribution Matrix"
        )

        st.write(
            f"Baseline Quantile Threshold Bound: "
            f"`{q_threshold:.4f}`"
        )

        st.write(
            f"Currently Active Quantile Bound: "
            f"`{q_active:.4f}`"
        )

        xai_df = pd.DataFrame(
            {
                "Biometric Feature Attribute":
                    feature_names,
                "Attribution Weight Score":
                    ensemble_importance,
            }
        ).sort_values(
            by="Attribution Weight Score",
            ascending=False,
        )

        st.dataframe(
            xai_df.style.format(
                {
                    "Attribution Weight Score":
                        "{:.2%}"
                }
            ).background_gradient(
                cmap="Purples",
                subset=[
                    "Attribution Weight Score"
                ],
            ),
            use_container_width=True,
        )

        st.write("---")

        st.markdown(
            "#### 🔍 Point-of-Care Patient-Specific Risk Breakdown"
        )

        selected_row = st.selectbox(
            "Choose Patient Record Index:",
            options=final_output_view.index,
        )

        patient_data = (
            final_output_view.loc[
                selected_row
            ]
        )

        p_prob = patient_data[
            "Consensus_GDM_Probability"
        ]

        p_audit = (
            "⚠️ FORCED MEDICAL AUDIT ENFORCED"
            if patient_data[
                "Requires_Human_Medical_Audit"
            ]
            else "✅ Automated Flag Decisive"
        )

        p_col1, p_col2 = st.columns(2)

        p_col1.metric(
            "Calculated GDM Prediction Risk "
            "Probability Score",
            value=p_prob,
        )

        p_col2.metric(
            "Conformal Validation Triage Outcome Set",
            value=str(
                patient_data[
                    "Conformal_Prediction_Interval"
                ]
            ),
            delta=p_audit,
            delta_color=(
                "inverse"
                if "AUDIT" in p_audit
                else "normal"
            ),
        )

    # ==============================================================================
    # TAB 3 — PERFORMANCE
    # ==============================================================================

    with tab_performance:

        st.subheader(
            "📈 Empirically Validated Model "
            "Performance Diagnostics"
        )

        st.markdown(
            "Metrics below were generated from the untouched "
            "25% testing subset used when the saved model artifact "
            "was trained. They are not recomputed from the current "
            "unlabeled incoming batch."
        )

        if has_ground_truth:

            st.info(
                f"Current uploaded dataset contains "
                f"{len(eval_df)} labeled records out of "
                f"{len(raw_dataframe)} total records. "
                "The current batch predictions are not used to "
                "recalculate the saved model's test-set metrics."
            )

        else:

            st.warning(
                "The current uploaded dataset contains no "
                "ground-truth GDM outcome column. "
                "Performance metrics shown here belong to "
                "the saved model artifact."
            )

        perf_col1, perf_col2, perf_col3 = (
            st.columns(3)
        )

        if not np.isnan(auc_score):

            perf_col1.metric(
                label="Area Under ROC (AUC-ROC)",
                value=f"{auc_score:.4f}",
                delta="Target Bound: ≥ 0.85",
            )

        else:

            perf_col1.metric(
                label="Area Under ROC (AUC-ROC)",
                value="N/A",
            )

        if not np.isnan(sensitivity):

            perf_col2.metric(
                label="Clinical Sensitivity (Recall)",
                value=f"{sensitivity * 100:.2f}%",
                delta="Target Bound: ≥ 90.0%",
            )

        else:

            perf_col2.metric(
                label="Clinical Sensitivity (Recall)",
                value="N/A",
            )

        if not np.isnan(brier_score):

            perf_col3.metric(
                label="Brier Calibration Fit Score",
                value=f"{brier_score:.4f}",
                delta=(
                    "Lower Scores Indicate "
                    "Superior Probability Calibration"
                ),
            )

        else:

            perf_col3.metric(
                label="Brier Calibration Fit Score",
                value="N/A",
            )

        st.write("---")

        st.markdown(
            "#### Diagnostic Confusion Matrix Boundaries"
        )

        (
            cm_col1,
            cm_col2,
            cm_col3,
            cm_col4,
        ) = st.columns(4)

        cm_col1.metric(
            "True Negatives (Healthy Checked)",
            value=int(tn),
        )

        cm_col2.metric(
            "False Positives (False Flags)",
            value=int(fp),
        )

        cm_col3.metric(
            "False Negatives (Missed Medical Targets)",
            value=int(fn),
        )

        cm_col4.metric(
            "True Positives (Correctly Captured GDM)",
            value=int(tp),
        )

        st.write("---")

        st.markdown(
            "#### 📊 Population Accounting"
        )

        population_col1, population_col2, population_col3 = (
            st.columns(3)
        )

        population_col1.metric(
            "Total Uploaded Records",
            value=len(raw_dataframe),
        )

        population_col2.metric(
            "Records With Known GDM Outcome",
            value=(
                int(
                    np.sum(labeled_mask)
                )
                if has_ground_truth
                else 0
            ),
        )

        population_col3.metric(
            "Records Scored by Model",
            value=len(consensus_prob),
        )

        st.caption(
            "A record can be scored by the model without having a "
            "known GDM outcome. Such records are included in inference "
            "and triage but excluded from supervised performance "
            "calculations."
        )

else:

    st.info(
        "👋 Upload the CBGS Excel or CSV dataset to begin."
    )

    st.markdown(
        """
        ### SafeTriage-GDM Pipeline

        **Inference population**

        Every uploaded patient record is processed and scored.

        **Evaluation population**

        Only records with a valid:

        `Gestational diabetes? = Yes / No`

        outcome are used for supervised performance analysis.

        **No synthetic labels are generated.**
        """
    )
