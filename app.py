import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stroke Classification Model Evaluator",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

METADATA_FILE = MODEL_DIR / "preprocessing_metadata.pkl"
SCALER_FILE = MODEL_DIR / "scaler.pkl"
TEST_LABELS_FILE = MODEL_DIR / "test_labels.csv"
TRAINING_RESULTS_FILE = MODEL_DIR / "evaluation_results.csv"


# ============================================================
# LOAD SAVED ARTIFACTS
# ============================================================

@st.cache_resource
def load_pickle_file(file_path):
    """Load a trusted local pickle artifact."""
    with open(file_path, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_all_models():
    """Load all saved classification models."""
    loaded_models = {}

    for model_name, file_name in MODEL_FILES.items():
        model_path = MODEL_DIR / file_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"Required model file is missing: {model_path}"
            )

        loaded_models[model_name] = load_pickle_file(model_path)

    return loaded_models


@st.cache_data
def load_csv_file(file_path):
    """Load a local CSV file."""
    return pd.read_csv(file_path)


# ============================================================
# VALIDATE PROJECT ARTIFACTS
# ============================================================

def validate_required_artifacts():
    """Return a list of missing deployment artifacts."""
    required_paths = [
        METADATA_FILE,
        SCALER_FILE,
        TEST_LABELS_FILE,
        *[MODEL_DIR / file_name for file_name in MODEL_FILES.values()],
    ]

    return [
        str(file_path)
        for file_path in required_paths
        if not file_path.exists()
    ]


# ============================================================
# PREPROCESS UPLOADED DATA
# ============================================================

def preprocess_uploaded_data(uploaded_df, metadata, scaler, model_name):
    """
    Apply the same preprocessing used during model training.

    Returns:
        model_input: Encoded data ready for the selected model.
        identifiers: Uploaded patient IDs, if available.
        uploaded_targets: Uploaded target values, if available.
        cleaned_df: Cleaned copy of the uploaded data.
    """
    data = uploaded_df.copy()

    identifier_column = metadata["identifier_column"]
    target_column = metadata["target_column"]
    continuous_columns = metadata["continuous_columns"]
    binary_columns = metadata["binary_columns"]
    categorical_columns = metadata["categorical_columns"]
    encoded_feature_columns = metadata["encoded_feature_columns"]
    bmi_median = metadata["bmi_median"]
    scaled_models = metadata["scaled_models"]

    required_input_columns = (
        continuous_columns
        + binary_columns
        + categorical_columns
    )

    missing_columns = [
        column
        for column in required_input_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Uploaded CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    identifiers = None
    if identifier_column in data.columns:
        identifiers = data[identifier_column].copy()

    uploaded_targets = None
    if target_column in data.columns:
        uploaded_targets = pd.to_numeric(
            data[target_column],
            errors="coerce",
        )

    # Convert continuous inputs to numeric values.
    for column in continuous_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # Use the training-data BMI median saved in metadata.
    data["bmi"] = data["bmi"].fillna(bmi_median)

    # Other continuous fields are required and should not be missing.
    missing_continuous = data[continuous_columns].isna().sum()
    invalid_continuous = missing_continuous[
        missing_continuous > 0
    ].to_dict()

    if invalid_continuous:
        raise ValueError(
            "Missing or non-numeric continuous values found: "
            f"{invalid_continuous}"
        )

    # Validate binary fields.
    for column in binary_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if data[column].isna().any():
            raise ValueError(
                f"Column '{column}' contains missing or non-numeric values."
            )

        invalid_values = set(data[column].unique()) - {0, 1}

        if invalid_values:
            raise ValueError(
                f"Column '{column}' must contain only 0 and 1. "
                f"Invalid values: {sorted(invalid_values)}"
            )

        data[column] = data[column].astype(int)

    feature_data = data.drop(
        columns=[identifier_column, target_column],
        errors="ignore",
    )

    encoded_data = pd.get_dummies(
        feature_data,
        columns=categorical_columns,
        drop_first=True,
        dtype=int,
    )

    # Match the exact columns and order used in model training.
    encoded_data = encoded_data.reindex(
        columns=encoded_feature_columns,
        fill_value=0,
    )

    model_input = encoded_data.copy()

    if model_name in scaled_models:
        model_input[continuous_columns] = scaler.transform(
            model_input[continuous_columns]
        )

    return (
        model_input,
        identifiers,
        uploaded_targets,
        data,
    )


# ============================================================
# OBTAIN ACTUAL LABELS FOR EVALUATION
# ============================================================

def obtain_actual_labels(
    uploaded_df,
    identifiers,
    uploaded_targets,
    metadata,
    saved_test_labels,
):
    """
    Obtain true labels either from the uploaded CSV or by matching
    uploaded IDs with the saved test-label file.
    """
    target_column = metadata["target_column"]
    identifier_column = metadata["identifier_column"]

    # Option 1: Target included in uploaded CSV.
    if uploaded_targets is not None:
        if uploaded_targets.isna().any():
            return None, (
                f"The uploaded '{target_column}' column contains "
                "missing or invalid values."
            )

        invalid_targets = set(uploaded_targets.unique()) - {0, 1}

        if invalid_targets:
            return None, (
                f"The uploaded '{target_column}' column must contain "
                "only 0 and 1."
            )

        return uploaded_targets.astype(int).to_numpy(), None

    # Option 2: Match IDs with the saved test labels.
    if identifiers is None:
        return None, (
            "Evaluation metrics cannot be calculated because the uploaded "
            f"file contains neither '{target_column}' nor "
            f"'{identifier_column}'. Predictions are still available."
        )

    label_lookup = saved_test_labels.set_index(
        identifier_column
    )[target_column]

    matched_labels = identifiers.map(label_lookup)

    if matched_labels.isna().any():
        unmatched_count = int(matched_labels.isna().sum())

        return None, (
            f"{unmatched_count} uploaded IDs were not found in the saved "
            "test-label file. Metrics are shown only when every uploaded "
            "record has a known true label."
        )

    return matched_labels.astype(int).to_numpy(), None


# ============================================================
# EVALUATION
# ============================================================

def calculate_metrics(y_actual, y_predicted, y_probability):
    """Calculate all six metrics required by the assignment."""
    metrics = {
        "Accuracy": accuracy_score(y_actual, y_predicted),
        "AUC": roc_auc_score(y_actual, y_probability),
        "Precision": precision_score(
            y_actual,
            y_predicted,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_actual,
            y_predicted,
            zero_division=0,
        ),
        "F1": f1_score(
            y_actual,
            y_predicted,
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(
            y_actual,
            y_predicted,
        ),
    }

    return metrics


def create_confusion_matrix_figure(y_actual, y_predicted, model_name):
    """Create a confusion-matrix figure."""
    matrix = confusion_matrix(
        y_actual,
        y_predicted,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(figsize=(6, 4.5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Stroke", "Stroke"],
        yticklabels=["No Stroke", "Stroke"],
        ax=axis,
    )

    axis.set_title(f"{model_name} Confusion Matrix")
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("Actual Class")

    figure.tight_layout()

    return figure


def create_classification_report(y_actual,y_predicted):
    """
    Create a correctly structured classification report.

    Accuracy is excluded because accuracy is a single overall
    metric and should not be displayed as a row under precision,
    recall, F1 score and support.
    """

    report = classification_report(y_actual,y_predicted,labels=[0, 1],target_names=["No Stroke","Stroke"],
        output_dict=True,
        zero_division=0
    )

    report_rows = {
        "No Stroke": report["No Stroke"],
        "Stroke": report["Stroke"],
        "Macro Average": report["macro avg"],
        "Weighted Average": report["weighted avg"]
    }

    report_df = pd.DataFrame.from_dict(
        report_rows,
        orient="index"
    )

    report_df = report_df[
        [
            "precision",
            "recall",
            "f1-score",
            "support"
        ]
    ]

    report_df = report_df.rename(
        columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1 Score",
            "support": "Support"
        }
    )

    report_df[
        [
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ] = report_df[
        [
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ].round(4)

    report_df["Support"] = (
        report_df["Support"]
        .round(0)
        .astype(int)
    )

    report_df.index.name = "Class / Average"

    return report_df


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("Stroke Classification Model Evaluator")

st.markdown(
    """
    Upload the provided **test_data.csv**, select a trained model, and
    review predictions and evaluation results. The application applies
    the same encoding and scaling rules used during model training.
    """
)

st.warning(
    "Academic demonstration only. This application is not a clinical "
    "diagnostic tool and must not be used for medical decisions."
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

missing_artifacts = validate_required_artifacts()

if missing_artifacts:
    st.error("Required trained artifacts are missing:")

    for artifact in missing_artifacts:
        st.code(artifact)

    st.info(
        "Run `python model_training.py` successfully before starting "
        "the Streamlit application."
    )

    st.stop()

try:
    metadata = load_pickle_file(METADATA_FILE)
    scaler = load_pickle_file(SCALER_FILE)
    models = load_all_models()
    saved_test_labels = load_csv_file(TEST_LABELS_FILE)

except Exception as error:
    st.error(f"Could not load project artifacts: {error}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Model Settings")

    selected_model_name = st.selectbox(
        "Select a classification model",
        options=list(MODEL_FILES.keys()),
    )

    uses_scaling = (
        selected_model_name
        in metadata["scaled_models"]
    )

    st.write(
        "Continuous feature scaling:",
        "Yes" if uses_scaling else "No",
    )

    st.caption(
        "Only age, average glucose level, and BMI are standardized "
        "for models configured to use scaled data."
    )

    if TRAINING_RESULTS_FILE.exists():
        st.divider()
        st.subheader("Saved Test Results")

        saved_results = load_csv_file(
            TRAINING_RESULTS_FILE
        )

        selected_saved_result = saved_results[
            saved_results["ML Model Name"]
            == selected_model_name
        ]

        if not selected_saved_result.empty:
            st.dataframe(
                selected_saved_result,
                hide_index=True,
                use_container_width=True,
            )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload test data in CSV format",
    type=["csv"],
    help="Use the test_data.csv generated by model_training.py.",
)

if uploaded_file is None:
    st.info(
        "Upload a CSV file to generate predictions and evaluation results."
    )

    with st.expander("Expected input columns"):
        expected_columns = (
            [metadata["identifier_column"]]
            + metadata["continuous_columns"]
            + metadata["binary_columns"]
            + metadata["categorical_columns"]
        )

        st.code("\n".join(expected_columns))

    st.stop()


# ============================================================
# READ AND VALIDATE UPLOAD
# ============================================================

try:
    uploaded_df = pd.read_csv(uploaded_file)

except Exception as error:
    st.error(f"Could not read the uploaded CSV: {error}")
    st.stop()

if uploaded_df.empty:
    st.error("The uploaded CSV contains no records.")
    st.stop()

st.success(
    f"CSV loaded successfully: {uploaded_df.shape[0]} rows and "
    f"{uploaded_df.shape[1]} columns."
)

with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(
        uploaded_df.head(20),
        use_container_width=True,
    )


# ============================================================
# PREPROCESS AND PREDICT
# ============================================================

try:
    (
        model_input,
        identifiers,
        uploaded_targets,
        cleaned_uploaded_df,
    ) = preprocess_uploaded_data(
        uploaded_df=uploaded_df,
        metadata=metadata,
        scaler=scaler,
        model_name=selected_model_name,
    )

except Exception as error:
    st.error(f"Preprocessing failed: {error}")
    st.stop()

selected_model = models[selected_model_name]

try:
    predictions = selected_model.predict(
        model_input
    ).astype(int)

    if hasattr(selected_model, "predict_proba"):
        probabilities = selected_model.predict_proba(
            model_input
        )[:, 1]

    elif hasattr(selected_model, "decision_function"):
        scores = selected_model.decision_function(
            model_input
        )

        score_min = scores.min()
        score_max = scores.max()

        if score_max == score_min:
            probabilities = np.zeros_like(
                scores,
                dtype=float,
            )
        else:
            probabilities = (
                (scores - score_min)
                / (score_max - score_min)
            )

    else:
        probabilities = predictions.astype(float)

except Exception as error:
    st.error(f"Model prediction failed: {error}")
    st.stop()


# ============================================================
# CREATE PREDICTION OUTPUT
# ============================================================

prediction_output = pd.DataFrame()

if identifiers is not None:
    prediction_output[
        metadata["identifier_column"]
    ] = identifiers.values

prediction_output["Predicted Stroke Class"] = predictions
prediction_output["Predicted Label"] = np.where(
    predictions == 1,
    "Stroke",
    "No Stroke",
)
prediction_output["Stroke Probability"] = np.round(
    probabilities,
    6,
)

actual_labels, label_message = obtain_actual_labels(
    uploaded_df=uploaded_df,
    identifiers=identifiers,
    uploaded_targets=uploaded_targets,
    metadata=metadata,
    saved_test_labels=saved_test_labels,
)

if actual_labels is not None:
    prediction_output["Actual Stroke Class"] = actual_labels
    prediction_output["Prediction Correct"] = (
        predictions == actual_labels
    )


# ============================================================
# SUMMARY
# ============================================================

st.subheader("Prediction Summary")

summary_column_1, summary_column_2, summary_column_3 = st.columns(3)

summary_column_1.metric(
    "Records Evaluated",
    f"{len(predictions):,}",
)

summary_column_2.metric(
    "Predicted No Stroke",
    f"{int((predictions == 0).sum()):,}",
)

summary_column_3.metric(
    "Predicted Stroke",
    f"{int((predictions == 1).sum()):,}",
)


# ============================================================
# METRICS AND REPORTS
# ============================================================

if actual_labels is not None:
    st.subheader("Evaluation Metrics")

    try:
        metrics = calculate_metrics(
            actual_labels,
            predictions,
            probabilities,
        )

        metric_order = [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC",
        ]

        metric_columns_ui = st.columns(3)

        for index, metric_name in enumerate(metric_order):
            metric_columns_ui[index % 3].metric(
                metric_name,
                f"{metrics[metric_name]:.4f}",
            )

    except ValueError as error:
        st.warning(
            "Some evaluation metrics could not be calculated. "
            f"Details: {error}"
        )

    report_column, matrix_column = st.columns([1.15, 1])

    with report_column:
        st.subheader("Classification Report")

        classification_report_df = create_classification_report(
            actual_labels,
            predictions,
        )

        st.dataframe(
            classification_report_df,
            use_container_width=True,
        )

    with matrix_column:
        st.subheader("Confusion Matrix")

        confusion_figure = create_confusion_matrix_figure(
            actual_labels,
            predictions,
            selected_model_name,
        )

        st.pyplot(confusion_figure)
        plt.close(confusion_figure)

else:
    st.warning(label_message)


# ============================================================
# PREDICTIONS AND DOWNLOAD
# ============================================================

st.subheader("Prediction Results")

st.dataframe(
    prediction_output,
    use_container_width=True,
    hide_index=True,
)

prediction_csv = prediction_output.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Predictions as CSV",
    data=prediction_csv,
    file_name=(
        selected_model_name
        .lower()
        .replace(" ", "_")
        + "_predictions.csv"
    ),
    mime="text/csv",
)


# ============================================================
# METHODOLOGY INFORMATION
# ============================================================

with st.expander("Preprocessing and evaluation methodology"):
    st.markdown(
        f"""
        - **Selected model:** {selected_model_name}
        - **Target variable:** `{metadata['target_column']}`
        - **Identifier excluded from training:** `{metadata['identifier_column']}`
        - **Continuous columns:** {', '.join(metadata['continuous_columns'])}
        - **Binary columns:** {', '.join(metadata['binary_columns'])}
        - **Categorical columns:** {', '.join(metadata['categorical_columns'])}
        - **BMI missing-value treatment:** Training-data median of {metadata['bmi_median']:.2f}
        - **Selective scaling used:** {'Yes' if uses_scaling else 'No'}
        - **Decision threshold:** Saved model default threshold
        """
    )

