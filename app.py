import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Classification Model Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "model"
DEFAULT_TEST_FILE = ROOT / "test_data.csv"


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    """
    <style>

    /* ========================================================
       GLOBAL PAGE
       ======================================================== */

    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e1e6ef;
    }

    section[data-testid="stSidebar"] * {
        color: #172033;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        background: #ffffff;
        border: 1px solid #dfe5ee;
        border-radius: 18px;
        padding: 32px 38px;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(20, 35, 60, 0.06);
    }

    .hero-tag {
        display: inline-block;
        background: #edf3ff;
        color: #315ea8;
        border: 1px solid #d9e5ff;
        border-radius: 20px;
        padding: 7px 14px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .hero-title {
        color: #14213d;
        font-size: 38px;
        line-height: 1.15;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #657286;
        font-size: 17px;
        line-height: 1.6;
        max-width: 950px;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-title {
        color: #14213d;
        font-size: 25px;
        font-weight: 750;
        margin-top: 32px;
        margin-bottom: 6px;
    }

    .section-description {
        color: #69778b;
        font-size: 15px;
        margin-bottom: 20px;
    }


    /* ========================================================
       INFORMATION CARDS
       ======================================================== */

    .info-card {
        background: #ffffff;
        border: 1px solid #dfe5ee;
        border-radius: 15px;
        padding: 21px;
        min-height: 125px;
        box-shadow: 0 5px 18px rgba(20, 35, 60, 0.045);
    }

    .info-label {
        color: #718096;
        font-size: 12px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 9px;
    }

    .info-value {
        color: #172033;
        font-size: 24px;
        line-height: 1.2;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .info-small {
        color: #7a8799;
        font-size: 13px;
        line-height: 1.4;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dfe5ee;
        border-radius: 15px;
        padding: 20px;
        min-height: 120px;
        box-shadow: 0 5px 18px rgba(20, 35, 60, 0.045);
    }

    div[data-testid="stMetricLabel"] {
        color: #69778b !important;
        font-size: 14px !important;
        font-weight: 650 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #172033 !important;
        font-size: 32px !important;
        font-weight: 800 !important;
    }


    /* ========================================================
       CUSTOM TABLE
       ======================================================== */

    .table-wrapper {
        width: 100%;
        overflow-x: auto;
        overflow-y: auto;
        background: #ffffff;
        border: 1px solid #dce3ed;
        border-radius: 14px;
        box-shadow: 0 5px 18px rgba(20, 35, 60, 0.045);
    }

    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
        font-size: 15px;
        color: #172033;
    }

    .custom-table th {
        background: #eef2f7;
        color: #334155;
        font-size: 14px;
        font-weight: 750;
        text-align: left;
        padding: 14px 16px;
        border-bottom: 2px solid #d8e0ea;
        white-space: nowrap;
    }

    .custom-table td {
        background: #ffffff;
        color: #273449;
        font-size: 15px;
        font-weight: 500;
        padding: 14px 16px;
        border-bottom: 1px solid #e7ebf1;
        white-space: nowrap;
    }

    .custom-table tr:hover td {
        background: #f8fafc;
    }

    .custom-table .number {
        text-align: right;
        font-variant-numeric: tabular-nums;
        font-weight: 600;
    }

    .custom-table .model-name {
        font-weight: 700;
        color: #1e3a5f;
    }


    /* ========================================================
       WINNER
       ======================================================== */

    .winner-card {
        background: #f0f8f3;
        border: 1px solid #c9e5d3;
        border-radius: 15px;
        padding: 21px 24px;
        margin-top: 22px;
        box-shadow: 0 5px 18px rgba(31, 95, 57, 0.04);
    }

    .winner-label {
        color: #4d765c;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .winner-name {
        color: #205b39;
        font-size: 25px;
        font-weight: 800;
        margin-top: 5px;
    }

    .winner-description {
        color: #547062;
        font-size: 15px;
        margin-top: 5px;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status {
        display: inline-block;
        background: #eef8f2;
        color: #28734b;
        border: 1px solid #d1e8da;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 750;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #8a95a5;
        font-size: 13px;
        padding-top: 40px;
    }

    </style>
    """
)


# ============================================================
# HELPER — HTML TABLE
# ============================================================

def render_html_table(
    dataframe,
    height=None,
    index=False,
    number_decimals=4,
):
    """
    Render a pandas dataframe as a readable HTML table.

    This is used instead of st.dataframe() because it gives us
    direct control over font size, padding and visibility.
    """

    df = dataframe.copy()

    if index:
        df = df.reset_index()

    # Round numeric columns
    for column in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            df[column] = df[column].round(
                number_decimals
            )

    html = df.to_html(
        index=False,
        classes="custom-table",
        border=0,
    )

    if height is not None:

        html = f"""
        <div
            class="table-wrapper"
            style="max-height:{height}px;"
        >
            {html}
        </div>
        """

    else:

        html = f"""
        <div class="table-wrapper">
            {html}
        </div>
        """

    st.html(html)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-tag">
            Machine Learning Assignment 2
        </div>

        <div class="hero-title">
            Classification Model Studio
        </div>

        <div class="hero-subtitle">
            Compare five machine learning classifiers on the
            Breast Cancer Wisconsin Diagnostic dataset using
            six standard evaluation metrics.
        </div>

    </div>
    """
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(model_path):

    return joblib.load(model_path)


@st.cache_data
def load_default_test():

    return pd.read_csv(
        DEFAULT_TEST_FILE
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## Experiment Setup")

    st.caption(
        "Configure the classifier and test dataset."
    )

    selected_model = st.selectbox(
        "Choose a model",
        list(MODEL_FILES.keys()),
    )

    st.markdown("---")

    st.markdown("### Test Dataset")

    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type=["csv"],
        help=(
            "CSV should contain the 30 feature columns. "
            "Include a target column when evaluation metrics "
            "are required."
        ),
    )

    if uploaded_file is None:

        st.info(
            "Using the bundled test_data.csv"
        )

    else:

        st.success(
            f"Using: {uploaded_file.name}"
        )

    st.markdown("---")

    st.markdown("### Models")

    for model_name in MODEL_FILES:

        if model_name == selected_model:

            st.markdown(
                f"**● {model_name}**"
            )

        else:

            st.markdown(
                f"○ {model_name}"
            )


# ============================================================
# LOAD DATASET
# ============================================================

try:

    if uploaded_file is not None:

        test_df = pd.read_csv(
            uploaded_file
        )

    else:

        test_df = load_default_test()

except Exception as exc:

    st.error(
        f"Unable to read the dataset: {exc}"
    )

    st.stop()


# ============================================================
# DATA VALIDATION
# ============================================================

if test_df.empty:

    st.error(
        "The dataset contains no rows."
    )

    st.stop()


missing_features = [
    feature
    for feature in FEATURES
    if feature not in test_df.columns
]


if missing_features:

    st.error(
        "The uploaded dataset is missing required feature columns."
    )

    st.write(
        missing_features
    )

    st.stop()


non_numeric = [
    feature
    for feature in FEATURES
    if not pd.api.types.is_numeric_dtype(
        test_df[feature]
    )
]


if non_numeric:

    st.error(
        "The following feature columns must be numeric:"
    )

    st.write(
        non_numeric
    )

    st.stop()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.html(
    """
    <div class="section-title">
        Dataset Overview
    </div>

    <div class="section-description">
        Information about the data currently being evaluated.
    </div>
    """
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.html(
        f"""
        <div class="info-card">

            <div class="info-label">
                Observations
            </div>

            <div class="info-value">
                {len(test_df)}
            </div>

            <div class="info-small">
                Rows available for evaluation
            </div>

        </div>
        """
    )


with col2:

    st.html(
        f"""
        <div class="info-card">

            <div class="info-label">
                Features
            </div>

            <div class="info-value">
                {len(FEATURES)}
            </div>

            <div class="info-small">
                Input variables used by the models
            </div>

        </div>
        """
    )


with col3:

    target_available = (
        "Available"
        if "target" in test_df.columns
        else "Not Available"
    )

    st.html(
        f"""
        <div class="info-card">

            <div class="info-label">
                Target
            </div>

            <div class="info-value">
                {target_available}
            </div>

            <div class="info-small">
                Required for model evaluation
            </div>

        </div>
        """
    )


with col4:

    st.html(
        f"""
        <div class="info-card">

            <div class="info-label">
                Selected Model
            </div>

            <div class="info-value">
                {selected_model}
            </div>

            <div class="info-small">
                Currently being evaluated
            </div>

        </div>
        """
    )


# ============================================================
# MODEL INPUT
# ============================================================

X = test_df[
    FEATURES
]


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    MODEL_FILES[
        selected_model
    ]
)


if not model_path.exists():

    st.error(
        f"Model file not found: {model_path}"
    )

    st.stop()


try:

    model = load_model(
        model_path
    )

except Exception as exc:

    st.error(
        f"Unable to load model: {exc}"
    )

    st.stop()


# ============================================================
# PREDICTIONS
# ============================================================

try:

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

except Exception as exc:

    st.error(
        f"Prediction failed: {exc}"
    )

    st.stop()


# ============================================================
# IF TARGET IS NOT AVAILABLE
# ============================================================

if "target" not in test_df.columns:

    prediction_df = X.copy()

    prediction_df[
        "prediction"
    ] = predictions

    prediction_df[
        "probability_class_1"
    ] = probabilities

    st.html(
        """
        <div class="section-title">
            Prediction Results
        </div>

        <div class="section-description">
            Predictions generated by the selected classifier.
        </div>
        """
    )

    render_html_table(
        prediction_df.head(20),
        height=550,
        index=False,
    )

    st.stop()


# ============================================================
# TARGET
# ============================================================

try:

    y = test_df[
        "target"
    ].astype(int)

except Exception:

    st.error(
        "The target column must contain binary numeric labels."
    )

    st.stop()


unique_targets = sorted(
    y.dropna().unique()
)


if len(unique_targets) != 2:

    st.error(
        "The target column must contain exactly two classes."
    )

    st.stop()


# ============================================================
# METRICS
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


metrics = {

    "Accuracy": accuracy_score(
        y,
        predictions,
    ),

    "AUC": roc_auc_score(
        y,
        probabilities,
    ),

    "Precision": precision_score(
        y,
        predictions,
        zero_division=0,
    ),

    "Recall": recall_score(
        y,
        predictions,
        zero_division=0,
    ),

    "F1 Score": f1_score(
        y,
        predictions,
        zero_division=0,
    ),

    "MCC": matthews_corrcoef(
        y,
        predictions,
    ),
}


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.html(
    """
    <div class="section-title">
        Model Performance
    </div>

    <div class="section-description">
        Evaluation metrics for the selected classifier.
    </div>
    """
)


heading_col, status_col = st.columns(
    [6, 1]
)


with heading_col:

    st.markdown(
        f"### {selected_model}"
    )


with status_col:

    st.html(
        """
        <div class="status">
            EVALUATED
        </div>
        """
    )


metric_columns = st.columns(6)


for column, (
    metric_name,
    metric_value,
) in zip(
    metric_columns,
    metrics.items(),
):

    with column:

        st.metric(
            metric_name,
            f"{metric_value:.4f}",
        )


# ============================================================
# DIAGNOSTIC ANALYSIS
# ============================================================

st.html(
    """
    <div class="section-title">
        Diagnostic Analysis
    </div>

    <div class="section-description">
        Detailed view of classification performance.
    </div>
    """
)


left, right = st.columns(
    2
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

with left:

    st.markdown(
        "#### Confusion Matrix"
    )

    cm = confusion_matrix(
        y,
        predictions,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual 0",
            "Actual 1",
        ],
        columns=[
            "Predicted 0",
            "Predicted 1",
        ],
    )

    render_html_table(
        cm_df,
        height=190,
        index=True,
        number_decimals=0,
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

with right:

    st.markdown(
        "#### Classification Report"
    )

    report = classification_report(
        y,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .round(4)
    )

    render_html_table(
        report_df,
        height=310,
        index=True,
        number_decimals=4,
    )


# ============================================================
# PREDICTION PREVIEW
# ============================================================

st.html(
    """
    <div class="section-title">
        Prediction Preview
    </div>

    <div class="section-description">
        First 20 observations with the original features,
        actual target, predicted class and prediction probability.
        Scroll horizontally to view all 30 features.
    </div>
    """
)


preview_df = test_df.copy()


preview_df[
    "prediction"
] = predictions


preview_df[
    "probability_class_1"
] = probabilities


render_html_table(
    preview_df.head(20),
    height=560,
    index=False,
    number_decimals=4,
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.html(
    """
    <div class="section-title">
        Model Comparison
    </div>

    <div class="section-description">
        Comparison of all five classifiers using the same
        test dataset and evaluation metrics.
    </div>
    """
)


comparison_rows = []


for model_name, filename in MODEL_FILES.items():

    comparison_path = (
        MODEL_DIR /
        filename
    )

    if not comparison_path.exists():

        continue


    try:

        comparison_model = load_model(
            comparison_path
        )

        comparison_predictions = (
            comparison_model.predict(X)
        )

        comparison_probabilities = (
            comparison_model.predict_proba(X)[:, 1]
        )

        comparison_rows.append(
            {

                "Model": model_name,

                "Accuracy": accuracy_score(
                    y,
                    comparison_predictions,
                ),

                "AUC": roc_auc_score(
                    y,
                    comparison_probabilities,
                ),

                "Precision": precision_score(
                    y,
                    comparison_predictions,
                    zero_division=0,
                ),

                "Recall": recall_score(
                    y,
                    comparison_predictions,
                    zero_division=0,
                ),

                "F1": f1_score(
                    y,
                    comparison_predictions,
                    zero_division=0,
                ),

                "MCC": matthews_corrcoef(
                    y,
                    comparison_predictions,
                ),

            }
        )

    except Exception as exc:

        st.warning(
            f"Could not evaluate {model_name}: {exc}"
        )


# ============================================================
# DISPLAY MODEL COMPARISON
# ============================================================

if comparison_rows:

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    render_html_table(
        comparison_df,
        height=330,
        index=False,
        number_decimals=4,
    )


    # ========================================================
    # OVERALL SCORE
    # ========================================================

    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]


    comparison_df[
        "Overall Score"
    ] = comparison_df[
        metric_columns
    ].mean(
        axis=1
    )


    winner_row = comparison_df.loc[
        comparison_df[
            "Overall Score"
        ].idxmax()
    ]


    winner = winner_row[
        "Model"
    ]

    winner_score = winner_row[
        "Overall Score"
    ]


    st.html(
        f"""
        <div class="winner-card">

            <div class="winner-label">
                Strongest Overall Metric Profile
            </div>

            <div class="winner-name">
                {winner}
            </div>

            <div class="winner-description">
                Average score across Accuracy, AUC,
                Precision, Recall, F1 Score and MCC:
                <strong>{winner_score:.4f}</strong>
            </div>

        </div>
        """
    )


else:

    st.warning(
        "No model comparison results are available."
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">
        Machine Learning Assignment 2
        &nbsp;•&nbsp;
        Classification Model Comparison
        &nbsp;•&nbsp;
        Breast Cancer Wisconsin Diagnostic Dataset
    </div>
    """
)