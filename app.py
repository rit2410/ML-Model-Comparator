# File: app.py
import io
import os
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.preprocess import prepare_dataset
from src.train import get_supported_models, fit_models
from src.evaluate import summarize_results_table
from src.utils import set_plot_theme, dataframe_hash

warnings.filterwarnings("ignore")

# ---------- Streamlit Page Config ----------
st.set_page_config(
    page_title="ML Model Comparator",
    page_icon="🧠",
    layout="wide",
)
set_plot_theme()

st.title("ML Model Comparator")
st.caption(
    "Upload a dataset, pick models, and compare performance with metrics and plots."
)

# ---------- Sidebar Controls ----------
st.sidebar.header("1) Upload or Load a Dataset")

def _load_builtin_dataset(name: str) -> pd.DataFrame:
    if name == "Breast Cancer (classification)":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        # Ensure target is last for UX
        if "target" in df.columns:
            cols = [c for c in df.columns if c != "target"] + ["target"]
            df = df[cols]
        return df
    elif name == "Wine (classification)":
        from sklearn.datasets import load_wine
        data = load_wine(as_frame=True)
        df = data.frame
        if "target" in df.columns:
            cols = [c for c in df.columns if c != "target"] + ["target"]
            df = df[cols]
        return df
    else:
        raise ValueError("Unknown built-in dataset")

builtin_choice = st.sidebar.selectbox(
    "Or pick a built-in dataset",
    ("None", "Breast Cancer (classification)", "Wine (classification)"),
    index=0,
)

uploaded = st.sidebar.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])  # noqa: E501

@st.cache_data(show_spinner=False)
def _read_file(file) -> Optional[pd.DataFrame]:
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

if uploaded is not None:
    df: Optional[pd.DataFrame] = _read_file(uploaded)
elif builtin_choice != "None":
    df = _load_builtin_dataset(builtin_choice)
else:
    df = None

if df is None:
    st.info("Upload a dataset or choose a built-in one to begin.")
    st.stop()

# Preview
st.subheader("Data Preview")
st.write("Shape:", df.shape)
st.dataframe(df.head(50), use_container_width=True)

# ---------- Target & Options ----------
st.sidebar.header("2) Configure Experiment")

target_col = st.sidebar.selectbox(
    "Target column (label)", options=df.columns, index=len(df.columns) - 1
)

test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

# Model selection
st.sidebar.header("3) Select Models")
SUPPORTED = get_supported_models()
model_names = list(SUPPORTED.keys())
chosen_models = st.sidebar.multiselect(
    "Choose one or more models",
    options=model_names,
    default=["Logistic Regression", "Random Forest"],
)

# Advanced options
with st.sidebar.expander("Advanced preprocessing", expanded=False):
    scale_numeric = st.checkbox("Scale numeric features", value=True)
    balance_classes = st.checkbox("Balance classes (class_weight='balanced' when supported)", value=True)  # noqa: E501

run_btn = st.sidebar.button("Train & Compare", type="primary")

with st.sidebar.expander("Hyperparameter Tuning", expanded=False):
    tuning_method = st.selectbox(
        "Tuning method",
        ["None", "Grid Search", "Random Search"],
        index=0
    )
    cv_folds = st.slider("Cross-validation folds", 2, 10, 3)
    n_iter = st.number_input("Iterations (for Random Search)", value=10, step=1)


if not run_btn:
    st.stop()

# ---------- Prepare Data ----------
with st.spinner("Preparing data…"):
    try:
        prep = prepare_dataset(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            random_state=int(random_state),
            scale_numeric=scale_numeric,
        )
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

X_train, X_test = prep.X_train, prep.X_test
y_train, y_test = prep.y_train, prep.y_test
preprocessor = prep.preprocessor
class_names = prep.class_names
n_classes = len(class_names)

# ---------- Train & Evaluate ----------
st.subheader("Results")
if len(chosen_models) == 0:
    st.warning("Please select at least one model.")
    st.stop()

with st.spinner("Training models…"):
    results = fit_models(
        model_names=chosen_models,
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        use_class_weight=balance_classes,
        tuning_method=tuning_method,
        cv_folds=cv_folds,
        n_iter=n_iter,
    )

# ---------- Display: Table & Plots ----------
summary_df = summarize_results_table(results)
st.dataframe(summary_df, use_container_width=True)

# Tabs for per-model details
tabs = st.tabs([r["model_name"] for r in results])
for tab, res in zip(tabs, results):
    with tab:
        st.markdown(f"### {res['model_name']}")
        cols = st.columns(2)
        with cols[0]:
            st.pyplot(res["confusion_fig"], use_container_width=True)
        with cols[1]:
            if res.get("roc_fig") is not None:
                st.pyplot(res["roc_fig"], use_container_width=True)
            else:
                st.info("ROC curve not available (model doesn't expose probabilities).")
        with st.expander("Classification report"):
            st.dataframe(pd.DataFrame(res["report"]).T, use_container_width=True)
        if res.get("best_params"):
            st.markdown("**Best parameters found:**")
            st.json(res["best_params"])


st.success("Done! You can tweak options on the left and re-run.")


