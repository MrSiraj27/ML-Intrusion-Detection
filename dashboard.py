"""Streamlit dashboard for the UNSW-NB15 intrusion detection proof-of-concept."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_CLEAN_PATH = DATA_DIR / "train_clean.csv"
TEST_CLEAN_PATH = DATA_DIR / "test_clean.csv"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.joblib"

sns.set_theme(style="whitegrid")

st.set_page_config(page_title="IDS ML Dashboard", layout="wide")


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------

@st.cache_data
def load_clean_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_preprocessing_artifacts():
    if not PREPROCESSING_PATH.exists():
        return None
    return joblib.load(PREPROCESSING_PATH)


@st.cache_resource
def load_joblib_model(path: Path):
    return joblib.load(path)


def list_saved_models():
    return sorted(MODELS_DIR.glob("*.joblib"))


# --------------------------------------------------------------------------
# Preprocessing helpers (mirrors notebooks/02_preprocessing.ipynb)
# --------------------------------------------------------------------------

def safe_label_transform(encoder, series: pd.Series) -> np.ndarray:
    known = set(encoder.classes_)
    mapped = series.where(series.isin(known), other="__unknown__")
    if "__unknown__" not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, "__unknown__")
    return encoder.transform(mapped)


def preprocess_raw_df(raw_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Apply the fitted encoders/scaler to a raw (unprocessed) UNSW-NB15-style dataframe."""
    df = raw_df.copy()

    numeric_cols = artifacts["numeric_cols"]
    onehot_cols = artifacts["onehot_cols"]
    scale_cols = artifacts["scale_cols"]
    medians = artifacts["medians"]
    feature_columns = artifacts["feature_columns"]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = medians[col]
    df[numeric_cols] = df[numeric_cols].fillna(medians)

    for col in ["proto"] + onehot_cols:
        if col not in df.columns:
            df[col] = "missing"
        df[col] = df[col].fillna("missing")

    df["proto_encoded"] = safe_label_transform(artifacts["proto_encoder"], df["proto"])

    onehot_encoder = artifacts["onehot_encoder"]
    onehot_arr = onehot_encoder.transform(df[onehot_cols])
    onehot_df = pd.DataFrame(
        onehot_arr,
        columns=onehot_encoder.get_feature_names_out(onehot_cols),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=onehot_cols), onehot_df], axis=1)

    df[scale_cols] = artifacts["scaler"].transform(df[scale_cols])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_columns]


# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------

if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "trained_model_name" not in st.session_state:
    st.session_state.trained_model_name = None
if "cv_results" not in st.session_state:
    st.session_state.cv_results = None


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------

st.sidebar.title("IDS ML Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Dataset Upload/Explorer", "Model Training", "Visualizations", "Predictions"],
)

st.sidebar.markdown("---")
if st.session_state.trained_model_name:
    st.sidebar.success(f"Active model: {st.session_state.trained_model_name}")
else:
    st.sidebar.info("No model trained yet this session.")


# --------------------------------------------------------------------------
# Page: Dataset Upload/Explorer
# --------------------------------------------------------------------------

if page == "Dataset Upload/Explorer":
    st.title("Dataset Upload / Explorer")

    source = st.radio(
        "Data source",
        ["Use bundled train_clean.csv", "Use bundled test_clean.csv", "Upload a CSV"],
        horizontal=True,
    )

    df = None
    if source == "Upload a CSV":
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
    elif source == "Use bundled train_clean.csv":
        if TRAIN_CLEAN_PATH.exists():
            df = load_clean_csv(TRAIN_CLEAN_PATH)
        else:
            st.warning("data/train_clean.csv not found. Run notebooks/02_preprocessing.ipynb first.")
    else:
        if TEST_CLEAN_PATH.exists():
            df = load_clean_csv(TEST_CLEAN_PATH)
        else:
            st.warning("data/test_clean.csv not found. Run notebooks/02_preprocessing.ipynb first.")

    if df is not None:
        col1, col2 = st.columns(2)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))

        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Column data types")
        st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)

        st.subheader("Null value counts")
        nulls = df.isnull().sum()
        st.dataframe(nulls[nulls > 0].rename("nulls") if nulls.sum() else pd.DataFrame({"nulls": ["None"]}))

        st.subheader("Summary statistics")
        st.dataframe(df.describe().T, use_container_width=True)

        if "label" in df.columns:
            st.subheader("Class balance (label)")
            counts = df["label"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(
                x=["Normal (0)", "Attack (1)"], y=counts.reindex([0, 1], fill_value=0).values,
                hue=["Normal (0)", "Attack (1)"], palette="Set2", legend=False, ax=ax,
            )
            ax.set_ylabel("Count")
            st.pyplot(fig)


# --------------------------------------------------------------------------
# Page: Model Training
# --------------------------------------------------------------------------

elif page == "Model Training":
    st.title("Model Training")

    if not TRAIN_CLEAN_PATH.exists():
        st.error("data/train_clean.csv not found. Run notebooks/02_preprocessing.ipynb first.")
        st.stop()

    train_df = load_clean_csv(TRAIN_CLEAN_PATH)
    drop_cols = [c for c in ["label", "attack_cat_encoded"] if c in train_df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["label"]

    st.caption(f"Training on data/train_clean.csv — {X_train.shape[0]:,} rows, {X_train.shape[1]} features")

    algo = st.selectbox("Algorithm", ["Decision Tree", "Random Forest"])

    with st.form("train_form"):
        if algo == "Decision Tree":
            max_depth = st.slider("max_depth (0 = no limit)", 0, 50, 10)
            min_samples_split = st.slider("min_samples_split", 2, 50, 2)
            criterion = st.selectbox("criterion", ["gini", "entropy"])
        else:
            n_estimators = st.slider("n_estimators", 10, 500, 200, step=10)
            max_depth = st.slider("max_depth (0 = no limit)", 0, 50, 10)
            min_samples_split = st.slider("min_samples_split", 2, 50, 2)
            criterion = st.selectbox("criterion", ["gini", "entropy"])

        submitted = st.form_submit_button("Train with 5-fold cross-validation")

    if submitted:
        depth_param = None if max_depth == 0 else max_depth

        if algo == "Decision Tree":
            model = DecisionTreeClassifier(
                max_depth=depth_param,
                min_samples_split=min_samples_split,
                criterion=criterion,
                random_state=42,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=depth_param,
                min_samples_split=min_samples_split,
                criterion=criterion,
                random_state=42,
                n_jobs=-1,
            )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        with st.spinner("Running 5-fold cross-validation..."):
            cv_results = cross_validate(
                model, X_train, y_train, cv=cv,
                scoring=["accuracy", "precision", "recall", "f1"],
            )

        st.subheader("Cross-validation results (5-fold)")
        metrics_df = pd.DataFrame({
            metric: [cv_results[f"test_{metric}"].mean(), cv_results[f"test_{metric}"].std()]
            for metric in ["accuracy", "precision", "recall", "f1"]
        }, index=["mean", "std"])
        st.dataframe(metrics_df.round(4), use_container_width=True)

        with st.spinner("Fitting final model on the full training set..."):
            model.fit(X_train, y_train)

        st.session_state.trained_model = model
        st.session_state.trained_model_name = f"{algo} (session)"
        st.session_state.cv_results = metrics_df

        st.success(f"{algo} trained and ready to use in Visualizations / Predictions.")

        save_name = st.text_input(
            "Filename to save this model as (in models/)",
            value="decision_tree.joblib" if algo == "Decision Tree" else "random_forest.joblib",
            key="save_name_input",
        )

    if st.session_state.trained_model is not None:
        st.markdown("---")
        default_name = "decision_tree.joblib" if isinstance(st.session_state.trained_model, DecisionTreeClassifier) else "random_forest.joblib"
        save_name = st.text_input("Save trained model as", value=default_name, key="save_name_persist")
        if st.button("Save model to models/"):
            out_path = MODELS_DIR / save_name
            joblib.dump(st.session_state.trained_model, out_path)
            st.success(f"Saved to {out_path}")


# --------------------------------------------------------------------------
# Page: Visualizations
# --------------------------------------------------------------------------

elif page == "Visualizations":
    st.title("Visualizations")

    if not TEST_CLEAN_PATH.exists():
        st.error("data/test_clean.csv not found. Run notebooks/02_preprocessing.ipynb first.")
        st.stop()

    test_df = load_clean_csv(TEST_CLEAN_PATH)
    drop_cols = [c for c in ["label", "attack_cat_encoded"] if c in test_df.columns]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["label"]

    model_choice = st.radio(
        "Model to evaluate",
        ["Use model trained this session", "Load a saved model"],
        horizontal=True,
    )

    model = None
    model_label = None

    if model_choice == "Use model trained this session":
        model = st.session_state.trained_model
        model_label = st.session_state.trained_model_name
        if model is None:
            st.warning("No model trained yet this session — train one on the Model Training page, or load a saved model instead.")
    else:
        saved = [p for p in list_saved_models() if p.name != "preprocessing.joblib"]
        if not saved:
            st.warning("No saved models found in models/.")
        else:
            chosen = st.selectbox("Saved model file", [p.name for p in saved])
            model = load_joblib_model(MODELS_DIR / chosen)
            model_label = chosen

    if model is not None:
        st.caption(f"Evaluating: {model_label} on data/test_clean.csv ({len(X_test):,} rows)")

        try:
            X_eval = X_test[model.feature_names_in_]
        except AttributeError:
            X_eval = X_test

        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        m2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        m3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        m4.metric("F1", f"{f1_score(y_test, y_pred):.4f}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"]).plot(ax=ax, cmap="Blues", colorbar=False)
            st.pyplot(fig)

        with col_b:
            st.subheader("ROC curve")
            if y_proba is not None:
                fig, ax = plt.subplots(figsize=(4, 4))
                RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                st.pyplot(fig)
            else:
                st.info("Model has no predict_proba — cannot draw an ROC curve.")

        st.subheader("Feature importance")
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=X_eval.columns)
            top = importances.sort_values(ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=top.values, y=top.index, hue=top.index, palette="viridis", legend=False, ax=ax)
            ax.set_xlabel("Importance")
            st.pyplot(fig)
        else:
            st.info("This model does not expose feature_importances_.")


# --------------------------------------------------------------------------
# Page: Predictions
# --------------------------------------------------------------------------

elif page == "Predictions":
    st.title("Predictions")

    artifacts = load_preprocessing_artifacts()
    if artifacts is None:
        st.error("models/preprocessing.joblib not found. Run notebooks/02_preprocessing.ipynb first.")
        st.stop()

    saved = [p for p in list_saved_models() if p.name != "preprocessing.joblib"]
    model_options = {}
    if st.session_state.trained_model is not None:
        model_options["Model trained this session"] = st.session_state.trained_model
    for p in saved:
        model_options[p.name] = p

    if not model_options:
        st.warning("No trained or saved models available. Train one on the Model Training page.")
        st.stop()

    chosen_key = st.selectbox("Model to use for predictions", list(model_options.keys()))
    chosen = model_options[chosen_key]
    model = chosen if not isinstance(chosen, Path) else load_joblib_model(chosen)

    tab_manual, tab_batch = st.tabs(["Manual input", "CSV batch upload"])

    with tab_manual:
        st.caption("Fill in raw traffic values below (defaults are training-set medians).")

        numeric_cols = artifacts["numeric_cols"]
        raw_stats = artifacts["raw_stats"]
        categorical_options = artifacts["categorical_options"]

        with st.form("manual_predict_form"):
            cat_values = {}
            cc1, cc2, cc3 = st.columns(3)
            cat_values["proto"] = cc1.selectbox("proto", categorical_options["proto"], index=categorical_options["proto"].index("tcp") if "tcp" in categorical_options["proto"] else 0)
            cat_values["service"] = cc2.selectbox("service", categorical_options["service"])
            cat_values["state"] = cc3.selectbox("state", categorical_options["state"])

            st.markdown("**Numeric traffic features**")
            numeric_values = {}
            cols = st.columns(3)
            for i, col_name in enumerate(numeric_cols):
                stats = raw_stats[col_name]
                target_col = cols[i % 3]
                numeric_values[col_name] = target_col.number_input(
                    col_name,
                    value=float(stats["median"]),
                    help=f"train range: {stats['min']:.3g} to {stats['max']:.3g}",
                )

            predict_clicked = st.form_submit_button("Predict")

        if predict_clicked:
            row = {**numeric_values, **cat_values}
            raw_df = pd.DataFrame([row])
            X_processed = preprocess_raw_df(raw_df, artifacts)

            try:
                X_processed = X_processed[model.feature_names_in_]
            except AttributeError:
                pass

            pred = model.predict(X_processed)[0]
            proba = model.predict_proba(X_processed)[0][1] if hasattr(model, "predict_proba") else None

            if pred == 1:
                st.error(f"Prediction: ATTACK" + (f" (probability {proba:.2%})" if proba is not None else ""))
            else:
                st.success(f"Prediction: NORMAL" + (f" (attack probability {proba:.2%})" if proba is not None else ""))

    with tab_batch:
        st.caption("Upload a CSV with raw UNSW-NB15-style columns (proto, service, state, and numeric traffic fields).")
        batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_csv")

        if batch_file is not None:
            raw_batch_df = pd.read_csv(batch_file)
            st.write(f"Loaded {len(raw_batch_df):,} rows.")

            if st.button("Run batch prediction"):
                with st.spinner("Preprocessing and predicting..."):
                    X_processed = preprocess_raw_df(raw_batch_df, artifacts)
                    try:
                        X_processed = X_processed[model.feature_names_in_]
                    except AttributeError:
                        pass

                    preds = model.predict(X_processed)
                    result_df = raw_batch_df.copy()
                    result_df["prediction"] = np.where(preds == 1, "Attack", "Normal")
                    if hasattr(model, "predict_proba"):
                        result_df["attack_probability"] = model.predict_proba(X_processed)[:, 1]

                st.dataframe(result_df.head(50), use_container_width=True)
                st.download_button(
                    "Download predictions as CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
