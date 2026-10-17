from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from src.common import ROOT, DATA_PATH, MODEL_PATH, METRICS_PATH, load_data, TARGET

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Prediction")
st.caption("Placement-ready machine learning project | IBM Telco sample dataset")

if not DATA_PATH.exists():
    st.warning("Dataset not found. In your terminal, run `python src/download_data.py` from the project folder, then refresh this page.")
    st.stop()
if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.warning("Model files are not available yet. Run `python src/train.py` first, then refresh.")
    st.stop()

df = load_data()
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
model = joblib.load(MODEL_PATH)

tab1, tab2, tab3 = st.tabs(["Overview", "Predict a Customer", "Model Evaluation"])
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers in cleaned dataset", f"{len(df):,}")
    c2.metric("Churn rate", f"{(df[TARGET] == 'Yes').mean():.1%}")
    c3.metric("Selected model", metrics["selected_model"])
    st.subheader("Churn distribution")
    st.bar_chart(df[TARGET].value_counts().rename_axis("Churn").to_frame("Customers"))
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.info("Churn rate is descriptive, not a causal explanation. Model predictions support prioritisation and should be reviewed by a human.")

with tab2:
    st.write("Enter customer details. Fields are generated from the actual dataset schema where possible.")
    feature_df = df.drop(columns=[TARGET])
    values = {}
    cols = st.columns(2)
    for i, col in enumerate(feature_df.columns):
        series = feature_df[col]
        with cols[i % 2]:
            if pd.api.types.is_numeric_dtype(series):
                med = float(series.median()) if series.notna().any() else 0.0
                values[col] = st.number_input(col, value=med)
            else:
                opts = sorted(series.dropna().astype(str).unique().tolist())
                if not opts:
                    opts = ["Unknown"]
                # Most common observed category is a sensible default.
                mode = series.dropna().astype(str).mode()
                default = mode.iloc[0] if len(mode) else opts[0]
                idx = opts.index(default) if default in opts else 0
                values[col] = st.selectbox(col, opts, index=idx)
    threshold = st.slider("Churn probability threshold", 0.10, 0.90, 0.50, 0.05)
    if st.button("Predict churn", type="primary"):
        one = pd.DataFrame([values], columns=feature_df.columns)
        probability = float(model.predict_proba(one)[0, 1])
        predicted = "Likely to churn" if probability >= threshold else "Less likely to churn"
        a, b = st.columns(2)
        a.metric("Prediction", predicted)
        b.metric("Estimated churn probability", f"{probability:.1%}")
        if probability >= threshold:
            st.warning("Suggested next step: review recent service issues, contract options, usage changes, and customer feedback before contacting the customer.")
        else:
            st.success("Continue normal customer-care monitoring. A low score does not guarantee retention.")
        st.caption("This probability is a model estimate, not certainty. Validate recommendations with business policy and customer consent.")

with tab3:
    st.subheader("Held-out test-set metrics")
    st.write(f"Selected model: **{metrics['selected_model']}**")
    st.dataframe(pd.DataFrame(metrics["model_comparison"]).set_index("model"), use_container_width=True)
    st.subheader("Confusion matrix")
    cm = metrics.get("selected_model_confusion_matrix")
    if cm:
        st.dataframe(pd.DataFrame(cm, index=["Actual: No Churn", "Actual: Churn"],
                                  columns=["Predicted: No Churn", "Predicted: Churn"]))
    report_path = ROOT / "reports" / "model_comparison.png"
    cm_path = ROOT / "reports" / "confusion_matrix.png"
    if report_path.exists():
        st.image(str(report_path), caption="Model comparison on held-out test data")
    if cm_path.exists():
        st.image(str(cm_path), caption="Selected model confusion matrix")
    st.caption("Metrics are populated only after running the training script on the dataset.")
