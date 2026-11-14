"""Inspect metrics and held-out predictions from a local run."""
from pathlib import Path

import pandas as pd
import streamlit as st

results = Path(__file__).resolve().parents[1] / "results"
st.set_page_config(page_title="BeforeTheNotice", layout="wide")
st.title("BeforeTheNotice")
metrics = results / "metrics.csv"
predictions = results / "predictions.csv"
if not (metrics.is_file() and predictions.is_file()):
    st.info("Run the pipeline in docs/data_decisions.md to view held-out results here.")
    st.stop()

st.caption("Historical holdout scores for observed California WARN notices.")
st.subheader("Model performance")
st.dataframe(pd.read_csv(metrics), hide_index=True)
st.subheader("Held-out company quarters")
scores = pd.read_csv(predictions, dtype={"company_key": str})
quarter = st.selectbox("Quarter", sorted(scores.quarter.unique(), reverse=True))
st.dataframe(scores[scores.quarter.eq(quarter)].sort_values("logistic_score", ascending=False), hide_index=True)
