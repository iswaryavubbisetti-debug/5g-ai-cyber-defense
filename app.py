import streamlit as st
import pandas as pd
import numpy as np
import os, sys
import json
from sklearn.preprocessing import StandardScaler

# Make sure src folder is visible
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from model import IsolationForestModel
    from response import automated_response
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

st.title("🔐 AI-Driven Threat Detection in 5G Networks")

uploaded_file = st.file_uploader("📂 Upload Network Flow CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Sample Data", df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in file")
        st.stop()

    X_raw = df[numeric_cols].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Train Isolation Forest model
    model = IsolationForestModel()
    model.train(X)
    scores = model.score(X)

    threshold = np.percentile(scores, 95)
    df["anomaly"] = (scores >= threshold).astype(int)

    st.subheader("🚨 Detection Summary")
    st.write(df["anomaly"].value_counts())

    # Save alerts
    alerts = [{"type": "ddos", "src": str(i)} for i in df.index[df["anomaly"] == 1]]
    with open("alerts.json", "w") as f:
        json.dump(alerts, f)

    st.download_button("⬇️ Download Alerts JSON",
                       json.dumps(alerts),
                       "alerts.json")

    # Trigger automated response
    if st.button("⚡ Trigger Response"):
        automated_response("alerts.json")
        st.success("Response executed ✅")
