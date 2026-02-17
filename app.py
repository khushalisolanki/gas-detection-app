import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from utils import detect_gas

st.set_page_config(layout="wide")

st.title("🔥 Gas / Explosive Sensor Intelligence Dashboard")

uploaded_file = st.file_uploader("Upload Sensor CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file, skiprows=2)

    # force numeric
    for col in ["MOX1(Ohms)", "MOX3(Ohms)", "MOX4(Ohms)", "Time(sec)"]:
              
       df[col] = pd.to_numeric(df[col], errors="coerce")

       # remove unstable start region
       df = df[df["Time(sec)"] > 3]

     # optional: reset index
     #df = df.reset_index(drop=True)

    st.success("File loaded successfully")

    # ---------------- SIDEBAR CONTROLS ----------------
    st.sidebar.header("Dashboard Controls")

    mox_cols = ["MOX1(Ohms)", "MOX3(Ohms)", "MOX4(Ohms)"]

    selected_sensors = st.sidebar.multiselect(
        "Select Sensors",
        mox_cols,
        default=mox_cols
    )

    plot_type = st.sidebar.radio(
        "Select Plot Type",
        ["Raw Only", "Slope Only", "Both"]
    )

    show_data = st.sidebar.checkbox("Show Raw Data Table")

    if show_data:
        st.dataframe(df.head())

    # ---------------- GAS DETECTION ----------------
    chunk_size = 90
    gas_detected = False
    detection_results = []

    start = 0
    while start <= len(df) - chunk_size:
        chunk = df.iloc[start + 7:start + chunk_size].copy()
        result = detect_gas(chunk)
        detection_results.append(result)

        if result != "No gas detected":
            gas_detected = True

        start += chunk_size + 1

    # ---------------- ALERT MESSAGE ----------------
    st.subheader("Gas Detection Status")

    if gas_detected:
        st.error("🚨 GAS DETECTED — See highlighted regions")
    else:
        st.success("✅ No gas detected")

    # ---------------- SIGNAL PLOTS ----------------
    st.subheader("Sensor Visualization")

    for col in selected_sensors:

        # ---------- RAW ----------
        if plot_type in ["Raw Only", "Both"]:

            fig = px.line(
                df,
                x="Time(sec)",
                y=col,
                title=f"{col} Raw Signal"
            )

            # change color if gas detected
            if gas_detected:
                fig.update_traces(line=dict(color="red"))

            st.plotly_chart(fig, use_container_width=True)

        # ---------- SLOPE ----------
        if plot_type in ["Slope Only", "Both"]:

            slope = np.gradient(df[col], df["Time(sec)"])

            fig = px.line(
                x=df["Time(sec)"],
                y=slope,
                title=f"{col} Slope"
            )

            if gas_detected:
                fig.update_traces(line=dict(color="orange"))

            st.plotly_chart(fig, use_container_width=True)

    # ---------------- RESULTS TABLE ----------------
    st.subheader("Detection Results")

    result_df = pd.DataFrame({"Detection Result": detection_results})
    st.dataframe(result_df)

    st.download_button(
        "Download Detection Report",
        result_df.to_csv(index=False),
        file_name="gas_detection_report.csv"
    )
