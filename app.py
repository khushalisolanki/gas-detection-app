import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils import detect_gas

st.set_page_config(layout="wide")
st.title("Gas Sensor Intelligence Dashboard")

uploaded_file = st.file_uploader("Upload Sensor CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, skiprows=2)

    # Force numeric — done ONCE before any filtering
    for col in ["MOX1(Ohms)", "MOX3(Ohms)", "MOX4(Ohms)", "Time(sec)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Time(sec)"])
    df = df.reset_index(drop=True)

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

    # ---------------- GAS DETECTION (matches original chunking logic) ----------------
    range_size = 90
    chunk_block_size = range_size + 1  # 91 rows per block, matching original script

    total_length = len(df)
    num_chunks = (total_length + 1) // chunk_block_size

    gas_detected = False
    detection_results = []
    chunk_meta = []  # stores (start_time, end_time, result) for highlighting

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_block_size
        end_idx = start_idx + range_size

        if end_idx > total_length:
            break

        chunk = df.iloc[start_idx:end_idx].copy()

        # Filter out first 3 seconds of each chunk, matching original behaviour
        t0 = chunk["Time(sec)"].iloc[0]
        chunk = chunk[chunk["Time(sec)"] >= t0 + 3]

        if chunk.empty:
            continue

        result = detect_gas(chunk)
        detection_results.append({
            "Chunk": chunk_idx + 1,
            "Start Time (sec)": round(chunk["Time(sec)"].iloc[0], 2),
            "End Time (sec)": round(chunk["Time(sec)"].iloc[-1], 2),
            "Detection Result": result,
        })

        if result != "No gas detected":
            gas_detected = True

        chunk_meta.append({
            "start": chunk["Time(sec)"].iloc[0],
            "end": chunk["Time(sec)"].iloc[-1],
            "gas": result != "No gas detected",
        })

    # ---------------- ALERT MESSAGE ----------------
    st.subheader("Gas Detection Status")
    if gas_detected:
        st.error(" GAS DETECTED — See highlighted regions below")
    else:
        st.success(" No gas detected")

    # ---------------- SIGNAL PLOTS ----------------
    st.subheader("Sensor Visualization")

    for col in selected_sensors:

        # ---------- RAW ----------
        if plot_type in ["Raw Only", "Both"]:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df["Time(sec)"],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(color="steelblue", width=1.5),
            ))

            # Highlight gas-detected chunks as red shaded regions
            for meta in chunk_meta:
                if meta["gas"]:
                    fig.add_vrect(
                        x0=meta["start"], x1=meta["end"],
                        fillcolor="red", opacity=0.15,
                        layer="below", line_width=0,
                    )

            fig.update_layout(
                title=f"{col} — Raw Signal",
                xaxis_title="Time (sec)",
                yaxis_title="Resistance (Ohms)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---------- SLOPE (np.diff matches original script) ----------
        if plot_type in ["Slope Only", "Both"]:
            time_vals = df["Time(sec)"].values
            raw_vals = df[col].values

            # np.diff gives N-1 points; pair with midpoint times
            slope = np.diff(raw_vals) / np.diff(time_vals)
            time_slope = (time_vals[:-1] + time_vals[1:]) / 2  # midpoints

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=time_slope,
                y=slope,
                mode="lines",
                name=f"Slope of {col}",
                line=dict(color="darkorange", width=1.5),
            ))

            for meta in chunk_meta:
                if meta["gas"]:
                    fig.add_vrect(
                        x0=meta["start"], x1=meta["end"],
                        fillcolor="red", opacity=0.15,
                        layer="below", line_width=0,
                    )

            fig.update_layout(
                title=f"{col} — Slope vs Time",
                xaxis_title="Time (sec)",
                yaxis_title=f"dR/dt  ({col})",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- RESULTS TABLE ----------------
    st.subheader("Detection Results by Chunk")
    result_df = pd.DataFrame(detection_results)
    st.dataframe(result_df, use_container_width=True)

    st.download_button(
        " Download Detection Report",
        result_df.to_csv(index=False),
        file_name="gas_detection_report.csv",
        mime="text/csv",
    )
