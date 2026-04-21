import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("Gas Sensor Intelligence Dashboard")


# ─────────────────────────────────────────────
# COLUMNS TO DROP  (matches original script)
# ─────────────────────────────────────────────
COLS_TO_DROP = [
    "MOX2(Ohms)", "UH1 Vtg(V)", "UH2 Vtg(V)", "UH3 Vtg(V)", "UH4 Vtg(V)",
    "Ambient Temperature(degC)", "Ambient Humdity(%%)", "Ambient Pressure(hPa)",
]
MOX_COLS   = ["MOX1(Ohms)", "MOX3(Ohms)", "MOX4(Ohms)"]
CHUNK_SIZE = 90


# ─────────────────────────────────────────────
# REAL GAS DETECTION  (ported from original script exactly)
# ─────────────────────────────────────────────
def detect_gas_chunk(chunk):
    """
    Run detection logic on a single 90-row chunk.
    Returns a list of finding strings (empty list = no gas).

    Original logic:
      - Uses np.gradient for slope & acceleration
      - MOX1: checks time_diff2_mox1 <= 39 + sub-rules for NO2/SO2/NH3
      - MOX3: checks time_diff2_mox3 < 30 and Min_slope3 < -500 → CH4 + concentration
      - else: No gas detected
    """
    findings = []

    # Slopes (np.gradient matches original — kept intentionally)
    chunk = chunk.copy()
    chunk["slope1"] = np.gradient(chunk["MOX1(Ohms)"].values, chunk["Time(sec)"].values)
    chunk["slope3"] = np.gradient(chunk["MOX3(Ohms)"].values, chunk["Time(sec)"].values)
    chunk["slope4"] = np.gradient(chunk["MOX4(Ohms)"].values, chunk["Time(sec)"].values)

    Min_slope1 = chunk["slope1"].min()
    Min_slope3 = chunk["slope3"].min()
    Min_slope4 = chunk["slope4"].min()

    start_time = chunk["Time(sec)"].iloc[0]

    # ── MOX1 ──
    min_time_mox1 = chunk.loc[chunk["MOX1(Ohms)"].idxmin(), "Time(sec)"]
    max_time_mox1 = chunk.loc[chunk["MOX1(Ohms)"].idxmax(), "Time(sec)"]
    time_diff1_mox1 = abs(min_time_mox1 - start_time)       # start → min
    time_diff2_mox1 = abs(max_time_mox1 - min_time_mox1)    # min → max

    # ── MOX3 ──
    min_time_mox3 = chunk.loc[chunk["MOX3(Ohms)"].idxmin(), "Time(sec)"]
    max_time_mox3 = chunk.loc[chunk["MOX3(Ohms)"].idxmax(), "Time(sec)"]
    time_diff1_mox3 = abs(min_time_mox3 - start_time)
    time_diff2_mox3 = abs(max_time_mox3 - min_time_mox3)

    # ── MOX4 ──
    min_time_mox4 = chunk.loc[chunk["MOX4(Ohms)"].idxmin(), "Time(sec)"]
    max_time_mox4 = chunk.loc[chunk["MOX4(Ohms)"].idxmax(), "Time(sec)"]
    time_diff1_mox4 = abs(min_time_mox4 - start_time)
    time_diff2_mox4 = abs(max_time_mox4 - min_time_mox4)

    # ── Detection rules (matches original if/elif/else exactly) ──
    if time_diff2_mox1 <= 39:
        if 1.5 <= time_diff1_mox1 <= 11:
            findings.append("MOX1: Gas detected — NO2 / SO2 / NH3")

        if 4 < time_diff1_mox1 < 7 and -450 < Min_slope1 < -200:
            findings.append("MOX1: NO2 detected — 5–7 ppm")

        elif 7 < time_diff1_mox1 < 12 and Min_slope1 < -450:
            findings.append("MOX1: NO2 detected — 10 ppm")

    elif time_diff2_mox3 < 30 and Min_slope3 < -500:
        findings.append("MOX3: CH4 detected")

        if -1000 < Min_slope3 < -500:
            findings.append("CH4 concentration: < 2000 ppm")
        elif -2500 < Min_slope3 < -1000:
            findings.append("CH4 concentration: 3000–5000 ppm")
        elif -3500 < Min_slope3 < -2500:
            findings.append("CH4 concentration: 5000–10000 ppm")
        elif Min_slope3 < -3500:
            findings.append("CH4 concentration: 10000–20000 ppm")

    # Return findings + debug metrics for the table
    metrics = {
        "min_time_mox1": round(min_time_mox1, 2),
        "max_time_mox1": round(max_time_mox1, 2),
        "time_diff1_mox1": round(time_diff1_mox1, 2),
        "time_diff2_mox1": round(time_diff2_mox1, 2),
        "Min_slope1": round(Min_slope1, 2),
        "min_time_mox3": round(min_time_mox3, 2),
        "max_time_mox3": round(max_time_mox3, 2),
        "time_diff1_mox3": round(time_diff1_mox3, 2),
        "time_diff2_mox3": round(time_diff2_mox3, 2),
        "Min_slope3": round(Min_slope3, 2),
        "min_time_mox4": round(min_time_mox4, 2),
        "max_time_mox4": round(max_time_mox4, 2),
        "time_diff1_mox4": round(time_diff1_mox4, 2),
        "time_diff2_mox4": round(time_diff2_mox4, 2),
        "Min_slope4": round(Min_slope4, 2),
    }
    return findings, metrics


# ─────────────────────────────────────────────
# CHUNKING  (matches original: chunk_size=90, step=91, skip first 7 rows per chunk)
# Bug fixed: original app used start+7 magic number instead of time-based filter;
#            original script also uses start+7 row skip — kept here to match exactly.
# ─────────────────────────────────────────────
def run_detection(df):
    results   = []
    chunk_meta = []   # (start_time, end_time, gas_detected)

    start_idx = 0
    chunk_num = 1

    while start_idx <= len(df) - CHUNK_SIZE:
        end_idx = start_idx + CHUNK_SIZE

        # Skip first 7 rows of chunk (matches original script)
        chunk = df.iloc[start_idx + 7: end_idx].copy()

        if chunk.empty or len(chunk) < 5:
            start_idx += CHUNK_SIZE + 1
            chunk_num += 1
            continue

        # Drop rows with NaN in key columns
        chunk = chunk.dropna(subset=MOX_COLS + ["Time(sec)"])
        if chunk.empty:
            start_idx += CHUNK_SIZE + 1
            chunk_num += 1
            continue

        findings, metrics = detect_gas_chunk(chunk)
        gas_found = len(findings) > 0

        result_label = "; ".join(findings) if findings else "No gas detected"

        results.append({
            "Chunk":            chunk_num,
            "Start Time (sec)": round(chunk["Time(sec)"].iloc[0], 2),
            "End Time (sec)":   round(chunk["Time(sec)"].iloc[-1], 2),
            "Detection Result": result_label,
            "Gas Detected":     gas_found,
            **metrics,
        })

        chunk_meta.append({
            "start": chunk["Time(sec)"].iloc[0],
            "end":   chunk["Time(sec)"].iloc[-1],
            "gas":   gas_found,
        })

        start_idx += CHUNK_SIZE + 1
        chunk_num += 1

    return pd.DataFrame(results), chunk_meta


# ─────────────────────────────────────────────
# PLOT BUILDERS
# ─────────────────────────────────────────────
def plot_raw(df, chunk_meta, selected_sensors, file_title=""):
    n = len(selected_sensors)
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=[f"{c} Raw" for c in selected_sensors])

    colors = ["steelblue", "seagreen", "mediumpurple"]

    for i, col in enumerate(selected_sensors, start=1):
        fig.add_trace(
            go.Scatter(x=df["Time(sec)"], y=df[col],
                       mode="lines", name=col,
                       line=dict(width=1.5, color=colors[(i - 1) % len(colors)])),
            row=1, col=i,
        )
        # Shade gas-detected regions
        for meta in chunk_meta:
            if meta["gas"]:
                fig.add_vrect(
                    x0=meta["start"], x1=meta["end"],
                    fillcolor="red", opacity=0.12,
                    layer="below", line_width=0,
                    row=1, col=i,
                )
        fig.update_xaxes(title_text="Time (sec)", row=1, col=i)
        fig.update_yaxes(title_text="Resistance (Ohms)", exponentformat="none", row=1, col=i)

    fig.update_layout(
        title_text=f"Raw Signal — {file_title}" if file_title else "Raw Signal",
        height=420, showlegend=False,
    )
    return fig


def plot_slope(df, chunk_meta, selected_sensors, file_title=""):
    n = len(selected_sensors)
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=[f"{c} Slope" for c in selected_sensors])

    for i, col in enumerate(selected_sensors, start=1):
        time_vals = df["Time(sec)"].values
        raw_vals  = df[col].values
        valid     = ~np.isnan(raw_vals)
        t = time_vals[valid]
        r = raw_vals[valid]

        if len(t) >= 2:
            # np.diff matches original script (not np.gradient)
            slope = np.diff(r) / np.diff(t)
            t_mid = (t[:-1] + t[1:]) / 2

            fig.add_trace(
                go.Scatter(x=t_mid, y=slope,
                           mode="lines", name=f"Slope {col}",
                           line=dict(width=1.5, color="darkorange")),
                row=1, col=i,
            )

        for meta in chunk_meta:
            if meta["gas"]:
                fig.add_vrect(
                    x0=meta["start"], x1=meta["end"],
                    fillcolor="red", opacity=0.12,
                    layer="below", line_width=0,
                    row=1, col=i,
                )
        fig.update_xaxes(title_text="Time (sec)", row=1, col=i)
        fig.update_yaxes(title_text="dR/dt", exponentformat="none", row=1, col=i)

    fig.update_layout(
        title_text=f"Slope — {file_title}" if file_title else "Slope",
        height=420, showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Sensor CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, skiprows=2)
    file_title = uploaded_file.name.replace(".csv", "")

    # Drop unused columns (ignore errors if some don't exist in this file)
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])

    # Force numeric — done ONCE
    for col in MOX_COLS + ["Time(sec)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Time(sec)"]).reset_index(drop=True)

    st.success(f"File loaded: **{uploaded_file.name}** — {len(df)} rows")

    # ── SIDEBAR ──
    st.sidebar.header("Dashboard Controls")
    selected_sensors = st.sidebar.multiselect(
        "Select Sensors", MOX_COLS, default=MOX_COLS
    )
    plot_type = st.sidebar.radio(
        "Select Plot Type", ["Raw Only", "Slope Only", "Both"]
    )
    show_metrics = st.sidebar.checkbox("Show Detection Metrics Table")
    show_data    = st.sidebar.checkbox("Show Raw Data Preview")

    if not selected_sensors:
        st.warning("Please select at least one sensor.")
        st.stop()

    # ── RUN DETECTION ──
    result_df, chunk_meta = run_detection(df)
    any_gas = result_df["Gas Detected"].any() if not result_df.empty else False

    # ── ALERT ──
    st.subheader("Gas Detection Status")
    if any_gas:
        gases = result_df[result_df["Gas Detected"]]["Detection Result"].unique().tolist()
        st.error(f" GAS DETECTED — {'; '.join(gases)}")
    else:
        st.success(" No gas detected")

    # ── PLOTS ──
    st.subheader("Sensor Visualization")
    if plot_type in ["Raw Only", "Both"]:
        st.plotly_chart(
            plot_raw(df, chunk_meta, selected_sensors, file_title),
            use_container_width=True,
        )
    if plot_type in ["Slope Only", "Both"]:
        st.plotly_chart(
            plot_slope(df, chunk_meta, selected_sensors, file_title),
            use_container_width=True,
        )

    # ── RESULTS TABLE ──
    st.subheader("Detection Results by Chunk")
    display_cols = ["Chunk", "Start Time (sec)", "End Time (sec)", "Detection Result", "Gas Detected"]
    def highlight_gas(row):
        color = "background-color: #ffcccc" if row["Gas Detected"] else ""
        return [color] * len(row)

    st.dataframe(
        result_df[display_cols].style.apply(highlight_gas, axis=1),
        use_container_width=True,
    )

    # ── METRICS TABLE (optional) ──
    if show_metrics:
        st.subheader("Detection Metrics per Chunk")
        metric_cols = [
            "Chunk", "Start Time (sec)", "End Time (sec)",
            "time_diff1_mox1", "time_diff2_mox1", "Min_slope1",
            "time_diff1_mox3", "time_diff2_mox3", "Min_slope3",
            "time_diff1_mox4", "time_diff2_mox4", "Min_slope4",
        ]
        st.dataframe(result_df[metric_cols], use_container_width=True)

    # ── DOWNLOAD ──
    st.download_button(
        " Download Detection Report",
        result_df[display_cols].to_csv(index=False),
        file_name="gas_detection_report.csv",
        mime="text/csv",
    )

    # ── DATA PREVIEW ──
    if show_data:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
