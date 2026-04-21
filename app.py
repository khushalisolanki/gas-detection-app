import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# PER-CHUNK PLOT  (matches original exactly)
# Layout: 2 rows × 3 cols
#   Row 0 → Raw Data:  MOX1, MOX3, MOX4
#   Row 1 → Slope:     MOX1, MOX3, MOX4
# Time filter: time >= t0 + 3  (same as original valid_idx)
# Slope: np.diff(raw) / np.diff(time)  (same as original)
# ─────────────────────────────────────────────
MOX_COLS_PLOT = ["MOX1(Ohms)", "MOX3(Ohms)", "MOX4(Ohms)"]

def build_chunk_figure(df, chunk_idx, file_title, gas_found):
    """
    Build one matplotlib figure (2×3) for a single chunk,
    exactly replicating the original script's output.
    """
    chunk_block_size = CHUNK_SIZE + 1
    start_idx = chunk_idx * chunk_block_size
    end_idx   = start_idx + CHUNK_SIZE

    if end_idx > len(df):
        return None

    # Time filter: keep rows where time >= first_time + 3  (original: valid_idx)
    time_chunk = df["Time(sec)"].iloc[start_idx:end_idx]
    if time_chunk.empty:
        return None

    valid_idx  = time_chunk >= (time_chunk.iloc[0] + 3)
    time_chunk = time_chunk[valid_idx]

    if time_chunk.empty:
        return None

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # Colour the figure border red if gas detected in this chunk
    border_color = "#d62728" if gas_found else "#2196F3"
    fig.patch.set_edgecolor(border_color)
    fig.patch.set_linewidth(3)

    for i, col in enumerate(MOX_COLS_PLOT):
        raw_values = df[col].iloc[start_idx:end_idx]
        raw_values = raw_values[valid_idx]

        # Slope: np.diff — matches original
        if len(raw_values) >= 2:
            slope      = np.diff(raw_values.values) / np.diff(time_chunk.values)
            time_slope = time_chunk.iloc[:-1]
        else:
            slope      = np.array([])
            time_slope = time_chunk.iloc[:-1]

        # ── Row 0: Raw Data ──
        axs[0, i].plot(time_chunk, raw_values, color="#1f77b4")
        axs[0, i].set_title(f"Raw Data: {col}")
        axs[0, i].set_xlabel("Time (sec)")
        axs[0, i].set_ylabel(col)
        axs[0, i].grid(True, alpha=0.3)

        # ── Row 1: Slope ──
        if len(slope) > 0:
            axs[1, i].plot(time_slope, slope, color="#ff7f0e")
        axs[1, i].set_title(f"Slope vs Time: {col}")
        axs[1, i].set_xlabel("Time (sec)")
        axs[1, i].set_ylabel(f"Slope of {col}")
        axs[1, i].grid(True, alpha=0.3)

    status = " GAS DETECTED" if gas_found else " No Gas"
    plt.suptitle(
        f"{file_title}  |  Chunk {chunk_idx + 1}  |  {status}",
        fontsize=12,
        color="red" if gas_found else "black",
        fontweight="bold" if gas_found else "normal",
    )
    plt.tight_layout()
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
    show_metrics = st.sidebar.checkbox("Show Detection Metrics Table")
    show_data    = st.sidebar.checkbox("Show Raw Data Preview")

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

    # ── PER-CHUNK PLOTS (2×3 grid per chunk, matching original script exactly) ──
    st.subheader("Sensor Visualization — Per Chunk")

    total_chunks = (len(df) + 1) // (CHUNK_SIZE + 1)

    for chunk_idx in range(total_chunks):
        # Look up whether this chunk had gas detected
        gas_found = (
            result_df.loc[result_df["Chunk"] == chunk_idx + 1, "Gas Detected"].values[0]
            if chunk_idx < len(result_df) else False
        )

        fig = build_chunk_figure(df, chunk_idx, file_title, gas_found)
        if fig is None:
            continue

        st.pyplot(fig)
        plt.close(fig)

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
