# Gas Sensor Intelligence Dashboard

An interactive data application for analyzing multi-sensor gas signals, detecting gas presence using chunk-based signal analysis, and visualizing raw and derived features (slope) in real time.

---

## Features

- Upload and analyze sensor datasets (CSV format)
- Multi-sensor support (MOX1, MOX3, MOX4)
- Interactive dashboard controls:
  - Sensor selection
  - Raw / Slope / Combined visualization
- Automatic preprocessing:
  - Numeric conversion
  - Removal of unstable initial signal region
- Chunk-based gas detection algorithm
- Slope-based signal analysis using numerical gradients
- Dynamic visualization with Plotly
- Downloadable detection report

---

## Problem

Gas sensor signals are noisy, time-dependent, and difficult to interpret manually.  
Detecting gas presence requires analyzing subtle signal variations across multiple sensors over time.

Manual inspection is:
- Time-consuming  
- Error-prone  
- Not scalable  

---

## Solution

This application automates gas detection by:

- Segmenting sensor data into fixed-size chunks  
- Applying a custom detection function (`detect_gas`)  
- Using slope (rate of change) as a key signal feature  
- Visualizing both raw signals and derived features  

The system enables fast, repeatable, and interpretable gas detection.

---

## How It Works

1. User uploads CSV file  
2. Data is cleaned and preprocessed:
   - Converts sensor values to numeric
   - Removes unstable initial region (Time < 3 sec)
3. Data is split into chunks (~90 rows each)
4. Each chunk is analyzed using a detection function
5. Results are aggregated and displayed
6. Signals and slopes are visualized interactively

---

## Architecture

Upload CSV  
→ Data Cleaning (Pandas)  
→ Chunk Segmentation  
→ Detection Logic (`detect_gas`)  
→ Feature Extraction (Slope via NumPy Gradient)  
→ Visualization (Plotly)  
→ Output (Dashboard + Report)

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Plotly  
- Streamlit  

---

## Key Techniques Used

- Time-series segmentation (chunk-based analysis)  
- Numerical differentiation (slope via `np.gradient`)  
- Multi-sensor signal comparison  
- Threshold-based detection logic

---

## How to Run

```bash
git clone https://github.com/your-username/gas-detection-dashboard.git
cd gas-detection-dashboard
pip install -r requirements.txt
streamlit run app.py
