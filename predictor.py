# skyline_app.py – Streamlit UI for SkyLine‑Delay Predictor
# -----------------------------------------------------------
# * Searches recursively for all nine CSV files (pattern `*-*2024.csv`)
#   anywhere inside the project directory, so you can keep them either
#   in `data/` **or** the repo root.
# * Builds a Linear Regression model to predict **dep_delay** (minutes).
#
# Run:
#   pip install streamlit pandas scikit-learn
#   streamlit run skyline_app.py

import glob
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

###############################################################################
# 1. CONSTANTS
###############################################################################
# Accept both root‑level CSVs *and* files inside ./data/
CSV_GLOB_PATTERN = "**/*-*2024.csv"  # recursive search

COLS = [
    "carrier", "date", "flight_num", "tail_num", "dest", "crs_dep_time",
    "dep_time", "dep_delay", "carrier_delay", "weather_delay", "nas_delay",
    "security_delay", "late_aircraft_delay",
]

###############################################################################
# 2. DATA LOADING
###############################################################################
@st.cache_data(show_spinner=True)
def load_and_merge() -> pd.DataFrame:
    """Load every `*-*2024.csv` found under the current working directory."""
    frames = []
    for fp in glob.glob(CSV_GLOB_PATTERN, recursive=True):
        # Skip non‑CSV that match pattern accidentally
        if not fp.lower().endswith(".csv"):
            continue
        df = pd.read_csv(fp, header=None, names=COLS, dtype=str)
        airport_code = Path(fp).stem.split("-")[1][:3]
        df["airport"] = airport_code
        frames.append(df)

    if not frames:
        st.error(
            "No CSV files matching pattern `*-*2024.csv` were found. "
            "Ensure the nine datasets (e.g., `AA-EWR2024.csv`) are either in the "
            "project root or a `data/` sub‑directory.")
        st.stop()

    data = pd.concat(frames, ignore_index=True)

    # Parse and clean
    data["date"] = pd.to_datetime(data["date"], format="%m/%d/%Y")
    num_cols = ["dep_delay", "carrier_delay", "weather_delay", "nas_delay",
                "security_delay", "late_aircraft_delay"]
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors="coerce")
    data["day_of_year"] = data["date"].dt.dayofyear

    return data.dropna(subset=["dep_delay"])


data = load_and_merge()

###############################################################################
# 3. MODEL TRAINING (cached)
###############################################################################
@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    X_cat = pd.get_dummies(df[["carrier", "airport"]], prefix=["airline", "airport"], drop_first=False)
    X_num = df[["day_of_year", "carrier_delay", "weather_delay", "nas_delay",
                "security_delay", "late_aircraft_delay"]]
    X = pd.concat([X_cat, X_num], axis=1)
    y = df["dep_delay"]

    model = LinearRegression().fit(X, y)
    return model, X.columns.tolist()


model, FEATURE_ORDER = train_model(data)

###############################################################################
# 4. Helper – typical component delays
###############################################################################

def typical_components(df: pd.DataFrame, carrier: str, airport: str):
    subset = df[(df.carrier == carrier) & (df.airport == airport)]
    if subset.empty:
        subset = df[df.carrier == carrier]
    if subset.empty:
        subset = df.copy()
    return subset[["carrier_delay", "weather_delay", "nas_delay",
                   "security_delay", "late_aircraft_delay"]].mean()

###############################################################################
# 5. STREAMLIT UI
###############################################################################

st.title("✈️ SkyLine‑Delay Predictor")

left, right = st.columns(2)
with left:
    airline = st.selectbox("Airline", sorted(data.carrier.unique()))
with right:
    airport = st.selectbox("Airport", sorted(data.airport.unique()))

date_choice = st.date_input(
    "Flight Date", dt.date.today(),
    min_value=data.date.min().date(),
    max_value=data.date.max().date())

defaults = typical_components(data, airline, airport)

with st.expander("Optional: override component delays"):
    carrier_d  = st.number_input("Carrier delay (min)",  value=float(round(defaults[0], 1)))
    weather_d  = st.number_input("Weather delay (min)", value=float(round(defaults[1], 1)))
    nas_d      = st.number_input("NAS delay (min)",     value=float(round(defaults[2], 1)))
    security_d = st.number_input("Security delay (min)",value=float(round(defaults[3], 1)))
    late_d     = st.number_input("Late‑aircraft delay (min)", value=float(round(defaults[4], 1)))

if st.button("Predict Delay"):
    feat = {col: 0 for col in FEATURE_ORDER}
    feat[f"airline_{airline}"] = 1
    feat[f"airport_{airport}"] = 1
    feat["day_of_year"] = pd.Timestamp(date_choice).dayofyear
    feat.update({
        "carrier_delay": carrier_d,
        "weather_delay": weather_d,
        "nas_delay": nas_d,
        "security_delay": security_d,
        "late_aircraft_delay": late_d,
    })

    X_pred = pd.DataFrame([feat])[FEATURE_ORDER]
    delay_pred = model.predict(X_pred)[0]

    st.metric("Predicted Departure Delay", f"{delay_pred:.1f} minutes")
    st.caption("Linear regression trained on 2024 NYC flight data.")
