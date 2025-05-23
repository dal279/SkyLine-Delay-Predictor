# skyline_app.py – Streamlit UI for SkyLine‑Delay Predictor
# -----------------------------------------------------------
# Predict departure delays for AA, B6, DL flights from NYC airports.
#
# * Recursively ingests all nine CSVs matching `*-*2024.csv` anywhere in the
#   repo.
# * Builds a Linear Regression on carrier/airport one‑hots, day_of_year,
#   and five delay‑component columns.
# * Allows selecting any flight date (past or future) — we removed the upper
#   bound so you can forecast beyond the training year.
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
CSV_GLOB_PATTERN = "**/*-*2024.csv"  # search everywhere

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
    frames = []
    for fp in glob.glob(CSV_GLOB_PATTERN, recursive=True):
        if not fp.lower().endswith(".csv"):
            continue
        df = pd.read_csv(fp, header=None, names=COLS, dtype=str)
        df["airport"] = Path(fp).stem.split("-")[1][:3]
        frames.append(df)

    if not frames:
        st.error("No CSV files like `AA-EWR2024.csv` found in project.")
        st.stop()

    data = pd.concat(frames, ignore_index=True)

    # Clean & parse
    data["date"] = pd.to_datetime(data["date"], format="%m/%d/%Y", errors="coerce")
    data = data.dropna(subset=["date"])

    num_cols = ["dep_delay", "carrier_delay", "weather_delay", "nas_delay",
                "security_delay", "late_aircraft_delay"]
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["dep_delay"])

    data["day_of_year"] = data["date"].dt.dayofyear
    return data


data = load_and_merge()

###############################################################################
# 3. MODEL TRAINING
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

# Allow any date the user wants (no upper bound)
min_dt = data.date.min().date()

date_choice = st.date_input("Flight Date", value=dt.date.today(), min_value=min_dt)

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
    st.caption("Linear regression trained on 2024 data; future dates use historical patterns via day‑of‑year feature.")
