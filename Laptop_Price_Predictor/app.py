import streamlit as st
import numpy as np
import pickle
import os

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

st.title("💻 Laptop Price Predictor")

# -------------------------------------------------
# File paths (FIXED)
# -------------------------------------------------
PIPE_PATH = "pipe.pkl"   # trained ML pipeline
DF_PATH = "df.pkl"       # dataframe used for dropdowns

# -------------------------------------------------
# Check files exist
# -------------------------------------------------
if not os.path.exists(PIPE_PATH) or not os.path.exists(DF_PATH):
    st.error("❌ Model files not found. Make sure `pipe.pkl` and `df.pkl` are in the same folder as app.py")
    st.stop()

# -------------------------------------------------
# Load model and dataframe
# -------------------------------------------------
with open(PIPE_PATH, "rb") as f:
    pipe = pickle.load(f)

with open(DF_PATH, "rb") as f:
    df = pickle.load(f)

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
company = st.selectbox("Brand", df["Company"].unique())
type_ = st.selectbox("Type", df["TypeName"].unique())
ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)

touchscreen = st.selectbox("Touchscreen", ["Yes", "No"])
ips = st.selectbox("IPS Display", ["Yes", "No"])

screen_size = st.selectbox(
    "Screen Size (inches)",
    [11.6, 12.0, 13.3, 13.6, 14.0, 15.6, 16.0, 16.1, 17.0, 17.3]
)

resolution = st.selectbox(
    "Screen Resolution",
    [
        "1920x1080", "1366x768", "1600x900", "3840x2160",
        "3200x1800", "2880x1800", "2560x1600",
        "2560x1440", "2304x1440"
    ]
)

cpu = st.selectbox("CPU Brand", df["CPU brand"].unique())
hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024])
gpu = st.selectbox("GPU Brand", df["Gpu brand"].unique())
os_ = st.selectbox("Operating System", df["os"].unique())

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Price 💰"):
    try:
        touchscreen_val = 1 if touchscreen == "Yes" else 0
        ips_val = 1 if ips == "Yes" else 0

        x_res, y_res = map(int, resolution.split("x"))
        ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size

        query = np.array([
            company,
            type_,
            ram,
            weight,
            touchscreen_val,
            ips_val,
            ppi,
            cpu,
            hdd,
            ssd,
            gpu,
            os_
        ], dtype=object).reshape(1, -1)

        # ---- CORRECT prediction ----
        prediction = pipe.predict(query)[0]
        predicted_price = int(np.exp(prediction))

        st.success(f"💰 Estimated Laptop Price: ₹{predicted_price:,}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
