import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

# --- File paths ---
pipe_path = os.path.join("model\df.pkl")
df_path = os.path.join("model\df.pkl")

# --- Check if model files exist ---
if not os.path.exists(pipe_path) or not os.path.exists(df_path):
    st.error("Model files not found. Please make sure 'pipe.pkl' and 'df.pkl' are in the same folder as this script.")
    # st.stop()

# --- Load model and dataframe ---
with open(pipe_path, "rb") as f:
    pipe = pickle.load(f)

with open(df_path, "rb") as f:
    df = pickle.load(f)

st.title("💻 Laptop Price Predictor")

# --- User inputs ---
company = st.selectbox("Brand", df["Company"].unique())
type_ = st.selectbox("Type", df["TypeName"].unique())
ram = st.selectbox("RAM (in GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.selectbox("Touch Screen", ["Yes", "No"])
ips = st.selectbox("IPS Display", ["Yes", "No"])
screen_size = st.selectbox("Screen Size (inches)", [11.6, 12.0, 13.3, 13.6, 14.0, 15.6, 16.0, 16.1, 17.0, 17.3])
resolution = st.selectbox(
    "Screen Resolution",
    ["1920x1080", "1366x768", "1600x900", "3840x2160", "3200x1800",
     "2880x1800", "2560x1600", "2560x1440", "2304x1440"]
)
cpu = st.selectbox("CPU", df["CPU brand"].unique())
hdd = st.selectbox("HDD (in GB)", [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox("SSD (in GB)", [0, 128, 256, 512, 1024])
gpu = st.selectbox("GPU", df["Gpu brand"].unique())
os_ = st.selectbox("Operating System", df["os"].unique())

# --- Prediction ---
if st.button("Predict Price"):
    touchscreen_val = 1 if touchscreen == "Yes" else 0
    ips_val = 1 if ips == "Yes" else 0

    x_res, y_res = map(int, resolution.split("x"))
    ppi = ((x_res**2 + y_res**2) ** 0.5) / screen_size

    query = np.array([company, type_, ram, weight, touchscreen_val, ips_val,
                      ppi, cpu, hdd, ssd, gpu, os_], dtype=object).reshape(1, -1)

    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.success(f"💰 Estimated Price: ₹{predicted_price:,}")

