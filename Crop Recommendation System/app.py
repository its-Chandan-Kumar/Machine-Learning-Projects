import os
import pickle
import streamlit as st
import numpy as np

BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "crop_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

# ------------------------------
# SIDEBAR SECTION
# ------------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.subheader("🌾 Crop Recommendation System")
st.sidebar.write("""
This ML system recommends the most suitable crop based on soil nutrients 
(N, P, K), temperature, humidity, pH, and rainfall.
""")

st.sidebar.subheader("📊 Dataset Details")
st.sidebar.write("""
- Source: Kaggle Crop Recommendation Dataset  
- Features:
    - N (Nitrogen)  
    - P (Phosphorus)  
    - K (Potassium)  
    - Temperature  
    - Humidity  
    - pH Level  
    - Rainfall  
- Target: 22 Crop Categories
""")

st.sidebar.subheader("🤖 Model Details")
st.sidebar.write("""
- Preprocessing: Label Encoding  
- Algorithm: Decision Tree 
- Training Accuracy: 100%  
- Testing Accuracy: 97.95%  
""")

st.sidebar.subheader("👨‍💻 Developer")
st.sidebar.write("""
Developed by: Chandan Kumar 
""")


# ------------------------------
# MAIN UI SECTION
# ------------------------------
st.title("🌱 Crop Recommendation System")
st.write("Enter the environmental and soil conditions below:")

# User Inputs
N = st.number_input("Nitrogen (N)", 0, 200,value=30)
P = st.number_input("Phosphorus (P)", 0, 200,value=45)
K = st.number_input("Potassium (K)", 0, 200,value=72)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0,value=40.6)
humidity = st.number_input("Humidity (%)", 0.0, 100.0,value=30.7)
ph = st.number_input("Soil pH Value", 0.0, 14.0,value=5.9)
rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0,value=80.6)

# Prediction Button
if st.button("🔍 Recommend Crop"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    pred_num = model.predict(features)[0]
    crop_name = str.title(le.inverse_transform([pred_num])[0])

    st.success(f"🌾 Recommended Crop: **{crop_name}**")
