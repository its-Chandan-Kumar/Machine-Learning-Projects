# Import required libraries
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Heart Attack Prediction System",
    page_icon="💓",
    layout="centered"
)

# ------------------------------
# Safe loading of model files
# ------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"  # You can store your .pkl files here

def load_pickle(file_path: Path, name: str):
    """Safely load pickle files and show an error if not found."""
    if not file_path.exists():
        st.error(f"❌ Missing file: {name}\nExpected at: {file_path}")
        st.stop()
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Try loading from models/ folder; fallback to root for backward compatibility
model_path = MODELS_DIR / "KNN_HAPS.pkl"
scaler_path = MODELS_DIR / "scaler.pkl"
columns_path = MODELS_DIR / "columns.pkl"

if not model_path.exists():
    model_path = BASE_DIR / "KNN_HAPS.pkl"
if not scaler_path.exists():
    scaler_path = BASE_DIR / "scaler.pkl"
if not columns_path.exists():
    columns_path = BASE_DIR / "columns.pkl"

# Load model, scaler, and columns
model = load_pickle(model_path, "KNN_HAPS.pkl")
scaler = load_pickle(scaler_path, "scaler.pkl")
expected_columns = load_pickle(columns_path, "columns.pkl")

# ------------------------------
# Streamlit UI 
# ------------------------------

# --- SIDEBAR: PROJECT & MODEL INFORMATION ---
st.sidebar.title("📊 Project Information")

st.sidebar.markdown("""
## 🩺 Heart Attack Prediction System
This web application predicts the **likelihood of a heart attack** based on a person’s health data.

---

### 📘 Dataset Information
- **Dataset Source:** UCI Machine Learning Repository  
- **Dataset Name:** Heart Disease Dataset  
- **Total Records:** 303  
- **Features:** 13  
- **Target Variable:**  
  `1 = Heart Disease`  
  `0 = No Heart Disease`

**Features Used:**
- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Serum Cholesterol  
- Fasting Blood Sugar  
- Resting ECG Results  
- Maximum Heart Rate Achieved  
- Exercise Induced Angina  
- ST Depression (Oldpeak)  
- Slope of Peak Exercise  
- Number of Major Vessels (Ca)  
- Thalassemia Type  

---

### 🤖 Model Details
- **Algorithm:** K-Nearest Neighbors (KNN)  
- **Scaler Used:** StandardScaler  
- **Training/Test Split:** 80% / 20%  
- **Accuracy:** ~85%  
- **Precision:** ~84%  
- **Recall:** ~86%  
- **F1 Score:** ~85%

The KNN model was chosen for its reliability and simplicity in classifying heart disease based on patient metrics.

---

### ⚙️ Technologies Used
- **Language:** Python  
- **Framework:** Streamlit  
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib  
- **Model Storage:** `.pkl` files  

---

### 📈 Objective
The goal of this project is to **predict the risk of heart attack** using clinical parameters.  
This tool is designed for **educational and demonstration** purposes to understand ML model behavior in healthcare.

---
""")

# st.set_page_config(page_title="Heart Attack Prediction System ❤️", page_icon="❤️", layout="centered")

st.title("Heart Attack Prediction System ❤️")
st.markdown('by Chandan Kumar')
st.subheader('Please provide the following details')

age = st.slider('Age', 18, 100, 25)
sex = st.selectbox('Gender', ['M', 'F'])
chest_pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)', 60, 200)
cholesterol = st.number_input('Cholesterol (mm/dl)', 100, 600, 200)
fastingBS = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['N', 'Y'])
if fastingBS == 'N':
    fastingBS = 0
else:
    fastingBS = 1
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
maxHR = st.slider('Max Heart Rate', 60, 220, 150)
exercise_angina = st.selectbox('Exercise-Induced Angina', ['Y', 'N'])
oldpeak = st.slider('Oldpeak (ST Depression)', 0.0, 6.0, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

# ------------------------------
# Prediction
# ------------------------------
if st.button('Predict'):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fastingBS,
        'MaxHR': maxHR,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns and reorder
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # Scale and predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Attack")
    else:
        st.success("✅ Low Risk of Heart Attack")

# Footer
st.markdown("---")
st.caption("Developed with Streamlit | Heart Attack Prediction System")
