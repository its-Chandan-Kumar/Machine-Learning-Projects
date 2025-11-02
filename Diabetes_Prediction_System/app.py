# Import Libraries
# ==============================
import streamlit as st
import numpy as np
import joblib
# ==============================

# Load Model
# ==============================
try:
    model = joblib.load('model/lightgbm_model.pkl')
except:
    st.error("Model file not found.")
    st.stop()
# ==============================


# Sidebar Information
# ==============================
st.sidebar.title("📊 About This App")

st.sidebar.markdown("""
### 🔍 **Overview**
This app predicts whether a person is likely to have **diabetes** based on key medical details.  
It uses a trained **Machine Learning model** built on real patient data.

### 🧠 **Model Details**
- **Algorithm:** LightGBM Classifier  
- **Accuracy:** ~85% on test data  
- **Goal:** Classify patients as *Diabetic* or *Non-Diabetic*  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score  

LightGBM is a gradient boosting model known for its speed and efficiency on structured (tabular) data.

### 📂 **Dataset Information**
- **Source:** Pima Indians Diabetes Dataset (UCI Machine Learning Repository / Kaggle)  
- **Records:** 768 samples  
- **Features Used:**  
  1. Pregnancies  
  2. Glucose Level  
  3. Blood Pressure  
  4. Skin Thickness  
  5. Insulin  
  6. BMI  
  7. Diabetes Pedigree Function  
  8. Age  

Each record is labeled as:
- **1 → Diabetic**  
- **0 → Non-Diabetic**

### 💡 **Note**
This app is for educational and research purposes only.  
It should **not** be used for medical diagnosis.
""")
# ==============================

# Title
# ==============================
st.title("🩺 Diabetes Prediction App")
st.markdown('by Chandan Kumar')
st.header('Please provide the following details')
# ==============================

# Input Fields
# ==============================
Pregnancies = st.number_input('Pregnancies',0,20,2)
Glucose = st.slider('Glucose (mg/dL)',50,200,70)
BloodPressure = st.slider("Diastolic BP (mmHg)", 40, 130, 80)
SkinThickness = st.slider("Skin Thickness (mm)", 7, 99, 25)
Insulin = st.slider("Insulin (µU/ml)", 0, 900, 150)
BMI = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=67.0, value=25.0)
DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function",0.0,2.5,0.5,0.01)
Age = st.number_input("Age (years)", min_value=1, max_value=100, value=25)
# ==============================

# Prediction
# ==============================
if st.button('Predict'):
    # Convert to array
    features = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

    try:
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.error("The person is likely to have Diabetes 😔")
        else:
             st.success("The person is unlikely to have Diabetes 😊")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ==============================



