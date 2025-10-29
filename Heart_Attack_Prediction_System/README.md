# 💓 Heart Attack Prediction System

This project is a **Machine Learning Web Application** that predicts the likelihood of a heart attack based on medical data provided by the user.  
It is built using **Python**, **Streamlit**, and a trained **K-Nearest Neighbors (KNN)** model.  

The system helps users and healthcare professionals quickly assess cardiovascular risk using simple health indicators.


## 🚀 Live Demo
👉 https://chandankumar-heart-attack-prediction-system.streamlit.app


## 📖 Overview
The **Heart Attack Prediction System** takes several health-related inputs — such as age, cholesterol level, resting blood pressure, heart rate, and other diagnostic parameters — and uses a machine learning model to predict whether a person is likely to have a heart attack.
This app is fully interactive and runs directly in the browser through Streamlit.

## 🧠 Machine Learning Model

- **Algorithm Used:** K-Nearest Neighbors (KNN)  
- **Model Files:**
  - `KNN_HAPS.pkl` — trained KNN classifier
  - `scaler.pkl` — `StandardScaler` used for feature normalization
  - `columns.pkl` — stores the expected column structure used during training
- **Libraries Used:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit

The model was trained using the **UCI Heart Disease Dataset** (`heart.csv`) and tuned for balanced performance between sensitivity and specificity.

---

## 🩺 Features

✅ Interactive Streamlit web interface  
✅ Real-time predictions with a trained KNN model  
✅ Automatically scales input data before prediction  
✅ Deployed seamlessly on Streamlit Cloud  
✅ Lightweight and responsive UI  

---

## 📊 Input Parameters

| Parameter | Description |
|------------|-------------|
| Age | Age of the person |
| Sex | 1 = Male, 0 = Female |
| Chest Pain Type | 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 3 = Asymptomatic |
| Resting Blood Pressure | Measured in mm Hg |
| Serum Cholesterol | Cholesterol level in mg/dl |
| Fasting Blood Sugar | 1 if FBS > 120 mg/dl else 0 |
| Resting ECG Results | 0 = Normal, 1 = ST-T abnormality, 2 = LVH |
| Maximum Heart Rate | Achieved during exercise |
| Exercise Induced Angina | 1 = Yes, 0 = No |
| Oldpeak | ST depression induced by exercise |
| Slope | 0–2, slope of peak exercise ST segment |
| CA | Number of major vessels (0–4) |
| Thalassemia | 1 = Normal, 2 = Fixed defect, 3 = Reversible defect |


<img width="574" height="828" alt="image" src="https://github.com/user-attachments/assets/4092f9fe-d3a7-4104-964e-b8d9033cacd3" />
<img width="574" height="828" alt="Heart Attack Prediction System ❤️" src="https://github.com/user-attachments/assets/76d379b1-83a4-437b-b061-f29f8dd4a9b0" />

