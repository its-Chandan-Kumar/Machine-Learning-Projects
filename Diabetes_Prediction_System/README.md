# 🩺 Diabetes Prediction System  

A **Machine Learning-based web application** that predicts the likelihood of **Diabetes** based on key medical parameters such as glucose level, blood pressure, insulin, BMI, and age.  
The system uses a trained **LightGBM Classifier** and is deployed using **Streamlit Cloud** for an interactive, browser-based experience.  

🔗 **Live App:** https://chandankumar-diabetes-prediction-system.streamlit.app/ 

---

## 📘 Overview  

This project is designed to assist in the early detection of diabetes using patient health data.  
It processes user inputs, runs them through a trained model, and instantly displays whether the person is **likely diabetic or not**.  

The model is built using the **Pima Indians Diabetes Dataset**, a widely used dataset for medical prediction tasks.  

---

## ⚙️ Technologies Used  
| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, LightGBM, Imbalanced-learn |
| **Model Deployment** | Streamlit |
| **Model Storage** | Joblib |

---

## 🧠 Model Details  
- **Algorithm:** LightGBM Classifier  
- **Accuracy:** ~85%  
- **Model File:** `lightgbm_model.pkl`  
- **Balanced Using:** SMOTE (Synthetic Minority Over-sampling Technique)  

---

## 📊 Dataset Information  
**Dataset:** Pima Indians Diabetes Database  
**Source:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  
**Target Variable:** `Outcome` → (1 = Diabetic, 0 = Non-Diabetic)

### Features Used:
| Feature | Description |
|----------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skinfold thickness |
| Insulin | Serum insulin (μU/ml) |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Family history function |
| Age | Age of the patient |

---

## 🧩 Project Workflow  

### 1. **Data Preprocessing**
- Replaced invalid zero values with mean/median values.  
- Removed outliers and unrealistic entries.  

### 2. **Feature Engineering**
- Created additional categorical features for BMI, glucose, and age (for analysis).  

### 3. **Data Balancing**
- Used **SMOTE** to handle class imbalance between diabetic and non-diabetic samples.  

### 4. **Model Training**
- Trained multiple ensemble models.  
- **LightGBM** achieved the best accuracy (~85%).  

### 5. **Evaluation**
- Measured performance using accuracy, recall, precision, and F1-score.  

### 6. **Deployment**
- Model exported using `joblib`.  
- Deployed with **Streamlit Cloud** for real-time predictions.

---

## 💾 Model Saving
```python
import joblib
joblib.dump(lgbm, 'lightgbm_model.pkl')
