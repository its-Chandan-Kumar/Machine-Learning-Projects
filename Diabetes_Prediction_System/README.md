# 🩺 Diabetes Prediction System  

An interactive **Machine Learning web application** that predicts whether a person is likely to have **diabetes** based on key medical attributes such as glucose level, blood pressure, insulin, BMI, and age.  
The system uses a trained **LightGBM Classifier** model and is deployed using **Streamlit** for a simple, browser-based interface.  

---

## 🚀 Live Demo

👉 

---
## 📘 Overview  
Diabetes is one of the most common chronic diseases worldwide.  
Early prediction can help individuals take preventive measures and make informed health decisions.  

This project aims to:
- Analyze health parameters of a patient  
- Train a machine learning model to predict diabetes likelihood  
- Provide a user-friendly web interface for real-time predictions  

---

## ⚙️ Technologies Used  
| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, LightGBM, Imbalanced-learn |
| **Model Deployment** | Streamlit |
| **Model Storage** | Joblib |

---

## 📂 Dataset Information  
**Dataset Name:** Pima Indians Diabetes Database  
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
**Rows:** 768  
**Target Column:** `Outcome` (1 = Diabetic, 0 = Non-Diabetic)

### Features Used
| Feature | Description |
|----------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | Serum insulin (μU/ml) |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Family history of diabetes |
| Age | Age of the patient (years) |

---

## 🧠 Model Training Steps  
1. **Data Preprocessing**  
   - Replaced zero values in key features with statistical measures (mean/median).  
   - Removed outliers and invalid entries.  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized feature distributions and correlations.  
   - Created new categorical features for Glucose, Age, and BMI ranges.  

3. **Data Balancing**  
   - Used **SMOTE** (Synthetic Minority Over-sampling Technique) to balance diabetic and non-diabetic samples.  

4. **Model Selection and Training**  
   - Tried ensemble algorithms.  
   - **LightGBM Classifier** performed best with an accuracy of **83–85%**.  

5. **Model Evaluation**  
   - Evaluated using accuracy, precision, recall, F1-score, and confusion matrix.  
   - Saved the trained model using `joblib` for deployment.  

---

## 📊 Model Performance  
| Metric | Score |
|---------|--------|
| Accuracy | 83% |
| Precision | 82% |
| Recall | 84% |
| F1-Score | 83% |

---

## 💾 Model Saving  
The trained model was saved as:  
```python
joblib.dump(lgbm, 'lightgbm_model.pkl')
