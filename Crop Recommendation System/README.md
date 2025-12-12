# 🌾 Crop Recommendation System

This project is a Machine Learning Web Application that predicts the most suitable crop based on environmental and soil nutrient data provided by the user.
It is built using Python, Streamlit, and a trained Decision Tree Classifier model.

The system helps farmers and agricultural experts make informed decisions by recommending the ideal crop for a given soil condition. This app fully runs in the browser through Streamlit.

---

## 🚀 Live Demo
👉 https://crop-recommendation-system-7ta3.onrender.com/

---

## 📖 Overview

The Crop Recommendation System takes several soil-related inputs — such as Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, soil pH, and rainfall — and uses a machine learning model to suggest the best crop to grow.
This improves agricultural productivity and assists in data-driven decision-making.

---

## 🧠 Machine Learning Model

Algorithm Used: Decision Tree Classifier
Model Files:
crop_model.pkl — trained Decision Tree model
label_encoder.pkl — converts numerical predictions back to crop names

Libraries Used:
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib / Seaborn (optional)

The model was trained using the Crop Recommendation Dataset from Kaggle and fine-tuned for high accuracy in crop classification.

---

## 🩺 Features
✅ Interactive Streamlit web interface  
✅ Real-time crop prediction using ML model
✅ Encoded label decoding applied automatically
✅ Trained on real agricultural dataset
✅ Lightweight and responsive UI 

---

## 📊 Input Parameters
| Parameter | Description |
|------------|-------------|
| N	Nitrogen | content in soil |
| P	Phosphorus | content in soil |
| K	Potassium | content in soil |
| Temperature | Temperature in °C |
| Humidity | Relative humidity (%) |
| pH | Soil pH level |
| Rainfall |	Annual rainfall (mm) |

---

## 📸 Screenshot of the App
<img width="770" height="773" alt="image" src="https://github.com/user-attachments/assets/f3482ff5-c7b8-4248-83b7-eec7fdc7e960" />


