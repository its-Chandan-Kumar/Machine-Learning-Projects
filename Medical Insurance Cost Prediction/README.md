# 🏥 Medical Insurance Cost Prediction (AWS Deployed)

## 🚀 Project Overview
This project is a **Machine Learning-powered web application** that predicts medical insurance costs based on user inputs such as age, sex, BMI, children, smoking habits, and region.

The application is built using:
- ⚡ FastAPI (Backend API)
- 🎨 Streamlit (Frontend UI)
- 🤖 Machine Learning Model (XgBoost)
- ☁️ AWS EC2 (Deployment) 

---

## 🎥 Demo Video
👉 
https://github.com/user-attachments/assets/a7f21da2-63f0-4746-a2dd-3ff0f78464b8

---

## 🧠 Features
- Predicts insurance cost in real-time
- User-friendly Streamlit interface
- API using FastAPI
- Deployed on AWS EC2
- Handles real-world user inputs

---

## 🛠️ Tech Stack

| Category        | Technology |
|----------------|-----------|
| Language       | Python 🐍 |
| Frontend       | Streamlit 🎨 |
| Backend        | FastAPI ⚡ |
| ML Model       | Scikit-learn 🤖 |
| Deployment     | AWS EC2 ☁️ |

---

## 📊 Input Features

- Age  
- Gender  
- BMI  
- Number of Children  
- Smoking Status  
- Region  

---

## 📦 Project Structure
Medical Insurance Cost Prediction/
- |── Medical Insurance Price Predictor.ipynb  # Jupyter Notebook
- │── app.py  # FastAPI backend
- │── frontend.py  # Streamlit frontend
- │── model.pkl  # Trained ML model
- │── requirements.txt  # Dependencies
- │── insurance.csv  # Dataset

---

## ☁️ AWS Deployment

This project is deployed on an AWS EC2 instance, where:
- FastAPI runs on port 8000
- Streamlit runs on port 8501
- Public IP is used to connect frontend & backend
