# 📩 Email / SMS Spam Detection System

A **Machine Learning-based web application** that predicts whether a given message is **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques.  
This system uses a trained **TF-IDF Vectorizer** and **Multinomial Naive Bayes Classifier**, and is deployed using **Streamlit Cloud** for an interactive user experience.

🔗 **Live App:** [SMS Spam Detection System](https://your-streamlit-app-link.streamlit.app/)

---

## 🔍 Overview

This project is designed to automatically classify SMS or email messages as *spam* or *not spam* using Machine Learning and NLP.  
It cleans and preprocesses text, converts it into numerical features using **TF-IDF**, and predicts the label using a trained **Naive Bayes model**.  

The dataset used is the **SMS Spam Collection Dataset**, which contains over **5,000 messages** labeled as either *ham* (non-spam) or *spam*.

---

## ⚙️ Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python |
| **Framework** | Streamlit |
| **Data Handling** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Natural Language Processing** | NLTK |
| **Model Storage** | Pickle |
| **Deployment** | Streamlit Cloud |

---

## 🧠 Model Details

| Component | Description |
|------------|-------------|
| **Algorithm** | Multinomial Naive Bayes |
| **Vectorizer** | TF-IDF (Term Frequency–Inverse Document Frequency) |
| **Dataset** | SMS Spam Collection Dataset |
| **Accuracy** | ~98% on test data |
| **Metrics Used** | Precision, Recall, F1 Score |

---

## 📊 Dataset Information

| Attribute | Description |
|------------|-------------|
| **label** | Indicates whether the message is spam or not |
| **message** | The actual text of the message |

**Dataset Source:** UCI Machine Learning Repository  
**Total Records:** ~5,500  

---

## 🧩 Project Workflow

1. **Data Preprocessing**  
   - Convert all text to lowercase  
   - Remove punctuation and stopwords  
   - Tokenize and stem words using **Porter Stemmer**

2. **Feature Extraction**  
   - Apply **TF-IDF Vectorization** to convert text into numerical form  

3. **Model Training**  
   - Train **Multinomial Naive Bayes** model on the processed data  

4. **Prediction**  
   - Input text → Preprocess → Vectorize → Predict → Output Spam/Not Spam  

---