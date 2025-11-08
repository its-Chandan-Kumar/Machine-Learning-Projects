# -------------------------------
# 📧 Email/SMS Spam Detection App
# -------------------------------

import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# -------------------------------
# NLTK Setup
# -------------------------------
# Download only if not already present
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path, exist_ok=True)

nltk.data.path.append(nltk_data_path)

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_path)
    nltk.download("punkt", download_dir=nltk_data_path)

# Initialize Porter Stemmer
ps = PorterStemmer()

# -------------------------------
# Text Preprocessing Function
# -------------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)


# -------------------------------
# Load Pre-trained Model & Vectorizer
# -------------------------------
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"❌ Required file not found: {e.filename}")
    st.stop()


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Ema_
