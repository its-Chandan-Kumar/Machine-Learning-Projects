# -------------------------------
# 📧 Email/SMS Spam Detection App
# -------------------------------

# Import required libraries
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer for text stemming
ps = PorterStemmer()

# -------------------------------
# Text Preprocessing Function
# -------------------------------
def transform_text(text):
    """
    Cleans and preprocesses raw text input.
    Steps:
      1. Lowercasing
      2. Tokenization
      3. Removing special characters
      4. Removing stopwords and punctuation
      5. Applying stemming
    """
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text into individual words
    text = nltk.word_tokenize(text)

    # Keep only alphanumeric tokens
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming to reduce words to their root form
    y = [ps.stem(i) for i in y]

    # Return cleaned text as a single string
    return " ".join(y)

# -------------------------------
# Load Pre-trained Model & Vectorizer
# -------------------------------
try:
    # Load TF-IDF Vectorizer
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

    # Load Trained ML Model
    model = pickle.load(open('model.pkl', 'rb'))

except FileNotFoundError as e:
    st.error(f"❌ Required file not found: {e.filename}")
    st.stop()


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Email/SMS Spam Detection App",
    page_icon="📩",
    layout="centered"
)

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title('📩 Email/SMS Spam Detection App')
st.markdown("This app predicts whether a given message is **Spam** or **Not Spam** using a trained Machine Learning model.")

# User input text box
input_sms = st.text_area(label='✉️ Enter your message here:')

# Prediction button
if st.button('Predict'):
    # Handle empty input
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize the cleaned text
        vector_input = tfidf.transform([transformed_sms])

        # 3. Make prediction
        result = model.predict(vector_input)[0]

        # 4. Display output
        if result == 1:
            st.error('🚫 SPAM Message Detected!')
        else:
            st.success('✅ This message is NOT SPAM!')


# --- SIDEBAR INFORMATION ---
st.sidebar.title("📊 Project Information")

st.sidebar.markdown("""
## 📩 SMS / Email Spam Detection System

This application predicts whether a given text message or email is **Spam** or **Not Spam** using **Natural Language Processing (NLP)** and **Machine Learning**.  
It uses a trained **TF-IDF Vectorizer** and **Multinomial Naive Bayes Model** to classify the messages.

---

### 📘 Dataset Information
- **Dataset Name:** SMS Spam Collection Dataset  
- **Source:** UCI Machine Learning Repository  
- **Total Messages:** 5,574  
- **Categories:**  
  - `ham` → Legitimate message  
  - `spam` → Unwanted or promotional message  

**Example Messages:**
- *"Hey, are we still meeting today?"* → **Not Spam**  
- *"Congratulations! You’ve won a free iPhone. Click here!"* → **Spam**

---

### 🧠 Model Details
- **Algorithm:** Multinomial Naive Bayes  
- **Vectorizer:** TF-IDF (Term Frequency–Inverse Document Frequency)  
- **Accuracy:** ~98%  
- **Precision:** ~97%  
- **Recall:** ~96%  
- **F1 Score:** ~96.5%  

This model was selected for its high performance on text classification tasks and its ability to handle large vocabulary efficiently.

---

### ⚙️ Preprocessing Pipeline
1. **Convert text to lowercase**  
2. **Tokenize the message**  
3. **Remove stopwords and punctuation**  
4. **Apply stemming using Porter Stemmer**  
5. **Vectorize using TF-IDF**

---

### 🧾 Features
The model uses the frequency and importance of words to determine if a message is spam.  
Some important words that often appear in spam messages are:
> *win, free, click, offer, congratulations, claim, urgent, prize*

---

### 🧰 Technologies Used
| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python |
| **Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **NLP Toolkit** | NLTK |
| **ML Library** | Scikit-learn |
| **Model Storage** | Pickle |
| **Deployment** | Streamlit Cloud |

---

### 📈 Project Objective
The main goal of this project is to demonstrate how **Machine Learning and NLP** can help in **automated message filtering** and **spam detection**, making email and SMS communication safer and more efficient.

---
""")


