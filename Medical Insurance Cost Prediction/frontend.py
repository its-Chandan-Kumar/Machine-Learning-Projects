import streamlit as st
import requests

API_URL = "http://13.201.28.130:8000/predict"

st.title("Medical Insurance Price Predictor")
st.markdown('Please provide your details below: ')

# Input Fields
age = st.number_input('Age', min_value=1, max_value=120, value=25)
sex = st.selectbox('Gender', options=['male', 'female'])
bmi = st.number_input("BMI", min_value=15.0, max_value=40.0,value=20.0)
children = st.select_slider('Children',options=list(range(0, 11)),value=2)
smoker = st.selectbox("Smoker",options=['yes','no'])
region = st.selectbox("Region",options=['southeast','southwest','northeast','northwest'])



if st.button("Predict Medical Insurance"):
    input_data = {
        'age':age,
        'sex':sex,
        'bmi':bmi,
        'children':children,
        'smoker':smoker,
        'region':region
    }

    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code==200:
            result = response.json()
            st.success(f"Your Predicted Medical Insurance Price: ₹{result['prediction']:.2f}")
        else:
            st.error(f'API Error: {response.status_code} - {response.text}')
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the FastAPI server. Make sure it's running on port 8000")
