import streamlit as st
import requests

# 🔥 Your EC2 FastAPI URL
API_URL = "http://13.201.28.130:8000/predict"

st.title("Medical Insurance Price Predictor")
st.markdown("Please provide your details below:")

# ✅ Input Fields
age = st.number_input('Age', min_value=1, max_value=120, value=25)
sex = st.selectbox('Gender', options=['male', 'female'])
bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=20.0)
children = st.select_slider('Children', options=list(range(0, 11)), value=2)
smoker = st.selectbox("Smoker", options=['yes', 'no'])
region = st.selectbox("Region", options=['southeast', 'southwest', 'northeast', 'northwest'])

# ✅ Predict Button
if st.button("Predict Medical Insurance"):

    # Prepare data
    input_data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }

    try:
        # Send request to FastAPI
        response = requests.post(API_URL, json=input_data)

        # Debug info (VERY IMPORTANT)
        st.write("Status Code:", response.status_code)

        # If success
        if response.status_code == 200:
            result = response.json()

            # 🔥 Show full response (for debugging)
            st.write("API Response:", result)

            # ✅ Handle different possible keys safely
            if 'prediction' in result:
                prediction = result['prediction']
            elif 'Predicted Insurance Cost' in result:
                prediction = result['Predicted Insurance Cost']
            else:
                st.error("Unexpected response format from API")
                st.stop()

            # Show result
            st.success(f"Your Predicted Medical Insurance Price: ₹{float(prediction):.2f}")

        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to FastAPI server. Make sure it's running.")

    except Exception as e:
        st.error(f"Unexpected Error: {e}")
