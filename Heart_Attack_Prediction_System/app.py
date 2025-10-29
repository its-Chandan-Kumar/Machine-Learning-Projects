# Import required libraries
import streamlit as st
import pandas as pd


import pickle

# Load the KNN model
with open('KNN_HAPS.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the expected columns
with open('columns.pkl', 'rb') as f:
    expected_columns = pickle.load(f)


# Making the UI of System
st.title("Heart Attack Prediction System ❤️")
st.markdown('by Chandan Kumar')
st.subheader('Please provide the following details')

age = st.slider('Age',18,100,25)
sex = st.selectbox('Gender',['M','F'])
chest_pain = st.selectbox('Chest Pain Type',['ATA','NAP','TA','ASY'])
resting_bp = st.number_input('Resting Blood Pressure (mm Hg)',60,200)
cholesterol = st.number_input('Cholesterol (mm/dl)',100,600,200)
fastingBS = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['N','Y'])
if fastingBS == 'N':
    fastingBS = 0
else: 
    fastingBS = 1
resting_ecg = st.selectbox('Resting ECG',['Normal','ST','LVH'])
maxHR = st.slider('Max Heart Rate',60,220,150)
exercise_angina = st.selectbox('Exercise-Induced Angina',['Y','N'])
oldpeak = st.slider('Oldpeak (ST Depression)', 0.0,6.0,1.0)
st_slope = st.selectbox('ST Slope', ['Up','Flat','Down'])


# When the 'Predict' button is clicked, collect all user inputs and make a prediction
if st.button('Predict'):
    # Create a dictionary with user inputs and one-hot encoded categorical values
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fastingBS,
        'MaxHR': maxHR,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Convert user inputs into a DataFrame for model prediction
    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns exist in input (add missing columns with 0)
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the model’s expected input format
    input_df = input_df[expected_columns]

    # Scale the input features using the previously saved StandardScaler
    scaled_input = scaler.transform(input_df)

    # Predict heart attack risk using the trained model
    prediction = model.predict(scaled_input)[0]

    # Display result based on prediction outcome
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Attack")
    else:
        st.success("✅ Low Risk of Heart Attack")
