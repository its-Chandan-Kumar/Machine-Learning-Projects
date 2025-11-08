import os
import streamlit as st
import pickle
import numpy as np

if not os.path.exists('pipe.pkl') or not os.path.exists('df.pkl'):
    st.error("Model files not found. Please ensure 'pipe.pkl' and 'df.pkl' are in the same directory.")
    st.stop()

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))


st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand',df['Company'].unique())

# Type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# RAM
ram = st.selectbox("RAM(in GB)",[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight')

# Touvhscreen
touchscreen = st.selectbox('Touch Screen',['Yes','No'])

# IPS
ips = st.selectbox("IPS",['Yes','No'])

# Screen size
screen_size = st.selectbox('Screen Size',[11.6,12.0,13.3,13.6,14.0,15.6,16.0,16.1,17.0,17.3])

# Resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# CPU
cpu = st.selectbox('CPU',df['CPU brand'].unique())

# Storage
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
ssd = st.selectbox('SSD(in GB)',[0,128,256,512,1024])

# GPU
gpu = st.selectbox('GPU',df['Gpu brand'].unique())

# OS 
os = st.selectbox("OS",df['os'].unique())

# Submit Button
if st.button('Predict Price'):
    # Query Point
    ppi=None

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
        
    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5/screen_size 

    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os,])

    query = query.reshape(1,12)
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.title(f'Predicted price of this configuration is: {predicted_price}')

    
