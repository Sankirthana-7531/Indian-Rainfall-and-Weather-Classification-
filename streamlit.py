import streamlit as st
import numpy as np
import joblib
st.title("Indian Rainfall Prediction")
st.write("Weather and Rainfall Classification Model")
st.success("Streamlit App is Working")

model = joblib.load("rainfall_model.pkl")
scalar = joblib.load("scalar.pkl")
st.title("Indian Rainfall Prediction")
st.subheader("Weather & Rainfall Classification")
st.write("Enter the weather condition below to predict rainfall.")
with st.form("Weather_form"):
 temperature_avg = st.number_input("Average Temperature", value=25.0)
 temperature_min = st.number_input("Minimum Temperature", value=20.0)
 temperature_max = st.number_input("Maximum Temperature", value=30.0)
 wind_speed = st.number_input("Wind Speed",value=5.0)
 air_pressure = st.number_input("Air pressure",value=1010.0)
 latitude =st.number_input("Latitude", value=17.4)
 longitude = st.number_input("Longitude", value=78.5)
 elevation =st.number_input("Elevation", value=500)
 month = st.selectbox("Month",list(range(1,12)))
 submit = st.form_submit_button("Predict Rainfall")

if submit:
    input_data = np.array([[
        temperature_max,
        temperature_min,
        air_pressure,
        elevation,
        month
    ]])
    input_scaled = scalar.transform(input_data)
    prediction = model.predict(input_scaled)
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("Rainfall Expected")
    else:
        st.info("No Rainfall Expected")

st.sidebar.write("Indaian Rainfall & Weather Prediction")
st.sidebar.write("Machine Learning Classification Project")