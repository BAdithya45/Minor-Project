import streamlit as st
import requests

st.title("Cooling Tower Efficiency Predictor")

temperature = st.slider("Temperature", 10.0, 50.0, 30.0)
flow_rate = st.slider("Flow Rate", 50, 200, 120)
humidity = st.slider("Humidity", 0, 100, 50)

if st.button("Predict"):
    data = {
        "temperature": temperature,
        "flow_rate": flow_rate,
        "humidity": humidity
    }
    response = requests.post("http://127.0.0.1:5000/predict", json=data)
    prediction = response.json()["predicted_efficiency"]
    st.success(f"Predicted Efficiency: {prediction:.2f}")
