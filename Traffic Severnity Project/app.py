# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import os

# ------------------- PATHS -------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "accident_model.pkl")

# ------------------- LOAD MODEL -------------------
pipe = joblib.load(MODEL_PATH)

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Accident Severity Prediction", page_icon="🚦", layout="wide")
st.title("🚦 Traffic Accident Severity Prediction App")
st.markdown("#### Provide accident details below to predict severity.")

# ------------------- USER INPUT -------------------
col1, col2 = st.columns(2)

with col1:
    weather = st.selectbox("🌦 Weather", ["Clear", "Overcast", "Rain", "Fog", "Snow", "Thunderstorm"])
    road_condition = st.selectbox("🛣 Road Condition", ["Dry", "Wet", "Icy", "Snowy", "Muddy"])
    road_type = st.selectbox("🛤 Road Type", ["Highway", "Main Road", "Residential", "Rural", "Service Road"])
    traffic_density = st.selectbox("🚗 Traffic Density", ["Low", "Medium", "High", "Very High", "Standstill"])
    road_light = st.selectbox("💡 Road Light Condition", ["Daylight", "Dark - Street Lights", "Dark - No Lights"])
    accident_type = st.selectbox("⚡ Accident Type", ["Rear-end", "Head-on", "Side-impact", "Single Vehicle", "Multi Vehicle"])

with col2:
    vehicle_type = st.selectbox("🚙 Vehicle Type", ["Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pedestrian"])
    number_of_vehicles = st.slider("🚘 Number of Vehicles", 1, 1000, 2)
    driver_age = st.slider("🧑 Driver Age", 16, 80, 30)
    driver_experience = st.slider("🎓 Driver Experience (years)", 0, 60, 5)
    driver_alcohol = st.selectbox("🍷 Driver under Alcohol influence?", ["No", "Yes"])
    speed_limit = st.slider("🚦 Speed Limit (km/h)", 20, 140, 60)
    time_of_day = st.selectbox("🕒 Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

# ------------------- PREPARE INPUT -------------------
input_data = pd.DataFrame([{
    "Weather": weather,
    "Road_Condition": road_condition,
    "Road_Type": road_type,
    "Traffic_Density": traffic_density,
    "Road_Light_Condition": road_light,
    "Vehicle_Type": vehicle_type,
    "Number_of_Vehicles": number_of_vehicles,
    "Driver_Age": driver_age,
    "Driver_Experience": driver_experience,
    "Driver_Alcohol": 1 if driver_alcohol == "Yes" else 0,
    "Speed_Limit": speed_limit,
    "Time_of_Day": time_of_day
}])

# ------------------- PREDICTION -------------------
if st.button("🔍 Predict Severity"):
    prediction = pipe.predict(input_data)[0]
    
    # Get probabilities if classifier supports it
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(input_data)[0]
        confidence = max(probs) * 100
    else:
        confidence = 100.0

    # Display result
    st.subheader("Prediction Result")
    st.info(f"⚡ Predicted Accident Type: **{prediction}** ({confidence:.2f}% confidence)")
