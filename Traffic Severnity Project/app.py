# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# ------------------- LOAD TRAINED MODEL -------------------
MODEL_PATH = "models/accident_model.pkl"

if os.path.exists(MODEL_PATH):
    pipe = joblib.load(MODEL_PATH)
else:
    st.error("❌ Model file not found! Please check the path and upload `models/accident_model.pkl`.")
    st.stop()

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Accident Severity Prediction", page_icon="🚦", layout="wide")

# ------------------- APP TITLE -------------------
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
    number_of_vehicles = st.slider("🚘 Number of Vehicles", 1, 1000, 2)   # Increased max to 1000
    driver_age = st.slider("🧑 Driver Age", 16, 80, 30)
    driver_experience = st.slider("🎓 Driver Experience (years)", 0, 60, 5)
    driver_alcohol = st.selectbox("🍷 Driver under Alcohol influence?", ["No", "Yes"])
    speed_limit = st.slider("🚦 Speed Limit (km/h)", 20, 140, 60)
    time_of_day = st.selectbox("🕒 Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

# Prepare input row
input_data = pd.DataFrame([{
    "Weather": weather,
    "Road_Condition": road_condition,
    "Road_Type": road_type,
    "Traffic_Density": traffic_density,
    "Road_Light_Condition": road_light,
    "Accident": accident_type,
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
    probs = pipe.predict_proba(input_data)[0]
    prediction = pipe.predict(input_data)[0]

    # Map numeric labels to text (adjust if dataset uses strings already)
    severity_map = {0: "Low", 1: "Medium", 2: "High"}
    severity_label = severity_map.get(int(prediction), str(prediction))
    confidence = probs[int(prediction)] * 100

    st.subheader("Prediction Result")
    if severity_label == "Low":
        st.success(f"✅ Predicted Severity: **{severity_label}** ({confidence:.2f}% chance of no accident !)")
    elif severity_label == "Medium":
        st.warning(f"⚠️ Predicted Severity: **{severity_label}** ({confidence:.2f}% chance of accident !)")
    else:
        st.error(f"🚨 Predicted Severity: **{severity_label}** ({confidence:.2f}% chance of accident !)")
