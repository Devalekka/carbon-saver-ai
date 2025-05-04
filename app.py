import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the model
model = pickle.load(open("rf_model.pkl", "rb"))

st.set_page_config(page_title="CarbonSaver AI 🌍", layout="centered")

# --- Title and Description ---
st.title("🌿 CarbonSaver AI – Personal Carbon Footprint Estimator")
st.markdown("Predict your carbon emissions and get lifestyle tips to reduce them.")

# --- User Input ---
diet = st.selectbox("🍽️ Diet Type", ["Vegan", "Vegetarian", "Meat-heavy"])
transport = st.selectbox("🚗 Primary Transport", ["Public Transport", "Private Car", "Cycle/Walk"])
heating = st.selectbox("🔥 Heating Energy Source", ["Electric", "Gas", "Coal"])
vehicle_type = st.selectbox("🚙 Vehicle Type", ["Electric", "Petrol", "Diesel"])
shower_freq = st.selectbox("🚿 How Often You Shower", ["Rarely", "Often", "Daily"])
flying_freq = st.selectbox("✈️ Frequency of Air Travel", ["Never", "Rarely", "Often"])
recycling = st.selectbox("♻️ Do You Recycle?", ["All", "Some", "None"])

# --- Predict Button ---
if st.button("💨 Calculate My CO₂ Emission"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Diet": diet,
        "Transport": transport,
        "Heating Energy Source": heating,
        "Vehicle Type": vehicle_type,
        "How Often Shower": shower_freq,
        "Frequency of Traveling by Air": flying_freq,
        "Recycling": recycling
    }])

    # Encode ordinal fields
    label_enc = LabelEncoder()
    for col in ["How Often Shower", "Frequency of Traveling by Air"]:
        input_data[col] = label_enc.fit_transform(input_data[col])

    # One-hot encode categorical features
    input_data = pd.get_dummies(input_data)

    # Make sure columns match model's training features
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_features]

    # Predict CO2 emission
    prediction = model.predict(input_data)[0]

    # --- Display Result ---
    st.success(f"🌎 Estimated CO₂ Emission: **{prediction:.2f} tons/year**")

    # --- Plot Graph ---
    fig, ax = plt.subplots()
    ax.bar(["You"], [prediction], color="mediumpurple")
    ax.axhline(3.0, color="green", linestyle="--", label="Global Avg CO₂ (tons/year)")
    ax.set_ylabel("CO₂ Emission (tons/year)")
    ax.set_title("Your Carbon Footprint")
    ax.legend()
    st.pyplot(fig)

    # --- Suggestions ---
    st.markdown("### 💡 Suggestions to Reduce Your Carbon Footprint")
    tips = []
    if diet == "Meat-heavy":
        tips.append("🥗 Try reducing meat consumption and explore plant-based meals.")
    if transport == "Private Car":
        tips.append("🚶‍♂️ Use public transport, walk, or bike when possible.")
    if heating == "Coal":
        tips.append("⚡ Switch to electric or solar-based heating if possible.")
    if recycling == "None":
        tips.append("♻️ Start sorting and recycling household waste.")

    if tips:
        for tip in tips:
            st.info(tip)
    else:
        st.success("You're already doing great! Keep it up 🌟")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by Devalekka")
