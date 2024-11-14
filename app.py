# app.py
import os
import streamlit as st
import tensorflow as tf
from backend.model_inference import generate_recommendations
from backend.data_processing import preprocess_data

# Suppress oneDNN and CPU optimization warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel('ERROR')

# Frontend Interface
st.image("Fitness_logo_watchos.png", width=100)
st.title("RevFit")
st.markdown("<h3 style='color: #4CAF50;'>Revitalize your Fitness Journey</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for User Inputs
st.sidebar.header("User Input")

# --- Distance Recommendation ---
st.subheader("Distance Recommendation Based on Calories")
target_calories = st.sidebar.number_input("Enter target calories to burn:", min_value=0, value=400, step=50)

if st.sidebar.button("Calculate Distance"):
    try:
        # Preprocess the input calories for model inference
        processed_calories = preprocess_data(target_calories)  # Directly pass single value
        recommendations = generate_recommendations(processed_calories)
        st.session_state.predicted_distance = f"**Recommended Distance**: {recommendations['predicted_distance'][0][0]:.2f} km 🚴"
    except Exception as e:
        st.error(f"Error in processing distance: {e}")

# Display previous results for distance
if "predicted_distance" in st.session_state:
    st.success(st.session_state.predicted_distance)

# --- Speed Sequence Recommendation ---
st.subheader("Speed Sequence Prediction (Steps/Minute)")
if st.sidebar.button("Generate Speed Sequence"):
    try:
        processed_calories = preprocess_data(target_calories)
        recommendations = generate_recommendations(processed_calories)
        st.session_state.predicted_speed = f"**Recommended Speed Sequence**: {recommendations['predicted_speed'][0]} steps/min 🚴"
    except Exception as e:
        st.error(f"Error in processing speed sequence: {e}")

# Display previous results for speed
if "predicted_speed" in st.session_state:
    st.success(st.session_state.predicted_speed)

# --- Heart Rate Prediction ---
st.subheader("Heart Rate Sequence Prediction")
if st.sidebar.button("Predict Heart Rate"):
    try:
        processed_calories = preprocess_data(target_calories)
        recommendations = generate_recommendations(processed_calories)
        st.session_state.predicted_heart_rate = f"**Recommended Heart Rate Sequence**: {recommendations['predicted_heart_rate'][0]} bpm ❤️"
    except Exception as e:
        st.error(f"Error in processing heart rate: {e}")

# Display previous results for heart rate
if "predicted_heart_rate" in st.session_state:
    st.success(st.session_state.predicted_heart_rate)
