import pickle
import streamlit as st
from os import path
import numpy as np

st.title("Mobile Price Classification App")

# Load the trained model
file_name = "svm_classifier.pkl"
with open(path.join( file_name), 'rb') as f:
 svm_classifier = pickle.load(f)

# Input features for 
# mobile price classification
battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=5000, step=50)
blue = st.selectbox("Bluetooth", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
clock_speed = st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.5, step=0.1)
dual_sim = st.selectbox("Dual SIM", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
fc = st.number_input("Front Camera (MP)", min_value=0, max_value=30, step=1)
four_g = st.selectbox("4G Support", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
int_memory = st.number_input("Internal Memory (GB)", min_value=2, max_value=256, step=1)
m_dep = st.number_input("Mobile Depth (cm)", min_value=0.1, max_value=1.0, step=0.01)
mobile_wt = st.number_input("Mobile Weight (g)", min_value=50, max_value=300, step=1)
n_cores = st.number_input("Number of Cores", min_value=1, max_value=8, step=1)
pc = st.number_input("Primary Camera (MP)", min_value=0, max_value=50, step=1)
px_height = st.number_input("Pixel Height", min_value=0, max_value=3000, step=10)
px_width = st.number_input("Pixel Width", min_value=0, max_value=3000, step=10)
ram = st.number_input("RAM (MB)", min_value=256, max_value=12000, step=256)
sc_h = st.number_input("Screen Height (cm)", min_value=0, max_value=30, step=1)
sc_w = st.number_input("Screen Width (cm)", min_value=0, max_value=20, step=1)
talk_time = st.number_input("Talk Time (hours)", min_value=0, max_value=50, step=1)
three_g = st.selectbox("3G Support", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
touch_screen = st.selectbox("Touch Screen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
wifi = st.selectbox("WiFi", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Convert dual sim to numeric (1 for Yes, 0 for No)
dual_sim = 1 if dual_sim == "Yes" else 0

# Button to trigger prediction
if st.button("Classify Mobile Price"):
    # Prepare the input data
    input_features = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep,
                          mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time,
                          three_g, touch_screen, wifi]])

    # Make prediction
    pred = svm_classifier.predict(input_features)

    # Display the prediction
    st.write("Predicted Mobile Price Category is:", pred[0])
