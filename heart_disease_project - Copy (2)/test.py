import streamlit as st
import pandas as pd
import joblib

# Load Model, Scaler and Columns
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title(" Heart Disease Prediction System")

st.write("Enter Patient Details")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=40)

sex = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
cp_dict = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
cp = cp_dict[cp]

restingbp = st.number_input("Resting Blood Pressure", value=120)

cholesterol = st.number_input("Cholesterol", value=200)

fastingbs = st.selectbox("Fasting Blood Sugar", [0, 1])

restecg = st.selectbox("Resting ECG", ["LVH", "Normal", "ST"])
ecg_dict = {"LVH": 0, "Normal": 1, "ST": 2}
restecg = ecg_dict[restecg]

maxhr = st.number_input("Maximum Heart Rate", value=150)

exercise = st.selectbox("Exercise Angina", ["No", "Yes"])
exercise = 1 if exercise == "Yes" else 0

oldpeak = st.number_input("Old Peak", value=0.0)

slope = st.selectbox("ST Slope", ["Down", "Flat", "Up"])
slope_dict = {"Down": 0, "Flat": 1, "Up": 2}
slope = slope_dict[slope]

# Prediction
if st.button("Predict"):

    data = pd.DataFrame([[age, sex, cp, restingbp, cholesterol,
                          fastingbs, restecg, maxhr,
                          exercise, oldpeak, slope]],
                        columns=columns)

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error(" Heart Disease Detected")
    else:
        st.success(" No Heart Disease")