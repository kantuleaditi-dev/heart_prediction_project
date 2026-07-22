import streamlit as st
import pandas as pd
import joblib
# Load Saved Files
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")
columns = joblib.load("diabetes_columns.pkl")
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
st.title("🩺 Diabetes Prediction System")
st.write("Enter Patient Details")
# Input Fields
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.000,
    max_value=3.000,
    value=0.500,
    format="%.3f"
)
Age = st.number_input("Age", min_value=1, max_value=120, value=30)
# Create DataFrame
input_data = pd.DataFrame([[
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age
]], columns=columns)

# Scale Data
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error(" Prediction: Person is Diabetic")
    else:
        st.success("Prediction: Person is Not Diabetic")