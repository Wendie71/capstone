import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Explainable Diabetes Risk Prediction")
st.markdown("""
This application predicts the likelihood of diabetes using
machine learning based on demographic and health indicators.
""")

st.write("Enter patient information below.")

# Inputs
age = st.number_input(
    "Age (in years)",
    min_value=1,
    max_value=120,
    value=30,
    help="Enter the patient's age."
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"],
    help="Select the patient's gender."
)

height = st.number_input(
    "Height (in cm)",
    min_value=30,
    max_value=300,
    value=170,
    help="Enter the patient's height."
)

weight = st.number_input(
    "Weight (in Kg)",
    min_value=1,
    max_value=300,
    value=60,
    help="Enter the patient's weight."
)

# Calculate BMI
bmi = weight / (height / 100) ** 2

st.write(f"Calculated BMI: {bmi:.2f}")

hba1c = st.number_input(
    "HbA1c (in percentage)",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    help="Enter the patient's HbA1c level."
)

glucose = st.number_input(
    "Random Blood Glucose Level (mg/dL)",
    min_value=1,
    max_value=1000,
    value=130,
    help="Enter the patient's random blood glucose level."
)

hypertension = st.selectbox(
    "Hypertension",
    ["Yes", "No"],
    help="Select whether the patient has hypertension."
)

heart_disease = st.selectbox(
    "Heart Disease",
    ["Yes", "No"],
    help="Select whether the patient has heart disease."
)

smoking_history = st.selectbox(
    "Patient's Smoking History",
    [
        "Current",
        "ever",
        "Former",
        "Never",
        "not current"
    ],
    help="Select the patient's smoking history."
)

# Prediction button
if st.button("Predict Diabetes Risk"):

     # Convert Yes/No variables to 0/1
    hypertension_value = 1 if hypertension == "Yes" else 0
    heart_disease_value = 1 if heart_disease == "Yes" else 0

    # One-hot encode smoking history
    smoking_current = 1 if smoking_history == "Current" else 0
    smoking_former = 1 if smoking_history == "Former" else 0
    smoking_never = 1 if smoking_history == "Never" else 0

    patient = pd.DataFrame({
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose],
        "smoking_history": [smoking_history]
    })

    patient[['age','bmi','HbA1c_level','blood_glucose_level','smoking_history']] = (
        scaler.transform(
            patient[['age','bmi','HbA1c_level','blood_glucose_level', 'smoking_history']]
        )
    )

    probability = model.predict_proba(patient)[0][1]

    st.subheader(f"Risk Score: {probability*100:.2f}%")

    if probability >= 0.3:
        st.error("High Risk")
    elif probability >= 0.2:
        st.warning("Moderate Risk")
    else:
        st.success("Low Risk")
