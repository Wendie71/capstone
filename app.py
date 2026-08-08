import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Explainable Diabetes Risk Prediction")

st.write("Enter patient information below.")

# Inputs
age = st.slider("Age", 18, 100, 40)
bmi = st.slider("BMI", 10.0, 60.0, 25.0)
hba1c = st.slider("HbA1c Level", 3.0, 15.0, 5.5)
glucose = st.slider("Blood Glucose Level", 70, 350, 120)

hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])

# Prediction button
if st.button("Predict Diabetes Risk"):

    patient = pd.DataFrame({
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose]
    })

    patient[['age','bmi','HbA1c_level','blood_glucose_level']] = (
        scaler.transform(
            patient[['age','bmi','HbA1c_level','blood_glucose_level']]
        )
    )

    probability = model.predict_proba(patient)[0][1]

    st.subheader(f"Risk Score: {probability*100:.2f}%")

    if probability >= 0.8:
        st.error("High Risk")
    elif probability >= 0.5:
        st.warning("Moderate Risk")
    else:
        st.success("Low Risk")
