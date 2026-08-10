import streamlit as st
import joblib
import pandas as pd



# -----------------------------
# Load model and scaler
# -----------------------------

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")


# -----------------------------
# Application title
# -----------------------------

st.title("Diabetes Risk Prediction")
if st.button("Reset Form"):
    st.session_state.clear()
    st.rerun()
st.write(
    "Welcome to the Diabetes Risk Prediction Application."
)


# -----------------------------
# Patient Information
# -----------------------------

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (in years)",
        min_value=1,
        max_value=120,
        value=30
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )


# -----------------------------
# Body Measurements
# -----------------------------

st.header("Body Measurements")

col1, col2 = st.columns(2)

with col1:
    height = st.number_input(
        "Height (in cm)",
        min_value=30.0,
        max_value=400.0,
        value=170.0
    )

with col2:
    weight = st.number_input(
        "Weight (in kg)",
        min_value=1.0,
        max_value=700.0,
        value=60.0
    )


# Calculate BMI

bmi = weight / (height / 100) ** 2

st.write(f"Calculated BMI: {bmi:.2f}")


# -----------------------------
# Clinical Data
# -----------------------------

st.header("Clinical Data")

col1, col2 = st.columns(2)

with col1:
    hba1c = st.number_input(
        "HbA1c (%)",
        min_value=1.0,
        max_value=50.0,
        value=5.0,
        step=0.1
    )

with col2:
    glucose = st.number_input(
        "Random Blood Glucose (mg/dL)",
        min_value=1.0,
        max_value=1000.0,
        value=130.0,
        step=1.0
    )


# -----------------------------
# Medical History
# -----------------------------

st.header("Medical History")

col1, col2 = st.columns(2)

with col1:
    hypertension = st.selectbox(
        "Hypertension",
        ["No", "Yes"]
    )

with col2:
    heart_disease = st.selectbox(
        "Heart Disease",
        ["No", "Yes"]
    )

smoking_history = st.selectbox(
    "Smoking History",
    [
        "Current",
        "ever",
        "Former",
        "Never",
        "not current"
    ]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Diabetes Risk"):

    try:

        # -----------------------------
        # Input validation
        # -----------------------------

        if height <= 0 or weight <= 0:
            st.error(
                "Please enter valid height and weight values."
            )
            st.stop()

        if hba1c <= 0 or glucose <= 0:
            st.error(
                "Please enter valid HbA1c and blood glucose values."
            )
            st.stop()


        # -----------------------------
        # Convert gender
        # -----------------------------

        gender_male = 1 if gender == "Male" else 0
        gender_other = 1 if gender == "Other" else 0


        # -----------------------------
        # Convert hypertension
        # -----------------------------

        hypertension_value = 1 if hypertension == "Yes" else 0


        # -----------------------------
        # Convert heart disease
        # -----------------------------

        heart_disease_value = 1 if heart_disease == "Yes" else 0


        # -----------------------------
        # Convert smoking history
        # -----------------------------

        smoking_current = (
            1 if smoking_history == "Current" else 0
        )

        smoking_ever = (
            1 if smoking_history == "ever" else 0
        )

        smoking_former = (
            1 if smoking_history == "Former" else 0
        )

        smoking_never = (
            1 if smoking_history == "Never" else 0
        )

        smoking_not_current = (
            1 if smoking_history == "not current" else 0
        )


        # -----------------------------
        # Create patient DataFrame
        # -----------------------------

        patient = pd.DataFrame({
            "age": [age],
            "hypertension": [hypertension_value],
            "heart_disease": [heart_disease_value],
            "bmi": [bmi],
            "HbA1c_level": [hba1c],
            "blood_glucose_level": [glucose],
            "gender_Male": [gender_male],
            "gender_Other": [gender_other],
            "smoking_history_current": [smoking_current],
            "smoking_history_ever": [smoking_ever],
            "smoking_history_former": [smoking_former],
            "smoking_history_never": [smoking_never],
            "smoking_history_not current": [smoking_not_current]
        })


        # -----------------------------
        # Scale numerical features
        # -----------------------------

        numerical_features = [
            "age",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level"
        ]

        patient[numerical_features] = scaler.transform(
            patient[numerical_features]
        )


        # -----------------------------
        # Make prediction
        # -----------------------------

        probability = model.predict_proba(patient)[0][1]


        # -----------------------------
        # Display prediction
        # -----------------------------

        st.header("Prediction Result")

        st.metric(
            label="Predicted Probability of Diabetes",
            value=f"{probability * 100:.2f}%"
        )


        # 0.30 is the model's internal decision threshold.
        # We do not expose the technical threshold here.

        if probability >= 0.30:

            st.error("Higher Predicted Risk")

            st.write(
                "The model's predicted probability is above "
                "the predefined threshold for this application."
            )

        else:

            st.success("Lower Predicted Risk")

            st.write(
                "The model's predicted probability is below "
                "the predefined threshold for this application."
            )


    except Exception:

        st.error(
            "An unexpected error occurred while generating "
            "the prediction."
        )


# -----------------------------
# Disclaimer
# -----------------------------

st.info(
    "This tool provides a machine-learning-based risk estimate "
    "and is intended to support, not replace, professional "
    "clinical assessment and diagnostic testing."
)


# -----------------------------
# Model Information
# -----------------------------

st.divider()

st.header("Model Details")

st.write("**Model:** XGBoost Classifier")

st.write(
    "**Purpose:** Estimate the probability of diabetes "
    "using demographic and health-related features."
)

st.write(
    "**Input features:** 13 features including age, BMI, "
    "HbA1c, blood glucose, hypertension, heart disease, "
    "gender, and smoking history."
)

st.write(
    "**Output:** A predicted probability representing the "
    "model's estimated likelihood of class 1 (diabetes)."
)

st.write(
    "**Decision rule:** A predefined probability threshold "
    "is used to classify predictions into lower or higher "
    "predicted risk."
)

st.write(
    "**Important limitation:** This model provides a "
    "machine-learning-based risk estimate and should not "
    "be interpreted as a standalone medical diagnosis."
)