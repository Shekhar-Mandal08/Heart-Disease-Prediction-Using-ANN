import streamlit as st
import pandas as pd
import numpy as np

import joblib

from tensorflow.keras.models import load_model


# -----------------------------
# Load Model and Preprocessor
# -----------------------------

model = load_model(
    "heart_ann_model.keras"
)

preprocessor = joblib.load(
    "preprocessor.pkl"
)


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("❤️ Heart Disease Prediction")
st.write(
    "ANN based Heart Disease Classification Model"
)


# User Inputs

Age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=50
)


Sex = st.selectbox(
    "Sex",
    ["M", "F"]
)


ChestPainType = st.selectbox(
    "Chest Pain Type",
    ["TA", "ATA", "NAP", "ASY"]
)


RestingBP = st.number_input(
    "Resting Blood Pressure",
    value=120
)


Cholesterol = st.number_input(
    "Cholesterol",
    value=200
)


FastingBS = st.selectbox(
    "Fasting Blood Sugar",
    [0,1]
)


RestingECG = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)


MaxHR = st.number_input(
    "Maximum Heart Rate",
    value=150
)


ExerciseAngina = st.selectbox(
    "Exercise Angina",
    ["Y","N"]
)


Oldpeak = st.number_input(
    "Oldpeak",
    value=1.0
)


ST_Slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)



# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Heart Disease"):


    input_df = pd.DataFrame({

        "Age":[Age],
        "Sex":[Sex],
        "ChestPainType":[ChestPainType],
        "RestingBP":[RestingBP],
        "Cholesterol":[Cholesterol],
        "FastingBS":[FastingBS],
        "RestingECG":[RestingECG],
        "MaxHR":[MaxHR],
        "ExerciseAngina":[ExerciseAngina],
        "Oldpeak":[Oldpeak],
        "ST_Slope":[ST_Slope]

    })


    # Apply same preprocessing
    input_processed = preprocessor.transform(
        input_df
    )


    # Prediction probability

    prediction_prob = model.predict(
        input_processed
    )[0][0]


    st.subheader(
        f"Heart Disease Probability: {prediction_prob:.2%}"
    )


    if prediction_prob >= 0.5:

        st.error(
            "⚠️ High Risk: Heart Disease Detected"
        )

    else:

        st.success(
            "✅ Low Risk: No Heart Disease Detected"
        )