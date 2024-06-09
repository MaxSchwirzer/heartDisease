import streamlit as st
import pandas as pd
import pickle
import os

# Load the pretrained model with error handling
model_filename = 'dssheart.pkl'

st.title("Heart Disease Prediction")

try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file {model_filename} not found. Please ensure it is in the correct directory.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Define the features and their descriptions
features = {
    'age': 'Age',
    'sex': 'Sex (1 = male; 0 = female)',
    'cp': 'Chest Pain Type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Serum Cholesterol in mg/dl',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)',
    'restecg': 'Resting Electrocardiographic Results (0 = normal, 1 = having ST-T wave abnormality, 2 = left ventricular hypertrophy)',
    'thalach': 'Maximum Heart Rate Achieved',
    'exang': 'Exercise Induced Angina (1 = yes; 0 = no)',
    'oldpeak': 'ST Depression Induced by Exercise Relative to Rest',
    'slope': 'Slope of the Peak Exercise ST Segment (0 = upsloping, 1 = flat, 2 = downsloping)',
    'ca': 'Number of Major Vessels (0-3) Colored by Fluoroscopy',
    'thal': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)'
}

st.write("Enter the patient's details to predict the likelihood of heart disease.")

# Create input fields for each feature
input_data = {}
for feature, description in features.items():
    input_data[feature] = st.number_input(description)

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Button for prediction
if st.button('Predict'):
    try:
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Display the prediction
        if prediction[0] == 1:
            st.write("The model predicts that the patient **has heart disease**.")
        else:
            st.write("The model predicts that the patient **does not have heart disease**.")

        st.write(f"Prediction Probability: {prediction_proba[0][1]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.write("Note: This tool is for educational purposes only and not a substitute for professional medical advice.")
