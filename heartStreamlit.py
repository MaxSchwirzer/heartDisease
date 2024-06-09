import streamlit as st
import pandas as pd
import pickle

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
    'age': 'Patient\'s age in years',
    'sex': 'Sex (1 = male; 0 = female)',
    'cp': 'Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)',
    'trestbps': 'Resting blood pressure (in mm Hg on admission to the hospital)',
    'chol': 'Serum cholesterol in mg/dl',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
    'restecg': 'Resting electrocardiographic results (0 = normal; 1 = having ST-T wave abnormality; 2 = \' criteria)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1 = yes; 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'The slope of the peak exercise ST segment (0 = upsloping; 1 = flat; 2 = downsloping)',
    'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
    'thal': 'Status of the heart (1 = normal; 2 = fixed defect; 3 = reversible defect)'
}

st.write("Enter the patient's details to predict the likelihood of heart disease.")

# Create input fields for each feature with specified constraints
input_data = {
    'age': st.number_input(features['age'], min_value=0, step=1),
    'sex': st.selectbox(features['sex'], options=[1, 0]),
    'cp': st.selectbox(features['cp'], options=[0, 1, 2, 3]),
    'trestbps': st.number_input(features['trestbps']),
    'chol': st.number_input(features['chol']),
    'fbs': st.selectbox(features['fbs'], options=[1, 0]),
    'restecg': st.selectbox(features['restecg'], options=[0, 1, 2]),
    'thalach': st.number_input(features['thalach']),
    'exang': st.selectbox(features['exang'], options=[1, 0]),
    'oldpeak': st.number_input(features['oldpeak']),
    'slope': st.selectbox(features['slope'], options=[0, 1, 2]),
    'ca': st.selectbox(features['ca'], options=[0, 1, 2, 3]),
    'thal': st.selectbox(features['thal'], options=[1, 2, 3])
}

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
