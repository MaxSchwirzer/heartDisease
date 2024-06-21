import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the pretrained model with error handling
model_filename = 'dssheart.knn'

st.title("Heart Disease Prediction")

try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file {model_filename} not found. Please ensure it is in the correct directory.")
except ImportError as e:
    st.error(f"An error occurred while loading the model: {e}. Ensure the required packages are installed.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Define the features and their descriptions
features = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Serum Cholesterol in mg/dl',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl',
    'restecg': 'Resting Electrocardiographic Results',
    'thalach': 'Maximum Heart Rate Achieved',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression Induced by Exercise Relative to Rest',
    'slope': 'Slope of the Peak Exercise ST Segment',
    'ca': 'Number of Major Vessels Colored by Fluoroscopy',
    'thal': 'Thalassemia'
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
    'thal': st.selectbox(features['thal'], options=[0, 1, 2, 3])  # Include 0 as 'unknown'
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Define the preprocessing steps
categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_vars),
        ('cat', categorical_transformer, categorical_vars)
    ]
)

# Preprocess the input data
input_preprocessed = preprocessor.fit_transform(input_df)

# Button for prediction
if 'model' in globals() and st.button('Predict'):
    try:
        # Make prediction
        prediction = model.predict(input_preprocessed)
        prediction_proba = model.predict_proba(input_preprocessed)

        # Display the prediction
        if prediction[0] == 1:
            st.write("The model predicts that the patient **has heart disease**.")
        else:
            st.write("The model predicts that the patient **does not have heart disease**.")

        st.write(f"Prediction Probability: {prediction_proba[0][1]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
elif 'model' not in globals():
    st.warning("Please ensure that the model is loaded successfully before making predictions.")

st.write("Note: This tool is for educational purposes only and not a substitute for professional medical advice.")
