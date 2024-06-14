import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and preprocessing pipeline
model_filename = 'dssheart.pkl'
with open(model_filename, 'rb') as file:
    best_model = pickle.load(file)

# Define a function to get user input
def get_user_input():
    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3], format_func=lambda x: ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'][x])
    trestbps = st.number_input('Resting Blood Pressure (in mm Hg)', min_value=80, max_value=200, value=120)
    chol = st.number_input('Serum Cholesterol in mg/dl', min_value=100, max_value=400, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'False' if x == 0 else 'True')
    restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2], format_func=lambda x: ['Normal', 'Abnormal', 'Ventricular hypertrophy'][x])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2], format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3], format_func=lambda x: ['Unknown', 'Normal', 'Fixed Defect', 'Reversible Defect'][x])

    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    return pd.DataFrame(user_data, index=[0])

# Title of the app
st.title('Heart Disease Prediction App')

# User input
user_input_df = get_user_input()

# Display the user input
st.subheader('User Input:')
st.write(user_input_df)

# Load preprocessing pipeline
preprocessor = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
        ]
    ))
])

# Preprocess the user input
user_input_preprocessed = preprocessor.fit_transform(user_input_df)

# Make prediction
prediction = best_model.predict(user_input_preprocessed)
prediction_proba = best_model.predict_proba(user_input_preprocessed)

# Display the prediction result
st.subheader('Prediction:')
if prediction[0] == 1:
    st.write('The model predicts that the patient **has heart disease**.')
else:
    st.write('The model predicts that the patient **does not have heart disease**.')

# Display the prediction probability
st.subheader('Prediction Probability:')
st.write(f"Probability of having heart disease: {prediction_proba[0][1]:.2f}")
st.write(f"Probability of not having heart disease: {prediction_proba[0][0]:.2f}")
