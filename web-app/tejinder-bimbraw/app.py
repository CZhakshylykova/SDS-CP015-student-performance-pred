import pandas as pd
import streamlit as st
import pickle
import os
import joblib
# Cache the model loading
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'xgboost_model.pkl')   
    model = joblib.load(model_path)
    return model

model = load_model()

# App title
st.title("Student Performance Predictor")

# Instructions
st.write("""
    This app predicts the target variable based on the features provided.
    Please enter values for the features and press 'Predict' to see the result.
""")

# Input fields for user to enter data
absences = st.number_input('Absences', min_value=0.0, value=0.0)
higher = st.number_input('Wants to take higher education (binary)', min_value=0, max_value=1, value=0)
famrel = st.number_input('Quality of family relationships (scale 1 to 5)', min_value=1, max_value=5, value=3)
G1 = st.number_input('First period grade (0 to 20)', min_value=0, max_value=20, value=0)
G2 = st.number_input('Second period grade (0 to 20)', min_value=0, max_value=20, value=0)

# Gather the input data into a DataFrame
input_data = pd.DataFrame([[absences, higher, famrel, G1, G2]], columns=['absences', 'higher', 'famrel', 'G1', 'G2'])

# Convert the columns to the correct data types (numeric)
input_data['higher'] = input_data['higher'].astype(int)
input_data['famrel'] = pd.to_numeric(input_data['famrel'], errors='coerce')
input_data['G1'] = pd.to_numeric(input_data['G1'], errors='coerce')
input_data['G2'] = pd.to_numeric(input_data['G2'], errors='coerce')

# Check if any conversion failed (i.e., if there are NaN values)
if input_data.isnull().any().any():
    st.error("Some of the input values could not be converted to numbers. Please check your inputs.")
else:
    # Predict when the button is clicked
    if st.button('Predict'):
        try:
            prediction = model.predict(input_data)
            st.write(f"Predicted value: {prediction[0]}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
