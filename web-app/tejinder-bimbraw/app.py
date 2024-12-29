

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Load the trained XGBoost model
model = joblib.load('xgboost_model.pkl')

# Streamlit app title
st.title("XGBoost Regression Model")

# Add some instructions for the user
st.write("""
    This app predicts the target variable based on the features provided.
    Please enter values for the features and press 'Predict' to see the result.
""")

# Input fields for user to enter data
# Adjust the feature names and input types according to your dataset
absences = st.number_input('absences', value=0.0)
higher = st.number_input('wants to take higher education input is binary', value=0.0)
famrel = st.number_input('quality of family relationships on scale of 1 to 5', value=0.0)
G1 = st.number_input('Firt period grade', value=0.0)
G2 = st.number_input('Second period grade', value=0.0)

# Gather the input data into a dataframe
input_data = pd.DataFrame([[absences, higher, famrel, G1,G2]], columns=['absences', 'higher', 'famrel', 'G1', 'G2'])

# Make predictions when the user clicks the 'Predict' button
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f"Predicted value: {prediction[0]}")
