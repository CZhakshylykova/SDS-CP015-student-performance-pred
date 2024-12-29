import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Set page configuration and title
st.set_page_config(page_title='Student Performance Prediction', layout='centered')

# Cache the model loading
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'xgboost_model.pkl')   
    model = joblib.load(model_path)
    return model

model = load_model()

# App title
st.title(":orange[Student Performance Prediction]")  # Title spans across the whole page


# Split the page into two columns
col1, col2 = st.columns([1, 2])  # Split into two columns: col1 takes 1/3, col2 takes 2/3

# Left column: About the study
with col1:
    st.header("About the Study")
    st.write("""
    This study explores various factors that affect student performance, 
    with the goal of predicting their final grades based on input features. 
    Some of the key features include:
    - Mother's education level
    - Alcohol consumption on workdays and weekends
    - Number of school absences
    - First and second period grades
    - Participation in extracurricular activities

    Link to Dataset: https://archive.ics.uci.edu/dataset/320/student+performance
    
    This project is a collaborative initiative brought by SuperDataScience community

 """)

# Right column: Feature input
with col2:

    st.header("Input Features")  # Feature input title is now part of the input container


    absences = st.number_input("Number of School Absences (0-93)", min_value=0, max_value=93, step=1)
    G1 = st.number_input("First Period Grade (0-20)", min_value=0, max_value=20, step=1)
    G2 = st.number_input("Second Period Grade (0-20)", min_value=0, max_value=20, step=1)
    higher = st.selectbox("Aiming for Higher Education (yes/no)", ['yes', 'no'])
    famrel = st.number_input("quality of family relationships on scale of 1 to 5 (5 being the best)",min_value=1,max_value=5,step = 1)

    # Columns definition for model prediction
    columns = [ 'absences', 'G1', 'G2', 'higher','famrel']

    def preprocess_input( absences, G1, G2, higher, famrel):
        # Encode categorical variables
        higher_encoded = 1 if higher == 'yes' else 0
        
                
        # Construct input array
        row = np.array([ absences, G1, G2, higher_encoded,famrel])
        return pd.DataFrame([row], columns=columns)

    # Prediction function
    def predict():
        X = preprocess_input( absences, G1, G2, higher,famrel)
        with st.spinner("Making prediction..."):
            prediction = model.predict(X)[0]
        st.success(f"Predicted Final Grade: {prediction:.2f}")

    # Prediction button
    if st.button("Predict"):
        predict()
