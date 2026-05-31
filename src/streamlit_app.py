import os
import streamlit as st
import joblib
import pandas as pd

# Resolve the path to models/model.pkl relative to this file's directory
MODEL_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../models/model.pkl'))

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at '{MODEL_PATH}'. Please train the model first by running the model building notebook or pipeline.")
    st.stop()

# Load the saved model
model = joblib.load(MODEL_PATH)

# Build the Web App UI
st.title("House Price Predictor 🏠")
st.write("Enter the details of the house below to estimate its price.")

# Create input sliders for the user
sqft = st.number_input("Square Feet", min_value=500, max_value=5000, value=1500)
beds = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=3)
age = st.slider("Age of House (Years)", min_value=0, max_value=100, value=10)

# Prediction Button
if st.button("Predict Price"):
    # Format the input exactly as the model expects it
    input_data = pd.DataFrame({
        'SquareFeet': [sqft],
        'Bedrooms': [beds],
        'AgeOfHouse': [age]
    })
    
    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    # Show the result on the screen
    st.success(f"The estimated price of the house is: ${prediction:,.2f}")
