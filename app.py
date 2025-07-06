# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st
import matplotlib.pyplot as plt

# Load the model and structure
model = joblib.load("Water-Quality-Prediction\\pollution_model.pkl")
model_cols = joblib.load("Water-Quality-Prediction\\model_columns (1).pkl")

# Let's create an User interface
st.title("Water Pollutants Predictor")
st.write("Predict the water pollutants based on Year and Station ID")


# User inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2024)
station_id = st.text_input("Enter Station ID", value='1')

# To encode and then predict
if st.button('Predict'):
    if not station_id:
        st.warning('Please enter the station ID')
    else:
        # Prepare the input
        input_df = pd.DataFrame({'year': [year_input], 'id':[station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])
        # Ensure all model columns are present
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reorder the columns to match the training data
        input_encoded = input_encoded[model_cols]


        # Predict
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        st.subheader(f"Predicted pollutant levels for the station '{station_id}' in {year_input}:")
        
        result_df = pd.DataFrame({
        'Pollutant': pollutants,
        'Predicted Value': predicted_pollutants
        })
        st.dataframe(result_df)

        fig, ax = plt.subplots()
        ax.bar(pollutants, predicted_pollutants, color='skyblue')
        ax.set_title("Predicted Pollutant Levels")
        st.pyplot(fig)
        



