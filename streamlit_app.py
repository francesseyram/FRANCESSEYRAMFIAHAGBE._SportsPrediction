import streamlit as st
import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor from scikit-learn

# Streamlit app interface
st.title('Player Rating Predictor')

# Create the file uploader widget
uploaded_file = st.file_uploader("Upload DecisionTreeRegressor.pkl", type="pkl")

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Load the model from the uploaded file
        model = pickle.load(uploaded_file, encoding='latin1')
        
        # Ensure the loaded model is of the correct type
        if not isinstance(model, DecisionTreeRegressor):
            raise TypeError("Uploaded model is not a DecisionTreeRegressor.")
        
        st.success("Model loaded successfully!")
        
    except pickle.UnpicklingError:
        st.error("Error in unpickling the file. The file might be corrupted.")
    except TypeError as e:
        st.error(f"Invalid model type: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
else:
    st.warning('Please upload a .pkl file.')

# Define the prediction function
def player_rating(model, features):
    try:
        # Ensure features are in the correct format and shape
        input_data = np.array(features).reshape(1, -1)
        
        # Make prediction using the loaded model
        prediction = model.predict(input_data)
        return prediction[0]  # Assuming model.predict() returns a single prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Input features from 
