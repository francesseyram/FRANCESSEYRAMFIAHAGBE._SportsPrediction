import streamlit as st
import numpy as np
import pickle

# Streamlit app interface
st.title('Player Rating Predictor')

# Create the file uploader widget
uploaded_file = st.file_uploader("Upload DecisionTreeRegressor.pkl", type="pkl")

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Load the model from the uploaded file
        with open(uploaded_file, 'rb') as file:
            model = pickle.load(file)
        st.success("Model loaded successfully!")
        
    except pickle.UnpicklingError:
        st.error("Error in unpickling the file. The file might be corrupted.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
else:
    st.warning('Please upload a .pkl file.')

# Define the prediction function
def player_rating(features):
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction[0]

# Input features from user
features = [ 'potential',  'mentality_vision','value_eur', 
    'wage_eur', 'age', 'league_level', 'weak_foot', 'skill_moves', 
    'international_reputation', 'pace', 'shooting', 'passing', 'dribbling', 
    'defending', 'physic', 'attacking_crossing', 'attacking_finishing', 
    'attacking_heading_accuracy', 'attacking_short_passing', 
    'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 
    'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 
    'movement_sprint_speed', 'movement_agility', 'movement_reactions', 
    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 
    'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 
    'mentality_positioning', 'mentality_penalties', 
    'mentality_composure', 'defending_marking_awareness', 
    'defending_standing_tackle', 'defending_sliding_tackle', 
    'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 
    'goalkeeping_positioning', 'goalkeeping_reflexes','cat_player_positions', 'catnationality_name', 'cat_preferred_foot', 
    'cat_work_rate', 'catls', 'catst', 'catrs', 'catlw', 'cat_lf', 
    'cat_cf', 'catrf', 'catrw', 'catlam', 'catcam', 'cat_ram', 
    'cat_lm', 'catlcm', 'catcm', 'catrcm', 'catrm', 'cat_lwb', 
    'cat_ldm', 'catcdm', 'catrdm', 'catrwb', 'catlb', 'cat_lcb', 
    'cat_cb', 'catrcb', 'catrb', 'cat_gk'
]
input_ = []

for i in features:
    value = st.number_input(f'Enter {i}', value=0.0)
    input_.append(value)

if st.button('Predict Rating'):
    rating = player_rating(input_)
    st.write(f'Predicted Player Rating: {rating}')
