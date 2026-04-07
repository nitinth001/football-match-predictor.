import streamlit as st
import requests
import joblib
import os

st.set_page_config(page_title="Football Predictor", page_icon="⚽")
st.title("⚽ Match Outcome Predictor")

# Load the name map
# Why: To populate the dropdown menu with team names
# Note: Use a relative path to find the file in the backend folder
path_to_map = os.path.join("..", "backend", "name_map.pkl")
name_map = joblib.load(path_to_map)

# Create a name-to-ID lookup
club_to_id = {name: id for id, name in name_map.items()}
club_list = sorted(list(club_to_id.keys()))

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", club_list)
with col2:
    away_team = st.selectbox("Away Team", club_list)

if st.button("Predict Result"):
    # Prepare data for API
    payload = {
        "home_id": int(club_to_id[home_team]),
        "away_id": int(club_to_id[away_team])
    }
    
    # URL: Use localhost for testing, change to Railway URL after deployment
    API_URL = "http://127.0.0.1:8000/predict"
    
    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()["prediction"]
        st.header(f"Result: {result}")
    except:
        st.error("Could not connect to the Backend. Make sure FastAPI is running!")