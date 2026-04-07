import streamlit as st
import joblib
import pandas as pd

# 1. Page Config (The "Face")
st.set_page_config(page_title="Football Predictor", page_icon="⚽")
st.title("⚽ Match Outcome Predictor")

# 2. Load the "Brain" directly (No FastAPI needed)
# We load these once at the start
@st.cache_resource
def load_assets():
    model = joblib.load("football_model.pkl")
    encoder = joblib.load("club-encoder.pkl")
    name_map = joblib.load("name_map.pkl")
    return model, encoder, name_map

model, encoder, name_map = load_assets()

# 3. Build the UI
club_to_id = {name: id for id, name in name_map.items()}
club_list = sorted(list(club_to_id.keys()))

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", club_list)
with col2:
    away_team = st.selectbox("Away Team", club_list)

# 4. The Logic (The "Chatbot" style)
if st.button("Predict Result"):
    try:
        # Get IDs from the names selected
        h_id = int(club_to_id[home_team])
        a_id = int(club_to_id[away_team])
        
        # Transform using the encoder
        h_idx = encoder.transform([h_id])[0]
        a_idx = encoder.transform([a_id])[0]
        
        # Make prediction
        prediction = model.predict([[h_idx, a_idx]])[0]
        
        # Show result
        outcomes = {1: "Home Win", 0: "Draw", 2: "Away Win"}
        st.header(f"Prediction: {outcomes[prediction]}")
        
    except Exception as e:
        st.error(f"Error: {e}")