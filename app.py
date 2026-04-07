import streamlit as st
import joblib
import pandas as pd
import os

# 1. Basic Page Setup (Standard & Stable)
st.set_page_config(page_title="Football Match Predictor", page_icon="⚽")

# 2. Header
st.title("⚽ Match Outcome Predictor")
st.markdown("---") # Simple visual separator

# 3. Robust Asset Loading
# This uses local paths to ensure it works on your PC and the Cloud.
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    
    try:
        model = joblib.load(os.path.join(base_path, "football_model.pkl"))
        encoder = joblib.load(os.path.join(base_path, "club-encoder.pkl"))
        name_map = joblib.load(os.path.join(base_path, "name_map.pkl"))
        return model, encoder, name_map
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, encoder, name_map = load_assets()

# 4. Main App Logic
if model is not None:
    # Prepare the dropdown list
    club_to_id = {name: id for id, name in name_map.items()}
    club_list = sorted(list(club_to_id.keys()))

    # UI Layout - Simple Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home Team")
        home_team = st.selectbox("Select Home Club", club_list, key="home")
        
    with col2:
        st.subheader("✈️ Away Team")
        away_team = st.selectbox("Select Away Club", club_list, key="away")

    st.write("") # Adds some space
    
    # 5. Prediction Button
    if st.button("Predict Result", use_container_width=True):
        try:
            # Process IDs
            h_id = int(club_to_id[home_team])
            a_id = int(club_to_id[away_team])
            
            # Encode for Model
            h_idx = encoder.transform([h_id])[0]
            a_idx = encoder.transform([a_id])[0]
            
            # Predict
            prediction = model.predict([[h_idx, a_idx]])[0]
            
            # Map Result
            outcomes = {1: "Home Win", 0: "Draw", 2: "Away Win"}
            result = outcomes[prediction]
            
            # Display Result in a standard Success box
            st.success(f"### Prediction: {result}")
            
            if result == "Home Win":
                st.write(f"📈 The model favors **{home_team}** due to home advantage and historical stats.")
            elif result == "Away Win":
                st.write(f"📈 The model favors **{away_team}** based on performance metrics.")
            else:
                st.write("📈 The model predicts a closely contested match ending in a **Draw**.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

else:
    st.warning("Please ensure your model files (.pkl) are in the same folder as app.py")