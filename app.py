import streamlit as st
import joblib
import pandas as pd
import os

# 1. PAGE SETUP
st.set_page_config(
    page_title="Match Day Predictor Pro",
    page_icon="⚽",
    layout="centered"
)

# 2. RELIABLE ASSET LOADING
@st.cache_resource
def load_assets():
    try:
        # Loading directly from root folder
        model = joblib.load("football_model.pkl")
        encoder = joblib.load("club-encoder.pkl")
        name_map = joblib.load("name_map.pkl")
        return model, encoder, name_map
    except Exception as e:
        st.error(f"⚠️ Asset Error: {e}")
        return None, None, None

model, encoder, name_map = load_assets()

# 3. UI HEADER
st.title("⚽ Match Day Predictor")
st.caption("AI-Powered Football Outcome Forecasting Engine with Confidence Scoring")
st.markdown("---")

if model:
    # Prepare Data
    club_to_id = {name: id for id, name in name_map.items()}
    club_list = sorted(list(club_to_id.keys()))

    # 4. TEAM SELECTION CARDS
    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏠 Home")
            home_team = st.selectbox("Select Host Club", club_list, key="h_team")
            
        with col2:
            st.markdown("### ✈️ Away")
            away_team = st.selectbox("Select Visiting Club", club_list, key="a_team")

    st.write("") 

    # 5. PREDICTION ENGINE
    if st.button("RUN PREDICTION ENGINE", type="primary", use_container_width=True):
        try:
            # Conversion Logic
            h_id = int(club_to_id[home_team])
            a_id = int(club_to_id[away_team])
            h_idx = encoder.transform([h_id])[0]
            a_idx = encoder.transform([a_id])[0]
            
            # Model Inference (Result + Probability)
            prediction = model.predict([[h_idx, a_idx]])[0]
            probabilities = model.predict_proba([[h_idx, a_idx]])[0]
            
            outcomes = {1: "HOME WIN", 0: "DRAW", 2: "AWAY WIN"}
            result = outcomes[prediction]
            
            # Confidence Calculation
            confidence = max(probabilities) * 100
            
            # 6. STYLED RESULT DISPLAY
            st.markdown("---")
            st.subheader("📊 Forecast Result")
            
            # Main Result Box
            if result == "HOME WIN":
                st.success(f"🏆 **{result}** predicted for {home_team}")
            elif result == "AWAY WIN":
                st.info(f"🏆 **{result}** predicted for {away_team}")
            else:
                st.warning(f"🤝 **{result}** predicted between both teams")
            
            # 7. CONFIDENCE METER (The Accuracy Part)
            st.write(f"**Model Confidence:** {confidence:.1f}%")
            st.progress(max(probabilities))
            
            st.caption("The confidence score represents the probability assigned by the Random Forest classifier based on historical match patterns.")
                
        except Exception as e:
            st.error(f"Logic Error: {e}")

else:
    st.error("Model files not found. Please check your GitHub root directory.")