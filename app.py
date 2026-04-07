import streamlit as st
import joblib
import pandas as pd
import os

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Football Match Outcome Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. SEPARATED CSS STYLING (Fixed for Python 3.14 compatibility)
football_style = """
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0c0f13;
        font-family: 'Montserrat', sans-serif;
    }
    .title-wrapper {
        background: linear-gradient(135deg, #1f2a3c, #1a1e26);
        border-bottom: 3px solid #00f901;
        color: white;
        padding: 1.5rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .title-wrapper h1 {
        margin: 0;
        font-size: 2.5rem !important;
        color: #00f901 !important;
    }
    div[data-testid="stColumn"] {
        background-color: #1a1e26;
        border: 1px solid #333;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
        margin: 0.5rem;
    }
    .stSelectbox label {
        color: #ddd;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-baseweb="select"] > div {
        background-color: #0c0f13 !important;
        border: 2px solid #555 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    .stButton>button {
        background-color: #00f901 !important;
        color: black !important;
        border: 2px solid #00f901 !important;
        border-radius: 25px !important;
        padding: 10px 30px !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: white !important;
        color: #00f901 !important;
        border: 2px solid white !important;
        box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.4);
    }
</style>
"""

# Apply the style
st.markdown(football_style, unsafe_allow_with_html=True)

# 3. CUSTOM HEADER
st.markdown('<div class="title-wrapper"><h1>⚽ Match Outcome Predictor</h1></div>', unsafe_allow_with_html=True)

# 4. LOAD ASSETS
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "football_model.pkl")
    encoder_path = os.path.join(base_path, "club-encoder.pkl")
    map_path = os.path.join(base_path, "name_map.pkl")
    
    if not all(os.path.exists(p) for p in [model_path, encoder_path, map_path]):
        st.error("Missing model files in the main directory!")
        st.stop()
        
    return joblib.load(model_path), joblib.load(encoder_path), joblib.load(map_path)

model, encoder, name_map = load_assets()

# 5. PREPARE DATA
club_to_id = {name: id for id, name in name_map.items()}
club_list = sorted(list(club_to_id.keys()))

# 6. UI LAYOUT
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏟️ Home Turf")
    home_team = st.selectbox("Select Home Club", club_list, key="home")

with col2:
    st.subheader("🚀 Away Challenge")
    away_team = st.selectbox("Select Away Club", club_list, key="away")

st.markdown("<br>", unsafe_allow_with_html=True)

# 7. PREDICTION LOGIC
if st.button("Predict Full Time Result"):
    try:
        h_id = int(club_to_id[home_team])
        a_id = int(club_to_id[away_team])
        
        h_idx = encoder.transform([h_id])[0]
        a_idx = encoder.transform([a_id])[0]
        
        prediction = model.predict([[h_idx, a_idx]])[0]
        outcomes = {1: "Home Win", 0: "Draw", 2: "Away Win"}
        
        result_icon = "⚽" if outcomes[prediction] == "Draw" else "🏆"
        
        # Display the result in a nice styled box
        st.markdown(f"""
            <div style="background-color: #1a1e26; padding: 20px; border-radius: 15px; border-left: 5px solid #00f901; margin-top: 20px;">
                <h2 style="color: white; margin: 0;">{result_icon} Prediction: {outcomes[prediction]}</h2>
            </div>
        """, unsafe_allow_with_html=True)
        
    except Exception as e:
        st.error(f"Error: {e}")