import streamlit as st
import joblib
import pandas as pd
import os

# --- PAGE CONFIGURATION (Crucial for first impression) ---
# This sets the browser tab title and a soccer ball icon.
st.set_page_config(
    page_title="Football Match Outcome Predictor",
    page_icon="⚽",
    layout="wide", # Use full width of the screen
    initial_sidebar_state="expanded",
)

# --- THE PITCH-SIDE STYLING (The magic CSS) ---
# We use custom CSS to apply a dark/green football aesthetic.
st.markdown("""
<style>
    /* 1. Overall Page Background and Font */
    [data-testid="stAppViewContainer"] {
        background-color: #0c0f13; /* Deep, dark charcoal */
        font-family: 'Montserrat', sans-serif; /* Modern, bold font */
    }

    /* 2. Top Title Bar Styling */
    .title-wrapper {
        background: linear-gradient(135deg, #1f2a3c, #1a1e26); /* Subtle gradient dark bar */
        border-bottom: 3px solid #00f901; /* VIBRANT Football Green border */
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
        color: #00f901 !important; /* Green title text */
    }

    /* 3. The Selectbox Containers (The "Input Stadiums") */
    div[data-testid="stColumn"] {
        background-color: #1a1e26; /* Slightly lighter dark gray */
        border: 1px solid #333; /* Soft border */
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.5); /* Subtle shadow for depth */
        margin: 0.5rem;
    }

    /* Style the selectbox labels and dropdowns */
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

    /* 4. The Predict Button Styling (Vibrant Green) */
    .stButton>button {
        background-color: #00f901 !important; /* Electric Football Green */
        color: black !important; /* Black text on green */
        border: 2px solid #00f901 !important;
        border-radius: 25px !important;
        padding: 10px 30px !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        width: 100% !important; /* Full width for dominance */
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        background-color: white !important; /* Hover effect: White text/green border */
        color: #00f901 !important;
        border: 2px solid white !important;
        box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.4);
    }
</style>
""", unsafe_allow_with_html=True)

# --- CUSTOM HEADER ---
# Replace standard st.title with our custom gradient bar
st.markdown('<div class="title-wrapper"><h1>⚽ Match Outcome Predictor</h1></div>', unsafe_allow_with_html=True)

# --- LOAD ASSETS (Your existing logic) ---
# Added basic error handling to match the deployment fixes.
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    
    # Define absolute paths for reliability
    model_path = os.path.join(base_path, "football_model.pkl")
    encoder_path = os.path.join(base_path, "club-encoder.pkl")
    map_path = os.path.join(base_path, "name_map.pkl")
    
    # Ensure all files exist
    if not all(os.path.exists(p) for p in [model_path, encoder_path, map_path]):
        st.error("Missing one or more model files. Verify 'football_model.pkl', 'club-encoder.pkl', and 'name_map.pkl' are in the main project folder.")
        st.stop()
        
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    name_map = joblib.load(map_path)
    
    return model, encoder, name_map

model, encoder, name_map = load_assets()

# --- PREPARE DATA (Your existing logic) ---
club_to_id = {name: id for id, name in name_map.items()}
club_list = sorted(list(club_to_id.keys()))

# --- THE INPUT LAYOUT (Two stylized columns) ---
# st.columns creates side-by-side elements. We add containers inside for styling.
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏟️ Home Turf")
    home_team = st.selectbox("Select Home Club", club_list)

with col2:
    st.subheader("🚀 Away Challenge")
    away_team = st.selectbox("Select Away Club", club_list)

# --- PADDING ---
# A simple way to add some empty space before the button.
st.markdown("<br>", unsafe_allow_with_html=True)

# --- THE PREDICT ACTION (Styled button and clear result) ---
if st.button("Predict Full Time Result"):
    try:
        # Get IDs from the names selected
        h_id = int(club_to_id[home_team])
        a_id = int(club_to_id[away_team])
        
        # Transform using the encoder
        h_idx = encoder.transform([h_id])[0]
        a_idx = encoder.transform([a_id])[0]
        
        # Make prediction
        prediction = model.predict([[h_idx, a_idx]])[0]
        
        # Define and display result clear
        outcomes = {1: "Home Win", 0: "Draw", 2: "Away Win"}
        
        # Use different icons based on the result
        result_icon = "⚽" if outcomes[prediction] == "Draw" else "🏆"
        
        st.header(f"{result_icon} Prediction: {outcomes[prediction]}")
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")