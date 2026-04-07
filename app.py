import streamlit as st
import joblib
import pandas as pd
import os

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Match Day Predictor Pro",
    page_icon="⚽",
    layout="wide"
)

# 2. SAFE CSS INJECTION (Broken into chunks to fix Python 3.14 TypeError)
def apply_styles():
    # Base Theme
    st.markdown("""<style>
        [data-testid="stAppViewContainer"] { background-color: #0c0f13; font-family: 'Segoe UI', sans-serif; }
        .stSelectbox label { color: #00f901 !important; font-weight: bold; text-transform: uppercase; }
    </style>""", unsafe_allow_with_html=True)
    
    # Header & Cards
    st.markdown("""<style>
        .title-wrapper {
            background: linear-gradient(90deg, #1f2a3c, #0c0f13);
            border-left: 10px solid #00f901;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        div[data-testid="stColumn"] {
            background-color: #1a1e26;
            border: 1px solid #333;
            border-radius: 20px;
            padding: 25px;
        }
    </style>""", unsafe_allow_with_html=True)

    # Buttons
    st.markdown("""<style>
        .stButton>button {
            background-color: #00f901 !important;
            color: black !important;
            font-weight: 900 !important;
            height: 3em;
            border-radius: 50px !important;
            text-transform: uppercase;
        }
        .stButton>button:hover {
            background-color: white !important;
            border: 2px solid #00f901 !important;
        }
    </style>""", unsafe_allow_with_html=True)

apply_styles()

# 3. HEADER UI
st.markdown('<div class="title-wrapper"><h1 style="color:white; margin:0;">⚽ MATCH DAY PREDICTOR</h1><p style="color:#888;">AI-Powered Statistical Analysis</p></div>', unsafe_allow_with_html=True)

# 4. DATA LOADING
@st.cache_resource
def load_data():
    base = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base, "football_model.pkl"))
        enc = joblib.load(os.path.join(base, "club-encoder.pkl"))
        n_map = joblib.load(os.path.join(base, "name_map.pkl"))
        return model, enc, n_map
    except Exception as e:
        st.error(f"Asset Load Error: {e}")
        return None, None, None

model, encoder, name_map = load_data()

if model:
    # 5. INPUTS
    club_to_id = {name: id for id, name in name_map.items()}
    club_list = sorted(list(club_to_id.keys()))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏠 HOME")
        home_team = st.selectbox("Select Home Club", club_list, key="h_sel")
    with col2:
        st.markdown("### ✈️ AWAY")
        away_team = st.selectbox("Select Away Club", club_list, key="a_sel")

    st.markdown("---")

    # 6. PREDICTION
    if st.button("CALCULATE PROBABILITY"):
        try:
            h_idx = encoder.transform([int(club_to_id[home_team])])[0]
            a_idx = encoder.transform([int(club_to_id[away_team])])[0]
            
            pred = model.predict([[h_idx, a_idx]])[0]
            outcomes = {1: "HOME WIN", 0: "DRAW", 2: "AWAY WIN"}
            res = outcomes[pred]
            
            # Stylized Result Display
            st.markdown(f"""
                <div style="background-color: #00f901; padding: 30px; border-radius: 15px; text-align: center;">
                    <h1 style="color: black; margin: 0; font-size: 3rem;">{res}</h1>
                    <p style="color: black; font-weight: bold; margin:0;">AI MODEL PREDICTION COMPLETE</p>
                </div>
            """, unsafe_allow_with_html=True)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")