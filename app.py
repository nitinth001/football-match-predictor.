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

# 2. THE STYLES (Using st.html for Python 3.14 Stability)
# We define the styles here and inject them using a more direct method.
custom_styles = """
<style>
    /* Dark Theme & Typography */
    [data-testid="stAppViewContainer"] { 
        background-color: #0c0f13; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    }
    
    /* Header Bar */
    .header-card {
        background: linear-gradient(90deg, #1f2a3c, #0c0f13);
        border-left: 8px solid #00f901;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        color: white;
    }

    /* Input Containers */
    div[data-testid="stColumn"] {
        background-color: #1a1e26;
        border: 1px solid #333;
        border-radius: 15px;
        padding: 25px;
    }

    /* Labels and Selectbox styling */
    label { color: #00f901 !important; font-weight: bold !important; }
    
    /* Predict Button Styling */
    .stButton > button {
        background-color: #00f901 !important;
        color: #000 !important;
        font-weight: 800 !important;
        border-radius: 30px !important;
        border: none !important;
        padding: 15px !important;
        width: 100% !important;
        text-transform: uppercase;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #ffffff !important;
        transform: scale(1.02);
    }
</style>
"""
st.html(custom_styles) # This is the "Safe" way for Python 3.14

# 3. HEADER UI
st.markdown("""
<div class="header-card">
    <h1 style="margin:0; font-size: 2.5rem;">⚽ MATCH DAY PREDICTOR</h1>
    <p style="margin:0; color: #888;">AI Analysis • Prediction Engine • v1.0</p>
</div>
""", unsafe_allow_with_html=True)

# 4. ASSET LOADING (Robust Error Handling)
@st.cache_resource
def load_assets():
    base = os.path.dirname(__file__)
    files = {
        "model": os.path.join(base, "football_model.pkl"),
        "encoder": os.path.join(base, "club-encoder.pkl"),
        "map": os.path.join(base, "name_map.pkl")
    }
    
    try:
        m = joblib.load(files["model"])
        e = joblib.load(files["encoder"])
        nm = joblib.load(files["map"])
        return m, e, nm
    except Exception as err:
        st.error(f"⚠️ Deployment Error: Could not find model files in {base}")
        return None, None, None

model, encoder, name_map = load_assets()

# 5. APP LOGIC
if model:
    # Prepare team list
    club_to_id = {name: id for id, name in name_map.items()}
    club_list = sorted(list(club_to_id.keys()))

    # Layout Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 HOME TEAM")
        home_team = st.selectbox("Choose Host", club_list, key="home_select")
        
    with col2:
        st.markdown("### ✈️ AWAY TEAM")
        away_team = st.selectbox("Choose Visitor", club_list, key="away_select")

    st.markdown("<br>", unsafe_allow_with_html=True)

    # 6. PREDICTION TRIGGER
    if st.button("RUN PREDICTION ENGINE"):
        try:
            # Data Processing
            h_id = int(club_to_id[home_team])
            a_id = int(club_to_id[away_team])
            h_idx = encoder.transform([h_id])[0]
            a_idx = encoder.transform([a_id])[0]
            
            # Inference
            prediction = model.predict([[h_idx, a_idx]])[0]
            outcomes = {1: "HOME WIN", 0: "DRAW", 2: "AWAY WIN"}
            result_text = outcomes[prediction]
            
            # Result Display
            st.markdown(f"""
                <div style="background-color: #00f901; padding: 25px; border-radius: 12px; text-align: center; margin-top: 20px;">
                    <span style="color: black; font-weight: 900; font-size: 1.2rem;">RESULT FORECAST</span>
                    <h1 style="color: black; margin: 0; font-size: 3.5rem;">{result_text}</h1>
                </div>
            """, unsafe_allow_with_html=True)
            
        except Exception as e:
            st.error(f"Prediction logic error: {e}")