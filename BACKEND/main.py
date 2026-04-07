from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the saved assets
model = joblib.load("football_model.pkl")
encoder = joblib.load("club-encoder.pkl")

app = FastAPI()

class MatchRequest(BaseModel):
    home_id: int
    away_id: int

@app.post("/predict")
def predict(data: MatchRequest):
    # The API receives raw IDs, converts them to trained indexes, and predicts
    h_idx = encoder.transform([data.home_id])[0]
    a_idx = encoder.transform([data.away_id])[0]
    
    res = model.predict([[h_idx, a_idx]])[0]
    
    mapping = {1: "Home Win", 0: "Draw", 2: "Away Win"}
    return {"prediction": mapping[res]}