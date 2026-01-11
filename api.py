from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

# Load data + model once at startup (fast)
df = pd.read_csv(BASE_DIR / "processed_fifa_df.csv")

with open(BASE_DIR / "knn_footballer_model.pkl", "rb") as f:
    knn, scaler, le_positions, le_national_team, feature_columns = pickle.load(f)

X_all = df[feature_columns]
X_all_scaled = scaler.transform(X_all)

app = FastAPI(title="Footballer Similarity API")

class Query(BaseModel):
    positions: str
    national_team: str
    preferred_foot: str  # "Left" or "Right"
    weak_foot: int
    overall_rating: float
    value_euro: float
    national_jersey_number: float
    height_weight_ratio: float
    winger_ability: float
    striker_ability: float
    midfielder_ability: float
    speed: float
    mental_physical_strength: float
    defensive_ability: float
    set_pieces: float
    top_k: int = 10

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(q: Query):
    positions_val = int(le_positions.transform([q.positions])[0])
    national_team_val = int(le_national_team.transform([q.national_team])[0])
    preferred_foot_val = 0 if q.preferred_foot == "Left" else 1

    user_dict = {
        "positions": positions_val,
        "overall_rating": q.overall_rating,
        "value_euro": q.value_euro,
        "preferred_foot": preferred_foot_val,
        "weak_foot(1-5)": q.weak_foot,
        "national_team": national_team_val,
        "national_jersey_number": q.national_jersey_number,
        "height_weight_ratio": q.height_weight_ratio,
        "winger_ability": q.winger_ability,
        "striker_ability": q.striker_ability,
        "midfielder_ability": q.midfielder_ability,
        "speed": q.speed,
        "mental_physical_strength": q.mental_physical_strength,
        "defensive_ability": q.defensive_ability,
        "set_pieces": q.set_pieces,
    }

    user_df = pd.DataFrame([user_dict])
    user_df = user_df[feature_columns]

    user_scaled = scaler.transform(user_df)

    k = int(q.top_k)
    if k < 1:
        k = 1
    if k > 50:
        k = 50

    distances, indices = knn.kneighbors(user_scaled, n_neighbors=k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[int(idx)]
        results.append({
            "name": row["name"],
            "overall_rating": float(row["overall_rating"]),
            "distance": float(dist),
        })

    return {
        "query": q.model_dump(),
        "results": results
    }
