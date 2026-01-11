# app.py
# Run: streamlit run app.py

import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Which Pro Footballer Am I?", layout="wide")

st.title("Which Pro Footballer Am I?")
st.write("This app finds the most similar footballers using KNN (nearest neighbors).")

# ---------------------------
# Load model bundle
# ---------------------------
@st.cache_resource
def load_bundle(pkl_path: str):
    with open(pkl_path, "rb") as f:
        knn, scaler, le_positions, le_national_team, feature_columns = pickle.load(f)
    return knn, scaler, le_positions, le_national_team, list(feature_columns)

knn, scaler, le_positions, le_national_team, feature_columns = load_bundle("knn_footballer_model.pkl")

# ---------------------------
# Load the same processed df you used for training
# IMPORTANT: this must match your training rows and order
# ---------------------------
@st.cache_data
def load_processed_df(csv_path: str):
    df = pd.read_csv(r'C:\Users\Acer\Developer\football\processed_fifa_df.csv')

    # IMPORTANT:
    # Your CSV here MUST already be the final processed dataframe that contains:
    # name, positions (encoded), national_team (encoded), preferred_foot (0/1), engineered ability columns, etc.
    # If your raw fifa_players.csv is not processed, export your final df to a new CSV and use that here.

    return df

st.sidebar.header("Data")
csv_path = st.sidebar.text_input(
    "Path to your processed CSV (same df used for training)",
    value=r"C:\Users\Acer\Developer\football\processed_fifa_df.csv"
)

df = load_processed_df(csv_path)

# Safety checks
missing_cols = [c for c in (["name"] + feature_columns) if c not in df.columns]
if len(missing_cols) > 0:
    st.error("Your CSV is missing required columns:")
    st.write(missing_cols)
    st.stop()

X_all = df[feature_columns]
X_all_scaled = scaler.transform(X_all)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Your Inputs")

top_k = st.sidebar.slider("How many matches to return?", min_value=3, max_value=20, value=10)

# Dropdowns using encoders (so user selects real labels)
positions_label = st.sidebar.selectbox("Position (from training encoder)", list(le_positions.classes_))
national_team_label = st.sidebar.selectbox("National team (from training encoder)", list(le_national_team.classes_))

# Preferred foot (your encoding: Left=0, Right=1)
preferred_foot_label = st.sidebar.selectbox("Preferred foot", ["Left", "Right"])
preferred_foot_val = 0 if preferred_foot_label == "Left" else 1

# Weak foot slider (exists in your features)
weak_foot_val = st.sidebar.slider("Weak foot (1-5)", 1, 5, 3)

# Numeric sliders with safe defaults based on your dataset ranges
def slider_from_col(col_name: str, label: str, default=None, step=1.0):
    col = df[col_name].astype(float)
    col_min = float(np.nanmin(col.values))
    col_max = float(np.nanmax(col.values))
    if default is None:
        default = float(np.nanmedian(col.values))
    return st.sidebar.slider(label, min_value=col_min, max_value=col_max, value=float(default), step=float(step))

overall_rating = slider_from_col("overall_rating", "Overall rating", step=1.0)
value_euro = slider_from_col("value_euro", "Value (euro)", step=100000.0)
national_jersey_number = slider_from_col("national_jersey_number", "National jersey number", step=1.0)

height_weight_ratio = slider_from_col("height_weight_ratio", "Height/Weight ratio", step=0.01)

winger_ability = slider_from_col("winger_ability", "Winger ability", step=0.1)
striker_ability = slider_from_col("striker_ability", "Striker ability", step=0.1)
midfielder_ability = slider_from_col("midfielder_ability", "Midfielder ability", step=0.1)
speed = slider_from_col("speed", "Speed", step=0.1)
mental_physical_strength = slider_from_col("mental_physical_strength", "Mental/Physical strength", step=0.1)
defensive_ability = slider_from_col("defensive_ability", "Defensive ability", step=0.1)
set_pieces = slider_from_col("set_pieces", "Set pieces", step=0.1)

# ---------------------------
# Build user row in correct feature order
# ---------------------------
positions_val = int(le_positions.transform([positions_label])[0])
national_team_val = int(le_national_team.transform([national_team_label])[0])

user_dict = {
    "positions": positions_val,
    "overall_rating": overall_rating,
    "value_euro": value_euro,
    "preferred_foot": preferred_foot_val,
    "weak_foot(1-5)": weak_foot_val,
    "national_team": national_team_val,
    "national_jersey_number": national_jersey_number,
    "height_weight_ratio": height_weight_ratio,
    "winger_ability": winger_ability,
    "striker_ability": striker_ability,
    "midfielder_ability": midfielder_ability,
    "speed": speed,
    "mental_physical_strength": mental_physical_strength,
    "defensive_ability": defensive_ability,
    "set_pieces": set_pieces,
}

# Ensure all needed columns exist in user_dict
for col in feature_columns:
    if col not in user_dict:
        st.error(f"Missing feature in user_dict: {col}. Add it to the app inputs.")
        st.stop()

user_df = pd.DataFrame([user_dict])
user_df = user_df[feature_columns]

# ---------------------------
# Predict neighbors
# ---------------------------
if st.button("Find my footballer matches"):
    user_scaled = scaler.transform(user_df)

    distances, indices = knn.kneighbors(user_scaled, n_neighbors=top_k)

    neighbor_idx = indices[0]
    neighbor_dist = distances[0]

    results = df.loc[neighbor_idx, ["name", "overall_rating", "positions", "preferred_foot"]].copy()
    results["distance"] = neighbor_dist

    # Decode positions back to labels for display
    results["positions_label"] = le_positions.inverse_transform(results["positions"].astype(int).values)

    # Decode preferred foot for display
    results["preferred_foot_label"] = results["preferred_foot"].map({0: "Left", 1: "Right"})

    results = results[["name", "overall_rating", "positions_label", "preferred_foot_label", "distance"]]

    st.subheader("Top matches")
    st.dataframe(results, use_container_width=True)

    st.subheader("Your input row (debug)")
    st.dataframe(user_df, use_container_width=True)

st.markdown("---")
st.caption("Tip: If the CSV you load is not your final processed df, export your processed df to a new CSV and point the app to it.")
