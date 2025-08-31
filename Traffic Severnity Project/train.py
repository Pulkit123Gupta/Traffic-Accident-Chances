# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import os

# ------------------- PATHS -------------------
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "data_accidents.csv")
MODEL_PATH = os.path.join(BASE_DIR, "accident_model.pkl")

# ------------------- FEATURES & TARGET -------------------
FEATURES = [
    "Weather","Road_Type","Road_Condition","Road_Light_Condition","Vehicle_Type",
    "Driver_Age","Driver_Experience","Driver_Alcohol","Speed_Limit",
    "Time_of_Day","Traffic_Density","Number_of_Vehicles"
]
TARGET = "Accident"  # Label column

# ------------------- LOAD DATA -------------------
df = pd.read_csv(CSV_PATH)

# Optional: rename columns if CSV differs
# df = df.rename(columns={"Road Type":"Road_Type", "Driver Age":"Driver_Age", ...})

# Drop missing values
df = df.dropna(subset=FEATURES + [TARGET]).copy()

# Normalize Driver_Alcohol
if df["Driver_Alcohol"].dtype == object:
    df["Driver_Alcohol"] = (
        df["Driver_Alcohol"].astype(str).str.strip().str.lower().map(
            {"yes":1,"y":1,"true":1,"1":1,"no":0,"n":0,"false":0,"0":0}
        ).fillna(0).astype(int)
    )

X = df[FEATURES]
y = df[TARGET]

# ------------------- PREPROCESS -------------------
cat_cols = ["Weather","Road_Type","Road_Condition","Road_Light_Condition","Vehicle_Type","Time_of_Day","Traffic_Density"]
num_cols = ["Driver_Age","Driver_Experience","Driver_Alcohol","Speed_Limit","Number_of_Vehicles"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

pipe = Pipeline([
    ("preprocess", pre),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
])

# ------------------- TRAIN/TEST SPLIT -------------------
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
)
pipe.fit(Xtr, ytr)

# ------------------- SAVE MODEL -------------------
joblib.dump(pipe, MODEL_PATH)
print(f"✅ Model trained and saved as {MODEL_PATH}")
