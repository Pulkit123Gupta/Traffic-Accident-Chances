# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

CSV_PATH = "data_accidents.csv"
MODEL_PATH = "accident_model.pkl"

# Features must match the app exactly
FEATURES = [
    "Weather","Road_Type","Road_Condition","Road_Light_Condition","Vehicle_Type",
    "Driver_Age","Driver_Experience","Driver_Alcohol","Speed_Limit",
    "Time_of_Day","Traffic_Density","Number_of_Vehicles"
]
TARGET = "Accident"  # change if your label is named differently

df = pd.read_csv(CSV_PATH)

# If your CSV column names differ, uncomment and edit the mapping:
# df = df.rename(columns={
#     "Road Type":"Road_Type",
#     "Road Condition":"Road_Condition",
#     "Road Light Condition":"Road_Light_Condition",
#     "Driver Age":"Driver_Age",
#     "Driver Experience":"Driver_Experience",
#     "Driver Alcohol":"Driver_Alcohol",
#     "Speed Limit":"Speed_Limit",
#     "Time of Day":"Time_of_Day",
#     "Traffic Density":"Traffic_Density",
#     "Number of Vehicles":"Number_of_Vehicles",
#     # If target differs:
#     # "Accident_Severity":"Accident",
# })

# Basic cleaning
df = df.dropna(subset=FEATURES + [TARGET]).copy()

# Normalize Driver_Alcohol to 0/1 if it’s text
if df["Driver_Alcohol"].dtype == object:
    df["Driver_Alcohol"] = (
        df["Driver_Alcohol"].astype(str).str.strip().str.lower().map(
            {"yes":1,"y":1,"true":1,"1":1,"no":0,"n":0,"false":0,"0":0}
        ).fillna(0).astype(int)
    )

X = df[FEATURES]
y = df[TARGET]

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

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
)
pipe.fit(Xtr, ytr)

joblib.dump(pipe, MODEL_PATH)
print(f"✅ Model trained and saved as {MODEL_PATH}")
