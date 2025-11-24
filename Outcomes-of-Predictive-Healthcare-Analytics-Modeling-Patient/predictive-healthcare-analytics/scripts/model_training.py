import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "data", "processed", "cleaned_data.csv")
)

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
)

df = pd.read_csv(DATA_PATH)

# REQUIRED columns (same as app.py)
required = [
    'age','bmi','bp','cholesterol','heart_rate',
    'diabetes','sex_Male','treatment_type_B','treatment_type_C'
]

# Select features
X = df[required]
y = df["outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric='logloss'
    ))
])

model.fit(X_train, y_train)

# Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("Model trained & saved at:", MODEL_PATH)
