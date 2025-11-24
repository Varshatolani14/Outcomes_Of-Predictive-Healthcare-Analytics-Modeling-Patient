import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle
import os

df = pd.read_csv("data/processed/merged_data.csv")
df = df.dropna()

X = df.drop(columns=["outcome", "patient_id"], errors="ignore")
y = df["outcome"].astype(int)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_pca, y)

# Grid Search
params = {
    "n_estimators": [200, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [4, 6]
}

model = XGBClassifier(random_state=42)
grid = GridSearchCV(model, params, cv=3, scoring="roc_auc", n_jobs=-1)

grid.fit(X_res, y_res)

print("Best params:", grid.best_params_)

os.makedirs("models", exist_ok=True)
pickle.dump(grid.best_estimator_, open("models/refined_model.pkl", "wb"))

print("✅ Refined model saved.")
