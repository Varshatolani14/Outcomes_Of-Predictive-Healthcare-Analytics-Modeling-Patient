import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import pickle
import os

df = pd.read_csv("data/processed/cleaned_data.csv")

X = df.drop("outcome", axis=1)
y = df["outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)
lgb_model.fit(X_train, y_train)
lgb_acc = accuracy_score(y_test, lgb_model.predict(X_test))

if xgb_acc >= lgb_acc:
    best_model = xgb_model
else:
    best_model = lgb_model

os.makedirs("models", exist_ok=True)

pickle.dump(xgb_model, open("models/xgb_model.pkl", "wb"))
pickle.dump(lgb_model, open("models/lightgbm_model.pkl", "wb"))
pickle.dump(best_model, open("models/best_model.pkl", "wb"))

print("Models saved successfully!")
