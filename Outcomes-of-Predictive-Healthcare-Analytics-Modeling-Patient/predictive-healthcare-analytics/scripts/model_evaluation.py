import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pickle

# Load model
model = pickle.load(open("models/best_model.pkl", "rb"))

# Load data
df = pd.read_csv("data/processed/merged_data.csv")
df = df.dropna()

y = df["outcome"]
X = df.drop(columns=["outcome", "patient_id"], errors="ignore")

# Predictions
y_prob = model.predict_proba(X)[:, 1]
y_pred = model.predict(X)

# Report
print("\nClassification Report:\n")
print(classification_report(y, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y, y_pred))

# ROC
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature importance
if hasattr(model, "feature_importances_"):
    imp = model.feature_importances_
    feats = X.columns
    for f, v in sorted(zip(feats, imp), key=lambda x:x[1], reverse=True):
        print(f"{f}: {v:.4f}")
