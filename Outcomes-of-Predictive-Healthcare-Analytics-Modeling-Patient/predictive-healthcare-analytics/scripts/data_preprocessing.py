import pandas as pd
import os

INPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "raw", "patient_data.csv")
)

OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_data.csv")
)

# Load raw data
df = pd.read_csv(INPUT_PATH)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=["sex", "treatment_type"], drop_first=True)

# Fill missing values (if any)
df = df.fillna(df.median(numeric_only=True))

# Save processed file
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("Cleaned dataset saved at:", OUTPUT_PATH)
