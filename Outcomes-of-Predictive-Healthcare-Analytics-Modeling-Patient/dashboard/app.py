import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Predictive Healthcare Analytics", layout="wide")

st.title("🩺 Predictive Healthcare Analytics Dashboard")
st.write("Real-time prediction of patient outcomes using ML models.")

# ---------------------------
# 🔍 Load Model Safely
# ---------------------------
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "predictive-healthcare-analytics", "models", "best_model.pkl"))

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ---------------------------
# 🧠 Load Training Feature Columns
# ---------------------------
feature_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "predictive-healthcare-analytics", "data", "processed", "cleaned_data.csv"))

df_train = pd.read_csv(feature_file)
training_columns = list(df_train.drop(columns=["outcome"]).columns)

# ---------------------------
# 📝 User Inputs
# ---------------------------
st.sidebar.header("Enter Patient Details")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
bp = st.sidebar.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
diabetes = st.sidebar.selectbox("Diabetes (1 = Yes, 0 = No)", [0, 1])
cholesterol = st.sidebar.number_input("Cholesterol", min_value=100, max_value=400, value=200)
heart_rate = st.sidebar.number_input("Heart Rate", min_value=40, max_value=150, value=80)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
treatment_type = st.sidebar.selectbox("Treatment Type", ["A", "B", "C"])

# ---------------------------
# 🧩 Prepare Input Row
# ---------------------------
input_df = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "bp": bp,
    "diabetes": diabetes,
    "cholesterol": cholesterol,
    "heart_rate": heart_rate,
    "sex": sex,
    "treatment_type": treatment_type
}])

# 🎯 One-hot encode same as training
input_df = pd.get_dummies(input_df)

# 🎯 Reindex to training columns
input_df = input_df.reindex(columns=training_columns, fill_value=0)

# ---------------------------
# 🧪 Predict Button
# ---------------------------
if st.button("Predict Outcome"):
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("🔮 Prediction Result")
        st.write(f"**Outcome:** `{pred}`")
        st.write(f"**Probability of High Risk:** `{prob:.2f}`")

        if pred == 1:
            st.error("⚠️ Patient is HIGH RISK")
        else:
            st.success("✅ Patient is LOW RISK")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

