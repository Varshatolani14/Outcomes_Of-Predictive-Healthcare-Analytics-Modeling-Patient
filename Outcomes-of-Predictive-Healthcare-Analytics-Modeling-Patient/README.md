
🩺 Predictive Healthcare Analytics — Patient Outcome Prediction

Machine Learning + Deep Learning framework to predict patient outcomes using structured clinical data.
Includes XGBoost, LightGBM, SMOTE, PCA, t-SNE, and a real-time Streamlit dashboard.

🚀 Project Overview

This project builds a complete ML pipeline to predict patient outcomes using demographic, clinical and treatment variables.
It includes:

Data cleaning & preprocessing

Feature engineering

Class balancing (SMOTE)

PCA for dimensionality reduction

t-SNE visualization

Multiple ML model training

Model evaluation

Best-model selection

A real-time Streamlit prediction dashboard

🎯 Problem Statement

Healthcare providers need reliable tools to identify high-risk patients early.
This system predicts whether a patient is likely to experience a negative outcome (0/1), helping hospitals:

Prioritize treatment

Allocate resources

Understand key risk factors

🧠 Features
Data Preprocessing

✔ Missing value handling
✔ One-hot encoding
✔ Normalization
✔ Train/Test split
✔ Cleaned dataset generated automatically

Machine Learning Models

This repo trains & compares:

XGBoost (best model — saved as best_model.pkl)

LightGBM

Random Forest

Logistic Regression

MLP Neural Network

Advanced Techniques
Technique	Used For
SMOTE	Handle class imbalance
PCA	Dimensionality reduction
t-SNE	Visualizing high-dimensional health features
GridSearch CV	Hyperparameter tuning
Feature Importance	Understanding clinical risk factors
🧪 Model Evaluation Metrics

Accuracy

Precision

Recall

F1-score

ROC-AUC

📊 Real-Time Prediction Dashboard

Built using Streamlit, allowing clinicians to input patient details and instantly receive a predicted outcome.

Run the dashboard:
cd dashboard
streamlit run app.py

🗂️ Repository Structure
📁 predictive-healthcare-analytics
│
├── data
│   ├── raw
│   │   └── patient_data.csv
│   └── processed
│       ├── cleaned_data.csv
│       └── merged_data.csv
│
├── models
│   ├── best_model.pkl
│   ├── xgb_model.pkl
│   └── lightgbm_model.pkl
│
├── scripts
│   ├── data_preprocessing.py
│   ├── visualization.py
│   └── (future utils)
│
├── model_training_pipeline.py
├── dashboard
│   └── app.py
│
└── README.md

⚡ How to Run the Entire Project (Baby Steps)
1. Install dependencies
pip install -r requirements.txt

1. Preprocess data
cd scripts
python data_preprocessing.py

1. Train models
cd ..
python model_training_pipeline.py

1. Launch dashboard
cd dashboard
streamlit run app.py

📝 Tech Stack

Python, Pandas, Scikit-learn, XGBoost,
LightGBM, SMOTE, PCA, t-SNE,
Streamlit, Matplotlib/Seaborn

📌 Perfect for Resume

This project demonstrates:

Production-ready ML pipeline

Feature engineering

Handling real-world clinical datasets

Model optimization

Deployment-ready dashboard

🌟 Contributions

Feel free to open issues, PRs, or discussions!