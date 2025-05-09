
README: Usage Instructions for Breast Cancer Classification Model

This folder contains the final trained model and preprocessing components 
used for breast cancer diagnosis classification.

Files:
-------
1. final_model_rf.pkl
   - Trained RandomForestClassifier with class_weight='balanced'
   - Trained on the full breast_cancer.csv dataset

2. final_model_scaler.pkl
   - StandardScaler used to normalize the input features

Usage Example (Python):
------------------------
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load scaler and model
scaler = joblib.load("final_model_scaler.pkl")
model = joblib.load("final_model_rf.pkl")

# Example: loading new data
new_data = pd.read_csv("new_patient_data.csv")  # Make sure it has 30 features
X_new_scaled = scaler.transform(new_data)
predictions = model.predict(X_new_scaled)

# Output diagnosis: 0 = Benign, 1 = Malignant

Notes:
-------
- This model expects all 30 numeric features from the original dataset.
- Features must be preprocessed exactly as during training (median imputation, scaling).
- RandomForest was chosen as the best-performing model after nested cross-validation.

Author: [Your Name]
