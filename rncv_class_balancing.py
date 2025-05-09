from rncv_pipeline import RepeatedNestedCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight

# Load and preprocess data
df = pd.read_csv("/mnt/data/breast_cancer.csv")
df = df.drop(columns=["id"])
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop(columns=['diagnosis']).values
y = df['diagnosis'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE balancing
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

# Class weights for classifiers that support it
sample_weights = compute_sample_weight(class_weight='balanced', y=y)

estimators = {
    "LogisticRegression": LogisticRegression(solver='saga', max_iter=10000, class_weight='balanced'),
    "GaussianNB": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": SVC(class_weight='balanced'),
    "RandomForest": RandomForestClassifier(class_weight='balanced'),
    "LightGBM": lgb.LGBMClassifier(class_weight='balanced')
}

param_spaces = {
    "LogisticRegression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.0, 0.5, 1.0],
        "penalty": ["elasticnet"]
    },
    "GaussianNB": {},
    "LDA": {},
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None]
    },
    "LightGBM": {
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [50, 100, 200]
    }
}

# Run rnCV with SMOTE-balanced data
print("Running with SMOTE balanced data:")
rncv_smote = RepeatedNestedCV(X_bal, y_bal, estimators, param_spaces, R=10, N=5, K=3, random_state=42)
results_smote = rncv_smote.run()

for model, metrics_list in results_smote.items():
    print(f"\nModel (SMOTE): {model}")
    for metric in ['AUC', 'F1', 'MCC', 'BA']:
        values = [m[metric] for m in metrics_list]
        print(f"{metric} - Median: {np.median(values):.3f}, Mean: {np.mean(values):.3f}, Std: {np.std(values):.3f}")

# Run rnCV with original data + class weights
print("\nRunning with class weights on original data:")
rncv_weighted = RepeatedNestedCV(X_scaled, y, estimators, param_spaces, R=10, N=5, K=3, random_state=42)
results_weighted = rncv_weighted.run()

for model, metrics_list in results_weighted.items():
    print(f"\nModel (Class Weights): {model}")
    for metric in ['AUC', 'F1', 'MCC', 'BA']:
        values = [m[metric] for m in metrics_list]
        print(f"{metric} - Median: {np.median(values):.3f}, Mean: {np.mean(values):.3f}, Std: {np.std(values):.3f}")
