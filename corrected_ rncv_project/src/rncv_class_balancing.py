#run_class_balancing.py

from rncv_pipeline import RepeatedNestedCV
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_csv("breast_cancer.csv")
df = df.drop(columns=["id"])
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop(columns=['diagnosis']).values
y = df['diagnosis'].values

# Estimators for SMOTE pipeline (no need for class_weight)
estimators_smote = {
    "LogisticRegression": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver='saga', max_iter=10000))
    ]),
    "GaussianNB": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ]),
    "LDA": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis())
    ]),
    "SVM": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("clf", SVC())
    ]),
    "RandomForest": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier())
    ]),
    "LightGBM": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("clf", lgb.LGBMClassifier())
    ])
}

# Estimators with class_weight
estimators_weighted = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver='saga', max_iter=10000, class_weight='balanced'))
    ]),
    "GaussianNB": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ]),
    "LDA": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis())
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(class_weight='balanced'))
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight='balanced'))
    ]),
    "LightGBM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", lgb.LGBMClassifier(class_weight='balanced'))
    ])
}

# Param spaces for classifiers (only for inner CV)
param_spaces = {
    "LogisticRegression": {
        "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "clf__l1_ratio": [0.0, 0.5, 1.0],
        "clf__penalty": ["elasticnet"]
    },
    "GaussianNB": {},
    "LDA": {},
    "SVM": {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"]
    },
    "RandomForest": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 10, None]
    },
    "LightGBM": {
        "clf__num_leaves": [15, 31, 63],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__n_estimators": [50, 100, 200]
    }
}

# Run rnCV with SMOTE
print("Running with SMOTE pipeline:")
rncv_smote = RepeatedNestedCV(X, y, estimators_smote, param_spaces, R=10, N=5, K=3, random_state=42)
results_smote = rncv_smote.run()

for model, metrics_list in results_smote.items():
    print(f"\nModel (SMOTE): {model}")
    for metric in ['AUC', 'F1', 'MCC', 'BA']:
        values = [m[metric] for m in metrics_list]
        print(f"{metric} - Median: {np.median(values):.3f}, Mean: {np.mean(values):.3f}, Std: {np.std(values):.3f}")

# Run rnCV with class weights
print("\nRunning with class weights pipeline:")
rncv_weighted = RepeatedNestedCV(X, y, estimators_weighted, param_spaces, R=10, N=5, K=3, random_state=42)
results_weighted = rncv_weighted.run()

for model, metrics_list in results_weighted.items():
    print(f"\nModel (Class Weights): {model}")
    for metric in ['AUC', 'F1', 'MCC', 'BA']:
        values = [m[metric] for m in metrics_list]
        print(f"{metric} - Median: {np.median(values):.3f}, Mean: {np.mean(values):.3f}, Std: {np.std(values):.3f}")
