
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from rncv_pipeline import RepeatedNestedCV
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# Load and preprocess data
df = pd.read_csv("breast_cancer.csv")
df = df.drop(columns=["id"])
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection Methods
skb_f = SelectKBest(score_func=f_classif, k=10)
skb_f.fit(X_scaled, y)
features_f = X.columns[skb_f.get_support()].tolist()

skb_mi = SelectKBest(score_func=mutual_info_classif, k=10)
skb_mi.fit(X_scaled, y)
features_mi = X.columns[skb_mi.get_support()].tolist()

rfe = RFE(LogisticRegression(max_iter=10000, solver='liblinear'), n_features_to_select=10)
rfe.fit(X_scaled, y)
features_rfe = X.columns[rfe.get_support()].tolist()

print("Selected features:")
print("F-test:", features_f)
print("Mutual Info:", features_mi)
print("RFE:", features_rfe)

# Choose which set to use for rnCV
selected_features = features_f  # Change to features_mi or features_rfe if needed

X_selected = df[selected_features].values
X_scaled_sel = scaler.fit_transform(X_selected)

# Define estimators and param spaces
estimators = {
    "LogisticRegression": LogisticRegression(solver='saga', max_iter=10000),
    "GaussianNB": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier(),
    "LightGBM": lgb.LGBMClassifier()
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

# Run rnCV
rncv = RepeatedNestedCV(X_scaled_sel, y, estimators, param_spaces, R=10, N=5, K=3, random_state=42)
results = rncv.run()

# Print results
for model, metrics_list in results.items():
    print(f"\nModel ({model})")
    for metric in ['AUC', 'F1', 'MCC', 'BA']:
        values = [m[metric] for m in metrics_list]
        print(f"{metric} - Median: {np.median(values):.3f}, Mean: {np.mean(values):.3f}, Std: {np.std(values):.3f}")
