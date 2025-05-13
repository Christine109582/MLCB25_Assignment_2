import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
import optuna
from collections import defaultdict

# Ignore warnings (e.g., from Optuna or sklearn)
warnings.filterwarnings("ignore")

class RepeatedNestedCV:
    def __init__(self, X, y, estimators, param_spaces, R=10, N=5, K=3, random_state=42):
        self.X = X
        self.y = y
        self.estimators = estimators
        self.param_spaces = param_spaces
        self.R = R  # Number of repetitions
        self.N = N  # Outer CV folds
        self.K = K  # Inner CV folds (for tuning)
        self.random_state = random_state
        self.results = defaultdict(list)

    def _evaluate(self, y_true, y_pred):
        return {
            'AUC': roc_auc_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'BA': balanced_accuracy_score(y_true, y_pred)
        }

    def _optimize(self, estimator, param_space, X_train, y_train):
        def objective(trial):
            # Define hyperparameters for trial
            params = {
                key: trial.suggest_categorical(key, values) if isinstance(values[0], str)
                else trial.suggest_float(key, *values)
                for key, values in param_space.items()
            }

            model = clone(estimator).set_params(**params)
            inner_cv = StratifiedKFold(n_splits=self.K, shuffle=True, random_state=self.random_state)
            scores = []

            for train_idx, val_idx in inner_cv.split(X_train, y_train):
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t, y_v = y_train[train_idx], y_train[val_idx]

                # Scaling ONLY within training fold (to avoid data leakage)
                scaler = StandardScaler()
                X_t_scaled = scaler.fit_transform(X_t)
                X_v_scaled = scaler.transform(X_v)

                model.fit(X_t_scaled, y_t)
                preds = model.predict(X_v_scaled)
                scores.append(f1_score(y_v, preds))

            return np.mean(scores)

        # Run Optuna optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        return study.best_params

    def run(self):
        for name, estimator in self.estimators.items():
            print(f"Running for model: {name}")
            for r in range(self.R):
                outer_cv = StratifiedKFold(n_splits=self.N, shuffle=True, random_state=self.random_state + r)

                for train_idx, test_idx in outer_cv.split(self.X, self.y):
                    X_train, X_test = self.X[train_idx], self.X[test_idx]
                    y_train, y_test = self.y[train_idx], self.y[test_idx]

                    # Inner optimization (on training set only)
                    best_params = self._optimize(estimator, self.param_spaces[name], X_train, y_train)
                    model = clone(estimator).set_params(**best_params)

                    # Scaling fitted ONLY on X_train and applied to both X_train and X_test
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    metrics = self._evaluate(y_test, y_pred)
                    self.results[name].append(metrics)

        return self.results
