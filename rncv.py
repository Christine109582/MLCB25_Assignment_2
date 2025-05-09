import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.base import clone
import optuna
from collections import defaultdict


class RepeatedNestedCV:
    def __init__(self, X, y, estimators, param_spaces, R=10, N=5, K=3, random_state=42):
        self.X = X
        self.y = y
        self.estimators = estimators
        self.param_spaces = param_spaces
        self.R = R
        self.N = N
        self.K = K
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
            params = {key: trial.suggest_categorical(key, values) if isinstance(values[0], str)
            else trial.suggest_float(key, *values) for key, values in param_space.items()}
            model = clone(estimator).set_params(**params)
            inner_cv = StratifiedKFold(n_splits=self.K, shuffle=True, random_state=self.random_state)
            scores = []
            for train_idx, val_idx in inner_cv.split(X_train, y_train):
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t, y_v = y_train[train_idx], y_train[val_idx]
                model.fit(X_t, y_t)
                preds = model.predict(X_v)
                scores.append(f1_score(y_v, preds))
            return np.mean(scores)

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

                    best_params = self._optimize(estimator, self.param_spaces[name], X_train, y_train)
                    model = clone(estimator).set_params(**best_params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = self._evaluate(y_test, y_pred)
                    self.results[name].append(metrics)
        return self.results
