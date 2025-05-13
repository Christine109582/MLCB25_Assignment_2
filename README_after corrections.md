 # Repeated Nested Cross-Validation Framework

A clean, reusable implementation of repeated nested cross-validation (RNCV) for classification, with:

- Scikit-learn pipelines
- Hyperparameter tuning with Optuna
- Feature selection (f-test, mutual information, RFE)
- Handling class imbalance (SMOTE, class weights)

## Structure

- `src/rncv_pipeline.py`: Core RNCV logic
- `src/run.py`: Main pipeline execution
- `src/feature_selection_rncv.py`: Feature selection comparisons
- `src/rncv_class_imbalance.py`: Handling class imbalance

## Requirements

```bash
pip install -r requirements.txt
