"""
Configuration settings for Banking Fraud Detection project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'raw_data_path': RAW_DATA_DIR / 'banking_transactions.csv',
    'processed_data_path': PROCESSED_DATA_DIR / 'fraud_data_processed.csv',
    'target_column': 'fraud_flag',
    'test_size': 0.2,
    'random_state': 42,
}

# Feature configuration
FEATURE_CONFIG = {
    'numerical_features': [
        'transaction_amount',
        'login_attempts',
        'device_risk_score',
        'transfer_frequency',
        'anomaly_score',
        'account_age_days',
        'transaction_time_hour',
        'failed_transactions_last_30d',
        'avg_monthly_balance'
    ],
    'features_to_drop': ['transaction_id'],
}

# Model configuration
MODEL_CONFIG = {
    'random_state': 42,
    'cv_folds': 5,
    'test_size': 0.2,
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced',
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 7,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'scale_pos_weight': 1,  # Adjust if class imbalance detected
        'eval_metric': 'logloss',
    },
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': 42,
        'solver': 'lbfgs',
    }
}

# Model paths
MODELS = {
    'best_model_path': MODELS_DIR / 'best_model.pkl',
    'model_metadata_path': MODELS_DIR / 'model_metadata.json',
}

# Results paths
RESULTS = {
    'model_performance_path': RESULTS_DIR / 'model_performance.json',
    'confusion_matrix_path': RESULTS_DIR / 'confusion_matrix.png',
    'roc_curve_path': RESULTS_DIR / 'roc_curve.png',
    'feature_importance_path': RESULTS_DIR / 'feature_importance.png',
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    'handle_outliers': True,
    'outlier_method': 'iqr',  # 'iqr' or 'zscore'
    'iqr_multiplier': 1.5,
    'zscore_threshold': 3,
    'normalize_features': True,
    'scaler_type': 'standard',  # 'standard' or 'minmax'
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'plot_roc_curve': True,
    'plot_confusion_matrix': True,
    'plot_feature_importance': True,
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': PROJECT_ROOT / 'logs' / 'project.log',
}

# Hyperparameter tuning configuration
HYPERPARAMETER_TUNING = {
    'enabled': False,  # Set to True to enable grid search
    'method': 'grid',  # 'grid' or 'random'
    'random_forest_params': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
    },
    'xgboost_params': {
        'n_estimators': [100, 200],
        'max_depth': [5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.3],
    }
}
