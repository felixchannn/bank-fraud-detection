# Application Guide: Implementing the Project

## Overview

This guide explains how to apply machine learning concepts to build a fraud detection system. It progresses from raw data to production-ready model.

## Phase 1: Data Loading and Validation

### Objective

Load raw data and validate it meets quality standards.

### Implementation

Create src/data_loader.py:

```python
import pandas as pd
import numpy as np
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load banking transactions dataset from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame containing all transactions and features
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is invalid
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate data quality and return report.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary containing validation report
    """
    validation_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'class_distribution': df['fraud_flag'].value_counts().to_dict()
    }
    
    # Print validation results
    print(f"Dataset Shape: {validation_report['shape']}")
    print(f"Missing Values: {sum(validation_report['missing_values'].values())}")
    print(f"Duplicate Rows: {validation_report['duplicates']}")
    print(f"Fraud Distribution: {validation_report['class_distribution']}")
    
    return validation_report
```

### Application in Notebooks

In notebooks/01_eda.ipynb:

```python
from src.data_loader import load_data, validate_data

# Load data
df = load_data('data/raw/banking_transactions.csv')

# Validate data quality
report = validate_data(df)

# Display summary statistics
print(df.describe())

# Examine feature correlations
correlation_matrix = df.corr()
fraud_correlations = correlation_matrix['fraud_flag'].sort_values(ascending=False)
print(fraud_correlations)
```

Key validation checks:
- Row count and column count reasonable
- No significant missing values
- No unexpected duplicates
- Data types correct for each column
- Class distribution shows both fraud and legitimate transactions

## Phase 2: Data Preprocessing

### Objective

Clean data and prepare it for model training by removing outliers, handling anomalies, and normalizing distributions.

### Implementation

Create src/preprocessing.py:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

class DataPreprocessor:
    """Handles all data cleaning and preparation operations."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.outlier_indices = None
        
    def remove_outliers_iqr(self, df: pd.DataFrame, 
                            multiplier: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using Interquartile Range method.
        
        Args:
            df: DataFrame to process
            multiplier: IQR multiplier (typically 1.5)
            
        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Identify and record outliers
            outlier_mask = (df_clean[column] < lower_bound) | \
                          (df_clean[column] > upper_bound)
            
            if outlier_mask.sum() > 0:
                print(f"{column}: Removed {outlier_mask.sum()} outliers")
            
            df_clean = df_clean[~outlier_mask]
        
        return df_clean
    
    def prepare_train_test(self, X: pd.DataFrame, y: pd.Series,
                          test_size: float = 0.2) -> Tuple:
        """
        Split data into train and test sets with stratification.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion for test set (0.2 = 20%)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Train fraud rate: {y_train.mean():.2%}")
        print(f"Test fraud rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame,
                      X_test: pd.DataFrame) -> Tuple:
        """
        Standardize features for algorithms requiring it.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def preprocess(self, df: pd.DataFrame, 
                   target_column: str = 'fraud_flag',
                   drop_columns: list = ['transaction_id']) -> Tuple:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            target_column: Name of target variable
            drop_columns: Columns to remove
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Remove unnecessary columns
        df = df.drop(drop_columns, axis=1)
        
        # Remove outliers
        df = self.remove_outliers_iqr(df)
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = self.prepare_train_test(X, y)
        
        # Scale features (for Logistic Regression)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
```

### Application in Notebooks

In notebooks/02_data_preprocessing.ipynb:

```python
from src.preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(random_state=42)

# Apply complete preprocessing
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = \
    preprocessor.preprocess(df)

# Verify results
print("Preprocessing complete")
print(f"Training samples: {X_train.shape[0]}")
print(f"Features per sample: {X_train.shape[1]}")
```

Key preprocessing decisions:
- Outlier removal improves model robustness
- Stratified split ensures representative test set
- Scaling required for distance-based algorithms
- All preprocessing fit on training data only

## Phase 3: Feature Engineering

### Objective

Create new features from raw features to improve predictive power.

### Implementation

Create src/feature_engineering.py:

```python
import pandas as pd
import numpy as np

class FeatureEngineer:
    """Creates domain-specific features for fraud detection."""
    
    @staticmethod
    def create_risk_score(X: pd.DataFrame) -> pd.Series:
        """
        Create composite risk score from multiple indicators.
        
        Combines device risk, anomaly detection, and login attempts
        with learned weights.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Risk score Series
        """
        risk_score = (
            X['device_risk_score'] * 0.4 +
            X['anomaly_score'] * 100 * 0.4 +
            X['login_attempts'] * 10 * 0.2
        )
        return risk_score
    
    @staticmethod
    def create_velocity_features(X: pd.DataFrame) -> pd.DataFrame:
        """
        Create features measuring transaction velocity.
        
        Unusual activity patterns (high frequency relative to account age)
        indicate potential fraud.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with new velocity features
        """
        X_new = X.copy()
        
        # Transaction velocity
        X_new['transaction_velocity'] = (
            X['transfer_frequency'] / (X['account_age_days'] + 1)
        )
        
        # Failed transaction rate
        X_new['failed_transaction_ratio'] = (
            X['failed_transactions_last_30d'] / (X['transfer_frequency'] + 1)
        )
        
        return X_new
    
    @staticmethod
    def create_balance_features(X: pd.DataFrame) -> pd.DataFrame:
        """
        Create features relating transaction size to account balance.
        
        Unusual amounts relative to account balance indicate suspicious activity.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with new balance features
        """
        X_new = X.copy()
        
        # Balance adequacy
        X_new['balance_to_transaction_ratio'] = (
            X['avg_monthly_balance'] / (X['transaction_amount'] + 1)
        )
        
        return X_new
    
    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            X: Original features DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        X_engineered = X.copy()
        
        # Add composite risk score
        X_engineered['risk_activity_score'] = self.create_risk_score(X)
        
        # Add velocity features
        X_engineered = self.create_velocity_features(X_engineered)
        
        # Add balance features
        X_engineered = self.create_balance_features(X_engineered)
        
        return X_engineered
```

### Application in Notebooks

In notebooks/03_feature_engineering.ipynb:

```python
from src.feature_engineering import FeatureEngineer

# Initialize engineer
engineer = FeatureEngineer()

# Create engineered features
X_train_engineered = engineer.engineer_features(X_train)
X_test_engineered = engineer.engineer_features(X_test)

# Check new features created
new_features = set(X_train_engineered.columns) - set(X_train.columns)
print(f"New features created: {new_features}")

# Verify improved correlation with fraud
print(X_train_engineered[list(new_features) + ['fraud_flag']].corr())
```

Feature engineering rationale:
- Domain knowledge: Features based on how fraud actually occurs
- Signal creation: Combining raw features improves predictive signal
- Normalization: Ratio features account for account characteristics

## Phase 4: Model Training

### Objective

Train multiple classification models and select best performer.

### Implementation

Create src/model_trainer.py:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from typing import Dict, Tuple

class ModelTrainer:
    """Trains and compares multiple classification models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.cv_scores = {}
    
    def train_logistic_regression(self, X_train_scaled, y_train) -> object:
        """
        Train Logistic Regression baseline model.
        
        Args:
            X_train_scaled: Scaled training features
            y_train: Training target
            
        Returns:
            Trained LogisticRegression model
        """
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train,
            cv=5, scoring='f1'
        )
        
        print(f"Logistic Regression CV Scores: {cv_scores}")
        print(f"Mean F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.models['logistic_regression'] = model
        self.cv_scores['logistic_regression'] = cv_scores
        
        return model
    
    def train_random_forest(self, X_train, y_train) -> object:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained RandomForestClassifier model
        """
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring='f1', n_jobs=-1
        )
        
        print(f"Random Forest CV Scores: {cv_scores}")
        print(f"Mean F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 5 Important Features:")
        print(feature_importance.head())
        
        self.models['random_forest'] = model
        self.cv_scores['random_forest'] = cv_scores
        
        return model
    
    def train_xgboost(self, X_train, y_train) -> object:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained XGBClassifier model
        """
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring='f1'
        )
        
        print(f"XGBoost CV Scores: {cv_scores}")
        print(f"Mean F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 5 Important Features:")
        print(feature_importance.head())
        
        self.models['xgboost'] = model
        self.cv_scores['xgboost'] = cv_scores
        
        return model
    
    def create_ensemble(self) -> VotingClassifier:
        """
        Create voting ensemble combining all models.
        
        Returns:
            VotingClassifier ensemble
        """
        ensemble = VotingClassifier(
            estimators=[
                ('lr', self.models['logistic_regression']),
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost'])
            ],
            voting='soft',
            weights=[1, 2, 2]
        )
        
        self.models['ensemble'] = ensemble
        return ensemble
```

### Application in Notebooks

In notebooks/04_model_training.ipynb:

```python
from src.model_trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(random_state=42)

# Train individual models
trainer.train_logistic_regression(X_train_scaled, y_train)
trainer.train_random_forest(X_train, y_train)
trainer.train_xgboost(X_train, y_train)

# Create ensemble
ensemble = trainer.create_ensemble()

# Save trained models
import pickle
for name, model in trainer.models.items():
    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
```

Model training process:
- Each model trained on same data to ensure fair comparison
- Cross-validation prevents overfitting detection
- Ensemble combines strengths of individual models
- Feature importance reveals what drives predictions

## Phase 5: Model Evaluation

### Objective

Measure model performance using appropriate metrics and create visualizations.

### Implementation

Create src/evaluation.py:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Evaluates and compares model performance."""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba) -> dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, model_name: str,
                      X_test_scaled=None) -> dict:
        """
        Complete evaluation of single model.
        
        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test labels
            model_name: Name of model for reporting
            X_test_scaled: Scaled test features (for LR)
            
        Returns:
            Dictionary containing metrics and predictions
        """
        # Use scaled features for Logistic Regression
        X_eval = X_test_scaled if X_test_scaled is not None else X_test
        
        # Get predictions
        y_pred = model.predict(X_eval)
        y_pred_proba = model.predict_proba(X_eval)[:, 1]
        
        # Calculate metrics
        metrics = ModelEvaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Print results
        print(f"\n{model_name} Evaluation Results:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric.upper():15} {value:.4f}")
        
        return {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name: str):
        """
        Create confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name for plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrix_{model_name}.png', dpi=300)
        plt.close()
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, model_name: str):
        """
        Create ROC curve visualization.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name for plot title
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC Curve (AUC={roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'results/roc_curve_{model_name}.png', dpi=300)
        plt.close()
```

### Application in Notebooks

In notebooks/04_model_training.ipynb:

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate each model
lr_results = evaluator.evaluate_model(
    trainer.models['logistic_regression'],
    X_test, y_test, 'Logistic Regression',
    X_test_scaled=X_test_scaled
)

rf_results = evaluator.evaluate_model(
    trainer.models['random_forest'],
    X_test, y_test, 'Random Forest'
)

xgb_results = evaluator.evaluate_model(
    trainer.models['xgboost'],
    X_test, y_test, 'XGBoost'
)

ensemble_results = evaluator.evaluate_model(
    trainer.models['ensemble'],
    X_test, y_test, 'Ensemble'
)

# Create visualizations
evaluator.plot_confusion_matrix(y_test, ensemble_results['y_pred'], 'Ensemble')
evaluator.plot_roc_curve(y_test, ensemble_results['y_pred_proba'], 'Ensemble')

# Compare models
comparison = pd.DataFrame({
    'Logistic Regression': lr_results['metrics'],
    'Random Forest': rf_results['metrics'],
    'XGBoost': xgb_results['metrics'],
    'Ensemble': ensemble_results['metrics']
}).T

print("\nModel Comparison:")
print(comparison)
```

Evaluation principles:
- Use appropriate metrics for problem (precision, recall, F1)
- Visualize performance for communication
- Compare models fairly using same test set
- Document results and rationale

## Production Deployment

### Creating Inference Script

Create scripts/predict.py for making predictions on new transactions:

```python
import pickle
import pandas as pd
from src.feature_engineering import FeatureEngineer

def predict_fraud(transaction_features: dict) -> dict:
    """
    Predict fraud probability for a transaction.
    
    Args:
        transaction_features: Dictionary of feature values
        
    Returns:
        Dictionary with prediction and probability
    """
    # Load trained model
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Convert to DataFrame
    X = pd.DataFrame([transaction_features])
    
    # Apply feature engineering
    engineer = FeatureEngineer()
    X_engineered = engineer.engineer_features(X)
    
    # Make prediction
    fraud_probability = model.predict_proba(X_engineered)[0, 1]
    prediction = model.predict(X_engineered)[0]
    
    return {
        'prediction': int(prediction),
        'fraud_probability': float(fraud_probability),
        'decision': 'FRAUD ALERT' if prediction == 1 else 'LEGITIMATE',
        'confidence': float(max(1 - fraud_probability, fraud_probability))
    }

# Example usage
if __name__ == '__main__':
    transaction = {
        'transaction_amount': 1500.0,
        'login_attempts': 2,
        'device_risk_score': 35.5,
        'transfer_frequency': 17,
        'anomaly_score': 0.26,
        'account_age_days': 2354,
        'transaction_time_hour': 22,
        'failed_transactions_last_30d': 0,
        'avg_monthly_balance': 112760.07
    }
    
    result = predict_fraud(transaction)
    print(f"Transaction Decision: {result['decision']}")
    print(f"Fraud Probability: {result['fraud_probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
```

## Summary of Application Process

1. Load and validate raw data
2. Clean data and handle outliers
3. Engineer features from domain knowledge
4. Train multiple models with cross-validation
5. Evaluate models using appropriate metrics
6. Select best model (ensemble)
7. Create inference script for predictions
8. Document results and learnings
9. Deploy to production environment

Each phase builds on previous one, ensuring reproducible, well-validated results at each step.