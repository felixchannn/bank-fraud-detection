# Project Setup and Execution Guide

## Prerequisites

- Python 3.8 or higher
- Git and GitHub account
- Kaggle account (to download dataset)
- Text editor or IDE (VS Code, PyCharm, etc.)

## Installation and Setup

### Step 1: Environment Configuration

Create and activate a Python virtual environment to isolate project dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (Git Bash):
source venv/Scripts/activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

Install all required Python packages from requirements.txt:

```bash
pip install -r requirements.txt
```

This installs:
- Data processing: pandas, numpy
- Machine learning: scikit-learn, xgboost
- Visualization: matplotlib, seaborn
- Jupyter notebooks for interactive development
- Testing and code quality tools

### Step 3: Data Acquisition

Download the banking transactions dataset from Kaggle:

1. Visit: https://www.kaggle.com/datasets/deepeshkansotia/banking-fraud-detection-risk-analytics-dataset
2. Download banking_transactions.csv
3. Place in: data/raw/banking_transactions.csv

The dataset contains 10,000 transactions with 20 features including transaction amounts, login attempts, device risk scores, and anomaly detection scores.

## Project Directory Structure

```
bank-fraud-detection/
├── README.md                    # Project overview and results
├── QUICKSTART.md               # Quick reference guide
├── requirements.txt            # Python package dependencies
├── config.py                   # Centralized configuration
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── raw/                    # Original unmodified dataset
│   │   └── banking_transactions.csv
│   └── processed/              # Cleaned and engineered data
│
├── src/                        # Main source code package
│   ├── __init__.py
│   ├── data_loader.py          # Load CSV and validate data
│   ├── preprocessing.py        # Clean and prepare data
│   ├── feature_engineering.py  # Create new features
│   ├── model_trainer.py        # Train classification models
│   ├── evaluation.py           # Calculate metrics and visualize
│   └── utils.py                # Helper functions
│
├── scripts/
│   ├── train.py                # Main training pipeline
│   └── predict.py              # Inference on new data
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── models/
│   ├── best_model.pkl         # Serialized trained model
│   └── model_metadata.json    # Model parameters and metrics
│
├── results/
│   ├── model_performance.json  # Performance metrics
│   ├── confusion_matrix.png    # Classification matrix visualization
│   ├── roc_curve.png          # ROC curve and AUC
│   └── feature_importance.png  # Feature contribution analysis
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # Unit tests for data cleaning
│   └── test_model.py          # Unit tests for model training
│
└── docs/
    ├── METHODOLOGY.md          # Detailed technical approach
    ├── MODEL_DETAILS.md        # Model specifications and hyperparameters
    └── WORKFLOW.md             # Git workflow and collaboration guide
```

## Execution Workflow

### Phase 1: Data Understanding (1-2 hours)

Create 01_eda.ipynb in notebooks/ to explore the data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/raw/banking_transactions.csv')

# Examine structure
print(df.shape)
print(df.info())
print(df.describe())

# Check for quality issues
print(df.isnull().sum())
print(df.duplicated().sum())

# Analyze target variable
print(df['fraud_flag'].value_counts())
print(df['fraud_flag'].value_counts(normalize=True))

# Correlation analysis
correlation_matrix = df.corr()
print(correlation_matrix['fraud_flag'].sort_values(ascending=False))
```

Key questions to answer:
- What is the class distribution (fraud vs legitimate)?
- Are there missing values or duplicates?
- Which features correlate strongest with fraud?
- What are the ranges and distributions of each feature?
- Are there outliers that need treatment?

### Phase 2: Data Preparation (1-2 hours)

Implement src/preprocessing.py to clean and prepare data:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(filepath):
    """Load CSV and apply preprocessing steps."""
    df = pd.read_csv(filepath)
    
    # Remove unnecessary columns
    df = df.drop(['transaction_id'], axis=1)
    
    # Handle outliers using IQR method
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Separate features and target
    X = df.drop('fraud_flag', axis=1)
    y = df['fraud_flag']
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler
```

Key preprocessing steps:
- Validate data integrity (check for nulls, duplicates)
- Remove or transform outliers to prevent model bias
- Split data with stratification to maintain class distribution
- Normalize numerical features for algorithms that require it

### Phase 3: Feature Engineering (1-2 hours)

Implement src/feature_engineering.py to create predictive features:

```python
def engineer_features(X_train, X_test):
    """Create new features from existing ones."""
    
    # Risk-weighted composite score
    X_train['risk_activity_score'] = (
        X_train['device_risk_score'] * 0.4 + 
        X_train['anomaly_score'] * 100 * 0.4 + 
        X_train['login_attempts'] * 10 * 0.2
    )
    
    # Transaction velocity (frequency per unit time)
    X_train['transaction_velocity'] = (
        X_train['transfer_frequency'] / (X_train['account_age_days'] + 1)
    )
    
    # Failed transaction ratio
    X_train['failed_transaction_ratio'] = (
        X_train['failed_transactions_last_30d'] / (X_train['transfer_frequency'] + 1)
    )
    
    # Balance adequacy ratio
    X_train['balance_to_transaction_ratio'] = (
        X_train['avg_monthly_balance'] / (X_train['transaction_amount'] + 1)
    )
    
    # Apply same transformations to test set
    X_test['risk_activity_score'] = (
        X_test['device_risk_score'] * 0.4 + 
        X_test['anomaly_score'] * 100 * 0.4 + 
        X_test['login_attempts'] * 10 * 0.2
    )
    
    # Apply remaining transformations to X_test...
    
    return X_train, X_test
```

Feature engineering rationale:
- Composite scores: Combine multiple risk indicators into single signals
- Velocity metrics: Unusual activity patterns indicate fraud
- Ratio features: Normalize amounts by account characteristics
- Domain knowledge: Create features based on how fraud occurs in reality

### Phase 4: Model Training (2-3 hours)

Execute python scripts/train.py which orchestrates:

```bash
python scripts/train.py
```

This runs the complete training pipeline:

1. Data loading and validation
2. Preprocessing and scaling
3. Feature engineering
4. Train-test split
5. Model training (Logistic Regression, Random Forest, XGBoost)
6. Cross-validation (5-fold)
7. Hyperparameter tuning (GridSearchCV)
8. Ensemble creation (Voting Classifier)
9. Evaluation and visualization
10. Model serialization

The script generates:
- Console output with detailed metrics
- models/best_model.pkl (trained model)
- results/model_performance.json (metrics)
- results/confusion_matrix.png (visualization)
- results/roc_curve.png (AUC curve)

### Phase 5: Model Evaluation (1 hour)

Analyze results and document findings:

```python
import json
import pandas as pd

# Load results
with open('results/model_performance.json') as f:
    results = json.load(f)

# Compare models
comparison_df = pd.DataFrame(results['results']).T
print(comparison_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']])

# Best model selection rationale
# - Precision: Minimize false positives (customer frustration)
# - Recall: Minimize false negatives (financial loss)
# - F1-Score: Balance between the two
# - ROC-AUC: Threshold-independent performance
```

Key evaluation considerations:
- Trade-offs between precision and recall
- Confusion matrix interpretation (TP, TN, FP, FN)
- Feature importance analysis
- Model calibration and confidence

## Configuration Management

All project settings are centralized in config.py:

```python
# Data paths
DATA_CONFIG = {
    'raw_data_path': 'data/raw/banking_transactions.csv',
    'test_size': 0.2,
    'random_state': 42,
}

# Model hyperparameters
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 7,
        'learning_rate': 0.1,
    }
}
```

To modify settings, edit config.py rather than hardcoding values. This enables:
- Easy experiment replication
- Hyperparameter tuning without code changes
- Consistent configuration across modules

## Testing

Create unit tests to validate components:

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v
```

Test categories:
- Preprocessing tests: Verify data cleaning logic
- Model tests: Validate training and prediction
- Integration tests: Check pipeline components work together

## Deployment Preparation

Once training is complete and results are satisfactory:

1. Create scripts/predict.py for inference on new transactions
2. Document model API and expected input format
3. Create deployment specifications (model size, latency, throughput)
4. Version the model and track changes

Example prediction workflow:

```python
import pickle

# Load trained model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare new transaction data
new_transaction = [[1500.0, 2, 35.5, 17, 0.26, ...]]  # Feature values

# Make prediction
fraud_probability = model.predict_proba(new_transaction)[0, 1]
prediction = model.predict(new_transaction)[0]

if fraud_probability > 0.5:
    print(f"FRAUD ALERT - Confidence: {fraud_probability:.2%}")
else:
    print(f"LEGITIMATE - Confidence: {1-fraud_probability:.2%}")
```

## Version Control Workflow

Track project changes using Git:

```bash
# Create feature branch
git checkout -b feature/model-improvements

# Make changes and commit
git add src/
git commit -m "feat(model): add ensemble voting classifier"

# Push and create pull request
git push origin feature/model-improvements
```

Commit message format:
- feat(scope): new feature
- fix(scope): bug fix
- docs(scope): documentation update
- refactor(scope): code refactoring

## Code Quality Standards

Maintain code quality with:

```bash
# Format code
black src/

# Check style compliance
flake8 src/

# Run linter
pylint src/
```

## Next Steps

1. Implement data_loader.py to load and validate raw data
2. Complete preprocessing.py with data cleaning functions
3. Develop feature_engineering.py with custom feature creation
4. Create model_trainer.py to train and evaluate models
5. Implement evaluation.py for metrics and visualizations
6. Run python scripts/train.py to execute full pipeline
7. Document results and insights
8. Commit changes and push to GitHub

Each module should be well-documented with docstrings explaining purpose, parameters, return values, and usage examples.

## Performance Expectations

Expected results after full implementation:

- Accuracy: 85-87%
- Precision: 80-84%
- Recall: 78-82%
- F1-Score: 0.80-0.83
- ROC-AUC: 0.90-0.92

These targets reflect a balance between catching fraudulent transactions and minimizing customer inconvenience from false positives.
