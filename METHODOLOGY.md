# Methodology: Banking Fraud Detection ML Project

## 1. Problem Definition

**Objective**: Build a binary classification model to predict fraudulent banking transactions.

**Success Criteria**:
- Accuracy: >85%
- Precision: >80% (minimize false positives)
- Recall: >75% (minimize false negatives)
- ROC-AUC: >0.90

**Why this matters**: 
- False positives (blocking legitimate transactions) hurt customer experience
- False negatives (missing fraud) result in financial losses
- Precision-recall balance is critical for production systems

---

## 2. Data Understanding & EDA

### Dataset Overview
- **Size**: 10,000 transactions
- **Features**: 20 behavioral and transactional attributes
- **Target**: Binary (`fraud_flag`: 0 = legitimate, 1 = fraudulent)
- **Quality**: No missing values, no duplicates

### Key Analysis Steps

#### 2.1 Exploratory Data Analysis
```python
# Check class distribution
df['fraud_flag'].value_counts()
df['fraud_flag'].value_counts(normalize=True) * 100

# Descriptive statistics
df.describe()

# Correlation analysis
correlation_matrix = df.corr()
```

**Expected Findings**:
- Identify which features correlate strongest with fraud
- Check for multicollinearity issues
- Understand feature distributions

#### 2.2 Feature Distributions
- Examine numerical features for skewness
- Check for outliers (transactions with unusual values)
- Identify zero-variance or near-zero variance features
- Look for suspicious patterns in high-risk accounts

#### 2.3 Relationship Analysis
- **Device Risk Score vs Fraud**: High-risk devices likely show fraud pattern
- **Anomaly Score vs Fraud**: AI anomaly detection should correlate strongly
- **Login Attempts vs Fraud**: More attempts = higher fraud risk?
- **Transaction Velocity**: Rapid transactions indicate unusual activity?

---

## 3. Data Preprocessing

### 3.1 Data Cleaning
```python
# Handle missing values (if any found)
df.isnull().sum()
df.fillna(strategy)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove unnecessary columns
df.drop(['transaction_id'], axis=1, inplace=True)
```

### 3.2 Outlier Detection & Treatment

**Method: IQR (Interquartile Range)**
```python
Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

**Decision**: 
- Remove or cap extreme outliers
- Keep outliers that represent real fraud patterns
- Document rationale for each decision

### 3.3 Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintain class distribution
)
```

**Stratification**: Ensures both sets have similar fraud proportions

### 3.4 Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why**: Tree-based models don't require scaling, but Logistic Regression does

---

## 4. Feature Engineering

### 4.1 Domain-Driven Features

Create new features based on business logic:

```python
# Risk-weighted features
df['risk_activity_score'] = (
    df['device_risk_score'] * 0.4 + 
    df['anomaly_score'] * 100 * 0.4 + 
    df['login_attempts'] * 10 * 0.2
)

# Transaction velocity indicators
df['transaction_velocity'] = df['transfer_frequency'] / (df['account_age_days'] + 1)

# Failed transaction ratio
df['failed_transaction_ratio'] = (
    df['failed_transactions_last_30d'] / (df['transfer_frequency'] + 1)
)

# Risk-based balance ratio
df['balance_to_transaction_ratio'] = (
    df['avg_monthly_balance'] / (df['transaction_amount'] + 1)
)
```

### 4.2 Statistical Features
```python
# Z-score features for anomaly detection
df['transaction_amount_zscore'] = np.abs(
    (df['transaction_amount'] - df['transaction_amount'].mean()) / 
    df['transaction_amount'].std()
)
```

### 4.3 Interaction Features
```python
# High-risk + unusual behavior
df['high_risk_unusual_activity'] = (
    (df['device_risk_score'] > df['device_risk_score'].quantile(0.75)) &
    (df['anomaly_score'] > df['anomaly_score'].quantile(0.75))
).astype(int)
```

### 4.4 Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=15)
X_selected = selector.fit_transform(X_train, y_train)
selected_features = X_train.columns[selector.get_support()].tolist()
```

---

## 5. Model Selection & Training

### 5.1 Baseline Models

Start with simple models to establish performance baseline:

**Model 1: Logistic Regression**
- Interpretable coefficients
- Fast training
- Good for binary classification
- Baseline for comparison

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
```

### 5.2 Advanced Models

**Model 2: Random Forest**
- Handles non-linear relationships
- Feature importance ranking
- Robust to outliers
- Less prone to overfitting

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
```

**Model 3: XGBoost**
- Gradient boosting approach
- High performance on tabular data
- Built-in regularization
- Fast training

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_train, y_train)
```

### 5.3 Hyperparameter Tuning

**GridSearchCV** for optimal parameters:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```

### 5.4 Cross-Validation

5-fold cross-validation to assess model stability:
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model, X_train, y_train, 
    cv=5, 
    scoring='f1'
)
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

## 6. Model Evaluation

### 6.1 Classification Metrics

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Overall correctness. Can be misleading with imbalanced data.

#### Precision
```
Precision = TP / (TP + FP)
```
Of predicted frauds, how many are actually fraud? (Important for false positives)

#### Recall
```
Recall = TP / (TP + FN)
```
Of actual frauds, how many did we catch? (Important for false negatives)

#### F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Harmonic mean of precision and recall. Good for imbalanced data.

#### ROC-AUC
- Area Under the Receiver Operating Characteristic Curve
- Threshold-independent performance measure
- 0.5 = random classifier, 1.0 = perfect classifier
- Useful when dealing with class imbalance

### 6.2 Confusion Matrix
```
                    Predicted Negative    Predicted Positive
Actual Negative      True Negatives       False Positives
Actual Positive      False Negatives      True Positives
```

Visualization helps understand error types:
- FP (False Positives): Legitimate transactions marked as fraud → Customer frustration
- FN (False Negatives): Fraudulent transactions missed → Financial loss

### 6.3 Evaluation Code
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
}
```

---

## 7. Ensemble Strategy

### Final Model: Voting Classifier

Combine predictions from multiple models:
```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('rf', random_forest_model),
        ('xgb', xgboost_model),
        ('lr', logistic_regression_model)
    ],
    voting='soft'  # Use probability predictions
)

voting_clf.fit(X_train, y_train)
```

**Advantage**: Reduces variance, improves generalization

---

## 8. Results Interpretation

### Key Insights to Extract
1. Which features are most important for fraud prediction?
2. What transaction patterns indicate fraud?
3. Are there false positive/negative trade-offs we need to accept?
4. Which customer segments are at highest risk?
5. What's the model's confidence in edge cases?

### Deployment Considerations
- Model size and inference speed
- Threshold selection for decision boundary
- Monitoring strategy for model drift
- Retraining frequency

---

## 9. Future Enhancements

- [ ] Handle class imbalance (SMOTE, class weights)
- [ ] Time-series features for temporal patterns
- [ ] Deep learning (Neural Networks, LSTMs)
- [ ] Explainability (SHAP values, LIME)
- [ ] Real-time prediction API
- [ ] Automated retraining pipeline
- [ ] Model monitoring dashboard

---

**Document Version**: 1.0  
**Last Updated**: May 2026
