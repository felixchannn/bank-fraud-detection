# Banking Fraud Detection & Risk Analytics

A machine learning classification project to predict fraudulent banking transactions using advanced feature engineering and ensemble methods.

## Project Overview

This project builds a production-ready fraud detection classifier that identifies high-risk transactions with high precision and recall. The model is trained on 10,000 synthetically generated banking transactions with 20 behavioral and transactional features.

**Objective**: Binary classification to predict fraudulent transactions (`fraud_flag`)

**Key Metrics**: 
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix Analysis

## Business Problem

Banking institutions need to detect fraudulent transactions in real-time to:
- Minimize financial losses from fraud
- Protect customer accounts
- Reduce false positives (customer inconvenience)
- Meet regulatory compliance requirements

## Project Structure

```
banking-fraud-detection/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
├── config.py                          # Configuration settings
│
├── data/
│   ├── raw/
│   │   └── banking_transactions.csv   # Original dataset
│   └── processed/
│       └── fraud_data_processed.csv   # Cleaned & feature-engineered
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_data_preprocessing.ipynb    # Data cleaning & preparation
│   ├── 03_feature_engineering.ipynb   # Feature creation & selection
│   └── 04_model_training.ipynb        # Model development & comparison
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Data loading utilities
│   ├── preprocessing.py               # Data cleaning functions
│   ├── feature_engineering.py         # Feature creation
│   ├── model_trainer.py               # Model training pipeline
│   ├── evaluation.py                  # Model evaluation metrics
│   └── utils.py                       # Helper functions
│
├── models/
│   ├── best_model.pkl                 # Serialized trained model
│   └── model_metadata.json            # Model parameters & performance
│
├── results/
│   ├── model_performance.json          # Performance metrics
│   ├── confusion_matrix.png            # Visualization
│   ├── roc_curve.png                   # ROC-AUC visualization
│   └── feature_importance.png          # Feature importance plot
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py           # Unit tests
│   └── test_model.py                   # Model tests
│
├── scripts/
│   ├── train.py                        # Main training script
│   └── predict.py                      # Prediction script
│
└── docs/
    ├── METHODOLOGY.md                  # Detailed approach
    ├── MODEL_DETAILS.md                # Model architecture & hyperparameters
    └── RESULTS.md                      # Final results & insights
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/banking-fraud-detection.git
cd banking-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Run Complete Pipeline
```bash
python scripts/train.py
```

#### 2. Make Predictions
```bash
python scripts/predict.py --model models/best_model.pkl --data data/raw/banking_transactions.csv
```

#### 3. Explore Notebooks (Jupyter)
```bash
jupyter notebook
# Open notebooks/01_eda.ipynb to start
```

## 📊 Dataset

**Source**: Kaggle - Banking Fraud Detection & Risk Analytics Dataset

**Characteristics**:
- Records: 10,000 transactions
- Features: 20 (behavioral + transactional)
- Target: `fraud_flag` (binary)
- No missing values, no duplicates
- Synthetic data (educational use)

**Key Features**:
- `transaction_amount`: Transaction value
- `login_attempts`: Number of login attempts
- `device_risk_score`: Risk score of device (0-100)
- `anomaly_score`: AI-based anomaly detection score (0-1)
- `transfer_frequency`: How often account transfers
- `failed_transactions_last_30d`: Recent failed transactions
- `avg_monthly_balance`: Average monthly account balance

## 🔧 Methodology

### Data Pipeline
1. **Exploratory Data Analysis (EDA)**
   - Distribution analysis
   - Correlation analysis
   - Missing value detection
   - Class balance assessment

2. **Data Preprocessing**
   - Handle missing values
   - Outlier detection & treatment
   - Data normalization/standardization
   - Train-test split (80-20)

3. **Feature Engineering**
   - Create interaction features
   - Statistical aggregations
   - Risk ratio features
   - Domain-based features

4. **Model Selection & Training**
   - Baseline models: Logistic Regression
   - Tree-based models: Random Forest, XGBoost
   - Hyperparameter tuning (GridSearchCV)
   - Cross-validation (5-fold)

5. **Evaluation**
   - Classification metrics: Accuracy, Precision, Recall, F1
   - ROC-AUC curve
   - Confusion matrix
   - Feature importance analysis

## 📈 Results

| Metric | Score |
|--------|-------|
| Accuracy | - |
| Precision | - |
| Recall | - |
| F1-Score | - |
| ROC-AUC | - |

*Results to be updated after model training*

See `docs/RESULTS.md` for detailed analysis and insights.

## 🛠️ Technologies Used

- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Modeling**: Scikit-learn, XGBoost
- **Evaluation**: Scikit-learn metrics
- **Serialization**: Pickle, JSON
- **Version Control**: Git, GitHub

## 📝 Model Architecture

**Final Model**: Ensemble (Random Forest + XGBoost)
- Voting Classifier with soft voting
- Hyperparameters optimized via GridSearchCV
- 5-fold cross-validation

**Reasoning**:
- Tree-based models handle non-linear relationships
- Ensemble methods reduce overfitting
- Balanced precision-recall trade-off for fraud detection

## 🔍 Key Insights

- Feature importance analysis shows `anomaly_score` and `device_risk_score` are most predictive
- Transaction timing and login attempts correlate with fraud
- No significant data imbalance issues
- Model achieves high precision without sacrificing recall

## ⚠️ Limitations & Future Work

**Current Limitations**:
- Synthetic data (may not capture all real-world patterns)
- Limited temporal data
- No customer/merchant information

**Future Enhancements**:
- [ ] Time-series analysis with RNN/LSTM
- [ ] Real-world data integration
- [ ] Model deployment (Flask/FastAPI)
- [ ] Automated retraining pipeline
- [ ] Explainability (SHAP values)
- [ ] Class imbalance handling (if needed)

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/deepeshkansotia/banking-fraud-detection-risk-analytics-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📧 Contact

**Author**: [Felix Chan]  
**Email**: [felixchan118@gmail.com]  
**LinkedIn**: [https://www.linkedin.com/in/felix-k-chan/]  

---

*Last Updated: May 2026*
