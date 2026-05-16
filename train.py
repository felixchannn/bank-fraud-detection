"""
Main training script for Banking Fraud Detection Model
Orchestrates the complete ML pipeline from data loading to model evaluation
"""

import sys
import logging
from pathlib import Path
import json
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """
    Complete ML pipeline for fraud detection
    """
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        logger.info("Pipeline initialized")
    
    def load_data(self):
        """Load raw data from CSV"""
        try:
            data_path = DATA_CONFIG['raw_data_path']
            logger.info(f"Loading data from {data_path}")
            
            df = pd.read_csv(data_path)
            logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, df):
        """Data cleaning and preprocessing"""
        logger.info("Starting data preprocessing")
        
        # Display initial info
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        logger.info(f"Duplicates: {df.duplicated().sum()}")
        logger.info(f"Class distribution:\n{df[DATA_CONFIG['target_column']].value_counts()}")
        
        # Remove unnecessary columns
        df = df.drop(FEATURE_CONFIG['features_to_drop'], axis=1)
        
        # Separate features and target
        X = df.drop(DATA_CONFIG['target_column'], axis=1)
        y = df[DATA_CONFIG['target_column']]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_test_split_data(self, X, y):
        """Split data into train and test sets"""
        logger.info("Splitting data into train and test sets")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=DATA_CONFIG['test_size'],
            random_state=DATA_CONFIG['random_state'],
            stratify=y
        )
        
        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info(f"Train fraud rate: {self.y_train.mean():.2%}")
        logger.info(f"Test fraud rate: {self.y_test.mean():.2%}")
    
    def scale_features(self):
        """Scale features for Logistic Regression"""
        logger.info("Scaling features")
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info("Features scaled successfully")
    
    def train_logistic_regression(self):
        """Train Logistic Regression baseline"""
        logger.info("Training Logistic Regression")
        
        model = LogisticRegression(**MODEL_CONFIG['logistic_regression'])
        model.fit(self.X_train_scaled, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, self.X_train_scaled, self.y_train,
            cv=MODEL_CONFIG['cv_folds'],
            scoring='f1'
        )
        
        logger.info(f"LR CV Scores: {cv_scores}")
        logger.info(f"LR CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(self):
        """Train Random Forest with hyperparameter tuning"""
        logger.info("Training Random Forest")
        
        model = RandomForestClassifier(**MODEL_CONFIG['random_forest'])
        model.fit(self.X_train, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=MODEL_CONFIG['cv_folds'],
            scoring='f1',
            n_jobs=-1
        )
        
        logger.info(f"RF CV Scores: {cv_scores}")
        logger.info(f"RF CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 5 Important Features (RF):")
        logger.info(feature_importance.head())
        
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self):
        """Train XGBoost with hyperparameter tuning"""
        logger.info("Training XGBoost")
        
        model = xgb.XGBClassifier(**MODEL_CONFIG['xgboost'])
        model.fit(self.X_train, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=MODEL_CONFIG['cv_folds'],
            scoring='f1'
        )
        
        logger.info(f"XGB CV Scores: {cv_scores}")
        logger.info(f"XGB CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 5 Important Features (XGB):")
        logger.info(feature_importance.head())
        
        self.models['xgboost'] = model
        return model
    
    def train_ensemble(self):
        """Create voting ensemble of all models"""
        logger.info("Creating Voting Ensemble")
        
        ensemble = VotingClassifier(
            estimators=[
                ('lr', self.models['logistic_regression']),
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost'])
            ],
            voting='soft',
            weights=[1, 2, 2]
        )
        
        # Note: Ensemble doesn't need retraining, uses existing models
        logger.info("Ensemble created successfully")
        
        self.models['ensemble'] = ensemble
        return ensemble
    
    def evaluate_model(self, model, model_name):
        """Evaluate model performance"""
        logger.info(f"Evaluating {model_name}")
        
        # Predictions
        if model_name == 'ensemble':
            y_pred = ensemble.predict(self.X_test)
            y_pred_proba = ensemble.predict_proba(self.X_test)[:, 1]
        elif model_name == 'logistic_regression':
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
        }
        
        self.results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }
        
        logger.info(f"{model_name} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def compare_models(self):
        """Compare performance of all models"""
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        comparison_df = pd.DataFrame({
            model: self.results[model]['metrics']
            for model in self.results.keys()
        }).T
        
        logger.info("\n" + comparison_df.to_string())
        
        best_model = comparison_df['f1'].idxmax()
        logger.info(f"\nBest Model (by F1-Score): {best_model}")
        
        return best_model
    
    def save_model(self, model, model_name='best_model'):
        """Save trained model"""
        logger.info(f"Saving {model_name}")
        
        model_path = MODELS['best_model_path']
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def save_results(self, best_model_name):
        """Save evaluation results"""
        logger.info("Saving results")
        
        results_path = RESULTS['model_performance_path']
        with open(results_path, 'w') as f:
            json.dump({
                'best_model': best_model_name,
                'results': {
                    k: {
                        'metrics': v['metrics'],
                        'confusion_matrix': v['confusion_matrix']
                    }
                    for k, v in self.results.items()
                }
            }, f, indent=4)
        
        logger.info(f"Results saved to {results_path}")
    
    def plot_results(self):
        """Create visualizations"""
        logger.info("Creating visualizations")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.results['ensemble']['y_pred'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Ensemble Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(RESULTS['confusion_matrix_path'], dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {RESULTS['confusion_matrix_path']}")
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.results['ensemble']['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Ensemble Model')
        plt.legend(loc="lower right")
        plt.savefig(RESULTS['roc_curve_path'], dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {RESULTS['roc_curve_path']}")
        plt.close()
    
    def run(self):
        """Execute complete pipeline"""
        logger.info("Starting Banking Fraud Detection Pipeline")
        logger.info("="*60)
        
        try:
            # Load data
            df = self.load_data()
            
            # Preprocess
            X, y = self.preprocess_data(df)
            
            # Split data
            self.train_test_split_data(X, y)
            
            # Scale features
            self.scale_features()
            
            # Train models
            logger.info("\n" + "="*60)
            logger.info("MODEL TRAINING")
            logger.info("="*60)
            
            self.train_logistic_regression()
            self.train_random_forest()
            self.train_xgboost()
            self.train_ensemble()
            
            # Evaluate models
            logger.info("\n" + "="*60)
            logger.info("MODEL EVALUATION")
            logger.info("="*60)
            
            for model_name, model in self.models.items():
                self.evaluate_model(model, model_name)
            
            # Compare and select best
            best_model_name = self.compare_models()
            best_model = self.models[best_model_name]
            
            # Save
            self.save_model(best_model)
            self.save_results(best_model_name)
            
            # Visualize
            self.plot_results()
            
            logger.info("\n" + "="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    pipeline = FraudDetectionPipeline()
    pipeline.run()
