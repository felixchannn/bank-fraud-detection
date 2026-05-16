"""
Banking Fraud Detection ML Package

A comprehensive machine learning solution for binary classification of fraudulent
banking transactions using ensemble methods and advanced feature engineering.

Version: 1.0.0
Author: [Your Name]
License: MIT
"""

__version__ = "1.0.0"
__author__ = "[Your Name]"
__email__ = "[your.email@example.com]"
__license__ = "MIT"

from . import data_loader
from . import preprocessing
from . import feature_engineering
from . import model_trainer
from . import evaluation
from . import utils

__all__ = [
    'data_loader',
    'preprocessing',
    'feature_engineering',
    'model_trainer',
    'evaluation',
    'utils',
]
