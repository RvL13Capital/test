# project/config.py
import os
import secrets
import torch
from pathlib import Path
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)

class SecureConfig:
    """2025 Security-Enhanced Configuration with proper secrets management"""
    
    # Required environment variables for security
    REQUIRED_ENV_VARS = [
        'SECRET_KEY', 'JWT_SECRET_KEY', 'CELERY_BROKER_URL', 
        'CELERY_RESULT_BACKEND', 'GCS_BUCKET_NAME'
    ]
    
    def __init__(self):
        self._validate_environment()
        self._setup_encryption()
    
    def _validate_environment(self):
        """Validate all required environment variables are set"""
        missing = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            # In development, generate temporary secrets
            if os.getenv('FLASK_ENV') == 'development':
                logger.warning("Development mode: generating temporary secrets")
                for var in missing:
                    if var in ['SECRET_KEY', 'JWT_SECRET_KEY']:
                        os.environ[var] = secrets.token_urlsafe(32)
            else:
                raise ValueError(f"Missing required environment variables: {missing}")
    
    def _setup_encryption(self):
        """Setup encryption for sensitive data"""
        key = os.getenv('ENCRYPTION_KEY')
        if not key:
            if os.getenv('FLASK_ENV') == 'development':
                key = Fernet.generate_key()
                logger.warning(f"Generated temporary encryption key: {key.decode()}")
                logger.warning("Set ENCRYPTION_KEY environment variable for production!")
            else:
                raise ValueError("ENCRYPTION_KEY environment variable required for production")
        
        self.cipher = Fernet(key if isinstance(key, bytes) else key.encode())
    
    # Security Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_urlsafe(32))
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
    JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 3600))  # 1 hour
    
    # System & Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    MODEL_SAVE_PATH = "saved_models/"
    FLASK_ENV = os.getenv('FLASK_ENV', 'production')
    
    # Security Headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'",
        'Referrer-Policy': 'strict-origin-when-cross-origin'
    }
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = os.getenv('REDIS_URL', os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1'))
    API_RATE_LIMIT = os.getenv('API_RATE_LIMIT', "200/hour")
    TRAINING_RATE_LIMIT = os.getenv('TRAINING_RATE_LIMIT', "10/hour")
    EXPENSIVE_OPERATION_LIMIT = "3/hour"
    
    # Input Validation
    MAX_DATAFRAME_SIZE = int(os.getenv('MAX_DATAFRAME_SIZE', 1_000_000))  # rows
    MAX_REQUEST_SIZE = int(os.getenv('MAX_REQUEST_SIZE', 16 * 1024 * 1024))  # 16MB
    MIN_DATAFRAME_SIZE = 100  # minimum rows for training
    ALLOWED_TICKERS = set(os.getenv('ALLOWED_TICKERS', 'AAPL,GOOGL,MSFT,TSLA,AMZN,META,NFLX,NVDA').split(','))
    
    # Data Processing
    DATA_WINDOW_SIZE = int(os.getenv('DATA_WINDOW_SIZE', 60))
    DATA_PREDICTION_LENGTH = int(os.getenv('DATA_PREDICTION_LENGTH', 5))
    MIN_VOLATILITY = float(os.getenv('MIN_VOLATILITY', 1e-9))
    
    # Hyperparameters with security constraints
    LSTM_HPARAMS = {
        'hidden_dim': int(os.getenv('LSTM_HIDDEN_DIM', 128)),
        'n_layers': int(os.getenv('LSTM_N_LAYERS', 2)),
        'dropout_prob': float(os.getenv('LSTM_DROPOUT', 0.3)),
        'learning_rate': float(os.getenv('LSTM_LR', 1e-3)),
        'epochs': int(os.getenv('LSTM_EPOCHS', 50)),  # Reduced for security
        'weight_decay': float(os.getenv('LSTM_WEIGHT_DECAY', 1e-5)),
        'teacher_forcing_ratio': float(os.getenv('LSTM_TEACHER_FORCING', 0.5)),
        'max_epochs': 200,  # Hard limit
        'max_hidden_dim': 512,  # Resource constraint
    }
    
    XGBOOST_HPARAMS = {
        'n_estimators': int(os.getenv('XGB_N_ESTIMATORS', 100)),  # Reduced
        'learning_rate': float(os.getenv('XGB_LR', 0.05)),
        'max_depth': int(os.getenv('XGB_MAX_DEPTH', 5)),
        'subsample': float(os.getenv('XGB_SUBSAMPLE', 0.8)),
        'colsample_bytree': float(os.getenv('XGB_COLSAMPLE', 0.8)),
        'objective': 'reg:squarederror',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'reg_alpha': float(os.getenv('XGB_REG_ALPHA', 0.1)),
        'reg_lambda': float(os.getenv('XGB_REG_LAMBDA', 1.0)),
        'max_n_estimators': 1000,  # Hard limit
    }
    
    # Training Control with timeouts
    EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE', 10))
    MAX_TRAINING_TIME = int(os.getenv('MAX_TRAINING_TIME', 3600))  # 1 hour max
    
    # Tuning & Backtesting with limits
    OPTUNA_N_TRIALS = int(os.getenv('OPTUNA_N_TRIALS', 20))  # Reduced for security
    OPTUNA_TIMEOUT = int(os.getenv('OPTUNA_TIMEOUT', 1800))  # 30 minutes max
    MAX_OPTUNA_TRIALS = 100  # Hard limit
    
    BACKTEST_INITIAL_CASH = float(os.getenv('BACKTEST_INITIAL_CASH', 100000.0))
    BACKTEST_COMMISSION = float(os.getenv('BACKTEST_COMMISSION', 0.001))
    RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', 0.01))
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', 0.01))
    
    # Celery Configuration
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    CELERY_TASK_SERIALIZER = 'json'
    CELERY_RESULT_SERIALIZER = 'json'
    CELERY_ACCEPT_CONTENT = ['json']
    CELERY_TIMEZONE = 'UTC'
    CELERY_ENABLE_UTC = True
    
    # Task Security
    CELERY_TASK_ROUTES = {
        'project.tasks.tune_and_train_async': {'queue': 'training'},
        'project.tasks.train_lstm_async': {'queue': 'training'},
        'project.tasks.train_xgboost_async': {'queue': 'training'},
    }
    CELERY_TASK_TIME_LIMIT = MAX_TRAINING_TIME
    CELERY_TASK_SOFT_TIME_LIMIT = MAX_TRAINING_TIME - 300  # 5 min buffer
    
    # GCS Configuration
    GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'your-ml-models-bucket')
    GCS_PROJECT_ID = os.getenv('GCS_PROJECT_ID', None)
    GCS_KMS_KEY_NAME = os.getenv('GCS_KMS_KEY_NAME', None)  # For server-side encryption
    MODEL_ENCRYPTION_KEY = os.getenv('MODEL_ENCRYPTION_KEY', None)  # For client-side encryption
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'SecureTradingSystem')
    
    # User Management (simplified for demo)
    DEFAULT_USERS = {
        'admin': {
            'password_hash': 'scrypt:32768:8:1$your_salt$your_hash',  # Change in production
            'role': 'admin',
            'permissions': ['train_model', 'tune_hyperparams', 'view_all_models']
        },
        'trader': {
            'password_hash': 'scrypt:32768:8:1$your_salt$your_hash',  # Change in production
            'role': 'trader', 
            'permissions': ['train_model', 'view_own_models']
        }
    }
    
    # Audit and Monitoring
    AUDIT_LOG_ENABLED = os.getenv('AUDIT_LOG_ENABLED', 'true').lower() == 'true'
    AUDIT_LOG_LEVEL = os.getenv('AUDIT_LOG_LEVEL', 'INFO')
    PROMETHEUS_METRICS_ENABLED = os.getenv('PROMETHEUS_METRICS_ENABLED', 'true').lower() == 'true'
    
    @classmethod
    def validate_hyperparams(cls, model_type, hparams):
        """Validate hyperparameters are within safe limits"""
        if model_type == 'lstm':
            if hparams.get('epochs', 0) > cls.LSTM_HPARAMS['max_epochs']:
                raise ValueError(f"LSTM epochs cannot exceed {cls.LSTM_HPARAMS['max_epochs']}")
            if hparams.get('hidden_dim', 0) > cls.LSTM_HPARAMS['max_hidden_dim']:
                raise ValueError(f"LSTM hidden_dim cannot exceed {cls.LSTM_HPARAMS['max_hidden_dim']}")
        elif model_type == 'xgboost':
            if hparams.get('n_estimators', 0) > cls.XGBOOST_HPARAMS['max_n_estimators']:
                raise ValueError(f"XGBoost n_estimators cannot exceed {cls.XGBOOST_HPARAMS['max_n_estimators']}")
        
        return True

# Global config instance
Config = SecureConfig()

# Security utility functions
def get_default_hparams(model_type):
    """Get default hyperparameters for a model type"""
    if model_type == 'lstm':
        return {k: v for k, v in Config.LSTM_HPARAMS.items() 
                if not k.startswith('max_')}
    elif model_type == 'xgboost':
        return {k: v for k, v in Config.XGBOOST_HPARAMS.items() 
                if not k.startswith('max_')}
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def estimate_training_time(model_type, tune_hyperparams=False):
    """Estimate training time for user feedback"""
    base_times = {'lstm': 300, 'xgboost': 120}  # seconds
    multiplier = 10 if tune_hyperparams else 1
    return base_times.get(model_type, 180) * multiplier