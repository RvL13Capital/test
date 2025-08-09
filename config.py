import os
import torch

class Config:
    GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "ignition-ki-csv-storage")
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "ignition-ki-csv-data-2025-user123")
    ALLOWED_CORS_ORIGINS = os.getenv("ALLOWED_CORS_ORIGINS", "http://localhost:3000,https://your-production-domain.com")
    LSTM_HPARAMS = {
        'input_dim': 17,
        'hidden_dim': 64,
        'n_layers': 1,
        'dropout_prob': 0.25,
        'learning_rate': 0.001,
        'epochs': 20
    }
    XGBOOST_HPARAMS = {
        'objective': 'reg:squarederror',
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 3,
        'subsample': 0.8,
        'tree_method': 'hist',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    DATA_WINDOW_SIZE = 75
    DATA_PREDICTION_LENGTH = 30
    CONSOLIDATION_BB_WINDOW = 20
    CONSOLIDATION_LOOKBACK_PERIOD = 50
    CONSOLIDATION_CHANNEL_THRESHOLD = 0.05
    ATR_MULTIPLIER = 2.5
    ADX_THRESHOLD = 25
    MIN_VOLATILITY = 1e-5
    MODEL_REGISTRY_PATH = 'model_registry.json'
    BACKTEST_INITIAL_CASH = 100000
    BACKTEST_COMMISSION = 0.001
    MAX_FILE_SIZE = 5 * 1024 * 1024
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    OPTUNA_TIMEOUT = int(os.getenv("OPTUNA_TIMEOUT", 3600))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    MLFLOW_EXPERIMENT_NAME = "trading_models"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")