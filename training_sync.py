"""
Training Synchronization Module - Contains training functions used by backtesting
Moved from backtesting.py for better separation of concerns
"""

import logging
from project.training import build_and_train_lstm, build_and_train_xgboost
from project.features import select_features

logger = logging.getLogger(__name__)

def train_models_sync(train_df, ticker_fold_id):
    """
    Synchrones Training von LSTM und XGBoost Modellen für Backtesting
    Verwendet die zentralisierten Trainingsfunktionen aus training.py
    
    Args:
        train_df: Training data DataFrame
        ticker_fold_id: Identifier for this training run (e.g., 'AAPL_fold_1')
    
    Returns:
        tuple: (lstm_model, scaler, xgboost_model, selected_features)
    
    Raises:
        Exception: If training fails
    """
    try:
        logger.info(f"Starting training for {ticker_fold_id}")
        
        # 1. Feature Selection - using real implementation from features.py
        selected_features = select_features(train_df)
        logger.info(f"Selected {len(selected_features)} features for {ticker_fold_id}")
        
        # 2. LSTM Training - verwendet zentrale Funktion
        logger.info(f"Training LSTM for {ticker_fold_id}")
        lstm_result = build_and_train_lstm(
            df=train_df,
            selected_features=selected_features,
            hparams=Config.LSTM_HPARAMS.copy(),
            validation_split=0.0,  # No validation split for backtesting
            update_callback=None   # No progress updates needed
        )
        
        lstm_model = lstm_result['model']
        scaler = lstm_result['scaler']
        logger.info(f"LSTM training completed for {ticker_fold_id} - Loss: {lstm_result['train_loss']:.4f}")
        
        # 3. XGBoost Training - verwendet zentrale Funktion
        logger.info(f"Training XGBoost for {ticker_fold_id}")
        xgboost_result = build_and_train_xgboost(
            df=train_df,
            hparams=Config.XGBOOST_HPARAMS.copy(),
            validation_split=0.0   # No validation split for backtesting
        )
        
        xgboost_model = xgboost_result['model']
        logger.info(f"XGBoost training completed for {ticker_fold_id} - Score: {xgboost_result['train_score']:.4f}")
        
        return lstm_model, scaler, xgboost_model, selected_features
        
    except Exception as e:
        logger.error(f"Training failed for {ticker_fold_id}: {str(e)}")
        raise Exception(f"Training failed for {ticker_fold_id}: {str(e)}")

# For backward compatibility
def get_trained_models(train_df, ticker_fold_id):
    """
    Legacy wrapper function for backward compatibility
    """
    return train_models_sync(train_df, ticker_fold_id)
