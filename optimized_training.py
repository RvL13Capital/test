"""
ENHANCED: training.py - Complete training module with all optimizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import traceback
import joblib
from pathlib import Path

# Project imports
from project.config import Config, get_device
from project.features import prepare_sequences, get_feature_columns, _create_features, select_features
from project.lstm_models import Seq2Seq, Encoder, Decoder, AsymmetricWeightedMSELoss
from project.storage import update_model_registry, get_model_registry
from project.web_setup import celery
from project.monitoring import setup_mlflow, start_mlflow_run

logger = logging.getLogger(__name__)

def validate_training_data(df, selected_features):
    """
    Validate training data before model training
    """
    if df is None or len(df) == 0:
        raise ValueError("Training data is empty")
    
    if len(df) < Config.DATA_WINDOW_SIZE + Config.DATA_PREDICTION_LENGTH:
        raise ValueError(f"Insufficient data: need at least {Config.DATA_WINDOW_SIZE + Config.DATA_PREDICTION_LENGTH} rows")
    
    # Check for required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for excessive NaN values
    nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if nan_ratio > 0.1:  # More than 10% NaN values
        logger.warning(f"High NaN ratio in data: {nan_ratio:.2%}")
    
    logger.info(f"Data validation passed: {len(df)} rows, {len(df.columns)} columns")

def build_and_train_lstm(df, selected_features, hparams, validation_split=0.2, update_callback=None):
    """
    ENHANCED: Build and train LSTM model with comprehensive error handling
    
    Args:
        df: Training DataFrame
        selected_features: List of selected feature columns
        hparams: Hyperparameters dictionary
        validation_split: Fraction of data for validation
        update_callback: Optional callback for progress updates
    
    Returns:
        dict: Training results with model, scaler, and metrics
    """
    try:
        logger.info("Starting LSTM training...")
        validate_training_data(df, selected_features)
        
        # Prepare sequences
        if validation_split > 0:
            train_size = int(len(df) * (1 - validation_split))
            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size:]
        else:
            train_df = df
            val_df = None
        
        src, trg, scaler = prepare_sequences(
            train_df, 
            Config.DATA_WINDOW_SIZE, 
            Config.DATA_PREDICTION_LENGTH, 
            selected_features
        )
        
        if src.shape[0] == 0:
            raise ValueError("No training sequences generated")
        
        logger.info(f"Generated {src.shape[0]} training sequences")
        
        # Validation sequences
        val_src, val_trg = None, None
        if val_df is not None and len(val_df) > Config.DATA_WINDOW_SIZE:
            val_src, val_trg, _ = prepare_sequences(
                val_df, 
                Config.DATA_WINDOW_SIZE, 
                Config.DATA_PREDICTION_LENGTH, 
                selected_features, 
                scaler=scaler
            )
            logger.info(f"Generated {val_src.shape[0] if val_src is not None else 0} validation sequences")
        
        # Model setup
        device = get_device()
        input_dim = src.shape[-1]
        
        # Use hyperparameters with fallbacks
        hidden_dim = hparams.get('hidden_dim', 128)
        n_layers = hparams.get('n_layers', 2)
        dropout_prob = hparams.get('dropout_prob', 0.2)
        learning_rate = hparams.get('learning_rate', 1e-3)
        epochs = hparams.get('epochs', 50)
        
        encoder = Encoder(input_dim, hidden_dim, n_layers, dropout_prob).to(device)
        decoder = Decoder(input_dim, hidden_dim, n_layers, dropout_prob).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Loss function setup
        criterion = AsymmetricWeightedMSELoss()
        
        # Prepare tensors
        src_tensor = torch.tensor(src, dtype=torch.float32).to(device)
        trg_tensor = torch.tensor(trg, dtype=torch.float32).to(device)
        
        # Calculate boundaries for asymmetric loss
        close_idx = selected_features.index('close') if 'close' in selected_features else 0
        last_known_close = src_tensor[:, -1, close_idx]
        
        historical_volatility = torch.clamp(
            src_tensor[:, :, close_idx].std(dim=1),
            min=Config.MIN_VOLATILITY,
            max=last_known_close * 0.5  # Cap at 50% of price
        )
        
        upper_bound_val = last_known_close + (Config.ATR_MULTIPLIER * historical_volatility)
        lower_bound_val = last_known_close - (Config.ATR_MULTIPLIER * historical_volatility)
        
        upper_bound = upper_bound_val.unsqueeze(1).unsqueeze(2).expand(-1, trg.shape[1], src.shape[2])
        lower_bound = lower_bound_val.unsqueeze(1).unsqueeze(2).expand(-1, trg.shape[1], src.shape[2])
        
        # Validation tensors
        val_src_tensor, val_trg_tensor = None, None
        val_upper_bound, val_lower_bound = None, None
        
        if val_src is not None:
            val_src_tensor = torch.tensor(val_src, dtype=torch.float32).to(device)
            val_trg_tensor = torch.tensor(val_trg, dtype=torch.float32).to(device)
            
            val_last_close = val_src_tensor[:, -1, close_idx]
            val_volatility = torch.clamp(
                val_src_tensor[:, :, close_idx].std(dim=1),
                min=Config.MIN_VOLATILITY,
                max=val_last_close * 0.5
            )
            
            val_upper_val = val_last_close + (Config.ATR_MULTIPLIER * val_volatility)
            val_lower_val = val_last_close - (Config.ATR_MULTIPLIER * val_volatility)
            
            val_upper_bound = val_upper_val.unsqueeze(1).unsqueeze(2).expand(-1, val_trg.shape[1], val_src.shape[2])
            val_lower_bound = val_lower_val.unsqueeze(1).unsqueeze(2).expand(-1, val_trg.shape[1], val_src.shape[2])
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()
            
            output = model(src_tensor, trg_tensor)
            train_loss = criterion(output, trg_tensor, upper_bound, lower_bound)
            
            # Check for NaN
            if torch.isnan(train_loss):
                logger.error("NaN loss detected during training")
                raise ValueError("Training produced NaN loss")
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(train_loss.item())
            
            # Validation phase
            val_loss = None
            if val_src_tensor is not None:
                model.eval()
                with torch.no_grad():
                    val_output = model(val_src_tensor, val_trg_tensor, teacher_forcing_ratio=0.0)
                    val_loss = criterion(val_output, val_trg_tensor, val_upper_bound, val_lower_bound)
                    val_losses.append(val_loss.item())
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model state
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                    
                    scheduler.step(val_loss)
                    
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        model.load_state_dict(best_model_state)
                        break
            else:
                scheduler.step(train_loss)
            
            # Progress callback
            if update_callback and epoch % 10 == 0:
                progress = (epoch + 1) / epochs
                update_callback(f"Training LSTM: Epoch {epoch + 1}/{epochs}", progress)
            
            # Logging
            if (epoch + 1) % 10 == 0:
                if val_loss is not None:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss.item():.6f}, "
                              f"Val Loss: {val_loss.item():.6f}")
                else:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss.item():.6f}")
        
        # Final evaluation
        model.eval()
        final_train_loss = train_losses[-1] if train_losses else float('inf')
        final_val_loss = val_losses[-1] if val_losses else None
        
        logger.info(f"LSTM training completed - Final train loss: {final_train_loss:.6f}")
        if final_val_loss is not None:
            logger.info(f"Final validation loss: {final_val_loss:.6f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'train_loss': final_train_loss,
            'val_loss': final_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_trained': len(train_losses),
            'hyperparameters': hparams
        }
        
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        logger.error(traceback.format_exc())
        raise

def build_and_train_xgboost(df, hparams, validation_split=0.2):
    """
    ENHANCED: Build and train XGBoost model with comprehensive error handling
    
    Args:
        df: Training DataFrame
        hparams: Hyperparameters dictionary
        validation_split: Fraction of data for validation
    
    Returns:
        dict: Training results with model and metrics
    """
    try:
        logger.info("Starting XGBoost training...")
        validate_training_data(df, get_feature_columns())
        
        # Feature engineering
        df_with_features = _create_features(df.copy())
        df_with_features['target'] = df_with_features['close'].shift(-1)
        df_with_features.dropna(inplace=True)
        
        if len(df_with_features) < 100:
            raise ValueError("Insufficient data after feature engineering")
        
        # Prepare features and target
        feature_columns = get_feature_columns()
        available_features = [col for col in feature_columns if col in df_with_features.columns]
        
        if len(available_features) < len(feature_columns) * 0.8:
            logger.warning(f"Many features missing: {len(available_features)}/{len(feature_columns)} available")
        
        X = df_with_features[available_features]
        y = df_with_features['target']
        
        logger.info(f"Training XGBoost with {len(available_features)} features on {len(X)} samples")
        
        # Train/validation split
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, shuffle=False
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Sample weighting for asymmetric loss
        close_values = X_train['close'] if 'close' in X_train.columns else y_train
        volatility = close_values.rolling(window=min(14, len(close_values)//4)).std()
        volatility = volatility.fillna(volatility.mean()).fillna(Config.MIN_VOLATILITY)
        
        upper_bound = close_values + (Config.ATR_MULTIPLIER * volatility)
        lower_bound = close_values - (Config.ATR_MULTIPLIER * volatility)
        
        # Calculate sample weights
        weights = np.ones_like(y_train)
        above_upper = (y_train > upper_bound)
        below_lower = (y_train < lower_bound)
        weights[above_upper | below_lower] = Config.ATR_MULTIPLIER
        
        logger.info(f"Applied asymmetric weighting to {np.sum(above_upper | below_lower)} samples")
        
        # Model setup with hyperparameters
        model_params = {
            'n_estimators': hparams.get('n_estimators', 100),
            'learning_rate': hparams.get('learning_rate', 0.1),
            'max_depth': hparams.get('max_depth', 6),
            'subsample': hparams.get('subsample', 0.8),
            'colsample_bytree': hparams.get('colsample_bytree', 0.8),
            'gamma': hparams.get('gamma', 0),
            'min_child_weight': hparams.get('min_child_weight', 1),
            'reg_alpha': hparams.get('reg_alpha', 0),
            'reg_lambda': hparams.get('reg_lambda', 1),
            'objective': hparams.get('objective', 'reg:squarederror'),
            'device': hparams.get('device', Config.XGBOOST_HPARAMS.get('device', 'cpu')),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**model_params)
        
        # Training with evaluation
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=eval_set,
            early_stopping_rounds=20 if eval_set else None,
            verbose=False
        )
        
        # Evaluation
        train_preds = model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_preds)
        train_mae = mean_absolute_error(y_train, train_preds)
        
        val_mse, val_mae = None, None
        if X_val is not None:
            val_preds = model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_preds)
            val_mae = mean_absolute_error(y_val, val_preds)
        
        # Feature importance
        feature_importance = dict(zip(available_features, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        logger.info(f"XGBoost training completed - Train MSE: {train_mse:.6f}")
        if val_mse is not None:
            logger.info(f"Validation MSE: {val_mse:.6f}")
        
        logger.info("Top 5 important features:")
        for feature, importance in top_features[:5]:
            logger.info(f"  {feature}: {importance:.4f}")
        
        return {
            'model': model,
            'train_score': -train_mse,  # Negative for consistency with sklearn convention
            'val_score': -val_mse if val_mse is not None else None,
            'train_mse': train_mse,
            'train_mae': train_mae,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'feature_importance': feature_importance,
            'top_features': top_features,
            'hyperparameters': hparams,
            'available_features': available_features
        }
        
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        logger.error(traceback.format_exc())
        raise

@celery.task(bind=True)
def train_lstm_async(self, df_json, hparams, ticker, selected_features):
    """
    ENHANCED: Async LSTM training with comprehensive error handling and model saving
    """
    try:
        logger.info(f"Starting async LSTM training for {ticker}")
        
        # Load and validate data
        df = pd.read_json(df_json)
        validate_training_data(df, selected_features)
        
        # Start MLflow run for tracking
        with start_mlflow_run(f"lstm_training_{ticker}"):
            # Train model
            result = build_and_train_lstm(
                df=df,
                selected_features=selected_features,
                hparams=hparams,
                validation_split=0.2,
                update_callback=lambda msg, progress: self.update_state(
                    state='PROGRESS',
                    meta={'message': msg, 'progress': progress}
                )
            )
            
            # Save model and scaler
            model_dir = Path(Config.MODEL_SAVE_PATH) / ticker
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save PyTorch model
            torch.save(result['model'].state_dict(), model_dir / 'lstm_model.pth')
            
            # Save scaler
            joblib.dump(result['scaler'], model_dir / 'scaler.pkl')
            
            # Save metadata
            metadata = {
                'model_type': 'lstm',
                'ticker': ticker,
                'selected_features': selected_features,
                'hyperparameters': result['hyperparameters'],
                'train_loss': result['train_loss'],
                'val_loss': result['val_loss'],
                'epochs_trained': result['epochs_trained'],
                'input_dim': len(selected_features),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            import json
            with open(model_dir / 'lstm_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update model registry
            update_model_registry(
                ticker=ticker,
                model_type="lstm_trained",
                model_path=str(model_dir / 'lstm_model.pth'),
                hparams=metadata
            )
            
            logger.info(f"LSTM model saved for {ticker} at {model_dir}")
            
            return {
                'status': 'SUCCESS',
                'ticker': ticker,
                'model_path': str(model_dir),
                'train_loss': result['train_loss'],
                'val_loss': result['val_loss'],
                'epochs_trained': result['epochs_trained']
            }
    
    except Exception as e:
        logger.error(f"Async LSTM training failed for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return {
            'status': 'FAILED',
            'ticker': ticker,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@celery.task(bind=True)
def train_xgboost_async(self, df_json, hparams, ticker):
    """
    ENHANCED: Async XGBoost training with comprehensive error handling and model saving
    """
    try:
        logger.info(f"Starting async XGBoost training for {ticker}")
        
        # Load and validate data
        df = pd.read_json(df_json)
        validate_training_data(df, get_feature_columns())
        
        # Start MLflow run for tracking
        with start_mlflow_run(f"xgboost_training_{ticker}"):
            # Train model
            result = build_and_train_xgboost(
                df=df,
                hparams=hparams,
                validation_split=0.2
            )
            
            # Save model
            model_dir = Path(Config.MODEL_SAVE_PATH) / ticker
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save XGBoost model
            result['model'].save_model(str(model_dir / 'xgboost_model.json'))
            
            # Alternative pickle save for compatibility
            joblib.dump(result['model'], model_dir / 'xgboost_model.pkl')
            
            # Save metadata
            metadata = {
                'model_type': 'xgboost',
                'ticker': ticker,
                'hyperparameters': result['hyperparameters'],
                'train_mse': result['train_mse'],
                'val_mse': result['val_mse'],
                'feature_importance': result['feature_importance'],
                'top_features': result['top_features'],
                'available_features': result['available_features'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            import json
            with open(model_dir / 'xgboost_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update model registry
            update_model_registry(
                ticker=ticker,
                model_type="xgboost_trained",
                model_path=str(model_dir / 'xgboost_model.pkl'),
                hparams=metadata
            )
            
            logger.info(f"XGBoost model saved for {ticker} at {model_dir}")
            
            return {
                'status': 'SUCCESS',
                'ticker': ticker,
                'model_path': str(model_dir),
                'train_mse': result['train_mse'],
                'val_mse': result['val_mse'],
                'feature_importance': result['top_features'][:5]
            }
    
    except Exception as e:
        logger.error(f"Async XGBoost training failed for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return {
            'status': 'FAILED',
            'ticker': ticker,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def load_trained_models(ticker):
    """
    Load trained models for a ticker from disk
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        dict: Dictionary containing loaded models and metadata
    """
    try:
        model_dir = Path(Config.MODEL_SAVE_PATH) / ticker
        
        if not model_dir.exists():
            raise FileNotFoundError(f"No saved models found for {ticker}")
        
        result = {}
        
        # Load LSTM model if available
        lstm_path = model_dir / 'lstm_model.pth'
        scaler_path = model_dir / 'scaler.pkl'
        lstm_meta_path = model_dir / 'lstm_metadata.json'
        
        if lstm_path.exists() and scaler_path.exists() and lstm_meta_path.exists():
            # Load metadata first
            import json
            with open(lstm_meta_path, 'r') as f:
                lstm_metadata = json.load(f)
            
            # Reconstruct model architecture
            input_dim = lstm_metadata['input_dim']
            hparams = lstm_metadata['hyperparameters']
            
            encoder = Encoder(
                input_dim,
                hparams.get('hidden_dim', 128),
                hparams.get('n_layers', 2),
                hparams.get('dropout_prob', 0.2)
            )
            
            decoder = Decoder(
                input_dim,
                hparams.get('hidden_dim', 128),
                hparams.get('n_layers', 2),
                hparams.get('dropout_prob', 0.2)
            )
            
            device = get_device()
            lstm_model = Seq2Seq(encoder, decoder, device)
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
            lstm_model.eval()
            
            # Load scaler
            scaler = joblib.load(scaler_path)
            
            result['lstm'] = {
                'model': lstm_model,
                'scaler': scaler,
                'metadata': lstm_metadata
            }
            
            logger.info(f"Loaded LSTM model for {ticker}")
        
        # Load XGBoost model if available
        xgb_path = model_dir / 'xgboost_model.pkl'
        xgb_meta_path = model_dir / 'xgboost_metadata.json'
        
        if xgb_path.exists() and xgb_meta_path.exists():
            # Load model and metadata
            xgb_model = joblib.load(xgb_path)
            
            import json
            with open(xgb_meta_path, 'r') as f:
                xgb_metadata = json.load(f)
            
            result['xgboost'] = {
                'model': xgb_model,
                'metadata': xgb_metadata
            }
            
            logger.info(f"Loaded XGBoost model for {ticker}")
        
        if not result:
            raise FileNotFoundError(f"No valid models found for {ticker}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to load models for {ticker}: {e}")
        raise

def get_training_status(ticker):
    """
    Get the current training status for a ticker
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        dict: Training status information
    """
    try:
        registry = get_model_registry()
        
        if ticker not in registry:
            return {'status': 'NO_MODELS', 'message': f'No training history for {ticker}'}
        
        ticker_data = registry[ticker]
        
        status = {
            'ticker': ticker,
            'lstm_trained': 'lstm_trained' in ticker_data,
            'xgboost_trained': 'xgboost_trained' in ticker_data,
            'feature_selection': 'feature_selection' in ticker_data,
            'last_updated': None
        }
        
        # Get timestamps if available
        timestamps = []
        for model_type in ['lstm_trained', 'xgboost_trained']:
            if model_type in ticker_data:
                model_data = ticker_data[model_type]
                if isinstance(model_data, dict) and 'hparams' in model_data:
                    timestamp = model_data['hparams'].get('timestamp')
                    if timestamp:
                        timestamps.append(pd.Timestamp(timestamp))
        
        if timestamps:
            status['last_updated'] = max(timestamps).isoformat()
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get training status for {ticker}: {e}")
        return {'status': 'ERROR', 'error': str(e)}

# UTILITY FUNCTIONS
def cleanup_old_models(ticker, keep_latest=3):
    """
    Clean up old model files, keeping only the latest versions
    """
    try:
        model_dir = Path(Config.MODEL_SAVE_PATH) / ticker
        if not model_dir.exists():
            return
        
        # This is a placeholder for more sophisticated cleanup logic
        # In a production system, you might want to keep models with timestamps
        logger.info(f"Cleanup functionality placeholder for {ticker}")
        
    except Exception as e:
        logger.error(f"Model cleanup failed for {ticker}: {e}")

def validate_model_compatibility(model_metadata, current_features):
    """
    Validate that a saved model is compatible with current feature set
    """
    if not isinstance(model_metadata, dict):
        return False
    
    saved_features = model_metadata.get('selected_features', [])
    if not saved_features:
        return False
    
    # Check if all saved features are available in current features
    missing_features = set(saved_features) - set(current_features)
    if missing_features:
        logger.warning(f"Model has missing features: {missing_features}")
        return False
    
    return True

# MODULE TEST FUNCTION
def test_training_module():
    """
    Test the training module functionality
    """
    try:
        # Test data validation
        test_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [98, 99, 100],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        
        validate_training_data(test_df, ['close', 'volume'])
        logger.info("✓ Data validation test passed")
        
        # Test feature columns
        feature_cols = get_feature_columns()
        assert len(feature_cols) > 0, "No feature columns found"
        logger.info(f"✓ Found {len(feature_cols)} feature columns")
        
        return True
        
    except Exception as e:
        logger.error(f"Training module test failed: {e}")
        return False

if __name__ == "__main__":
    print("Enhanced Training Module")
    print("=" * 40)
    print("Features:")
    print("- Robust LSTM training with early stopping")
    print("- XGBoost training with asymmetric loss weighting")
    print("- Comprehensive error handling and validation")
    print("- Model persistence and loading")
    print("- Progress tracking and MLflow integration")
    print("- Async training with Celery")
    
    # Run module test
    if test_training_module():
        print("✓ Training module loaded and tested successfully")
    else:
        print("✗ Training module test failed")