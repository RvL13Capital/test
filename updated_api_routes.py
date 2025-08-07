from flask import request, jsonify
from project.web_setup import app, r
from project.tuning import tune_hyperparameters_async
from celery.result import AsyncResult
import pandas as pd
import joblib
import torch
import json
from project.config import Config, get_device
from project.features import prepare_inference_from_df, _create_features, get_feature_columns
from project.signals import calculate_breakout_signal
from project.lstm_models import Seq2Seq, Encoder, Decoder
import os
import logging
from jsonschema import validate, ValidationError
from project.storage import get_model_registry
from datetime import datetime, timedelta
import numpy as np
# Import GCS storage utilities
from project.gcs_storage import get_gcs_storage
from google.cloud.exceptions import NotFound

logger = logging.getLogger(__name__)

TRAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "data": {"type": "string"},
        "ticker": {"type": "string"}
    },
    "required": ["data"]
}

# Globaler Modell-Cache
model_cache = {}
CACHE_TTL = 300  # 5 Minuten

@app.before_request
def validate_data():
    if request.method == 'POST' and request.path == '/train':
        try:
            data = request.get_json()
            validate(instance=data, schema=TRAIN_SCHEMA)
            
            df = pd.read_json(data['data'])
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return jsonify({'error': f'Missing required columns: {required_cols}'}), 400
                
            if len(df) < Config.DATA_WINDOW_SIZE + Config.DATA_PREDICTION_LENGTH:
                return jsonify({'error': 'Insufficient data points'}), 400
                
        except ValidationError as e:
            return jsonify({'error': f'Invalid request: {e.message}'}), 400
        except Exception as e:
            return jsonify({'error': f'Data validation failed: {str(e)}'}), 400

def load_model_from_registry(ticker, device):
    """
    Load models from model registry
    UPDATED: Now loads models from GCS instead of local files
    """
    # Cache prüfen
    cache_key = f"{ticker}_models"
    if cache_key in model_cache:
        cache_entry = model_cache[cache_key]
        if (datetime.now() - cache_entry['timestamp']).seconds < CACHE_TTL:
            logger.info(f"Loading models for {ticker} from cache")
            return cache_entry['models']
    
    logger.info(f"Loading models for {ticker} from registry and GCS")
    
    registry = get_model_registry(ticker)
    if not registry:
        logger.warning(f"No model registry found for ticker: {ticker}")
        return None, None, None, None
    
    lstm_entry = registry.get('lstm')
    xgb_entry = registry.get('xgboost')
    scaler_entry = registry.get('lstm_scaler')
    feature_entry = registry.get('feature_selection')
    
    if not all([lstm_entry, xgb_entry, scaler_entry, feature_entry]):
        logger.warning(f"Incomplete model registry for ticker {ticker}: "
                      f"lstm={bool(lstm_entry)}, xgb={bool(xgb_entry)}, "
                      f"scaler={bool(scaler_entry)}, features={bool(feature_entry)}")
        return None, None, None, None
    
    selected_features = feature_entry['hparams']['features']
    
    try:
        # UPDATED: Load models from GCS instead of local files
        gcs_storage = get_gcs_storage()
        
        # Load scaler from GCS
        logger.info(f"Loading scaler from GCS: {scaler_entry['path']}")
        try:
            scaler = gcs_storage.download_joblib_model(scaler_entry['path'])
        except NotFound:
            logger.error(f"Scaler not found in GCS: {scaler_entry['path']}")
            return None, None, None, None
        except Exception as e:
            logger.error(f"Failed to load scaler from GCS: {e}")
            return None, None, None, None
        
        # Load LSTM model from GCS
        logger.info(f"Loading LSTM model from GCS: {lstm_entry['path']}")
        try:
            # Download model state dict from GCS
            lstm_state_dict = gcs_storage.download_pytorch_model(lstm_entry['path'], device)
            
            # Reconstruct LSTM model architecture
            lstm_hparams = lstm_entry['hparams']
            encoder = Encoder(
                lstm_hparams['input_dim'],
                lstm_hparams['hidden_dim'],
                lstm_hparams['n_layers'],
                lstm_hparams['dropout_prob']
            ).to(device)
            
            decoder = Decoder(
                lstm_hparams['input_dim'],
                lstm_hparams['hidden_dim'],
                lstm_hparams['n_layers'],
                lstm_hparams['dropout_prob']
            ).to(device)
            
            model = Seq2Seq(encoder, decoder, device).to(device)
            model.load_state_dict(lstm_state_dict)
            model.eval()
            
        except NotFound:
            logger.error(f"LSTM model not found in GCS: {lstm_entry['path']}")
            return None, None, None, None
        except Exception as e:
            logger.error(f"Failed to load LSTM model from GCS: {e}")
            return None, None, None, None
        
        # Load XGBoost model from GCS
        logger.info(f"Loading XGBoost model from GCS: {xgb_entry['path']}")
        try:
            xgboost_model = gcs_storage.download_joblib_model(xgb_entry['path'])
        except NotFound:
            logger.error(f"XGBoost model not found in GCS: {xgb_entry['path']}")
            return None, None, None, None
        except Exception as e:
            logger.error(f"Failed to load XGBoost model from GCS: {e}")
            return None, None, None, None
        
        # Cache loaded models for future use
        model_cache[cache_key] = {
            'models': (model, xgboost_model, scaler, selected_features),
            'timestamp': datetime.now()
        }
        
        logger.info(f"Successfully loaded all models for {ticker} from GCS")
        return model, xgboost_model, scaler, selected_features
        
    except Exception as e:
        logger.error(f"Unexpected error loading models for {ticker}: {str(e)}")
        return None, None, None, None