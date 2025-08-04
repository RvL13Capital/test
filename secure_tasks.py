# project/tasks.py
import pandas as pd
import logging
import hashlib
import json
import redis
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from celery import Task
from celery.exceptions import Retry
import optuna

# Import from other project modules
from .extensions import celery
from .training import build_and_train_lstm, build_and_train_xgboost
from .features import get_feature_columns
from .storage import get_gcs_storage
from .tuning import objective_lstm, objective_xgboost
from .config import Config, get_default_hparams

logger = logging.getLogger(__name__)

class SecureTask(Task):
    """Base class for secure, idempotent tasks with audit logging"""
    
    def __init__(self):
        # Redis client for deduplication and caching
        try:
            self.redis_client = redis.from_url(Config.CELERY_RESULT_BACKEND)
            self.redis_client.ping()  # Test connection
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _generate_request_hash(self, **kwargs) -> str:
        """Generate unique hash for request deduplication"""
        # Remove non-deterministic fields
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['user_id', 'request_timestamp']}
        
        # Create deterministic string
        request_str = json.dumps(clean_kwargs, sort_keys=True, default=str)
        return hashlib.sha256(request_str.encode()).hexdigest()
    
    def _is_duplicate_request(self, request_hash: str, user_id: str) -> bool:
        """Check if request was already processed recently"""
        if not self.redis_client:
            return False
        
        key = f"request_processed:{user_id}:{request_hash}"
        return self.redis_client.exists(key)
    
    def _mark_request_processed(self, request_hash: str, user_id: str, ttl_hours: int = 24):
        """Mark request as processed with TTL"""
        if not self.redis_client:
            return
        
        key = f"request_processed:{user_id}:{request_hash}"
        self.redis_client.setex(key, timedelta(hours=ttl_hours), "processed")
    
    def _validate_user_permissions(self, user_id: str, operation: str) -> bool:
        """Validate user has permission for operation"""
        # In a real system, this would check against a proper user database
        # For now, we'll implement basic validation
        if not user_id or user_id == 'anonymous':
            return False
        
        # Check if operation is allowed
        allowed_operations = ['train_model', 'tune_hyperparams', 'view_models']
        return operation in allowed_operations
    
    def _audit_log(self, event_type: str, user_id: str, details: Dict[str, Any], success: bool = True):
        """Log audit events for security compliance"""
        if Config.AUDIT_LOG_ENABLED:
            audit_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': event_type,
                'user_id': user_id,
                'task_id': self.request.id if hasattr(self, 'request') else None,
                'success': success,
                'details': details
            }
            
            # In production, send to proper audit system (SIEM, etc.)
            logger.info(f"AUDIT: {json.dumps(audit_data)}")
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame meets security requirements"""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if len(df) < Config.MIN_DATAFRAME_SIZE:
            raise ValueError(f"DataFrame too small: {len(df)} < {Config.MIN_DATAFRAME_SIZE}")
        
        if len(df) > Config.MAX_DATAFRAME_SIZE:
            raise ValueError(f"DataFrame too large: {len(df)} > {Config.MAX_DATAFRAME_SIZE}")
        
        # Check for required columns (basic validation)
        required_columns = ['close', 'volume']  # Minimum required
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return True
    
    def _validate_hyperparameters(self, model_type: str, hparams: Dict) -> Dict:
        """Validate and sanitize hyperparameters"""
        # Use config validation
        Config.validate_hyperparams(model_type, hparams)
        
        # Additional security checks
        if model_type == 'lstm':
            # Ensure reasonable bounds
            hparams['hidden_dim'] = min(hparams.get('hidden_dim', 128), Config.LSTM_HPARAMS['max_hidden_dim'])
            hparams['epochs'] = min(hparams.get('epochs', 50), Config.LSTM_HPARAMS['max_epochs'])
            hparams['n_layers'] = min(max(hparams.get('n_layers', 2), 1), 5)  # 1-5 layers
            hparams['dropout_prob'] = max(min(hparams.get('dropout_prob', 0.3), 0.8), 0.0)  # 0-0.8
            
        elif model_type == 'xgboost':
            hparams['n_estimators'] = min(hparams.get('n_estimators', 100), Config.XGBOOST_HPARAMS['max_n_estimators'])
            hparams['max_depth'] = min(max(hparams.get('max_depth', 5), 1), 15)  # 1-15 depth
            hparams['learning_rate'] = max(min(hparams.get('learning_rate', 0.05), 1.0), 0.001)  # 0.001-1.0
        
        return hparams

@celery.task(bind=True, base=SecureTask, autoretry_for=(Exception,), 
             retry_kwargs={'max_retries': 3, 'countdown': 60}, 
             time_limit=Config.CELERY_TASK_TIME_LIMIT,
             soft_time_limit=Config.CELERY_TASK_SOFT_TIME_LIMIT)
def tune_and_train_async(self, df_json: str, model_type: str, ticker: str, 
                        user_id: str, request_hash: Optional[str] = None):
    """
    Secure hyperparameter tuning and training task with comprehensive validation
    """
    start_time = datetime.utcnow()
    
    try:
        # Generate request hash if not provided
        if not request_hash:
            request_hash = self._generate_request_hash(
                df_json=df_json[:100],  # Use first 100 chars for hash
                model_type=model_type,
                ticker=ticker
            )
        
        # Check for duplicate requests
        if self._is_duplicate_request(request_hash, user_id):
            self._audit_log('duplicate_request', user_id, {
                'operation': 'tune_and_train',
                'ticker': ticker,
                'model_type': model_type
            })
            return {
                'status': 'DUPLICATE',
                'message': 'Request already processed',
                'ticker': ticker
            }
        
        # Validate permissions
        if not self._validate_user_permissions(user_id, 'tune_hyperparams'):
            self._audit_log('permission_denied', user_id, {
                'operation': 'tune_and_train',
                'ticker': ticker,
                'model_type': model_type
            }, success=False)
            return {
                'status': 'FORBIDDEN',
                'message': 'Insufficient permissions for hyperparameter tuning'
            }
        
        self.update_state(state='PROGRESS', meta={
            'status': 'Validating data and permissions...',
            'progress': 5,
            'user_id': user_id
        })
        
        # Load and validate data
        df = pd.read_json(df_json, orient='split')
        self._validate_dataframe(df)
        
        # Validate model type and ticker
        if model_type not in ['lstm', 'xgboost']:
            raise ValueError(f"Invalid model type: {model_type}")
        
        if ticker not in Config.ALLOWED_TICKERS:
            raise ValueError(f"Ticker {ticker} not allowed")
        
        # Select objective function
        if model_type == 'lstm':
            objective_func = objective_lstm
            build_func = build_and_train_lstm
        else:
            objective_func = objective_xgboost
            build_func = build_and_train_xgboost
        
        # --- Tuning Phase ---
        self.update_state(state='PROGRESS', meta={
            'status': f'Starting Optuna tuning for {model_type}...',
            'progress': 20,
            'estimated_time': Config.OPTUNA_TIMEOUT
        })
        
        # Limit trials based on config
        n_trials = min(Config.OPTUNA_N_TRIALS, Config.MAX_OPTUNA_TRIALS)
        
        # Create study with timeout
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective_func(trial, df), 
            n_trials=n_trials,
            timeout=Config.OPTUNA_TIMEOUT
        )
        
        best_hparams = study.best_params
        
        # Validate optimized hyperparameters
        best_hparams = self._validate_hyperparameters(model_type, best_hparams)
        
        self._audit_log('tuning_completed', user_id, {
            'ticker': ticker,
            'model_type': model_type,
            'best_params': best_hparams,
            'n_trials': len(study.trials)
        })
        
        logger.info(f"Optuna tuning completed for {model_type}: {best_hparams}")
        
        # --- Training Phase ---
        self.update_state(state='PROGRESS', meta={
            'status': f'Training final {model_type} model with optimized parameters...',
            'progress': 70
        })
        
        # Train final model with best parameters
        if model_type == 'lstm':
            # Add epochs from config (not tuned for time reasons)
            best_hparams['epochs'] = Config.LSTM_HPARAMS['epochs']
            final_result = build_func(df, get_feature_columns(), best_hparams)
        else:  # xgboost
            final_result = build_func(df, best_hparams)
        
        # --- Storage Phase ---
        self.update_state(state='PROGRESS', meta={
            'status': 'Securely storing optimized model...',
            'progress': 90
        })
        
        gcs = get_gcs_storage()
        model_paths = {}
        
        if gcs.client:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            
            if model_type == 'lstm':
                model_path = f"models/{ticker}/lstm_tuned_{timestamp}.pth"
                scaler_path = f"models/{ticker}/scaler_tuned_{timestamp}.pkl"
                
                # Upload with user context
                gcs.upload_pytorch_model(
                    final_result['model'].state_dict(), 
                    model_path, 
                    user_id=user_id,
                    model_info={
                        'ticker': ticker,
                        'model_type': model_type,
                        'hyperparams': best_hparams,
                        'tuned': True
                    }
                )
                gcs.upload_joblib(final_result['scaler'], scaler_path, user_id=user_id)
                
                model_paths = {
                    'model_path': model_path,
                    'scaler_path': scaler_path
                }
            else:  # xgboost
                model_path = f"models/{ticker}/xgboost_tuned_{timestamp}.joblib"
                gcs.upload_joblib(final_result['model'], model_path, user_id=user_id)
                model_paths = {'model_path': model_path}
        
        # Mark request as processed
        self._mark_request_processed(request_hash, user_id)
        
        # Final audit log
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        self._audit_log('training_completed', user_id, {
            'ticker': ticker,
            'model_type': model_type,
            'execution_time_seconds': execution_time,
            'model_paths': model_paths,
            'tuned': True
        })
        
        return {
            'status': 'SUCCESS',
            'ticker': ticker,
            'model_type': model_type,
            'best_params': best_hparams,
            'model_paths': model_paths,
            'execution_time': execution_time,
            'user_id': user_id
        }
    
    except Exception as e:
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        self._audit_log('training_failed', user_id, {
            'ticker': ticker,
            'model_type': model_type,
            'error': str(e),
            'execution_time_seconds': execution_time
        }, success=False)
        
        logger.error(f"Tune-and-train task failed for user {user_id}: {e}")
        
        # Exponential backoff for retries
        if self.request.retries < self.max_retries:
            countdown = 2 ** self.request.retries * 60
            raise self.retry(countdown=countdown, exc=e)
        
        return {
            'status': 'FAILED',
            'error': str(e),
            'ticker': ticker,
            'execution_time': execution_time
        }

@celery.task(bind=True, base=SecureTask, autoretry_for=(Exception,),
             retry_kwargs={'max_retries': 2, 'countdown': 30},
             time_limit=Config.CELERY_TASK_TIME_LIMIT,
             soft_time_limit=Config.CELERY_TASK_SOFT_TIME_LIMIT)
def train_lstm_async(self, df_json: str, hparams: Dict, ticker: str, user_id: str):
    """
    Secure LSTM training task with default hyperparameters
    """
    start_time = datetime.utcnow()
    
    try:
        # Generate request hash for deduplication
        request_hash = self._generate_request_hash(
            df_json=df_json[:100],
            hparams=hparams,
            ticker=ticker,
            model_type='lstm'
        )
        
        # Check for duplicates
        if self._is_duplicate_request(request_hash, user_id):
            return {'status': 'DUPLICATE', 'message': 'Request already processed'}
        
        # Validate permissions
        if not self._validate_user_permissions(user_id, 'train_model'):
            self._audit_log('permission_denied', user_id, {
                'operation': 'train_lstm',
                'ticker': ticker
            }, success=False)
            return {'status': 'FORBIDDEN', 'message': 'Insufficient permissions'}
        
        self.update_state(state='PROGRESS', meta={
            'status': 'Loading and validating data...',
            'progress': 10,
            'user_id': user_id
        })
        
        # Load and validate data
        df = pd.read_json(df_json, orient='split')
        self._validate_dataframe(df)
        
        # Validate and sanitize hyperparameters
        hparams = self._validate_hyperparameters('lstm', hparams)
        
        if ticker not in Config.ALLOWED_TICKERS:
            raise ValueError(f"Ticker {ticker} not allowed")
        
        self.update_state(state='PROGRESS', meta={
            'status': 'Training LSTM model with default parameters...',
            'progress': 30
        })
        
        # Train model
        result = build_and_train_lstm(df, get_feature_columns(), hparams)
        
        self.update_state(state='PROGRESS', meta={
            'status': 'Securely storing model...',
            'progress': 80
        })
        
        # Store model securely
        gcs = get_gcs_storage()
        model_paths = {}
        
        if gcs.client:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_path = f"models/{ticker}/lstm_{timestamp}.pth"
            scaler_path = f"models/{ticker}/scaler_{timestamp}.pkl"
            
            gcs.upload_pytorch_model(
                result['model'].state_dict(),
                model_path,
                user_id=user_id,
                model_info={
                    'ticker': ticker,
                    'model_type': 'lstm',
                    'hyperparams': hparams,
                    'tuned': False
                }
            )
            gcs.upload_joblib(result['scaler'], scaler_path, user_id=user_id)
            
            model_paths = {
                'model_path': model_path,
                'scaler_path': scaler_path
            }
        
        # Mark as processed
        self._mark_request_processed(request_hash, user_id)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        self._audit_log('training_completed', user_id, {
            'ticker': ticker,
            'model_type': 'lstm',
            'execution_time_seconds': execution_time,
            'model_paths': model_paths,
            'tuned': False
        })
        
        return {
            'status': 'SUCCESS',
            'ticker': ticker,
            'model_type': 'lstm',
            'model_paths': model_paths,
            'execution_time': execution_time,
            'user_id': user_id
        }
    
    except Exception as e:
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        self._audit_log('training_failed', user_id, {
            'ticker': ticker,
            'model_type': 'lstm',
            'error': str(e),
            'execution_time_seconds': execution_time
        }, success=False)
        
        logger.error(f"LSTM training task failed for user {user_id}: {e}")
        
        if self.request.retries < self.max_retries:
            countdown = 2 ** self.request.retries * 30
            raise self.retry(countdown=countdown, exc=e)
        
        return {
            'status': 'FAILED',
            'error': str(e),
            'ticker': ticker,
            'execution_time': execution_time
        }

@celery.task(bind=True, base=SecureTask, autoretry_for=(Exception,),
             retry_kwargs={'max_retries': 2, 'countdown': 30},
             time_limit=Config.CELERY_TASK_TIME_LIMIT,
             soft_time_limit=Config.CELERY_TASK_SOFT_TIME_LIMIT)
def train_xgboost_async(self, df_json: str, hparams: Dict, ticker: str, user_id: str):
    """
    Secure XGBoost training task with default hyperparameters
    """
    start_time = datetime.utcnow()
    
    try:
        # Generate request hash for deduplication
        request_hash = self._generate_request_hash(
            df_json=df_json[:100],
            hparams=hparams,
            ticker=ticker,
            model_type='xgboost'
        )
        
        # Check for duplicates
        if self._is_duplicate_request(request_hash, user_id):
            return {'status': 'DUPLICATE', 'message': 'Request already processed'}
        
        # Validate permissions
        if not self._validate_user_permissions(user_id, 'train_model'):
            self._audit_log('permission_denied', user_id, {
                'operation': 'train_xgboost',
                'ticker': ticker
            }, success=False)
            return {'status': 'FORBIDDEN', 'message': 'Insufficient permissions'}
        
        self.update_state(state='PROGRESS', meta={
            'status': 'Loading and validating data...',
            'progress': 10,
            'user_id': user_id
        })
        
        # Load and validate data
        df = pd.read_json(df_json, orient='split')
        self._validate_dataframe(df)
        
        # Validate hyperparameters
        hparams = self._validate_hyperparameters('xgboost', hparams)
        
        if ticker not in Config.ALLOWED_TICKERS:
            raise ValueError(f"Ticker {ticker} not allowed")
        
        self.update_state(state='PROGRESS', meta={
            'status': 'Training XGBoost model with default parameters...',
            'progress': 30
        })
        
        # Train model
        result = build_and_train_xgboost(df, hparams)
        
        self.update_state(state='PROGRESS', meta={
            'status': 'Securely storing model...',
            'progress': 80
        })
        
        # Store model securely
        gcs = get_gcs_storage()
        model_paths = {}
        
        if gcs.client:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_path = f"models/{ticker}/xgboost_{timestamp}.joblib"
            
            gcs.upload_joblib(
                result['model'],
                model_path,
                user_id=user_id
            )
            
            model_paths = {'model_path': model_path}
        
        # Mark as processed
        self._mark_request_processed(request_hash, user_id)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        self._audit_log('training_completed', user_id, {
            'ticker': ticker,
            'model_type': 'xgboost',
            'execution_time_seconds': execution_time,
            'model_paths': model_paths,
            'tuned': False
        })
        
        return {
            'status': 'SUCCESS',
            'ticker': ticker,
            'model_type': 'xgboost',
            'model_paths': model_paths,
            'execution_time': execution_time,
            'user_id': user_id
        }
    
    except Exception as e:
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        self._audit_log('training_failed', user_id, {
            'ticker': ticker,
            'model_type': 'xgboost',
            'error': str(e),
            'execution_time_seconds': execution_time
        }, success=False)
        
        logger.error(f"XGBoost training task failed for user {user_id}: {e}")
        
        if self.request.retries < self.max_retries:
            countdown = 2 ** self.request.retries * 30
            raise self.retry(countdown=countdown, exc=e)
        
        return {
            'status': 'FAILED',
            'error': str(e),
            'ticker': ticker,
            'execution_time': execution_time
        }