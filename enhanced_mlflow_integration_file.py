# project/enhanced_mlflow_integration.py
"""
Enhanced MLflow Integration for comprehensive experiment tracking
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
import torch
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import wraps
import traceback
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for MLflow experiments"""
    experiment_name: str
    run_name: str
    tags: Dict[str, str]
    nested: bool = False
    parent_run_id: Optional[str] = None

class MLflowTracker:
    """
    Comprehensive MLflow tracking for all ML operations
    """
    
    def __init__(self, tracking_uri: str, default_experiment: str = "trading_system"):
        self.tracking_uri = tracking_uri
        self.default_experiment = default_experiment
        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        
        # Create default experiment if not exists
        try:
            mlflow.create_experiment(default_experiment)
        except:
            pass  # Experiment already exists
    
    def track_training(self, func):
        """Decorator for automatic training tracking"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract relevant parameters
            model_type = kwargs.get('model_type', 'unknown')
            ticker = kwargs.get('ticker', 'unknown')
            hparams = kwargs.get('hparams', {})
            
            # Generate run name
            run_name = f"{model_type}_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name):
                try:
                    # Log parameters
                    mlflow.log_params({
                        "model_type": model_type,
                        "ticker": ticker,
                        **hparams
                    })
                    
                    # Log data characteristics
                    if 'df' in kwargs:
                        df = kwargs['df']
                        mlflow.log_params({
                            "data_rows": len(df),
                            "data_columns": len(df.columns),
                            "data_start_date": str(df.index[0]) if hasattr(df, 'index') else 'N/A',
                            "data_end_date": str(df.index[-1]) if hasattr(df, 'index') else 'N/A',
                        })
                    
                    # Execute training
                    result = func(*args, **kwargs)
                    
                    # Log metrics
                    if isinstance(result, dict):
                        self._log_training_metrics(result)
                        self._log_model(result, model_type)
                    
                    # Mark run as successful
                    mlflow.set_tag("status", "success")
                    
                    return result
                    
                except Exception as e:
                    # Log failure
                    mlflow.set_tag("status", "failed")
                    mlflow.log_text(traceback.format_exc(), "error.txt")
                    mlflow.log_metric("error", 1)
                    raise
                    
        return wrapper
    
    def _log_training_metrics(self, result: Dict[str, Any]):
        """Log comprehensive training metrics"""
        # Basic metrics
        if 'train_loss' in result:
            mlflow.log_metric("train_loss", result['train_loss'])
        if 'val_loss' in result:
            mlflow.log_metric("val_loss", result['val_loss'])
        if 'training_time' in result:
            mlflow.log_metric("training_time_seconds", result['training_time'])
        
        # Advanced metrics
        if 'metrics' in result:
            metrics = result['metrics']
            if hasattr(metrics, 'to_dict'):
                metrics = metrics.to_dict()
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                elif isinstance(value, list) and value:
                    # Log time series metrics
                    for i, v in enumerate(value):
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"{key}_step", v, step=i)
        
        # Financial metrics
        if 'sharpe_ratio' in result:
            mlflow.log_metric("sharpe_ratio", result['sharpe_ratio'])
        if 'max_drawdown' in result:
            mlflow.log_metric("max_drawdown", result['max_drawdown'])
        if 'profit_factor' in result:
            mlflow.log_metric("profit_factor", result['profit_factor'])
    
    def _log_model(self, result: Dict[str, Any], model_type: str):
        """Log model artifacts with signatures"""
        try:
            model = result.get('model')
            if model is None:
                return
            
            # Prepare sample input for signature
            sample_input = None
            signature = None
            
            if model_type == 'lstm' and 'scaler' in result:
                # PyTorch model
                if hasattr(model, 'module'):
                    model_to_log = model.module
                else:
                    model_to_log = model
                
                # Create sample input
                sample_input = torch.randn(1, 30, 10)  # batch_size=1, seq_len=30, features=10
                signature = infer_signature(sample_input.numpy(), model_to_log(sample_input).detach().numpy())
                
                mlflow.pytorch.log_model(
                    model_to_log,
                    "model",
                    signature=signature,
                    input_example=sample_input.numpy()
                )
                
                # Log scaler separately
                mlflow.sklearn.log_model(result['scaler'], "scaler")
                
            elif model_type == 'xgboost':
                # XGBoost model
                if 'selected_features' in result:
                    feature_names = result['selected_features']
                    sample_input = pd.DataFrame(
                        np.random.randn(1, len(feature_names)),
                        columns=feature_names
                    )
                    signature = infer_signature(sample_input, model.predict(sample_input))
                
                mlflow.xgboost.log_model(
                    model,
                    "model",
                    signature=signature,
                    input_example=sample_input
                )
                
                # Log feature importance
                if 'feature_importance' in result:
                    mlflow.log_dict(result['feature_importance'], "feature_importance.json")
            
            # Log additional artifacts
            if 'hyperparameters' in result:
                mlflow.log_dict(result['hyperparameters'], "hyperparameters.json")
                
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            mlflow.log_text(str(e), "model_logging_error.txt")
    
    def track_hyperparameter_tuning(self, study_name: str, ticker: str):
        """Track Optuna hyperparameter tuning"""
        def decorator(objective_func):
            @wraps(objective_func)
            def wrapper(trial):
                # Create nested run for each trial
                with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                    # Log trial parameters
                    mlflow.log_params(trial.params)
                    mlflow.log_param("trial_number", trial.number)
                    
                    # Execute objective
                    result = objective_func(trial)
                    
                    # Log result
                    if isinstance(result, tuple):
                        for i, value in enumerate(result):
                            mlflow.log_metric(f"objective_{i}", value)
                    else:
                        mlflow.log_metric("objective", result)
                    
                    return result
            return wrapper
        return decorator
    
    def track_backtesting(self, strategy_name: str, ticker: str, 
                          timeframe: str, config: Dict[str, Any]):
        """Track backtesting results comprehensively"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                run_name = f"backtest_{strategy_name}_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                with mlflow.start_run(run_name=run_name):
                    # Log configuration
                    mlflow.log_params({
                        "strategy": strategy_name,
                        "ticker": ticker,
                        "timeframe": timeframe,
                        **config
                    })
                    
                    # Execute backtesting
                    result = func(*args, **kwargs)
                    
                    # Log comprehensive metrics
                    if isinstance(result, dict):
                        self._log_backtest_metrics(result)
                        
                        # Log equity curve
                        if 'equity_curve' in result:
                            self._log_equity_curve(result['equity_curve'])
                        
                        # Log trade analysis
                        if 'trades' in result:
                            self._log_trade_analysis(result['trades'])
                    
                    return result
            return wrapper
        return decorator
    
    def _log_backtest_metrics(self, result: Dict[str, Any]):
        """Log comprehensive backtesting metrics"""
        metrics_to_log = [
            'total_return', 'annual_return', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'win_rate', 'profit_factor', 'total_trades',
            'winning_trades', 'losing_trades', 'avg_win', 'avg_loss',
            'largest_win', 'largest_loss', 'consecutive_wins', 'consecutive_losses',
            'exposure_time', 'calmar_ratio', 'omega_ratio', 'ulcer_index'
        ]
        
        for metric in metrics_to_log:
            if metric in result:
                mlflow.log_metric(metric, result[metric])
        
        # Log risk metrics
        if 'risk_metrics' in result:
            for key, value in result['risk_metrics'].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"risk_{key}", value)
    
    def _log_equity_curve(self, equity_curve: pd.Series):
        """Log equity curve as artifact and metrics"""
        # Save as CSV
        equity_curve.to_csv("equity_curve.csv")
        mlflow.log_artifact("equity_curve.csv")
        
        # Log key points
        mlflow.log_metric("equity_start", equity_curve.iloc[0])
        mlflow.log_metric("equity_end", equity_curve.iloc[-1])
        mlflow.log_metric("equity_max", equity_curve.max())
        mlflow.log_metric("equity_min", equity_curve.min())
        
        # Log time series (sample if too long)
        step_size = max(1, len(equity_curve) // 1000)  # Max 1000 points
        for i in range(0, len(equity_curve), step_size):
            mlflow.log_metric("equity", equity_curve.iloc[i], step=i)
    
    def _log_trade_analysis(self, trades: List[Dict]):
        """Log detailed trade analysis"""
        if not trades:
            return
        
        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame(trades)
        
        # Save trades log
        trades_df.to_csv("trades.csv")
        mlflow.log_artifact("trades.csv")
        
        # Analyze by various dimensions
        if 'symbol' in trades_df.columns:
            symbol_performance = trades_df.groupby('symbol')['pnl'].agg(['sum', 'mean', 'std'])
            mlflow.log_dict(symbol_performance.to_dict(), "symbol_performance.json")
        
        if 'strategy_signal' in trades_df.columns:
            signal_performance = trades_df.groupby('strategy_signal')['pnl'].agg(['count', 'sum', 'mean'])
            mlflow.log_dict(signal_performance.to_dict(), "signal_performance.json")
    
    def log_model_comparison(self, models: List[Tuple[str, Dict]], 
                            test_data: pd.DataFrame):
        """Compare multiple models on same test data"""
        comparison_run_name = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=comparison_run_name):
            comparison_results = {}
            
            for model_name, model_info in models:
                with mlflow.start_run(nested=True, run_name=f"compare_{model_name}"):
                    # Log model info
                    mlflow.log_params({
                        "model_name": model_name,
                        "model_type": model_info.get('type', 'unknown')
                    })
                    
                    # Get predictions
                    model = model_info['model']
                    predictions = model_info['predict_func'](model, test_data)
                    
                    # Calculate metrics
                    metrics = self._calculate_comparison_metrics(
                        test_data['target'].values,
                        predictions
                    )
                    
                    # Log metrics
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(metric_name, value)
                    
                    comparison_results[model_name] = metrics
            
            # Log comparison summary
            mlflow.log_dict(comparison_results, "comparison_results.json")
            
            # Find best model
            best_model = min(comparison_results.items(), 
                           key=lambda x: x[1].get('mse', float('inf')))
            mlflow.set_tag("best_model", best_model[0])
            
            return comparison_results
    
    def _calculate_comparison_metrics(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive comparison metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'directional_accuracy': np.mean(np.sign(y_true[1:] - y_true[:-1]) == 
                                           np.sign(y_pred[1:] - y_pred[:-1]))
        }
        
        return metrics
    
    def create_model_registry_entry(self, run_id: str, model_name: str, 
                                   tags: Dict[str, str] = None):
        """Register model in MLflow Model Registry"""
        # Register model
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Add tags to model version
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    model_name, 
                    model_version.version,
                    key, 
                    value
                )
        
        return model_version
    
    def get_production_model(self, model_name: str):
        """Get current production model from registry"""
        client = MlflowClient()
        
        # Get latest production version
        versions = client.get_latest_versions(
            model_name, 
            stages=["Production"]
        )
        
        if not versions:
            return None
        
        # Load model
        model_uri = f"models:/{model_name}/Production"
        return mlflow.pytorch.load_model(model_uri)

# Integration with existing training code
class EnhancedTrainingManager:
    """
    Enhanced training manager with full MLflow integration
    """
    
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.tracker = mlflow_tracker
        self.active_experiments = {}
    
    def train_model_with_tracking(self, df: pd.DataFrame, model_type: str,
                                 selected_features: List[str], hparams: Dict,
                                 ticker: str, validation_split: float = 0.2,
                                 **kwargs):
        """
        Train model with comprehensive MLflow tracking
        """
        # Add tracking decorator
        @self.tracker.track_training
        def _train():
            # Import actual training functions
            from project.training_optimized import (
                build_and_train_lstm_optimized,
                build_and_train_xgboost_optimized
            )
            
            if model_type == 'lstm':
                result = build_and_train_lstm_optimized(
                    df, selected_features, hparams, validation_split
                )
            elif model_type == 'xgboost':
                result = build_and_train_xgboost_optimized(
                    df, hparams, validation_split
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return result
        
        # Execute with tracking
        kwargs.update({
            'model_type': model_type,
            'ticker': ticker,
            'hparams': hparams,
            'df': df
        })
        
        return _train(**kwargs)

# Usage example
def setup_mlflow_tracking(config: Dict[str, Any]) -> MLflowTracker:
    """
    Setup MLflow tracking with configuration
    """
    tracker = MLflowTracker(
        tracking_uri=config.get('mlflow_tracking_uri', 'http://localhost:5001'),
        default_experiment=config.get('default_experiment', 'trading_system')
    )
    
    # Set global tags
    mlflow.set_tags({
        "environment": config.get('environment', 'development'),
        "version": config.get('version', '1.0.0'),
        "team": "trading-ml"
    })
    
    return tracker

# Integrate with Celery tasks
def track_celery_task(tracker: MLflowTracker):
    """Decorator for Celery task tracking"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            task_name = func.__name__
            run_name = f"celery_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("task_type", "celery")
                mlflow.set_tag("task_name", task_name)
                
                try:
                    result = func(*args, **kwargs)
                    mlflow.set_tag("status", "success")
                    return result
                except Exception as e:
                    mlflow.set_tag("status", "failed")
                    mlflow.log_text(str(e), "error.txt")
                    raise
        return wrapper
    return decorator
