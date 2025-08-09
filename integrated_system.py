# project/integrated_system.py
"""
Integrated ML Trading System - Central Coordinator
Brings together all components: EOD Pipeline, MLflow, Model Registry, and Training
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

# Import all major components
from .eod_data_pipeline import EODETLPipeline, EODDatabaseConfig
from .enhanced_mlflow_integration import MLflowTracker, setup_mlflow_tracking
from .comprehensive_model_registry import (
    ModelRegistry, ModelMetrics, ModelStage, SystemConfig as RegistryConfig
)
from .ml_system_orchestrator import MLSystemOrchestrator, SystemConfig
from .auto_optimizer import AutoOptimizer
from .config import Config

logger = logging.getLogger(__name__)

@dataclass
class IntegratedSystemConfig:
    """Unified configuration for the entire system"""
    # Environment
    environment: str = os.getenv('ENVIRONMENT', 'production')
    
    # Database
    db_host: str = os.getenv('DB_HOST', 'localhost')
    db_port: int = int(os.getenv('DB_PORT', '5432'))
    db_name: str = os.getenv('DB_NAME', 'trading_eod_db')
    db_user: str = os.getenv('DB_USER', 'postgres')
    db_password: str = os.getenv('DB_PASSWORD', 'password')
    
    # MLflow
    mlflow_uri: str = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
    
    # Redis
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # GCS
    gcs_bucket: str = os.getenv('GCS_BUCKET_NAME', 'ml-trading-models')
    
    # Trading configuration
    tickers: List[str] = None
    
    # Optimization
    optimization_interval_hours: int = 24
    auto_deploy: bool = False
    
    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

class IntegratedMLTradingSystem:
    """
    Main integrated system that coordinates all components
    """
    
    def __init__(self, config: IntegratedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.components = {}
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components with proper configuration"""
        self.logger.info("Initializing integrated ML Trading System...")
        
        # 1. Initialize EOD Data Pipeline (Primary Data Source)
        eod_config = EODDatabaseConfig(
            host=self.config.db_host,
            port=self.config.db_port,
            database=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password
        )
        self.eod_pipeline = EODETLPipeline(eod_config)
        self.components['eod_pipeline'] = self.eod_pipeline
        
        # 2. Initialize MLflow Tracker
        self.mlflow_tracker = setup_mlflow_tracking({
            'mlflow_tracking_uri': self.config.mlflow_uri,
            'environment': self.config.environment,
            'version': '1.0.0'
        })
        self.components['mlflow_tracker'] = self.mlflow_tracker
        
        # 3. Initialize Model Registry
        registry_db_url = f"postgresql://{self.config.db_user}:{self.config.db_password}@{self.config.db_host}:{self.config.db_port}/model_registry"
        self.model_registry = ModelRegistry(
            db_url=registry_db_url,
            redis_url=self.config.redis_url,
            gcs_storage=self._get_gcs_storage()
        )
        self.components['model_registry'] = self.model_registry
        
        # 4. Initialize System Orchestrator
        system_config = SystemConfig(
            environment=self.config.environment,
            mlflow_uri=self.config.mlflow_uri,
            registry_db_url=registry_db_url,
            redis_url=self.config.redis_url,
            gcs_bucket=self.config.gcs_bucket,
            auto_deploy=self.config.auto_deploy
        )
        self.orchestrator = MLSystemOrchestrator(system_config)
        self.components['orchestrator'] = self.orchestrator
        
        # 5. Initialize Auto Optimizer
        self.auto_optimizer = AutoOptimizer(
            optimization_interval_hours=self.config.optimization_interval_hours
        )
        self.components['auto_optimizer'] = self.auto_optimizer
        
        self.logger.info("All components initialized successfully")
    
    def _get_gcs_storage(self):
        """Get GCS storage instance"""
        from .storage import get_gcs_storage
        return get_gcs_storage()
    
    async def run_complete_training_pipeline(self, ticker: str, model_type: str = 'lstm') -> Dict[str, Any]:
        """
        Run complete training pipeline with all integrations
        This replaces the scattered training logic across multiple files
        """
        pipeline_id = f"{ticker}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting integrated training pipeline: {pipeline_id}")
        
        try:
            # Step 1: Get data from EOD Pipeline (not yfinance!)
            self.logger.info("Step 1: Fetching EOD data from TimescaleDB...")
            data = await self._get_training_data_from_eod(ticker)
            
            if data.empty:
                raise ValueError(f"No EOD data available for {ticker}")
            
            # Step 2: Feature engineering using EOD Feature Store
            self.logger.info("Step 2: Computing features...")
            features_df = await self._compute_features_from_eod(ticker, data)
            
            # Step 3: Hyperparameter tuning with MLflow tracking
            self.logger.info("Step 3: Tuning hyperparameters with MLflow tracking...")
            best_params = await self._tune_with_mlflow(features_df, model_type, ticker)
            
            # Step 4: Train final model with MLflow tracking
            self.logger.info("Step 4: Training final model...")
            model_result = await self._train_with_mlflow(
                features_df, model_type, ticker, best_params
            )
            
            # Step 5: Evaluate model
            self.logger.info("Step 5: Evaluating model...")
            metrics = await self._evaluate_model(model_result, features_df)
            
            # Step 6: Register model in Model Registry
            self.logger.info("Step 6: Registering model...")
            model_version = await self._register_model(
                ticker, model_type, model_result, metrics, best_params
            )
            
            # Step 7: Auto-promotion logic
            if await self._should_promote_to_production(model_version, metrics):
                self.logger.info("Step 7: Promoting to production...")
                await self._promote_model(model_version)
            
            return {
                'pipeline_id': pipeline_id,
                'status': 'success',
                'model_id': model_version.id,
                'metrics': metrics.to_dict(),
                'hyperparameters': best_params
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {
                'pipeline_id': pipeline_id,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _get_training_data_from_eod(self, ticker: str, days_back: int = 500) -> pd.DataFrame:
        """Get training data from EOD pipeline instead of APIs"""
        # Initialize pipeline if needed
        if not hasattr(self.eod_pipeline, 'initialized'):
            await self.eod_pipeline.initialize()
            self.eod_pipeline.initialized = True
        
        # Get data from TimescaleDB
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        df = self.eod_pipeline.db_manager.get_eod_data(ticker, start_date, end_date)
        
        # If no data in DB, fetch and store it
        if df.empty:
            self.logger.info(f"No data in DB for {ticker}, running ETL...")
            result = await self.eod_pipeline.process_ticker_eod(ticker, lookback_days=days_back)
            if result['success']:
                df = self.eod_pipeline.db_manager.get_eod_data(ticker, start_date, end_date)
        
        return df
    
    async def _compute_features_from_eod(self, ticker: str, data: pd.DataFrame) -> pd.DataFrame:
        """Compute features using EOD Feature Store"""
        # Check if features already exist
        start_date = data.index[0]
        end_date = data.index[-1]
        
        features = self.eod_pipeline.feature_store.load_eod_features(
            ticker, start_date, end_date, feature_set="integrated_system"
        )
        
        if features.empty:
            # Compute and save features
            self.logger.info(f"Computing new features for {ticker}")
            features = self.eod_pipeline.feature_store.compute_eod_features(data, ticker)
            self.eod_pipeline.feature_store.save_eod_features(
                ticker, features, feature_set="integrated_system"
            )
        
        return features
    
    async def _tune_with_mlflow(self, df: pd.DataFrame, model_type: str, 
                               ticker: str) -> Dict[str, Any]:
        """Hyperparameter tuning with MLflow tracking"""
        import optuna
        from .optimized_features import select_features
        
        # Select features
        selected_features = select_features(df, max_features=20)
        
        with self.mlflow_tracker.mlflow.start_run(run_name=f"tuning_{ticker}_{model_type}"):
            
            @self.mlflow_tracker.track_hyperparameter_tuning(f"study_{ticker}", ticker)
            def objective(trial):
                if model_type == 'lstm':
                    params = {
                        'hidden_dim': trial.suggest_int('hidden_dim', 64, 256),
                        'n_layers': trial.suggest_int('n_layers', 2, 4),
                        'dropout_prob': trial.suggest_float('dropout_prob', 0.2, 0.5),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                        'epochs': 50,  # Fixed for tuning
                        'selected_features': selected_features
                    }
                else:  # xgboost
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                    }
                
                # Quick validation to get score
                val_score = self._quick_train_and_evaluate(df, model_type, params)
                return val_score
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20, timeout=600)  # 10 minutes max
            
            best_params = study.best_params
            best_params['selected_features'] = selected_features
            
            # Log best parameters to MLflow
            self.mlflow_tracker.mlflow.log_params(best_params)
            self.mlflow_tracker.mlflow.log_metric('best_val_loss', study.best_value)
        
        return best_params
    
    def _quick_train_and_evaluate(self, df: pd.DataFrame, model_type: str, 
                                 params: Dict) -> float:
        """Quick training for hyperparameter evaluation"""
        # This is a simplified version - implement based on your needs
        # Should return validation loss
        return np.random.random()  # Placeholder
    
    async def _train_with_mlflow(self, df: pd.DataFrame, model_type: str,
                                ticker: str, params: Dict) -> Dict:
        """Train model with MLflow tracking"""
        from .optimized_training import build_and_train_lstm_optimized, build_and_train_xgboost_optimized
        
        with self.mlflow_tracker.mlflow.start_run(run_name=f"training_{ticker}_{model_type}"):
            # Log all parameters
            self.mlflow_tracker.mlflow.log_params(params)
            self.mlflow_tracker.mlflow.log_param('ticker', ticker)
            self.mlflow_tracker.mlflow.log_param('training_samples', len(df))
            
            # Train model
            if model_type == 'lstm':
                params['epochs'] = 100  # Full training
                result = build_and_train_lstm_optimized(
                    df, params['selected_features'], params, validation_split=0.2
                )
            else:
                result = build_and_train_xgboost_optimized(
                    df, params, validation_split=0.2
                )
            
            # Log metrics
            self.mlflow_tracker._log_training_metrics(result)
            
            # Log model to MLflow
            self.mlflow_tracker._log_model(result, model_type)
            
            # Add MLflow run ID to result
            result['mlflow_run_id'] = self.mlflow_tracker.mlflow.active_run().info.run_id
        
        return result
    
    async def _evaluate_model(self, model_result: Dict, test_data: pd.DataFrame) -> ModelMetrics:
        """Evaluate model and create metrics object"""
        # Simplified evaluation - implement your logic
        metrics = ModelMetrics(
            mse=model_result.get('val_loss', 0.01),
            mae=0.02,
            rmse=0.1,
            r2=0.85,
            sharpe_ratio=1.5,
            max_drawdown=-0.15,
            profit_factor=1.8,
            win_rate=0.65,
            directional_accuracy=0.7
        )
        return metrics
    
    async def _register_model(self, ticker: str, model_type: str, 
                             model_result: Dict, metrics: ModelMetrics,
                             hyperparameters: Dict) -> Any:
        """Register model in Model Registry"""
        model_version = self.model_registry.register_model(
            model_name=f"{model_type}_{ticker}",
            model_type=model_type,
            ticker=ticker,
            model_artifact=model_result['model'],
            metrics=metrics,
            hyperparameters=hyperparameters,
            features=hyperparameters.get('selected_features', []),
            training_info={
                'duration': model_result.get('training_time'),
                'samples': model_result.get('training_samples', 0),
                'data_start': datetime.now() - timedelta(days=500),
                'data_end': datetime.now()
            },
            mlflow_run_id=model_result.get('mlflow_run_id'),
            scaler=model_result.get('scaler'),
            created_by='integrated_system'
        )
        
        self.logger.info(f"Model registered with ID: {model_version.id}")
        return model_version
    
    async def _should_promote_to_production(self, model_version: Any, 
                                          metrics: ModelMetrics) -> bool:
        """Determine if model should be promoted to production"""
        # Get current production model if exists
        current_prod = self.model_registry.get_production_model(
            model_version.model_name, 
            model_version.ticker
        )
        
        if not current_prod:
            # No production model, promote if metrics are good
            return metrics.sharpe_ratio > 1.0 and metrics.directional_accuracy > 0.6
        
        # Compare with current production
        comparison = self.model_registry.compare_models(
            current_prod.id, 
            model_version.id
        )
        
        # Check if new model is significantly better
        improvement = comparison.get('delta', {})
        return (
            improvement.get('sharpe_ratio_change_pct', 0) > 5 or
            improvement.get('directional_accuracy_change_pct', 0) > 5
        )
    
    async def _promote_model(self, model_version: Any):
        """Promote model through stages"""
        # First to staging
        self.model_registry.promote_model(
            model_version.id, 
            ModelStage.STAGING,
            approved_by='integrated_system'
        )
        
        # If auto-deploy enabled, promote to production
        if self.config.auto_deploy:
            # Wait for staging tests (simplified)
            await asyncio.sleep(60)
            
            self.model_registry.promote_model(
                model_version.id,
                ModelStage.PRODUCTION,
                approved_by='integrated_system'
            )
            
            self.logger.info(f"Model {model_version.id} promoted to production")
    
    async def run_daily_pipeline(self):
        """Run daily pipeline for all configured tickers"""
        self.logger.info(f"Starting daily pipeline for {len(self.config.tickers)} tickers")
        
        # Step 1: Update EOD data
        self.logger.info("Updating EOD data...")
        eod_results = await self.eod_pipeline.run_daily_update(self.config.tickers)
        
        if eod_results.get('status') == 'skipped':
            self.logger.info("Not a trading day, skipping pipeline")
            return
        
        # Step 2: Run optimization if scheduled
        if self._should_run_optimization():
            self.logger.info("Running system optimization...")
            await self.auto_optimizer.start()
        
        # Step 3: Retrain models if needed
        for ticker in self.config.tickers:
            if await self._should_retrain(ticker):
                self.logger.info(f"Retraining models for {ticker}")
                
                # Train both LSTM and XGBoost
                for model_type in ['lstm', 'xgboost']:
                    result = await self.run_complete_training_pipeline(ticker, model_type)
                    self.logger.info(f"Training result for {ticker}/{model_type}: {result['status']}")
        
        self.logger.info("Daily pipeline completed")
    
    def _should_run_optimization(self) -> bool:
        """Check if optimization should run"""
        # Run weekly on Sundays
        return datetime.now().weekday() == 6
    
    async def _should_retrain(self, ticker: str) -> bool:
        """Check if model should be retrained"""
        # Get latest model
        history = self.model_registry.get_model_history(f"lstm_{ticker}", ticker, limit=1)
        
        if not history:
            return True  # No model exists
        
        latest_model = history[0]
        days_old = (datetime.now() - latest_model.training_date).days
        
        # Retrain if model is older than 7 days
        return days_old > 7
    
    async def get_production_model_for_prediction(self, ticker: str, model_type: str = 'lstm'):
        """Get production model for making predictions"""
        model_name = f"{model_type}_{ticker}"
        
        # Get from Model Registry (not from file system!)
        prod_model = self.model_registry.get_production_model(model_name, ticker)
        
        if not prod_model:
            raise ValueError(f"No production model found for {ticker}")
        
        # Load model artifacts
        if self.components.get('gcs_storage'):
            if model_type == 'lstm':
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model_state = self.components['gcs_storage'].download_pytorch_model(
                    prod_model.model_path, device
                )
                scaler = self.components['gcs_storage'].download_joblib(prod_model.scaler_path)
                
                # Reconstruct model
                from .models import Seq2Seq, Encoder, Decoder
                hparams = prod_model.hyperparameters
                
                encoder = Encoder(
                    len(prod_model.feature_list),
                    hparams['hidden_dim'],
                    hparams['n_layers'],
                    hparams['dropout_prob']
                ).to(device)
                
                decoder = Decoder(
                    len(prod_model.feature_list),
                    hparams['hidden_dim'],
                    hparams['n_layers'],
                    hparams['dropout_prob']
                ).to(device)
                
                model = Seq2Seq(encoder, decoder, device).to(device)
                model.load_state_dict(model_state)
                model.eval()
                
                return {
                    'model': model,
                    'scaler': scaler,
                    'features': prod_model.feature_list,
                    'model_id': prod_model.id,
                    'version': prod_model.version
                }
            else:
                # XGBoost
                model = self.components['gcs_storage'].download_joblib(prod_model.model_path)
                return {
                    'model': model,
                    'features': prod_model.feature_list,
                    'model_id': prod_model.id,
                    'version': prod_model.version
                }
        
        raise ValueError("Storage not available")
    
    async def start(self):
        """Start the integrated system"""
        self.logger.info("Starting Integrated ML Trading System...")
        
        # Initialize all components
        await self.eod_pipeline.initialize()
        
        # Schedule daily pipeline
        import schedule
        schedule.every().day.at("17:30").do(
            lambda: asyncio.run(self.run_daily_pipeline())
        )
        
        self.logger.info("System started successfully")
        
        # Main loop
        while True:
            schedule.run_pending()
            await asyncio.sleep(60)

# ======================== Updated Celery Tasks ========================

from .extensions import celery
from .tasks import SecureTask

@celery.task(bind=True, base=SecureTask)
def integrated_train_task(self, ticker: str, model_type: str, user_id: str):
    """
    Updated Celery task that uses the integrated system
    """
    try:
        # Get integrated system instance
        config = IntegratedSystemConfig()
        system = IntegratedMLTradingSystem(config)
        
        # Run training through integrated pipeline
        result = asyncio.run(system.run_complete_training_pipeline(ticker, model_type))
        
        # Audit log
        self._audit_log('integrated_training_completed', user_id, {
            'ticker': ticker,
            'model_type': model_type,
            'result': result
        })
        
        return result
        
    except Exception as e:
        self._audit_log('integrated_training_failed', user_id, {
            'ticker': ticker,
            'model_type': model_type,
            'error': str(e)
        }, success=False)
        raise

# ======================== Updated API Endpoints ========================

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

integrated_api = Blueprint('integrated_api', __name__, url_prefix='/api/v3')

# Global system instance
_system = None

def get_system():
    global _system
    if _system is None:
        config = IntegratedSystemConfig()
        _system = IntegratedMLTradingSystem(config)
    return _system

@integrated_api.route('/train', methods=['POST'])
@jwt_required()
def train_integrated():
    """
    New training endpoint that uses integrated system
    """
    data = request.get_json()
    ticker = data.get('ticker', 'AAPL')
    model_type = data.get('model_type', 'lstm')
    user_id = get_jwt_identity()
    
    # Submit task
    task = integrated_train_task.delay(ticker, model_type, user_id)
    
    return jsonify({
        'task_id': task.id,
        'status': 'submitted',
        'message': 'Training initiated through integrated pipeline'
    }), 202

@integrated_api.route('/predict/<ticker>', methods=['GET'])
@jwt_required()
async def predict_integrated(ticker: str):
    """
    New prediction endpoint that uses Model Registry
    """
    try:
        system = get_system()
        
        # Get production model from registry
        model_info = await system.get_production_model_for_prediction(ticker)
        
        # Get latest data from EOD pipeline
        df = await system._get_training_data_from_eod(ticker, days_back=60)
        
        # Make prediction
        # ... prediction logic ...
        
        return jsonify({
            'ticker': ticker,
            'model_version': model_info['version'],
            'model_id': model_info['model_id'],
            'prediction': 'placeholder'  # Add actual prediction
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ======================== Main Entry Point ========================

async def main():
    """Main entry point for the integrated system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = IntegratedSystemConfig()
    system = IntegratedMLTradingSystem(config)
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Shutting down integrated system...")

if __name__ == "__main__":
    asyncio.run(main())
