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
from .eod_data_pipeline import get_eod_pipeline  # ANPASSUNG: Import der Singleton-Funktion
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
        self.eod_pipeline = get_eod_pipeline()  # ANPASSUNG: Nutze Singleton für EOD-Pipeline
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
    
    async def _get_training_data_from_eod(self, ticker: str, days_back: int = 252 * 5) -> pd.DataFrame:
        """Fetch training data from EOD pipeline"""
        try:
            pipeline = get_eod_pipeline()  # ANPASSUNG: Nutze Singleton für Pipeline-Zugriff
            data = pipeline.get_data_for_training(ticker, days_back=days_back)  # ANPASSUNG: Übergabe von days_back für Flexibilität
            
            if data.empty:
                raise ValueError(f"No training data available for {ticker}")
            
            self.logger.info(f"Fetched {len(data)} training records for {ticker}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to fetch training data for {ticker}: {e}")
            raise
    
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
                raise ValueError(f"No training data available for {ticker}")
            
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
            self.logger.error(f"Training pipeline failed: {e}")
            raise

    async def start(self):
        """Start the integrated system"""
        self.logger.info("Starting Integrated ML Trading System...")
        
        # Initialize all components
        await self.eod_pipeline.initialize()  # ANPASSUNG: Async-Initialisierung der Pipeline
        
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
