# ml_system_orchestrator.py
"""
Complete ML Trading System Orchestrator
Integrates CI/CD, MLflow tracking, and Model Registry
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import yaml
from dataclasses import dataclass
import logging

# Import all components
from enhanced_mlflow_integration import MLflowTracker, setup_mlflow_tracking
from comprehensive_model_registry import (
    ModelRegistry, ModelMetrics, ModelStage, 
    DeploymentStatus
)
from project.storage import get_gcs_storage
from project.config import Config

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """Central configuration for the ML system"""
    environment: str  # development, staging, production
    mlflow_uri: str
    registry_db_url: str
    redis_url: str
    gcs_bucket: str
    monitoring_enabled: bool = True
    auto_deploy: bool = False
    canary_duration_hours: int = 24
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Load configuration from environment"""
        return cls(
            environment=os.getenv('ENVIRONMENT', 'development'),
            mlflow_uri=os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001'),
            registry_db_url=os.getenv('REGISTRY_DB_URL', 'postgresql://localhost/model_registry'),
            redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
            gcs_bucket=os.getenv('GCS_BUCKET_NAME', 'ml-trading-models'),
            monitoring_enabled=os.getenv('MONITORING_ENABLED', 'true').lower() == 'true',
            auto_deploy=os.getenv('AUTO_DEPLOY', 'false').lower() == 'true',
            canary_duration_hours=int(os.getenv('CANARY_DURATION_HOURS', '24'))
        )

class MLSystemOrchestrator:
    """
    Main orchestrator for the entire ML system
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Initialize components
        self.mlflow_tracker = setup_mlflow_tracking({
            'mlflow_tracking_uri': config.mlflow_uri,
            'environment': config.environment,
            'version': '1.0.0'
        })
        
        self.model_registry = ModelRegistry(
            db_url=config.registry_db_url,
            redis_url=config.redis_url,
            gcs_storage=get_gcs_storage()
        )
        
        self.active_pipelines = {}
        
    async def run_training_pipeline(self, ticker: str, 
                                   model_type: str,
                                   data_source: str = 'yfinance') -> Dict[str, Any]:
        """
        Complete training pipeline with tracking and registration
        """
        pipeline_id = f"{ticker}_{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        logger.info(f"Starting training pipeline {pipeline_id}")
        
        try:
            # 1. Data Collection
            logger.info("Step 1: Collecting data...")
            df = await self._collect_data(ticker, data_source)
            
            # 2. Feature Engineering
            logger.info("Step 2: Engineering features...")
            from project.features_optimized import OptimizedFeatureEngine, select_features
            
            feature_engine = OptimizedFeatureEngine()
            df_with_features = feature_engine.create_features_optimized(df)
            selected_features = select_features(df_with_features)
            
            # 3. Hyperparameter Tuning (with MLflow tracking)
            logger.info("Step 3: Tuning hyperparameters...")
            best_params = await self._tune_hyperparameters(
                df_with_features, selected_features, model_type, ticker
            )
            
            # 4. Model Training (with MLflow tracking)
            logger.info("Step 4: Training model...")
            with self.mlflow_tracker.mlflow.start_run(run_name=f"train_{pipeline_id}"):
                # Log all parameters
                self.mlflow_tracker.mlflow.log_params({
                    'ticker': ticker,
                    'model_type': model_type,
                    'features': len(selected_features),
                    **best_params
                })
                
                # Train model
                from project.training_optimized import build_and_train_lstm_optimized, build_and_train_xgboost_optimized
                
                if model_type == 'lstm':
                    result = build_and_train_lstm_optimized(
                        df_with_features, selected_features, best_params, validation_split=0.2
                    )
                else:
                    result = build_and_train_xgboost_optimized(
                        df_with_features, best_params, validation_split=0.2
                    )
                
                # Log metrics
                self.mlflow_tracker._log_training_metrics(result)
                
                # Get MLflow run ID
                mlflow_run_id = self.mlflow_tracker.mlflow.active_run().info.run_id
            
            # 5. Model Validation
            logger.info("Step 5: Validating model...")
            validation_results = await self._validate_model(result, df_with_features)
            
            # 6. Register Model
            logger.info("Step 6: Registering model...")
            metrics = ModelMetrics(
                mse=result.get('val_loss', 0),
                mae=result.get('val_mae', 0),
                rmse=result.get('val_rmse', 0),
                r2=result.get('val_r2', 0),
                sharpe_ratio=validation_results.get('sharpe_ratio'),
                directional_accuracy=validation_results.get('directional_accuracy')
            )
            
            model_version = self.model_registry.register_model(
                model_name=f"{model_type}_{ticker}",
                model_type=model_type,
                ticker=ticker,
                model_artifact=result['model'],
                metrics=metrics,
                hyperparameters=best_params,
                features=selected_features,
                training_info={
                    'duration': result.get('training_time'),
                    'samples': len(df_with_features),
                    'data_start': df.index[0],
                    'data_end': df.index[-1]
                },
                mlflow_run_id=mlflow_run_id,
                scaler=result.get('scaler')
            )
            
            # 7. Auto-promote to staging if performance meets criteria
            if self._should_auto_promote(metrics):
                logger.info("Step 7: Auto-promoting to staging...")
                self.model_registry.promote_model(
                    model_version.id, 
                    ModelStage.STAGING
                )
                
                # 8. Run A/B test in staging
                if self.config.environment in ['staging', 'production']:
                    logger.info("Step 8: Running A/B test...")
                    ab_results = await self._run_ab_test(model_version.id)
                    
                    # 9. Deploy if A/B test successful
                    if ab_results['success'] and self.config.auto_deploy:
                        logger.info("Step 9: Deploying to production...")
                        await self._deploy_model(model_version.id)
            
            return {
                'pipeline_id': pipeline_id,
                'model_id': model_version.id,
                'status': 'success',
                'metrics': metrics.to_dict(),
                'mlflow_run_id': mlflow_run_id
            }
            
        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} failed: {e}")
            return {
                'pipeline_id': pipeline_id,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _collect_data(self, ticker: str, source: str) -> pd.DataFrame:
        """Collect training data"""
        if source == 'yfinance':
            import yfinance as yf
            data = yf.download(ticker, period='2y', interval='1d')
            return data
        else:
            # Load from other sources
            pass
    
    async def _tune_hyperparameters(self, df: pd.DataFrame, 
                                   features: List[str],
                                   model_type: str,
                                   ticker: str) -> Dict:
        """Run hyperparameter tuning with MLflow tracking"""
        import optuna
        
        # Create MLflow parent run for tuning
        with self.mlflow_tracker.mlflow.start_run(run_name=f"tuning_{ticker}_{model_type}"):
            
            def objective(trial):
                # Use nested MLflow runs for each trial
                with self.mlflow_tracker.mlflow.start_run(
                    nested=True, 
                    run_name=f"trial_{trial.number}"
                ):
                    if model_type == 'lstm':
                        params = {
                            'hidden_dim': trial.suggest_int('hidden_dim', 32, 256),
                            'n_layers': trial.suggest_int('n_layers', 1, 3),
                            'dropout_prob': trial.suggest_float('dropout_prob', 0.1, 0.5),
                            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                            'epochs': 20  # Fixed for tuning
                        }
                    else:  # xgboost
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                            'max_depth': trial.suggest_int('max_depth', 3, 10),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                        }
                    
                    # Log trial parameters
                    self.mlflow_tracker.mlflow.log_params(params)
                    
                    # Quick training for evaluation
                    # ... training code ...
                    
                    # Return validation loss
                    val_loss = 0.1  # Placeholder
                    self.mlflow_tracker.mlflow.log_metric('val_loss', val_loss)
                    
                    return val_loss
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)
            
            # Log best parameters
            self.mlflow_tracker.mlflow.log_params(study.best_params)
            self.mlflow_tracker.mlflow.log_metric('best_val_loss', study.best_value)
            
            return study.best_params
    
    async def _validate_model(self, model_result: Dict, 
                            test_data: pd.DataFrame) -> Dict:
        """Comprehensive model validation"""
        validation = {}
        
        # Backtesting
        # ... backtesting code ...
        
        # Risk metrics
        # ... risk calculation ...
        
        return validation
    
    def _should_auto_promote(self, metrics: ModelMetrics) -> bool:
        """Determine if model should be auto-promoted"""
        criteria = [
            metrics.mse < 0.01,
            metrics.sharpe_ratio and metrics.sharpe_ratio > 1.5,
            metrics.directional_accuracy and metrics.directional_accuracy > 0.6
        ]
        return sum(criteria) >= 2
    
    async def _run_ab_test(self, model_id: str) -> Dict:
        """Run A/B test in staging/canary"""
        logger.info(f"Starting A/B test for model {model_id}")
        
        # Deploy as canary (10% traffic)
        deployment = self.model_registry.deploy_model(
            model_id,
            environment='canary',
            config={
                'traffic_percentage': 10,
                'endpoint_url': f"/api/v1/models/{model_id}/predict"
            }
        )
        
        # Monitor for specified duration
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=self.config.canary_duration_hours)
        
        metrics_history = []
        while datetime.utcnow() < end_time:
            # Collect metrics
            metrics = self.model_registry.get_deployment_metrics(
                model_id,
                start_time=datetime.utcnow() - timedelta(minutes=5)
            )
            metrics_history.append(metrics)
            
            # Check for anomalies
            if self._detect_anomalies(metrics):
                logger.warning(f"Anomalies detected in canary {model_id}")
                # Rollback
                self.model_registry.rollback_deployment(
                    deployment.id,
                    reason="Anomalies detected during A/B test"
                )
                return {'success': False, 'reason': 'anomalies_detected'}
            
            await asyncio.sleep(300)  # Check every 5 minutes
        
        # Analyze results
        ab_results = self._analyze_ab_test(metrics_history)
        
        return {
            'success': ab_results['improvement'] > 0,
            'improvement': ab_results['improvement'],
            'metrics': ab_results
        }
    
    def _detect_anomalies(self, metrics: Dict) -> bool:
        """Detect anomalies in deployment metrics"""
        return (
            metrics.get('error_rate', 0) > 0.05 or
            metrics.get('p99_latency_ms', 0) > 1000
        )
    
    def _analyze_ab_test(self, metrics_history: List[Dict]) -> Dict:
        """Analyze A/B test results"""
        # Statistical analysis of canary vs baseline
        # ... implementation ...
        return {'improvement': 0.1}  # Placeholder
    
    async def _deploy_model(self, model_id: str):
        """Deploy model to production"""
        # Promote to production stage
        self.model_registry.promote_model(
            model_id,
            ModelStage.PRODUCTION,
            approved_by='auto_deploy'
        )
        
        # Deploy to production environment
        self.model_registry.deploy_model(
            model_id,
            environment='production',
            config={
                'replicas': 3,
                'cpu_limit': 2.0,
                'memory_limit': 4.0,
                'endpoint_url': f"/api/v1/models/{model_id}/predict"
            }
        )
        
        logger.info(f"Model {model_id} deployed to production")
    
    async def run_continuous_training(self, tickers: List[str], 
                                     interval_hours: int = 24):
        """
        Run continuous training loop
        """
        logger.info(f"Starting continuous training for {tickers}")
        
        while True:
            for ticker in tickers:
                for model_type in ['lstm', 'xgboost']:
                    try:
                        result = await self.run_training_pipeline(
                            ticker, model_type
                        )
                        logger.info(f"Pipeline completed: {result}")
                    except Exception as e:
                        logger.error(f"Pipeline failed for {ticker}/{model_type}: {e}")
            
            # Wait for next iteration
            await asyncio.sleep(interval_hours * 3600)
    
    def cleanup_old_resources(self, days: int = 30):
        """Clean up old models and experiments"""
        # Archive old models
        archived = self.model_registry.archive_old_models(days)
        logger.info(f"Archived {archived} old models")
        
        # Clean up MLflow experiments
        # ... implementation ...

# CLI Interface
import click

@click.group()
def cli():
    """ML Trading System CLI"""
    pass

@cli.command()
@click.option('--ticker', required=True, help='Stock ticker')
@click.option('--model-type', type=click.Choice(['lstm', 'xgboost']), required=True)
@click.option('--auto-deploy', is_flag=True, help='Auto-deploy if successful')
def train(ticker, model_type, auto_deploy):
    """Train a new model"""
    config = SystemConfig.from_env()
    config.auto_deploy = auto_deploy
    
    orchestrator = MLSystemOrchestrator(config)
    result = asyncio.run(orchestrator.run_training_pipeline(ticker, model_type))
    
    click.echo(f"Training completed: {result}")

@cli.command()
@click.option('--model-id', required=True, help='Model ID to promote')
@click.option('--stage', type=click.Choice(['staging', 'production']), required=True)
def promote(model_id, stage):
    """Promote a model to new stage"""
    config = SystemConfig.from_env()
    orchestrator = MLSystemOrchestrator(config)
    
    stage_enum = ModelStage[stage.upper()]
    success = orchestrator.model_registry.promote_model(model_id, stage_enum)
    
    if success:
        click.echo(f"Model {model_id} promoted to {stage}")
    else:
        click.echo(f"Failed to promote model {model_id}")

@cli.command()
@click.option('--baseline', required=True, help='Baseline model ID')
@click.option('--candidate', required=True, help='Candidate model ID')
def compare(baseline, candidate):
    """Compare two models"""
    config = SystemConfig.from_env()
    orchestrator = MLSystemOrchestrator(config)
    
    results = orchestrator.model_registry.compare_models(baseline, candidate)
    click.echo(json.dumps(results, indent=2, default=str))

@cli.command()
@click.option('--tickers', multiple=True, required=True, help='Stock tickers')
@click.option('--interval', default=24, help='Training interval in hours')
def continuous(tickers, interval):
    """Run continuous training"""
    config = SystemConfig.from_env()
    orchestrator = MLSystemOrchestrator(config)
    
    asyncio.run(orchestrator.run_continuous_training(list(tickers), interval))

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run CLI
    cli()

import json  # Add this import at the top