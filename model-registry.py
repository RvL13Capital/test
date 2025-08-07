# comprehensive_model_registry.py
"""
Comprehensive Model Registry with versioning, lifecycle management, and deployment tracking
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import redis
from functools import lru_cache
import logging

Base = declarative_base()
logger = logging.getLogger(__name__)

class ModelStage(Enum):
    """Model lifecycle stages"""
    EXPERIMENTAL = "experimental"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class ModelMetrics:
    """Performance metrics for a model"""
    mse: float
    mae: float
    rmse: float
    r2: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None
    win_rate: Optional[float] = None
    directional_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def compare(self, other: 'ModelMetrics') -> Dict[str, float]:
        """Compare metrics with another model"""
        comparison = {}
        for key in self.to_dict():
            if hasattr(other, key) and getattr(other, key) is not None:
                self_val = getattr(self, key)
                other_val = getattr(other, key)
                if self_val != 0:
                    comparison[f"{key}_change_pct"] = ((other_val - self_val) / abs(self_val)) * 100
                else:
                    comparison[f"{key}_change"] = other_val - self_val
        return comparison

# Database Models
class ModelVersion(Base):
    __tablename__ = 'model_versions'
    
    id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    model_type = Column(String, nullable=False)  # lstm, xgboost, etc.
    ticker = Column(String, nullable=False)
    
    # Training information
    training_date = Column(DateTime, default=datetime.utcnow)
    training_duration_seconds = Column(Float)
    training_data_start = Column(DateTime)
    training_data_end = Column(DateTime)
    training_samples = Column(Integer)
    
    # Model artifacts
    model_path = Column(String)  # GCS/S3 path
    scaler_path = Column(String)
    feature_list = Column(JSON)
    hyperparameters = Column(JSON)
    
    # Performance metrics
    train_metrics = Column(JSON)
    validation_metrics = Column(JSON)
    test_metrics = Column(JSON)
    
    # Lifecycle
    stage = Column(String, default=ModelStage.EXPERIMENTAL.value)
    promoted_date = Column(DateTime)
    deprecated_date = Column(DateTime)
    
    # Deployment info
    deployment_status = Column(String, default=DeploymentStatus.PENDING.value)
    deployment_date = Column(DateTime)
    deployment_endpoint = Column(String)
    
    # Metadata
    created_by = Column(String)
    tags = Column(JSON)
    description = Column(String)
    mlflow_run_id = Column(String)
    git_commit = Column(String)
    
    def __repr__(self):
        return f"<ModelVersion {self.model_name}:{self.version} ({self.stage})>"

class ModelComparison(Base):
    __tablename__ = 'model_comparisons'
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Models being compared
    baseline_model_id = Column(String)
    candidate_model_id = Column(String)
    
    # Comparison results
    metrics_comparison = Column(JSON)
    performance_delta = Column(JSON)
    
    # A/B test results (if performed)
    ab_test_start = Column(DateTime)
    ab_test_end = Column(DateTime)
    ab_test_results = Column(JSON)
    
    # Decision
    winner = Column(String)  # baseline or candidate
    promotion_approved = Column(Boolean)
    decision_reason = Column(String)
    approved_by = Column(String)

class ModelDeployment(Base):
    __tablename__ = 'model_deployments'
    
    id = Column(String, primary_key=True)
    model_version_id = Column(String)
    
    # Deployment details
    environment = Column(String)  # staging, production, canary
    deployed_at = Column(DateTime, default=datetime.utcnow)
    deployed_by = Column(String)
    
    # Configuration
    endpoint_url = Column(String)
    replicas = Column(Integer)
    cpu_limit = Column(Float)
    memory_limit = Column(Float)
    
    # Traffic routing
    traffic_percentage = Column(Float)
    
    # Health & monitoring
    health_check_url = Column(String)
    last_health_check = Column(DateTime)
    health_status = Column(String)
    
    # Rollback info
    previous_version_id = Column(String)
    can_rollback = Column(Boolean, default=True)
    rollback_deadline = Column(DateTime)

class ModelRegistry:
    """
    Comprehensive model registry with full lifecycle management
    """
    
    def __init__(self, db_url: str, redis_url: str = None, gcs_storage=None):
        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Redis for caching
        self.redis_client = None
        if redis_url:
            self.redis_client = redis.from_url(redis_url)
        
        # Storage backend (from project.storage)
        self.storage = gcs_storage
        
        # In-memory cache for frequent lookups
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    def register_model(self, model_name: str, model_type: str, ticker: str,
                       model_artifact: Any, metrics: ModelMetrics,
                       hyperparameters: Dict, features: List[str],
                       training_info: Dict, **kwargs) -> ModelVersion:
        """
        Register a new model version
        """
        session = self.SessionLocal()
        try:
            # Generate unique ID
            model_id = self._generate_model_id(model_name, ticker)
            
            # Get next version number
            version = self._get_next_version(session, model_name)
            
            # Store model artifacts
            model_path = None
            scaler_path = None
            
            if self.storage:
                model_key = f"models/{model_name}/{version}/model.pkl"
                model_path = self.storage.upload_joblib(model_artifact, model_key)
                
                if 'scaler' in kwargs:
                    scaler_key = f"models/{model_name}/{version}/scaler.pkl"
                    scaler_path = self.storage.upload_joblib(kwargs['scaler'], scaler_key)
            
            # Create model version entry
            model_version = ModelVersion(
                id=model_id,
                model_name=model_name,
                version=version,
                model_type=model_type,
                ticker=ticker,
                model_path=model_path,
                scaler_path=scaler_path,
                feature_list=features,
                hyperparameters=hyperparameters,
                train_metrics=metrics.to_dict() if isinstance(metrics, ModelMetrics) else metrics,
                training_date=datetime.utcnow(),
                training_duration_seconds=training_info.get('duration'),
                training_data_start=training_info.get('data_start'),
                training_data_end=training_info.get('data_end'),
                training_samples=training_info.get('samples'),
                created_by=kwargs.get('created_by', 'system'),
                tags=kwargs.get('tags', {}),
                description=kwargs.get('description'),
                mlflow_run_id=kwargs.get('mlflow_run_id'),
                git_commit=kwargs.get('git_commit')
            )
            
            session.add(model_version)
            session.commit()
            
            # Invalidate cache
            self._invalidate_cache(model_name)
            
            logger.info(f"Registered model {model_name}:{version} with ID {model_id}")
            return model_version
            
        finally:
            session.close()
    
    def promote_model(self, model_id: str, target_stage: ModelStage,
                     comparison_results: Optional[Dict] = None,
                     approved_by: Optional[str] = None) -> bool:
        """
        Promote model to a new stage with validation
        """
        session = self.SessionLocal()
        try:
            model = session.query(ModelVersion).filter_by(id=model_id).first()
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            # Validate promotion path
            if not self._validate_promotion(model.stage, target_stage):
                raise ValueError(f"Invalid promotion from {model.stage} to {target_stage.value}")
            
            # If promoting to production, ensure comparison was done
            if target_stage == ModelStage.PRODUCTION:
                if not comparison_results:
                    # Auto-compare with current production
                    current_prod = self.get_production_model(model.model_name, model.ticker)
                    if current_prod:
                        comparison_results = self.compare_models(current_prod.id, model_id)
                        
                        # Check if new model is better
                        if not self._is_model_better(comparison_results):
                            logger.warning(f"Model {model_id} does not outperform current production")
                            return False
                
                # Demote current production model
                self._demote_current_production(session, model.model_name, model.ticker)
            
            # Update model stage
            old_stage = model.stage
            model.stage = target_stage.value
            model.promoted_date = datetime.utcnow()
            
            # Log comparison if exists
            if comparison_results:
                comparison = ModelComparison(
                    id=self._generate_comparison_id(),
                    baseline_model_id=comparison_results.get('baseline_id'),
                    candidate_model_id=model_id,
                    metrics_comparison=comparison_results.get('metrics'),
                    performance_delta=comparison_results.get('delta'),
                    winner='candidate',
                    promotion_approved=True,
                    decision_reason=comparison_results.get('reason'),
                    approved_by=approved_by
                )
                session.add(comparison)
            
            session.commit()
            
            # Trigger deployment if moving to production
            if target_stage == ModelStage.PRODUCTION:
                self._trigger_deployment(model_id)
            
            logger.info(f"Promoted model {model_id} from {old_stage} to {target_stage.value}")
            return True
            
        finally:
            session.close()
    
    def compare_models(self, baseline_id: str, candidate_id: str,
                       test_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Compare two model versions
        """
        session = self.SessionLocal()
        try:
            baseline = session.query(ModelVersion).filter_by(id=baseline_id).first()
            candidate = session.query(ModelVersion).filter_by(id=candidate_id).first()
            
            if not baseline or not candidate:
                raise ValueError("One or both models not found")
            
            comparison = {
                'baseline_id': baseline_id,
                'candidate_id': candidate_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Compare training metrics
            baseline_metrics = ModelMetrics(**baseline.train_metrics)
            candidate_metrics = ModelMetrics(**candidate.train_metrics)
            
            comparison['metrics'] = {
                'baseline': baseline_metrics.to_dict(),
                'candidate': candidate_metrics.to_dict()
            }
            
            comparison['delta'] = baseline_metrics.compare(candidate_metrics)
            
            # If test data provided, run live comparison
            if test_data is not None and self.storage:
                comparison['live_test'] = self._run_live_comparison(
                    baseline, candidate, test_data
                )
            
            # Generate recommendation
            comparison['recommendation'] = self._generate_recommendation(comparison)
            
            return comparison
            
        finally:
            session.close()
    
    def _run_live_comparison(self, baseline: ModelVersion, 
                            candidate: ModelVersion,
                            test_data: pd.DataFrame) -> Dict:
        """
        Run live comparison on test data
        """
        results = {}
        
        try:
            # Load models
            baseline_model = self.storage.download_joblib(baseline.model_path)
            candidate_model = self.storage.download_joblib(candidate.model_path)
            
            # Get predictions
            baseline_preds = self._get_predictions(baseline_model, test_data, baseline)
            candidate_preds = self._get_predictions(candidate_model, test_data, candidate)
            
            # Calculate live metrics
            if 'target' in test_data.columns:
                y_true = test_data['target'].values
                
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                
                results['baseline_live_mse'] = mean_squared_error(y_true, baseline_preds)
                results['candidate_live_mse'] = mean_squared_error(y_true, candidate_preds)
                results['improvement_pct'] = ((results['baseline_live_mse'] - 
                                              results['candidate_live_mse']) / 
                                             results['baseline_live_mse'] * 100)
            
            # Calculate prediction divergence
            results['prediction_correlation'] = np.corrcoef(baseline_preds, candidate_preds)[0, 1]
            results['mean_absolute_diff'] = np.mean(np.abs(baseline_preds - candidate_preds))
            
        except Exception as e:
            logger.error(f"Live comparison failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_production_model(self, model_name: str, ticker: str) -> Optional[ModelVersion]:
        """
        Get current production model for a given name and ticker
        """
        # Check cache first
        cache_key = f"prod_{model_name}_{ticker}"
        if cache_key in self._cache:
            cached_time, cached_model = self._cache[cache_key]
            if datetime.utcnow() - cached_time < timedelta(seconds=self._cache_ttl):
                return cached_model
        
        session = self.SessionLocal()
        try:
            model = session.query(ModelVersion).filter_by(
                model_name=model_name,
                ticker=ticker,
                stage=ModelStage.PRODUCTION.value
            ).order_by(ModelVersion.promoted_date.desc()).first()
            
            # Update cache
            self._cache[cache_key] = (datetime.utcnow(), model)
            
            return model
        finally:
            session.close()
    
    def get_model_history(self, model_name: str, ticker: str = None,
                         limit: int = 10) -> List[ModelVersion]:
        """
        Get model version history
        """
        session = self.SessionLocal()
        try:
            query = session.query(ModelVersion).filter_by(model_name=model_name)
            
            if ticker:
                query = query.filter_by(ticker=ticker)
            
            models = query.order_by(ModelVersion.version.desc()).limit(limit).all()
            return models
        finally:
            session.close()
    
    def deploy_model(self, model_id: str, environment: str,
                    config: Dict[str, Any]) -> ModelDeployment:
        """
        Deploy a model to specified environment
        """
        session = self.SessionLocal()
        try:
            model = session.query(ModelVersion).filter_by(id=model_id).first()
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            # Create deployment record
            deployment = ModelDeployment(
                id=self._generate_deployment_id(),
                model_version_id=model_id,
                environment=environment,
                deployed_by=config.get('deployed_by', 'system'),
                endpoint_url=config.get('endpoint_url'),
                replicas=config.get('replicas', 1),
                cpu_limit=config.get('cpu_limit', 1.0),
                memory_limit=config.get('memory_limit', 2.0),
                traffic_percentage=config.get('traffic_percentage', 100.0),
                health_check_url=config.get('health_check_url')
            )
            
            # Update model deployment status
            model.deployment_status = DeploymentStatus.DEPLOYED.value
            model.deployment_date = datetime.utcnow()
            model.deployment_endpoint = deployment.endpoint_url
            
            session.add(deployment)
            session.commit()
            
            logger.info(f"Deployed model {model_id} to {environment}")
            return deployment
            
        finally:
            session.close()
    
    def rollback_deployment(self, deployment_id: str, reason: str) -> bool:
        """
        Rollback a deployment
        """
        session = self.SessionLocal()
        try:
            deployment = session.query(ModelDeployment).filter_by(id=deployment_id).first()
            if not deployment:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            if not deployment.can_rollback:
                raise ValueError("Deployment cannot be rolled back")
            
            # Get previous version
            if deployment.previous_version_id:
                # Restore previous version
                prev_model = session.query(ModelVersion).filter_by(
                    id=deployment.previous_version_id
                ).first()
                
                if prev_model:
                    prev_model.deployment_status = DeploymentStatus.DEPLOYED.value
                    prev_model.deployment_date = datetime.utcnow()
            
            # Mark current as rolled back
            current_model = session.query(ModelVersion).filter_by(
                id=deployment.model_version_id
            ).first()
            
            if current_model:
                current_model.deployment_status = DeploymentStatus.ROLLED_BACK.value
            
            session.commit()
            
            logger.info(f"Rolled back deployment {deployment_id}: {reason}")
            return True
            
        finally:
            session.close()
    
    def get_deployment_metrics(self, model_id: str, 
                              start_time: datetime = None,
                              end_time: datetime = None) -> Dict:
        """
        Get deployment metrics for a model
        """
        # This would integrate with your monitoring system
        # (Prometheus, CloudWatch, etc.)
        metrics = {
            'model_id': model_id,
            'period': {
                'start': start_time or datetime.utcnow() - timedelta(hours=24),
                'end': end_time or datetime.utcnow()
            },
            'inference_count': 0,
            'avg_latency_ms': 0,
            'p99_latency_ms': 0,
            'error_rate': 0,
            'predictions': []
        }
        
        # Placeholder for actual metrics collection
        # Would integrate with Prometheus/Grafana as shown in docker-compose
        
        return metrics
    
    def archive_old_models(self, days_old: int = 90) -> int:
        """
        Archive models older than specified days
        """
        session = self.SessionLocal()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            old_models = session.query(ModelVersion).filter(
                ModelVersion.training_date < cutoff_date,
                ModelVersion.stage.in_([
                    ModelStage.EXPERIMENTAL.value,
                    ModelStage.DEPRECATED.value
                ])
            ).all()
            
            archived_count = 0
            for model in old_models:
                model.stage = ModelStage.ARCHIVED.value
                archived_count += 1
                
                # Clean up storage if configured
                if self.storage and model.model_path:
                    # Move to archive storage (cold storage)
                    archive_path = model.model_path.replace('/models/', '/archive/')
                    # Implementation depends on storage backend
                    logger.info(f"Archived model {model.id} to {archive_path}")
            
            session.commit()
            logger.info(f"Archived {archived_count} old models")
            return archived_count
            
        finally:
            session.close()
    
    # Helper methods
    def _generate_model_id(self, model_name: str, ticker: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        hash_input = f"{model_name}_{ticker}_{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    
    def _generate_comparison_id(self) -> str:
        """Generate unique comparison ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        return f"comp_{timestamp}"
    
    def _generate_deployment_id(self) -> str:
        """Generate unique deployment ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        return f"deploy_{timestamp}"
    
    def _get_next_version(self, session: Session, model_name: str) -> int:
        """Get next version number for model"""
        max_version = session.query(
            func.max(ModelVersion.version)
        ).filter_by(model_name=model_name).scalar()
        
        return (max_version or 0) + 1
    
    def _validate_promotion(self, current_stage: str, 
                          target_stage: ModelStage) -> bool:
        """Validate if promotion is allowed"""
        valid_paths = {
            ModelStage.EXPERIMENTAL.value: [ModelStage.STAGING],
            ModelStage.STAGING.value: [ModelStage.PRODUCTION, ModelStage.DEPRECATED],
            ModelStage.PRODUCTION.value: [ModelStage.DEPRECATED, ModelStage.ARCHIVED],
            ModelStage.DEPRECATED.value: [ModelStage.ARCHIVED]
        }
        
        return target_stage in valid_paths.get(current_stage, [])
    
    def _is_model_better(self, comparison: Dict) -> bool:
        """Determine if candidate model is better than baseline"""
        delta = comparison.get('delta', {})
        
        # Define improvement criteria
        criteria = [
            delta.get('mse_change_pct', 0) < -5,  # 5% improvement in MSE
            delta.get('sharpe_ratio_change_pct', 0) > 5,  # 5% improvement in Sharpe
            delta.get('directional_accuracy_change_pct', 0) > 2  # 2% improvement in accuracy
        ]
        
        # At least 2 out of 3 criteria should be met
        return sum(criteria) >= 2
    
    def _demote_current_production(self, session: Session, 
                                  model_name: str, ticker: str):
        """Demote current production model"""
        current = session.query(ModelVersion).filter_by(
            model_name=model_name,
            ticker=ticker,
            stage=ModelStage.PRODUCTION.value
        ).first()
        
        if current:
            current.stage = ModelStage.DEPRECATED.value
            current.deprecated_date = datetime.utcnow()
    
    def _trigger_deployment(self, model_id: str):
        """Trigger deployment process"""
        # This would integrate with your CI/CD pipeline
        # Could trigger GitHub Actions, Jenkins, etc.
        logger.info(f"Triggering deployment for model {model_id}")
    
    def _invalidate_cache(self, model_name: str):
        """Invalidate cache entries for a model"""
        keys_to_remove = [k for k in self._cache.keys() if model_name in k]
        for key in keys_to_remove:
            del self._cache[key]
    
    def _get_predictions(self, model: Any, data: pd.DataFrame, 
                        model_version: ModelVersion) -> np.ndarray:
        """Get predictions from model"""
        # Handle different model types
        if model_version.model_type == 'xgboost':
            features = model_version.feature_list
            return model.predict(data[features])
        elif model_version.model_type == 'lstm':
            # Handle LSTM preprocessing
            # This would need actual implementation based on your LSTM structure
            pass
        
        return np.array([])

# CLI interface for model registry operations
class ModelRegistryCLI:
    """
    Command-line interface for model registry
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def list_models(self, model_name: str = None, stage: str = None):
        """List models with optional filters"""
        # Implementation
        pass
    
    def promote(self, model_id: str, target_stage: str):
        """Promote a model"""
        stage = ModelStage[target_stage.upper()]
        success = self.registry.promote_model(model_id, stage)
        if success:
            print(f"Successfully promoted {model_id} to {target_stage}")
        else:
            print(f"Failed to promote {model_id}")
    
    def compare(self, baseline_id: str, candidate_id: str):
        """Compare two models"""
        results = self.registry.compare_models(baseline_id, candidate_id)
        print(json.dumps(results, indent=2, default=str))
    
    def deploy(self, model_id: str, environment: str):
        """Deploy a model"""
        config = {
            'endpoint_url': f"https://api.trading.com/v1/models/{model_id}",
            'replicas': 3 if environment == 'production' else 1
        }
        deployment = self.registry.deploy_model(model_id, environment, config)
        print(f"Deployed to {deployment.endpoint_url}")

from sqlalchemy import func  # Add this import at the top

if __name__ == "__main__":
    # Example usage
    db_url = "postgresql://user:pass@localhost/model_registry"
    registry = ModelRegistry(db_url)
    
    # Register a model
    model_metrics = ModelMetrics(
        mse=0.001,
        mae=0.03,
        rmse=0.032,
        r2=0.95,
        sharpe_ratio=1.8,
        directional_accuracy=0.65
    )
    
    model_version = registry.register_model(
        model_name="lstm_trading_model",
        model_type="lstm",
        ticker="AAPL",
        model_artifact=None,  # Would be actual model
        metrics=model_metrics,
        hyperparameters={'hidden_dim': 128, 'layers': 3},
        features=['close', 'volume', 'rsi', 'macd'],
        training_info={'duration': 3600, 'samples': 10000}
    )
    
    print(f"Registered model: {model_version.id}")