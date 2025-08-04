# staging_environment.py
"""
Staging Environment für sicheres Deployment und Konfigurationsmanagement
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import hashlib
import difflib

class DeploymentStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"

@dataclass
class APISchema:
    """Schema-Definition für API-Endpoints"""
    endpoint: str
    method: str
    request_schema: Dict[str, Any]
    response_schema: Dict[str, Any]
    version: str
    last_validated: datetime

@dataclass
class ModelVersion:
    """Versionierung für ML-Modelle"""
    model_id: str
    version: str
    training_date: datetime
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    status: str  # 'staging', 'production', 'deprecated'
    
class SchemaValidator:
    """Validiert API-Schemas und erkennt Breaking Changes"""
    
    def __init__(self):
        self.schema_history = {}
        
    def validate_response(self, response: Dict, expected_schema: Dict) -> Tuple[bool, List[str]]:
        """Validiert eine API-Response gegen das erwartete Schema"""
        errors = []
        
        def check_schema(data, schema, path=""):
            if isinstance(schema, dict):
                if not isinstance(data, dict):
                    errors.append(f"{path}: Expected dict, got {type(data).__name__}")
                    return
                    
                # Check required fields
                for key, value in schema.items():
                    if key not in data:
                        errors.append(f"{path}.{key}: Missing required field")
                    else:
                        check_schema(data[key], value, f"{path}.{key}")
                        
                # Check for unexpected fields
                for key in data:
                    if key not in schema:
                        errors.append(f"{path}.{key}: Unexpected field")
                        
            elif isinstance(schema, list):
                if not isinstance(data, list):
                    errors.append(f"{path}: Expected list, got {type(data).__name__}")
                elif len(schema) > 0 and len(data) > 0:
                    # Validate first element as sample
                    check_schema(data[0], schema[0], f"{path}[0]")
                    
            elif isinstance(schema, type):
                if not isinstance(data, schema):
                    errors.append(f"{path}: Expected {schema.__name__}, got {type(data).__name__}")
                    
        check_schema(response, expected_schema)
        return len(errors) == 0, errors
        
    def detect_breaking_changes(self, old_schema: Dict, new_schema: Dict) -> List[str]:
        """Erkennt Breaking Changes zwischen Schema-Versionen"""
        breaking_changes = []
        
        def compare_schemas(old, new, path=""):
            if type(old) != type(new):
                breaking_changes.append(f"{path}: Type changed from {type(old).__name__} to {type(new).__name__}")
                return
                
            if isinstance(old, dict):
                # Removed fields are breaking changes
                for key in old:
                    if key not in new:
                        breaking_changes.append(f"{path}.{key}: Field removed")
                    else:
                        compare_schemas(old[key], new[key], f"{path}.{key}")
                        
                # New required fields might be breaking
                for key in new:
                    if key not in old:
                        # This could be breaking if the field is required
                        breaking_changes.append(f"{path}.{key}: New required field added")
                        
        compare_schemas(old_schema, new_schema)
        return breaking_changes

class StagingEnvironment:
    """Hauptklasse für Staging-Umgebung und sichere Deployments"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.schema_validator = SchemaValidator()
        self.staging_models = {}
        self.production_models = {}
        self.canary_models = {}
        self.api_schemas = {}
        self.deployment_history = []
        
    async def validate_api_compatibility(self, provider: str) -> Dict[str, Any]:
        """Validiert API-Kompatibilität vor Deployment"""
        results = {
            "provider": provider,
            "timestamp": datetime.now(),
            "passed": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Test API endpoints
            test_endpoints = self.get_test_endpoints(provider)
            
            for endpoint in test_endpoints:
                response = await self.test_api_endpoint(provider, endpoint)
                expected_schema = self.api_schemas.get(f"{provider}:{endpoint}")
                
                if expected_schema:
                    valid, errors = self.schema_validator.validate_response(
                        response, 
                        expected_schema.response_schema
                    )
                    
                    if not valid:
                        results["passed"] = False
                        results["errors"].extend([
                            f"{endpoint}: {error}" for error in errors
                        ])
                        
                else:
                    results["warnings"].append(
                        f"{endpoint}: No schema defined for validation"
                    )
                    
        except Exception as e:
            results["passed"] = False
            results["errors"].append(f"API test failed: {str(e)}")
            
        return results
        
    def validate_model_deployment(self, model_id: str, 
                                 new_version: ModelVersion) -> Dict[str, Any]:
        """Validiert Model-Deployment durch Parallel-Run"""
        validation_results = {
            "model_id": model_id,
            "new_version": new_version.version,
            "timestamp": datetime.now(),
            "passed": True,
            "metrics": {}
        }
        
        # Get current production model
        prod_model = self.production_models.get(model_id)
        if not prod_model:
            validation_results["warnings"] = ["No production model for comparison"]
            return validation_results
            
        # Run parallel predictions on test data
        test_data = self.load_test_data(model_id)
        
        try:
            prod_predictions = prod_model.predict(test_data)
            new_predictions = new_version.model.predict(test_data)
            
            # Calculate divergence metrics
            metrics = self.calculate_model_divergence(
                prod_predictions, 
                new_predictions
            )
            
            validation_results["metrics"] = metrics
            
            # Check thresholds
            if metrics["mean_absolute_diff"] > 0.1:
                validation_results["passed"] = False
                validation_results["errors"] = [
                    f"High prediction divergence: {metrics['mean_absolute_diff']:.3f}"
                ]
                
            if metrics["correlation"] < 0.95:
                validation_results["warnings"] = [
                    f"Low correlation with production: {metrics['correlation']:.3f}"
                ]
                
        except Exception as e:
            validation_results["passed"] = False
            validation_results["errors"] = [f"Model validation failed: {str(e)}"]
            
        return validation_results
        
    def calculate_model_divergence(self, pred1: np.ndarray, 
                                  pred2: np.ndarray) -> Dict[str, float]:
        """Berechnet Divergenz-Metriken zwischen Modell-Outputs"""
        return {
            "mean_absolute_diff": np.mean(np.abs(pred1 - pred2)),
            "max_absolute_diff": np.max(np.abs(pred1 - pred2)),
            "correlation": np.corrcoef(pred1.flatten(), pred2.flatten())[0, 1],
            "rmse": np.sqrt(np.mean((pred1 - pred2) ** 2))
        }
        
    async def canary_deployment(self, component: str, 
                               new_version: Any,
                               traffic_percentage: float = 0.1,
                               duration_hours: int = 24) -> Dict[str, Any]:
        """Führt Canary Deployment durch"""
        canary_id = f"canary_{component}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        deployment = {
            "canary_id": canary_id,
            "component": component,
            "traffic_percentage": traffic_percentage,
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(hours=duration_hours),
            "metrics": {},
            "status": "running"
        }
        
        # Route traffic
        await self.setup_traffic_routing(component, new_version, traffic_percentage)
        
        # Monitor performance
        monitoring_task = asyncio.create_task(
            self.monitor_canary_performance(canary_id, component)
        )
        
        self.deployment_history.append(deployment)
        return deployment
        
    async def monitor_canary_performance(self, canary_id: str, component: str):
        """Überwacht Canary Deployment Performance"""
        deployment = next(d for d in self.deployment_history if d["canary_id"] == canary_id)
        
        while deployment["status"] == "running":
            # Collect metrics
            metrics = await self.collect_performance_metrics(component)
            deployment["metrics"][datetime.now().isoformat()] = metrics
            
            # Check for anomalies
            if self.detect_anomalies(metrics):
                deployment["status"] = "rolled_back"
                await self.rollback_deployment(component)
                break
                
            # Check if duration exceeded
            if datetime.now() > deployment["end_time"]:
                deployment["status"] = "completed"
                break
                
            await asyncio.sleep(60)  # Check every minute
            
    def detect_anomalies(self, metrics: Dict[str, float]) -> bool:
        """Erkennt Anomalien in Performance-Metriken"""
        # Error rate threshold
        if metrics.get("error_rate", 0) > 0.05:  # 5% error rate
            return True
            
        # Latency threshold
        if metrics.get("p99_latency", 0) > 1000:  # 1 second
            return True
            
        # Custom business metrics
        if metrics.get("prediction_accuracy", 1) < 0.8:
            return True
            
        return False
        
    async def rollback_deployment(self, component: str):
        """Führt Rollback auf vorherige Version durch"""
        print(f"Rolling back {component} deployment")
        # Implement rollback logic
        
    def create_deployment_plan(self, changes: List[Dict]) -> Dict[str, Any]:
        """Erstellt Deployment-Plan mit Abhängigkeiten"""
        plan = {
            "id": hashlib.sha256(str(changes).encode()).hexdigest()[:8],
            "created": datetime.now(),
            "stages": [],
            "dependencies": {}
        }
        
        # Analyze dependencies
        for change in changes:
            if change["type"] == "model":
                plan["stages"].append({
                    "component": change["model_id"],
                    "action": "deploy_model",
                    "version": change["new_version"],
                    "stage": DeploymentStage.STAGING
                })
            elif change["type"] == "api_update":
                plan["stages"].append({
                    "component": change["api"],
                    "action": "update_schema",
                    "version": change["new_schema_version"],
                    "stage": DeploymentStage.STAGING
                })
                
        return plan
        
    def generate_rollback_plan(self, deployment_id: str) -> Dict[str, Any]:
        """Generiert Rollback-Plan für fehlgeschlagenes Deployment"""
        deployment = self.get_deployment(deployment_id)
        
        rollback_plan = {
            "deployment_id": deployment_id,
            "created": datetime.now(),
            "steps": []
        }
        
        # Reverse deployment steps
        for stage in reversed(deployment["stages"]):
            if stage["action"] == "deploy_model":
                rollback_plan["steps"].append({
                    "action": "restore_model",
                    "component": stage["component"],
                    "target_version": self.get_previous_version(stage["component"])
                })
                
        return rollback_plan

class ConfigurationManager:
    """Verwaltet Konfigurationen über verschiedene Umgebungen"""
    
    def __init__(self):
        self.configs = {}
        self.version_history = []
        
    def load_config(self, environment: DeploymentStage) -> Dict[str, Any]:
        """Lädt Konfiguration für spezifische Umgebung"""
        base_config = self.load_base_config()
        env_config = self.load_environment_config(environment)
        
        # Merge configurations
        merged_config = self.merge_configs(base_config, env_config)
        
        # Validate configuration
        self.validate_config(merged_config)
        
        return merged_config
        
    def track_config_change(self, config: Dict[str, Any], 
                           change_reason: str):
        """Trackt Konfigurations-Änderungen"""
        version = {
            "timestamp": datetime.now(),
            "config_hash": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
            "change_reason": change_reason,
            "config": config.copy()
        }
        
        self.version_history.append(version)
        
    def diff_configs(self, config1: Dict[str, Any], 
                    config2: Dict[str, Any]) -> List[str]:
        """Zeigt Unterschiede zwischen Konfigurationen"""
        json1 = json.dumps(config1, indent=2, sort_keys=True).splitlines()
        json2 = json.dumps(config2, indent=2, sort_keys=True).splitlines()
        
        diff = difflib.unified_diff(json1, json2, lineterm='')
        return list(diff)
        
    def validate_config(self, config: Dict[str, Any]):
        """Validiert Konfiguration auf Vollständigkeit und Korrektheit"""
        required_keys = [
            "api_endpoints",
            "model_parameters",
            "risk_limits",
            "monitoring_thresholds"
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration: {key}")
                
        # Validate specific values
        if config.get("risk_limits", {}).get("max_position_size", 0) > 0.1:
            raise ValueError("Max position size exceeds safe limit")
            
    def merge_configs(self, base: Dict[str, Any], 
                     override: Dict[str, Any]) -> Dict[str, Any]:
        """Merged Konfigurationen mit Override-Logik"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result