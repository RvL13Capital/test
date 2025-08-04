# comprehensive_monitoring.py
"""
Umfassendes Monitoring-System für Trading-Operationen
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from prometheus_client import Counter, Gauge, Histogram, Summary, Info
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import aiohttp
import logging

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class Alert:
    metric_name: str
    severity: AlertSeverity
    condition: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    resolved: bool = False
    
@dataclass
class DriftDetection:
    metric: str
    baseline_mean: float
    baseline_std: float
    current_mean: float
    current_std: float
    drift_score: float
    is_drifting: bool
    timestamp: datetime

class TradingMetrics:
    """Definiert alle Trading-spezifischen Metriken"""
    
    def __init__(self):
        # API Metrics
        self.api_requests = Counter(
            'trading_api_requests_total',
            'Total API requests',
            ['provider', 'endpoint', 'status']
        )
        
        self.api_latency = Histogram(
            'trading_api_latency_seconds',
            'API request latency',
            ['provider', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        )
        
        self.api_costs = Gauge(
            'trading_api_costs_usd',
            'API costs in USD',
            ['provider', 'billing_period']
        )
        
        self.api_errors = Counter(
            'trading_api_errors_total',
            'Total API errors',
            ['provider', 'error_type']
        )
        
        # Data Pipeline Metrics
        self.pipeline_latency = Histogram(
            'trading_pipeline_latency_seconds',
            'Data pipeline end-to-end latency',
            ['pipeline_stage'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30]
        )
        
        self.data_quality_score = Gauge(
            'trading_data_quality_score',
            'Data quality score (0-1)',
            ['data_source', 'data_type']
        )
        
        self.missing_data_points = Counter(
            'trading_missing_data_total',
            'Total missing data points',
            ['data_source', 'field']
        )
        
        # Model Performance Metrics
        self.prediction_distribution = Histogram(
            'trading_model_predictions',
            'Distribution of model predictions',
            ['model_id', 'model_version'],
            buckets=np.linspace(0, 1, 21).tolist()
        )
        
        self.model_accuracy = Gauge(
            'trading_model_accuracy',
            'Model accuracy score',
            ['model_id', 'model_version', 'timeframe']
        )
        
        self.feature_importance_drift = Gauge(
            'trading_feature_importance_drift',
            'Feature importance drift score',
            ['model_id', 'feature_name']
        )
        
        # Portfolio Performance Metrics
        self.portfolio_pnl = Gauge(
            'trading_portfolio_pnl_usd',
            'Portfolio P&L in USD',
            ['strategy', 'timeframe']
        )
        
        self.portfolio_returns = Histogram(
            'trading_portfolio_returns',
            'Portfolio returns distribution',
            ['strategy', 'timeframe'],
            buckets=[-0.1, -0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05, 0.1]
        )
        
        self.drawdown = Gauge(
            'trading_drawdown_percent',
            'Current drawdown percentage',
            ['strategy']
        )
        
        self.sharpe_ratio = Gauge(
            'trading_sharpe_ratio',
            'Rolling Sharpe ratio',
            ['strategy', 'window']
        )
        
        self.win_rate = Gauge(
            'trading_win_rate',
            'Win rate percentage',
            ['strategy', 'window']
        )
        
        # Execution Metrics
        self.order_latency = Histogram(
            'trading_order_latency_ms',
            'Order execution latency',
            ['broker', 'order_type'],
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500]
        )
        
        self.slippage_bps = Histogram(
            'trading_slippage_bps',
            'Execution slippage in basis points',
            ['strategy', 'order_type'],
            buckets=[0, 1, 2, 5, 10, 20, 50, 100]
        )
        
        self.fill_rate = Gauge(
            'trading_fill_rate',
            'Order fill rate',
            ['strategy', 'order_type']
        )
        
        # Risk Metrics
        self.var_utilization = Gauge(
            'trading_var_utilization',
            'VaR utilization percentage',
            ['portfolio']
        )
        
        self.leverage_ratio = Gauge(
            'trading_leverage_ratio',
            'Current leverage ratio',
            ['portfolio']
        )
        
        self.position_concentration = Gauge(
            'trading_position_concentration',
            'Largest position as percentage of portfolio',
            ['portfolio']
        )
        
        # System Health Metrics
        self.system_cpu_usage = Gauge(
            'trading_system_cpu_percent',
            'CPU usage percentage',
            ['component']
        )

class ModelDriftMonitor:
    """Überwacht Model Drift und Performance-Degradation"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.prediction_history = defaultdict(lambda: deque(maxlen=window_size))
        self.feature_history = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_distributions = {}
        self.drift_thresholds = {
            "kolmogorov_smirnov": 0.1,
            "jensen_shannon": 0.1,
            "population_stability_index": 0.2
        }
        
    def record_prediction(self, model_id: str, prediction: float, 
                         features: Dict[str, float]):
        """Zeichnet Modell-Vorhersage auf"""
        self.prediction_history[model_id].append({
            "timestamp": datetime.now(),
            "prediction": prediction,
            "features": features
        })
        
    def detect_prediction_drift(self, model_id: str) -> DriftDetection:
        """Erkennt Drift in Modell-Vorhersagen"""
        if model_id not in self.prediction_history:
            return None
            
        predictions = [p["prediction"] for p in self.prediction_history[model_id]]
        
        if len(predictions) < 100:
            return None
            
        # Split into baseline and current
        split_point = len(predictions) // 2
        baseline = predictions[:split_point]
        current = predictions[split_point:]
        
        # Kolmogorov-Smirnov Test
        ks_statistic, ks_pvalue = stats.ks_2samp(baseline, current)
        
        # Jensen-Shannon Divergence
        js_divergence = self.calculate_js_divergence(baseline, current)
        
        # Population Stability Index
        psi = self.calculate_psi(baseline, current)
        
        is_drifting = (
            ks_statistic > self.drift_thresholds["kolmogorov_smirnov"] or
            js_divergence > self.drift_thresholds["jensen_shannon"] or
            psi > self.drift_thresholds["population_stability_index"]
        )
        
        return DriftDetection(
            metric=f"model_{model_id}_predictions",
            baseline_mean=np.mean(baseline),
            baseline_std=np.std(baseline),
            current_mean=np.mean(current),
            current_std=np.std(current),
            drift_score=max(ks_statistic, js_divergence, psi),
            is_drifting=is_drifting,
            timestamp=datetime.now()
        )
        
    def calculate_js_divergence(self, dist1: List[float], 
                               dist2: List[float]) -> float:
        """Berechnet Jensen-Shannon Divergence"""
        # Create histograms
        bins = np.linspace(0, 1, 20)
        hist1, _ = np.histogram(dist1, bins=bins, density=True)
        hist2, _ = np.histogram(dist2, bins=bins, density=True)
        
        # Normalize
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # Calculate JS divergence
        m = 0.5 * (hist1 + hist2)
        divergence = 0.5 * stats.entropy(hist1, m) + 0.5 * stats.entropy(hist2, m)
        
        return divergence
        
    def calculate_psi(self, baseline: List[float], 
                     current: List[float]) -> float:
        """Berechnet Population Stability Index"""
        bins = np.linspace(0, 1, 10)
        
        baseline_percents = np.histogram(baseline, bins=bins)[0] / len(baseline)
        current_percents = np.histogram(current, bins=bins)[0] / len(current)
        
        # Avoid division by zero
        baseline_percents = np.where(baseline_percents == 0, 0.0001, baseline_percents)
        current_percents = np.where(current_percents == 0, 0.0001, current_percents)
        
        psi = np.sum((current_percents - baseline_percents) * 
                     np.log(current_percents / baseline_percents))
        
        return psi

class DataQualityMonitor:
    """Überwacht Datenqualität und -integrität"""
    
    def __init__(self):
        self.quality_checks = {
            "completeness": self.check_completeness,
            "consistency": self.check_consistency,
            "timeliness": self.check_timeliness,
            "accuracy": self.check_accuracy
        }
        self.quality_history = defaultdict(list)
        
    def check_data_quality(self, data: pd.DataFrame, 
                         source: str) -> Dict[str, float]:
        """Führt Qualitätsprüfungen durch"""
        results = {}
        
        for check_name, check_func in self.quality_checks.items():
            score = check_func(data)
            results[check_name] = score
            
        overall_score = np.mean(list(results.values()))
        
        self.quality_history[source].append({
            "timestamp": datetime.now(),
            "scores": results,
            "overall": overall_score
        })
        
        return results
        
    def check_completeness(self, data: pd.DataFrame) -> float:
        """Prüft Datenvollständigkeit"""
        total_cells = data.size
        missing_cells = data.isna().sum().sum()
        return 1 - (missing_cells / total_cells)
        
    def check_consistency(self, data: pd.DataFrame) -> float:
        """Prüft Datenkonsistenz"""
        consistency_score = 1.0
        
        # Check for price consistency
        if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
            invalid_prices = ((data['close'] > data['high']) | 
                            (data['close'] < data['low'])).sum()
            consistency_score *= (1 - invalid_prices / len(data))
            
        # Check for volume consistency
        if 'volume' in data.columns:
            negative_volumes = (data['volume'] < 0).sum()
            consistency_score *= (1 - negative_volumes / len(data))
            
        return consistency_score
        
    def check_timeliness(self, data: pd.DataFrame) -> float:
        """Prüft Datenaktualität"""
        if 'timestamp' not in data.columns:
            return 1.0
            
        latest_timestamp = pd.to_datetime(data['timestamp']).max()
        delay = datetime.now() - latest_timestamp
        
        # Score decreases with delay
        if delay < timedelta(minutes=1):
            return 1.0
        elif delay < timedelta(minutes=5):
            return 0.9
        elif delay < timedelta(minutes=15):
            return 0.7
        elif delay < timedelta(hours=1):
            return 0.5
        else:
            return 0.0
            
    def check_accuracy(self, data: pd.DataFrame) -> float:
        """Prüft Datengenauigkeit (vereinfacht)"""
        accuracy_score = 1.0
        
        # Check for outliers
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['volume', 'close', 'high', 'low', 'open']:
                # Simple outlier detection using IQR
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 3 * IQR)) | 
                           (data[col] > (Q3 + 3 * IQR))).sum()
                accuracy_score *= (1 - outliers / len(data))
                
        return accuracy_score

class AlertingSystem:
    """Verwaltet Alerts und Benachrichtigungen"""
    
    def __init__(self):
        self.alert_rules = []
        self.active_alerts = []
        self.alert_history = deque(maxlen=10000)
        self.notification_channels = {}
        
    def add_alert_rule(self, name: str, metric: str, 
                      condition: Callable, threshold: float,
                      severity: AlertSeverity, message_template: str):
        """Fügt neue Alert-Regel hinzu"""
        self.alert_rules.append({
            "name": name,
            "metric": metric,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "message_template": message_template
        })
        
    async def check_alerts(self, metrics: Dict[str, float]):
        """Prüft alle Alert-Regeln"""
        for rule in self.alert_rules:
            metric_value = metrics.get(rule["metric"])
            
            if metric_value is None:
                continue
                
            if rule["condition"](metric_value, rule["threshold"]):
                await self.trigger_alert(rule, metric_value)
            else:
                await self.resolve_alert(rule["name"])
                
    async def trigger_alert(self, rule: Dict, current_value: float):
        """Löst Alert aus"""
        # Check if alert already active
        existing = next((a for a in self.active_alerts 
                        if a.metric_name == rule["name"]), None)
        
        if existing:
            return  # Alert already active
            
        alert = Alert(
            metric_name=rule["name"],
            severity=rule["severity"],
            condition=f"{rule['metric']} {rule['condition'].__name__} {rule['threshold']}",
            current_value=current_value,
            threshold=rule["threshold"],
            message=rule["message_template"].format(
                metric=rule["metric"],
                value=current_value,
                threshold=rule["threshold"]
            ),
            timestamp=datetime.now()
        )
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        await self.send_notifications(alert)
        
    async def resolve_alert(self, alert_name: str):
        """Löst Alert auf"""
        alert = next((a for a in self.active_alerts 
                     if a.metric_name == alert_name), None)
        
        if alert:
            alert.resolved = True
            self.active_alerts.remove(alert)
            await self.send_notifications(alert, resolved=True)
            
    async def send_notifications(self, alert: Alert, resolved: bool = False):
        """Sendet Benachrichtigungen über konfigurierte Kanäle"""
        for channel_name, channel_func in self.notification_channels.items():
            try:
                await channel_func(alert, resolved)
            except Exception as e:
                logging.error(f"Failed to send notification via {channel_name}: {e}")

class MonitoringDashboard:
    """Hauptklasse für das Monitoring-Dashboard"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = TradingMetrics()
        self.drift_monitor = ModelDriftMonitor()
        self.quality_monitor = DataQualityMonitor()
        self.alerting = AlertingSystem()
        
        # Initialize alert rules
        self.setup_alert_rules()
        
        # Monitoring tasks
        self.monitoring_tasks = []
        
    def setup_alert_rules(self):
        """Konfiguriert Standard-Alert-Regeln"""
        
        # API Cost Alerts
        self.alerting.add_alert_rule(
            name="high_api_costs",
            metric="api_costs_per_hour",
            condition=lambda x, t: x > t,
            threshold=100,  # $100/hour
            severity=AlertSeverity.WARNING,
            message_template="High API costs detected: ${value:.2f}/hour (threshold: ${threshold})"
        )
        
        # Error Rate Alerts
        self.alerting.add_alert_rule(
            name="high_error_rate",
            metric="api_error_rate",
            condition=lambda x, t: x > t,
            threshold=0.05,  # 5% error rate
            severity=AlertSeverity.CRITICAL,
            message_template="High error rate: {value:.1%} (threshold: {threshold:.1%})"
        )
        
        # Model Drift Alerts
        self.alerting.add_alert_rule(
            name="model_drift_detected",
            metric="model_drift_score",
            condition=lambda x, t: x > t,
            threshold=0.15,
            severity=AlertSeverity.WARNING,
            message_template="Model drift detected: score={value:.3f} (threshold: {threshold})"
        )
        
        # Drawdown Alerts
        self.alerting.add_alert_rule(
            name="excessive_drawdown",
            metric="portfolio_drawdown",
            condition=lambda x, t: x < t,  # Drawdown is negative
            threshold=-0.10,  # 10% drawdown
            severity=AlertSeverity.CRITICAL,
            message_template="Excessive drawdown: {value:.1%} (threshold: {threshold:.1%})"
        )
        
        # Latency Alerts
        self.alerting.add_alert_rule(
            name="high_latency",
            metric="p99_latency_seconds",
            condition=lambda x, t: x > t,
            threshold=2.0,  # 2 seconds
            severity=AlertSeverity.WARNING,
            message_template="High latency detected: {value:.2f}s (threshold: {threshold}s)"
        )
        
    async def start_monitoring(self):
        """Startet alle Monitoring-Tasks"""
        self.monitoring_tasks = [
            asyncio.create_task(self.monitor_apis()),
            asyncio.create_task(self.monitor_models()),
            asyncio.create_task(self.monitor_portfolio()),
            asyncio.create_task(self.monitor_system_health()),
            asyncio.create_task(self.check_alerts_loop())
        ]
        
        await asyncio.gather(*self.monitoring_tasks)
        
    async def monitor_apis(self):
        """Überwacht API-Performance und Kosten"""
        while True:
            try:
                # Collect API metrics
                api_metrics = await self.collect_api_metrics()
                
                # Update Prometheus metrics
                for provider, metrics in api_metrics.items():
                    self.metrics.api_costs.labels(
                        provider=provider,
                        billing_period="hourly"
                    ).set(metrics["cost_per_hour"])
                    
                    self.metrics.api_errors.labels(
                        provider=provider,
                        error_type="all"
                    )._value._value = metrics["error_count"]
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Error monitoring APIs: {e}")
                await asyncio.sleep(60)
                
    async def monitor_models(self):
        """Überwacht Model Performance und Drift"""
        while True:
            try:
                # Check each model for drift
                for model_id in self.get_active_models():
                    drift_result = self.drift_monitor.detect_prediction_drift(model_id)
                    
                    if drift_result and drift_result.is_drifting:
                        self.metrics.feature_importance_drift.labels(
                            model_id=model_id,
                            feature_name="all"
                        ).set(drift_result.drift_score)
                        
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logging.error(f"Error monitoring models: {e}")
                await asyncio.sleep(300)
                
    async def monitor_portfolio(self):
        """Überwacht Portfolio-Performance"""
        while True:
            try:
                portfolio_metrics = await self.collect_portfolio_metrics()
                
                # Update metrics
                self.metrics.portfolio_pnl.labels(
                    strategy="all",
                    timeframe="daily"
                ).set(portfolio_metrics["daily_pnl"])
                
                self.metrics.drawdown.labels(
                    strategy="all"
                ).set(portfolio_metrics["current_drawdown"])
                
                self.metrics.sharpe_ratio.labels(
                    strategy="all",
                    window="30d"
                ).set(portfolio_metrics["sharpe_ratio_30d"])
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logging.error(f"Error monitoring portfolio: {e}")
                await asyncio.sleep(30)
                
    async def monitor_system_health(self):
        """Überwacht System-Gesundheit"""
        while True:
            try:
                health_metrics = await self.collect_system_health()
                
                for component, metrics in health_metrics.items():
                    self.metrics.system_cpu_usage.labels(
                        component=component
                    ).set(metrics["cpu_percent"])
                    
                    self.metrics.system_memory_usage.labels(
                        component=component
                    ).set(metrics["memory_percent"])
                    
                    self.metrics.component_health.labels(
                        component=component
                    ).set(metrics["health_score"])
                    
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"Error monitoring system health: {e}")
                await asyncio.sleep(10)
                
    async def check_alerts_loop(self):
        """Prüft kontinuierlich Alert-Bedingungen"""
        while True:
            try:
                current_metrics = await self.collect_all_metrics()
                await self.alerting.check_alerts(current_metrics)
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Error checking alerts: {e}")
                await asyncio.sleep(30)
                
    # Helper methods (mock implementations)
    async def collect_api_metrics(self) -> Dict[str, Dict[str, float]]:
        """Mock implementation"""
        return {
            "alpaca": {"cost_per_hour": 25.5, "error_count": 12},
            "polygon": {"cost_per_hour": 45.2, "error_count": 3}
        }
        
    def get_active_models(self) -> List[str]:
        """Mock implementation"""
        return ["model_v1", "model_v2"]
        
    async def collect_portfolio_metrics(self) -> Dict[str, float]:
        """Mock implementation"""
        return {
            "daily_pnl": 15234.56,
            "current_drawdown": -0.032,
            "sharpe_ratio_30d": 1.85
        }
        
    async def collect_system_health(self) -> Dict[str, Dict[str, float]]:
        """Mock implementation"""
        return {
            "data_pipeline": {"cpu_percent": 45.2, "memory_percent": 62.1, "health_score": 0.95},
            "model_server": {"cpu_percent": 78.5, "memory_percent": 81.3, "health_score": 0.88},
            "execution_engine": {"cpu_percent": 23.1, "memory_percent": 41.2, "health_score": 0.99}
        }
        
    async def collect_all_metrics(self) -> Dict[str, float]:
        """Mock implementation"""
        return {
            "api_costs_per_hour": 70.7,
            "api_error_rate": 0.02,
            "model_drift_score": 0.08,
            "portfolio_drawdown": -0.032,
            "p99_latency_seconds": 1.2
        }
        
        self.system_memory_usage = Gauge(
            'trading_system_memory_percent',
            'Memory usage percentage',
            ['component']
        )
        
        self.system_disk_usage = Gauge(
            'trading_system_disk_percent',
            'Disk usage percentage',
            ['mount_point']
        )
        
        self.component_health = Gauge(
            'trading_component_health',
            'Component health score (0-1)',
            ['component']
        )