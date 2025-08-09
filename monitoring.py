# project/monitoring.py
import time
import logging
import psutil
import torch
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, asdict
import redis
import json

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, start_http_server,
    CONTENT_TYPE_LATEST
)
from flask import Response
import threading

from .config import Config

logger = logging.getLogger(__name__)

# ============================================================================
#                           PROMETHEUS METRICS DEFINITIONS
# ============================================================================

# Custom registry for better control
registry = CollectorRegistry()

# Request metrics
HTTP_REQUESTS_TOTAL = Counter(
    'ml_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

HTTP_REQUEST_DURATION = Histogram(
    'ml_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    registry=registry
)

# Training metrics
TRAINING_REQUESTS_TOTAL = Counter(
    'ml_training_requests_total',
    'Total training requests',
    ['model_type', 'status'],
    registry=registry
)

TRAINING_DURATION = Histogram(
    'ml_training_duration_seconds',
    'Training duration in seconds',
    ['model_type', 'ticker'],
    buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600],
    registry=registry
)

TRAINING_MEMORY_PEAK = Gauge(
    'ml_training_memory_peak_bytes',
    'Peak memory usage during training',
    ['model_type'],
    registry=registry
)

MODEL_PERFORMANCE = Gauge(
    'ml_model_performance_score',
    'Model performance score (accuracy/loss)',
    ['model_type', 'ticker', 'metric_type'],
    registry=registry
)

# Feature engineering metrics
FEATURE_COMPUTATION_DURATION = Histogram(
    'ml_feature_computation_duration_seconds',
    'Feature computation duration',
    ['feature_type', 'data_size_category'],
    registry=registry
)

FEATURE_CACHE_OPERATIONS = Counter(
    'ml_feature_cache_operations_total',
    'Feature cache operations',
    ['operation', 'result'],
    registry=registry
)

# System resource metrics
SYSTEM_CPU_USAGE = Gauge(
    'ml_system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=registry
)

SYSTEM_MEMORY_USAGE = Gauge(
    'ml_system_memory_usage_bytes',
    'System memory usage in bytes',
    registry=registry
)

SYSTEM_MEMORY_AVAILABLE = Gauge(
    'ml_system_memory_available_bytes',
    'System available memory in bytes',
    registry=registry
)

GPU_MEMORY_ALLOCATED = Gauge(
    'ml_gpu_memory_allocated_bytes',
    'GPU memory allocated in bytes',
    ['device'],
    registry=registry
)

GPU_UTILIZATION = Gauge(
    'ml_gpu_utilization_percent',
    'GPU utilization percentage',
    ['device'],
    registry=registry
)

# Celery task metrics
CELERY_TASKS_TOTAL = Counter(
    'ml_celery_tasks_total',
    'Total Celery tasks',
    ['task_name', 'status'],
    registry=registry
)

CELERY_TASK_DURATION = Histogram(
    'ml_celery_task_duration_seconds',
    'Celery task duration',
    ['task_name'],
    registry=registry
)

CELERY_ACTIVE_TASKS = Gauge(
    'ml_celery_active_tasks',
    'Currently active Celery tasks',
    ['queue'],
    registry=registry
)

# Cache metrics
CACHE_OPERATIONS = Counter(
    'ml_cache_operations_total',
    'Cache operations',
    ['cache_type', 'operation', 'result'],
    registry=registry
)

CACHE_HIT_RATIO = Gauge(
    'ml_cache_hit_ratio',
    'Cache hit ratio',
    ['cache_type'],
    registry=registry
)

# Model storage metrics
MODEL_STORAGE_OPERATIONS = Counter(
    'ml_model_storage_operations_total',
    'Model storage operations',
    ['operation', 'status'],
    registry=registry
)

MODEL_STORAGE_SIZE = Gauge(
    'ml_model_storage_size_bytes',
    'Model storage size in bytes',
    ['model_type', 'ticker'],
    registry=registry
)

# Application info
APPLICATION_INFO = Info(
    'ml_application_info',
    'Application information',
    registry=registry
)

# ============================================================================
#                           MONITORING DECORATORS
# ============================================================================

def monitor_training(model_type: str):
    """Decorator to monitor training functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            ticker = kwargs.get('ticker', 'unknown')
            start_time = time.time()
            
            # Record training start
            TRAINING_REQUESTS_TOTAL.labels(model_type=model_type, status='started').inc()
            
            try:
                # Monitor memory before training
                memory_before = psutil.virtual_memory().used
                
                result = func(*args, **kwargs)
                
                # Record success metrics
                duration = time.time() - start_time
                TRAINING_DURATION.labels(model_type=model_type, ticker=ticker).observe(duration)
                TRAINING_REQUESTS_TOTAL.labels(model_type=model_type, status='success').inc()
                
                # Record memory usage
                memory_after = psutil.virtual_memory().used
                memory_peak = memory_after - memory_before
                TRAINING_MEMORY_PEAK.labels(model_type=model_type).set(memory_peak)
                
                # Record model performance if available
                if isinstance(result, dict):
                    if 'val_loss' in result:
                        MODEL_PERFORMANCE.labels(
                            model_type=model_type, 
                            ticker=ticker, 
                            metric_type='val_loss'
                        ).set(result['val_loss'])
                    
                    if 'train_score' in result:
                        MODEL_PERFORMANCE.labels(
                            model_type=model_type, 
                            ticker=ticker, 
                            metric_type='train_score'
                        ).set(result['train_score'])
                
                return result
                
            except Exception as e:
                # Record failure metrics
                TRAINING_REQUESTS_TOTAL.labels(model_type=model_type, status='failed').inc()
                logger.error(f"Training failed for {model_type}: {e}")
                raise
                
        return wrapper
    return decorator

def monitor_http_requests(func: Callable) -> Callable:
    """Decorator to monitor HTTP requests"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Extract request info (this is Flask-specific)
        from flask import request
        method = request.method
        endpoint = request.endpoint or 'unknown'
        
        try:
            response = func(*args, **kwargs)
            
            # Determine status code
            if hasattr(response, 'status_code'):
                status_code = str(response.status_code)
            else:
                status_code = '200'  # Assume success if no explicit status
            
            # Record metrics
            HTTP_REQUESTS_TOTAL.labels(
                method=method, 
                endpoint=endpoint, 
                status_code=status_code
            ).inc()
            
            duration = time.time() - start_time
            HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            
            return response
            
        except Exception as e:
            # Record error
            HTTP_REQUESTS_TOTAL.labels(
                method=method, 
                endpoint=endpoint, 
                status_code='500'
            ).inc()
            raise
            
    return wrapper

def monitor_feature_computation(feature_type: str):
    """Decorator to monitor feature computation"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Determine data size category
            data_size = 0
            if args and hasattr(args[0], '__len__'):
                data_size = len(args[0])
            
            if data_size < 1000:
                size_category = 'small'
            elif data_size < 10000:
                size_category = 'medium'
            else:
                size_category = 'large'
            
            try:
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                FEATURE_COMPUTATION_DURATION.labels(
                    feature_type=feature_type,
                    data_size_category=size_category
                ).observe(duration)
                
                return result
                
            except Exception as e:
                logger.error(f"Feature computation failed for {feature_type}: {e}")
                raise
                
        return wrapper
    return decorator

# ============================================================================
#                           SYSTEM METRICS COLLECTOR
# ============================================================================

class SystemMetricsCollector:
    """Collects system-level metrics periodically"""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.running = False
        self.thread = None
        
    def start(self):
        """Start collecting metrics in background thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
        self.thread.start()
        logger.info("System metrics collection started")
    
    def stop(self):
        """Stop collecting metrics"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("System metrics collection stopped")
    
    def _collect_metrics_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_gpu_metrics()
                self._collect_cache_metrics()
            except Exception as e:
                logger.warning(f"Error collecting system metrics: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system CPU and memory metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        SYSTEM_CPU_USAGE.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        SYSTEM_MEMORY_USAGE.set(memory.used)
        SYSTEM_MEMORY_AVAILABLE.set(memory.available)
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics if available"""
        if not torch.cuda.is_available():
            return
        
        try:
            for i in range(torch.cuda.device_count()):
                device_name = f"cuda:{i}"
                
                # GPU memory
                memory_allocated = torch.cuda.memory_allocated(i)
                GPU_MEMORY_ALLOCATED.labels(device=device_name).set(memory_allocated)
                
                # GPU utilization (if nvidia-ml-py is available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    GPU_UTILIZATION.labels(device=device_name).set(utilization.gpu)
                except ImportError:
                    # nvidia-ml-py not available, skip utilization
                    pass
                    
        except Exception as e:
            logger.debug(f"Error collecting GPU metrics: {e}")
    
    def _collect_cache_metrics(self):
        """Collect cache metrics"""
        try:
            # Redis cache metrics
            redis_client = redis.from_url(Config.CELERY_RESULT_BACKEND)
            info = redis_client.info()
            
            # Cache hit ratio calculation would go here
            # This is a placeholder - implement based on your cache implementation
            
        except Exception as e:
            logger.debug(f"Error collecting cache metrics: {e}")

# ============================================================================
#                           CELERY MONITORING INTEGRATION
# ============================================================================

class CeleryMonitor:
    """Monitor Celery tasks and queues"""
    
    @staticmethod
    def monitor_task_start(task_name: str):
        """Record task start"""
        CELERY_TASKS_TOTAL.labels(task_name=task_name, status='started').inc()
    
    @staticmethod
    def monitor_task_success(task_name: str, duration: float):
        """Record task success"""
        CELERY_TASKS_TOTAL.labels(task_name=task_name, status='success').inc()
        CELERY_TASK_DURATION.labels(task_name=task_name).observe(duration)
    
    @staticmethod
    def monitor_task_failure(task_name: str, duration: float):
        """Record task failure"""
        CELERY_TASKS_TOTAL.labels(task_name=task_name, status='failed').inc()
        CELERY_TASK_DURATION.labels(task_name=task_name).observe(duration)
    
    @staticmethod
    def update_active_tasks():
        """Update active task counts"""
        try:
            from .extensions import celery
            inspect = celery.control.inspect()
            active_tasks = inspect.active()
            
            if active_tasks:
                # Count tasks by queue
                queue_counts = {}
                for worker, tasks in active_tasks.items():
                    for task in tasks:
                        queue = task.get('delivery_info', {}).get('routing_key', 'default')
                        queue_counts[queue] = queue_counts.get(queue, 0) + 1
                
                # Update metrics
                for queue, count in queue_counts.items():
                    CELERY_ACTIVE_TASKS.labels(queue=queue).set(count)
                    
        except Exception as e:
            logger.debug(f"Error updating active task metrics: {e}")

# ============================================================================
#                           ENHANCED TASK MONITORING
# ============================================================================

class MonitoredSecureTask:
    """Enhanced SecureTask with comprehensive monitoring"""
    
    def __init__(self, task_func):
        self.task_func = task_func
        self.task_name = task_func.__name__
    
    def __call__(self, *args, **kwargs):
        start_time = time.time()
        
        # Record task start
        CeleryMonitor.monitor_task_start(self.task_name)
        
        try:
            # Execute task
            result = self.task_func(*args, **kwargs)
            
            # Record success
            duration = time.time() - start_time
            CeleryMonitor.monitor_task_success(self.task_name, duration)
            
            # Extract metrics from result if available
            if isinstance(result, dict):
                self._record_task_metrics(result, kwargs)
            
            return result
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            CeleryMonitor.monitor_task_failure(self.task_name, duration)
            
            logger.error(f"Monitored task {self.task_name} failed: {e}")
            raise
    
    def _record_task_metrics(self, result: dict, kwargs: dict):
        """Record task-specific metrics"""
        model_type = kwargs.get('model_type', 'unknown')
        ticker = kwargs.get('ticker', 'unknown')
        
        # Training metrics
        if 'train_loss' in result:
            MODEL_PERFORMANCE.labels(
                model_type=model_type, 
                ticker=ticker, 
                metric_type='train_loss'
            ).set(result['train_loss'])
        
        if 'val_loss' in result:
            MODEL_PERFORMANCE.labels(
                model_type=model_type, 
                ticker=ticker, 
                metric_type='val_loss'
            ).set(result['val_loss'])
        
        # Memory usage
        if 'peak_memory_mb' in result:
            TRAINING_MEMORY_PEAK.labels(model_type=model_type).set(
                result['peak_memory_mb'] * 1024 * 1024  # Convert to bytes
            )

# ============================================================================
#                           CACHE MONITORING
# ============================================================================

class CacheMonitor:
    """Monitor cache operations and performance"""
    
    @staticmethod
    def record_cache_operation(cache_type: str, operation: str, hit: bool):
        """Record cache operation"""
        result = 'hit' if hit else 'miss'
        CACHE_OPERATIONS.labels(
            cache_type=cache_type, 
            operation=operation, 
            result=result
        ).inc()
    
    @staticmethod
    def record_feature_cache_operation(operation: str, hit: bool):
        """Record feature cache operation"""
        result = 'hit' if hit else 'miss'
        FEATURE_CACHE_OPERATIONS.labels(operation=operation, result=result).inc()
    
    @staticmethod
    def update_cache_hit_ratios():
        """Update cache hit ratios"""
        try:
            # This would be implemented based on your specific cache metrics
            # For now, we'll use placeholder logic
            
            # Feature cache hit ratio
            feature_hits = FEATURE_CACHE_OPERATIONS._value.get(('get', 'hit'), 0)
            feature_misses = FEATURE_CACHE_OPERATIONS._value.get(('get', 'miss'), 0)
            
            if feature_hits + feature_misses > 0:
                hit_ratio = feature_hits / (feature_hits + feature_misses)
                CACHE_HIT_RATIO.labels(cache_type='features').set(hit_ratio)
                
        except Exception as e:
            logger.debug(f"Error updating cache hit ratios: {e}")

# ============================================================================
#                           MONITORING MANAGER
# ============================================================================

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_system_metrics: bool = True
    system_metrics_interval: int = 30
    enable_prometheus_server: bool = True
    prometheus_port: int = 8000
    enable_detailed_logging: bool = True

class MonitoringManager:
    """Central monitoring manager"""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.system_collector = SystemMetricsCollector(self.config.system_metrics_interval)
        self.prometheus_server_started = False
        
        # Initialize application info
        APPLICATION_INFO.info({
            'version': '2025.1.0',
            'environment': Config.FLASK_ENV,
            'gpu_available': str(torch.cuda.is_available()),
            'start_time': datetime.utcnow().isoformat()
        })
    
    def start_monitoring(self):
        """Start all monitoring services"""
        logger.info("Starting comprehensive monitoring system...")
        
        # Start system metrics collection
        if self.config.enable_system_metrics:
            self.system_collector.start()
        
        # Start Prometheus HTTP server
        if self.config.enable_prometheus_server and not self.prometheus_server_started:
            try:
                start_http_server(self.config.prometheus_port, registry=registry)
                self.prometheus_server_started = True
                logger.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {e}")
        
        logger.info("Monitoring system fully initialized")
    
    def stop_monitoring(self):
        """Stop all monitoring services"""
        logger.info("Stopping monitoring system...")
        
        if self.config.enable_system_metrics:
            self.system_collector.stop()
        
        logger.info("Monitoring system stopped")
    
    def get_metrics_summary(self) -> Dict:
        """Get comprehensive metrics summary"""
        return {
            'system': {
                'cpu_usage': SYSTEM_CPU_USAGE._value._value if hasattr(SYSTEM_CPU_USAGE._value, '_value') else 0,
                'memory_usage_gb': (SYSTEM_MEMORY_USAGE._value._value / (1024**3)) if hasattr(SYSTEM_MEMORY_USAGE._value, '_value') else 0,
                'gpu_available': torch.cuda.is_available()
            },
            'training': {
                'total_requests': sum(TRAINING_REQUESTS_TOTAL._value.values()) if hasattr(TRAINING_REQUESTS_TOTAL, '_value') else 0,
                'active_tasks': sum(CELERY_ACTIVE_TASKS._value.values()) if hasattr(CELERY_ACTIVE_TASKS, '_value') else 0
            },
            'cache': {
                'feature_cache_enabled': True,  # Based on your implementation
            },
            'prometheus': {
                'server_running': self.prometheus_server_started,
                'port': self.config.prometheus_port
            }
        }
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(registry).decode('utf-8')

# ============================================================================
#                           FLASK INTEGRATION
# ============================================================================

def setup_flask_monitoring(app):
    """Setup Flask monitoring endpoints"""
    
    @app.route('/metrics')
    def metrics():
        """Prometheus metrics endpoint"""
        return Response(
            monitoring_manager.export_metrics(),
            mimetype=CONTENT_TYPE_LATEST
        )
    
    @app.route('/health/detailed')
    def detailed_health():
        """Detailed health check with metrics"""
        from flask import jsonify
        
        metrics_summary = monitoring_manager.get_metrics_summary()
        
        health = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics_summary,
            'checks': {
                'database': True,  # Add your checks here
                'cache': True,
                'gpu': torch.cuda.is_available()
            }
        }
        
        return jsonify(health)
    
    @app.before_request
    def before_request():
        """Record request start time"""
        from flask import g
        g.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        """Record request metrics"""
        from flask import request, g
        
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            
            method = request.method
            endpoint = request.endpoint or 'unknown'
            status_code = str(response.status_code)
            
            HTTP_REQUESTS_TOTAL.labels(
                method=method, 
                endpoint=endpoint, 
                status_code=status_code
            ).inc()
            
            HTTP_REQUEST_DURATION.labels(
                method=method, 
                endpoint=endpoint
            ).observe(duration)
        
        return response

# ============================================================================
#                           GLOBAL MONITORING INSTANCE
# ============================================================================

# Global monitoring manager instance
monitoring_config = MonitoringConfig(
    enable_prometheus_server=Config.PROMETHEUS_METRICS_ENABLED,
    prometheus_port=int(Config.__dict__.get('PROMETHEUS_PORT', 8000))
)

monitoring_manager = MonitoringManager(monitoring_config)

# Convenience functions for backward compatibility
def start_monitoring():
    """Start monitoring system"""
    monitoring_manager.start_monitoring()

def stop_monitoring():
    """Stop monitoring system"""
    monitoring_manager.stop_monitoring()

def get_metrics():
    """Get metrics summary"""
    return monitoring_manager.get_metrics_summary()

# ============================================================================
#                           USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Starting monitoring system test...")
    
    # Start monitoring
    monitoring_manager.start_monitoring()
    
    # Simulate some metrics
    TRAINING_REQUESTS_TOTAL.labels(model_type='lstm', status='started').inc()
    TRAINING_DURATION.labels(model_type='lstm', ticker='AAPL').observe(120.5)
    MODEL_PERFORMANCE.labels(model_type='lstm', ticker='AAPL', metric_type='val_loss').set(0.025)
    
    # Get metrics summary
    summary = monitoring_manager.get_metrics_summary()
    print(f"Metrics summary: {summary}")
    
    # Export metrics
    metrics_text = monitoring_manager.export_metrics()
    print(f"Exported metrics (first 500 chars): {metrics_text[:500]}...")
    
    print("Monitoring system test completed")
    
    # Clean shutdown
    time.sleep(2)
    monitoring_manager.stop_monitoring()