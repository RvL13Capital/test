# production/gunicorn_config.py
"""
PRODUCTION: Optimized Gunicorn configuration for ML Trading System
Supports high concurrency, memory management, and monitoring
"""

import multiprocessing
import os
from pathlib import Path

# ============================================================================
#                           BASIC CONFIGURATION
# ============================================================================

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = os.getenv('GUNICORN_WORKER_CLASS', 'gevent')  # Async for I/O-bound ML tasks
worker_connections = int(os.getenv('GUNICORN_WORKER_CONNECTIONS', 1000))

# Worker lifecycle
max_requests = int(os.getenv('GUNICORN_MAX_REQUESTS', 1000))
max_requests_jitter = int(os.getenv('GUNICORN_MAX_REQUESTS_JITTER', 100))
preload_app = True  # Important for memory sharing and faster startup

# Timeouts (important for ML workloads)
timeout = int(os.getenv('GUNICORN_TIMEOUT', 300))  # 5 minutes for training tasks
keepalive = int(os.getenv('GUNICORN_KEEPALIVE', 5))
graceful_timeout = int(os.getenv('GUNICORN_GRACEFUL_TIMEOUT', 30))

# ============================================================================
#                           PERFORMANCE OPTIMIZATION
# ============================================================================

# Memory management
max_worker_memory = int(os.getenv('MAX_WORKER_MEMORY_MB', 2048)) * 1024 * 1024  # 2GB default
worker_tmp_dir = '/dev/shm'  # Use shared memory for better performance

# Threading (for gevent workers)
threads = int(os.getenv('GUNICORN_THREADS', 4))
thread_map_async = True

# SSL (if needed)
# keyfile = '/path/to/ssl.key'
# certfile = '/path/to/ssl.crt'

# ============================================================================
#                           LOGGING CONFIGURATION
# ============================================================================

# Logging
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')
access_log_format = ('%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s '
                    '"%(f)s" "%(a)s" %(D)s %(p)s')

# Log files
log_dir = Path(os.getenv('LOG_DIR', '/var/log/ml-trading'))
log_dir.mkdir(exist_ok=True, parents=True)

accesslog = str(log_dir / 'gunicorn-access.log')
errorlog = str(log_dir / 'gunicorn-error.log')
capture_output = True

# ============================================================================
#                           MONITORING & HEALTH CHECKS
# ============================================================================

# Monitoring
statsd_host = os.getenv('STATSD_HOST')  # e.g., 'localhost:8125'
if statsd_host:
    statsd_prefix = 'ml_trading.gunicorn'

# Prometheus metrics (if prometheus_flask_exporter is used)
enable_prometheus = os.getenv('ENABLE_PROMETHEUS', 'true').lower() == 'true'

# ============================================================================
#                           HOOKS FOR ML SYSTEM INTEGRATION
# ============================================================================

def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("ML Trading System starting up...")
    
    # Initialize any global resources here
    # e.g., warm up model cache, initialize monitoring

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading ML Trading System...")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("ML Trading System is ready to serve requests")
    
    # Start background services
    try:
        from project.monitoring import start_monitoring
        start_monitoring()
        server.log.info("Monitoring system started")
    except ImportError:
        server.log.warning("Monitoring system not available")

def worker_init(worker):
    """Called just after a worker has been forked."""
    worker.log.info(f"Worker {worker.pid} initialized")
    
    # Initialize per-worker resources
    # e.g., database connections, model loading
    
    # Set memory limits for ML workloads
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (max_worker_memory, max_worker_memory))
    except Exception as e:
        worker.log.warning(f"Failed to set memory limit: {e}")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.debug(f"About to fork worker {worker.pid}")

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.debug(f"Worker {worker.pid} forked")
    
    # Initialize ML-specific resources per worker
    try:
        import torch
        if torch.cuda.is_available():
            # Set CUDA device for this worker if multiple GPUs available
            device_id = worker.age % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            worker.log.info(f"Worker {worker.pid} using CUDA device {device_id}")
    except Exception as e:
        worker.log.warning(f"CUDA setup failed for worker {worker.pid}: {e}")

def worker_exit(server, worker):
    """Called just after a worker has been exited, in the master process."""
    server.log.info(f"Worker {worker.pid} exited")
    
    # Cleanup worker-specific resources
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def on_exit(server):
    """Called just before exiting Gunicorn."""
    server.log.info("ML Trading System shutting down...")
    
    # Cleanup global resources
    try:
        from project.monitoring import stop_monitoring
        stop_monitoring()
        server.log.info("Monitoring system stopped")
    except ImportError:
        pass

# ============================================================================
#                           DEVELOPMENT VS PRODUCTION
# ============================================================================

if os.getenv('FLASK_ENV') == 'development':
    # Development settings
    reload = True
    reload_extra_files = ['project/']
    workers = 2  # Fewer workers for development
    loglevel = 'debug'
    timeout = 120  # Shorter timeout for development
else:
    # Production settings
    reload = False
    preload_app = True
    
    # Security settings
    limit_request_line = 4094
    limit_request_fields = 100
    limit_request_field_size = 8190

# ============================================================================
#                           REDIS CLUSTER CONFIGURATION
# ============================================================================

# production/redis_cluster_config.py
"""
Redis Cluster Configuration for ML Trading System
Provides high availability and horizontal scaling for caching and task queues
"""

import os
from typing import List, Dict

class RedisClusterConfig:
    """Redis Cluster configuration for production deployment"""
    
    # Cluster nodes (configure based on your deployment)
    CLUSTER_NODES = [
        {'host': os.getenv('REDIS_NODE_1_HOST', 'redis-node-1'), 'port': int(os.getenv('REDIS_NODE_1_PORT', 6379))},
        {'host': os.getenv('REDIS_NODE_2_HOST', 'redis-node-2'), 'port': int(os.getenv('REDIS_NODE_2_PORT', 6379))},
        {'host': os.getenv('REDIS_NODE_3_HOST', 'redis-node-3'), 'port': int(os.getenv('REDIS_NODE_3_PORT', 6379))},
        {'host': os.getenv('REDIS_NODE_4_HOST', 'redis-node-4'), 'port': int(os.getenv('REDIS_NODE_4_PORT', 6379))},
        {'host': os.getenv('REDIS_NODE_5_HOST', 'redis-node-5'), 'port': int(os.getenv('REDIS_NODE_5_PORT', 6379))},
        {'host': os.getenv('REDIS_NODE_6_HOST', 'redis-node-6'), 'port': int(os.getenv('REDIS_NODE_6_PORT', 6379))},
    ]
    
    # Connection settings
    CONNECTION_POOL_KWARGS = {
        'max_connections': int(os.getenv('REDIS_MAX_CONNECTIONS', 50)),
        'retry_on_timeout': True,
        'health_check_interval': int(os.getenv('REDIS_HEALTH_CHECK_INTERVAL', 30)),
        'socket_timeout': float(os.getenv('REDIS_SOCKET_TIMEOUT', 5.0)),
        'socket_connect_timeout': float(os.getenv('REDIS_CONNECT_TIMEOUT', 5.0)),
    }
    
    # Cluster settings
    CLUSTER_KWARGS = {
        'startup_nodes': CLUSTER_NODES,
        'decode_responses': False,  # Important for binary data (pickled models)
        'skip_full_coverage_check': True,
        'max_connections': int(os.getenv('REDIS_CLUSTER_MAX_CONNECTIONS', 100)),
        'readonly_mode': False,
        'reinitialize_steps': int(os.getenv('REDIS_REINIT_STEPS', 10)),
    }
    
    # Celery-specific configuration
    CELERY_BROKER_URL = f"redis+cluster://{':'.join([f'{node['host']}:{node['port']}' for node in CLUSTER_NODES[:3]])}/0"
    CELERY_RESULT_BACKEND = f"redis+cluster://{':'.join([f'{node['host']}:{node['port']}' for node in CLUSTER_NODES[3:]])}/0"
    
    # Cache-specific configuration
    CACHE_REDIS_URL = f"redis+cluster://{':'.join([f'{node['host']}:{node['port']}' for node in CLUSTER_NODES])}/1"

# production/celery_config_optimized.py
"""
Optimized Celery Configuration for ML Trading System
Handles task routing, memory management, and monitoring
"""

import os
from datetime import timedelta
from kombu import Queue, Exchange

# Import Redis cluster config
from .redis_cluster_config import RedisClusterConfig

class OptimizedCeleryConfig:
    """Production-optimized Celery configuration"""
    
    # Broker settings (Redis Cluster)
    broker_url = RedisClusterConfig.CELERY_BROKER_URL
    result_backend = RedisClusterConfig.CELERY_RESULT_BACKEND
    
    # Serialization
    task_serializer = 'json'
    result_serializer = 'json'
    accept_content = ['json']
    result_accept_content = ['json']
    
    # Timezone
    timezone = 'UTC'
    enable_utc = True
    
    # Task execution
    task_acks_late = True
    task_reject_on_worker_lost = True
    task_ignore_result = False
    
    # Worker configuration
    worker_prefetch_multiplier = 1  # Critical for ML tasks - prevents memory issues
    worker_max_tasks_per_child = int(os.getenv('CELERY_MAX_TASKS_PER_CHILD', 50))  # Memory leak prevention
    worker_disable_rate_limits = False
    
    # Task routing and queues
    task_default_queue = 'default'
    task_default_exchange = 'default'
    task_default_exchange_type = 'direct'
    task_default_routing_key = 'default'
    
    # Define exchanges
    task_exchanges = [
        Exchange('ml_tasks', type='direct'),
        Exchange('preprocessing', type='direct'),
        Exchange('training', type='direct'),
        Exchange('inference', type='direct'),
    ]
    
    # Define queues with different priorities
    task_queues = [
        # High priority queue for real-time inference
        Queue('inference', Exchange('inference'), routing_key='inference',
              queue_arguments={'x-max-priority': 10}),
        
        # Medium priority for preprocessing
        Queue('preprocessing', Exchange('preprocessing'), routing_key='preprocessing',
              queue_arguments={'x-max-priority': 5}),
        
        # Lower priority for training (resource intensive)
        Queue('training_cpu', Exchange('training'), routing_key='training.cpu',
              queue_arguments={'x-max-priority': 2}),
        
        Queue('training_gpu', Exchange('training'), routing_key='training.gpu',
              queue_arguments={'x-max-priority': 3}),
        
        # Default queue
        Queue('default', Exchange('default'), routing_key='default',
              queue_arguments={'x-max-priority': 1}),
    ]
    
    # Task routing
    task_routes = {
        # Training tasks
        'project.tasks.tune_and_train_async': {
            'queue': 'training_gpu',
            'routing_key': 'training.gpu'
        },
        'project.tasks.train_lstm_async': {
            'queue': 'training_gpu',
            'routing_key': 'training.gpu'
        },
        'project.tasks.train_xgboost_async': {
            'queue': 'training_cpu',
            'routing_key': 'training.cpu'
        },
        
        # Feature computation tasks
        'project.tasks.compute_features_async': {
            'queue': 'preprocessing',
            'routing_key': 'preprocessing'
        },
        
        # Inference tasks (highest priority)
        'project.tasks.predict_async': {
            'queue': 'inference',
            'routing_key': 'inference'
        },
    }
    
    # Result backend settings
    result_expires = int(os.getenv('CELERY_RESULT_EXPIRES', 3600))  # 1 hour
    result_cache_max = int(os.getenv('CELERY_RESULT_CACHE_MAX', 10000))
    
    # Task time limits (important for ML workloads)
    task_time_limit = int(os.getenv('CELERY_TASK_TIME_LIMIT', 3600))  # 1 hour
    task_soft_time_limit = int(os.getenv('CELERY_TASK_SOFT_TIME_LIMIT', 3300))  # 55 minutes
    
    # Monitoring and logging
    worker_send_task_events = True
    task_send_sent_event = True
    
    # Memory management
    worker_max_memory_per_child = int(os.getenv('CELERY_MAX_MEMORY_MB', 2048)) * 1024  # 2GB in KB
    
    # Beat schedule for periodic tasks
    beat_schedule = {
        'cleanup-expired-results': {
            'task': 'project.tasks.cleanup_expired_results',
            'schedule': timedelta(hours=1),
            'options': {'queue': 'default'}
        },
        'update-model-performance-metrics': {
            'task': 'project.tasks.update_performance_metrics',
            'schedule': timedelta(minutes=15),
            'options': {'queue': 'preprocessing'}
        },
        'health-check-models': {
            'task': 'project.tasks.health_check_models',
            'schedule': timedelta(minutes=30),
            'options': {'queue': 'default'}
        },
    }
    
    # Security
    worker_hijack_root_logger = False
    worker_log_color = False
    
    # Advanced settings for production
    broker_connection_retry_on_startup = True
    broker_connection_retry = True
    broker_connection_max_retries = 10
    
    # Task compression
    task_compression = 'gzip'
    result_compression = 'gzip'

# production/docker-compose.yml
"""
Docker Compose configuration for production deployment
"""

version: '3.8'

services:
  # Redis Cluster Nodes
  redis-node-1:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --port 6379
    ports:
      - "7001:6379"
    volumes:
      - redis-node-1-data:/data

  redis-node-2:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --port 6379
    ports:
      - "7002:6379"
    volumes:
      - redis-node-2-data:/data

  redis-node-3:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --port 6379
    ports:
      - "7003:6379"
    volumes:
      - redis-node-3-data:/data

  redis-node-4:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --port 6379
    ports:
      - "7004:6379"
    volumes:
      - redis-node-4-data:/data

  redis-node-5:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --port 6379
    ports:
      - "7005:6379"
    volumes:
      - redis-node-5-data:/data

  redis-node-6:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --port 6379
    ports:
      - "7006:6379"
    volumes:
      - redis-node-6-data:/data

  # ML Trading Application
  ml-trading-app:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "5000:5000"
      - "8000:8000"  # Prometheus metrics
    environment:
      - FLASK_ENV=production
      - REDIS_NODE_1_HOST=redis-node-1
      - REDIS_NODE_2_HOST=redis-node-2
      - REDIS_NODE_3_HOST=redis-node-3
      - REDIS_NODE_4_HOST=redis-node-4
      - REDIS_NODE_5_HOST=redis-node-5
      - REDIS_NODE_6_HOST=redis-node-6
      - GUNICORN_WORKERS=4
      - CELERY_MAX_TASKS_PER_CHILD=50
      - PROMETHEUS_METRICS_ENABLED=true
    volumes:
      - ./logs:/var/log/ml-trading
      - model-storage:/app/saved_models
    depends_on:
      - redis-node-1
      - redis-node-2
      - redis-node-3
      - redis-node-4
      - redis-node-5
      - redis-node-6
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  # Celery Workers - CPU-intensive tasks
  celery-worker-cpu:
    build:
      context: .
      dockerfile: Dockerfile.production
    command: celery -A project.extensions:celery worker --loglevel=info --queues=training_cpu,preprocessing,default --concurrency=2 --hostname=cpu-worker@%h
    environment:
      - FLASK_ENV=production
      - REDIS_NODE_1_HOST=redis-node-1
      - REDIS_NODE_2_HOST=redis-node-2
      - REDIS_NODE_3_HOST=redis-node-3
      - REDIS_NODE_4_HOST=redis-node-4
      - REDIS_NODE_5_HOST=redis-node-5
      - REDIS_NODE_6_HOST=redis-node-6
      - CELERY_MAX_TASKS_PER_CHILD=20
    volumes:
      - ./logs:/var/log/ml-trading
      - model-storage:/app/saved_models
    depends_on:
      - redis-node-1
      - redis-node-2
      - redis-node-3
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 3G
          cpus: '2.0'

  # Celery Workers - GPU-intensive tasks
  celery-worker-gpu:
    build:
      context: .
      dockerfile: Dockerfile.production
    command: celery -A project.extensions:celery worker --loglevel=info --queues=training_gpu,inference --concurrency=1 --hostname=gpu-worker@%h
    environment:
      - FLASK_ENV=production
      - REDIS_NODE_1_HOST=redis-node-1
      - REDIS_NODE_2_HOST=redis-node-2
      - REDIS_NODE_3_HOST=redis-node-3
      - REDIS_NODE_4_HOST=redis-node-4
      - REDIS_NODE_5_HOST=redis-node-5
      - REDIS_NODE_6_HOST=redis-node-6
      - CELERY_MAX_TASKS_PER_CHILD=10
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./logs:/var/log/ml-trading
      - model-storage:/app/saved_models
    depends_on:
      - redis-node-1
      - redis-node-2
      - redis-node-3
    deploy:
      resources:
        limits:
          memory: 6G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Celery Beat Scheduler
  celery-beat:
    build:
      context: .
      dockerfile: Dockerfile.production
    command: celery -A project.extensions:celery beat --loglevel=info --scheduler=django_celery_beat.schedulers:DatabaseScheduler
    environment:
      - FLASK_ENV=production
      - REDIS_NODE_1_HOST=redis-node-1
      - REDIS_NODE_2_HOST=redis-node-2
      - REDIS_NODE_3_HOST=redis-node-3
    volumes:
      - ./logs:/var/log/ml-trading
    depends_on:
      - redis-node-1

  # Flower - Celery monitoring
  flower:
    build:
      context: .
      dockerfile: Dockerfile.production
    command: celery -A project.extensions:celery flower --port=5555
    ports:
      - "5555:5555"
    environment:
      - FLASK_ENV=production
      - REDIS_NODE_1_HOST=redis-node-1
      - REDIS_NODE_2_HOST=redis-node-2
      - REDIS_NODE_3_HOST=redis-node-3
    depends_on:
      - redis-node-1

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  redis-node-1-data:
  redis-node-2-data:
  redis-node-3-data:
  redis-node-4-data:
  redis-node-5-data:
  redis-node-6-data:
  model-storage:
  prometheus-data:
  grafana-data:

networks:
  default:
    driver: bridge

# production/Dockerfile.production
"""
Production-optimized Dockerfile for ML Trading System
"""

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements/production.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to app user
RUN chown -R app:app /app

# Switch to app user
USER app

# Expose ports
EXPOSE 5000 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["gunicorn", "--config", "production/gunicorn_config.py", "project.server:app"]

# production/requirements.txt
"""
Production requirements for ML Trading System
"""

# Core ML and Data Science
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
xgboost>=1.7.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
numba>=0.57.0

# Technical Analysis
talib-binary>=0.4.25

# Web Framework
Flask>=2.3.0
Flask-JWT-Extended>=4.5.0
Flask-CORS>=4.0.0
Flask-Limiter>=3.5.0
gunicorn>=21.2.0
gevent>=23.7.0

# Task Queue
celery>=5.3.0
redis>=4.6.0
redis-py-cluster>=2.1.3
kombu>=5.3.0

# Monitoring and Metrics
prometheus-client>=0.17.0
prometheus-flask-exporter>=0.23.0
psutil>=5.9.0

# Data Storage
google-cloud-storage>=2.10.0
joblib>=1.3.0

# Security
cryptography>=41.0.0
marshmallow>=3.20.0
werkzeug>=2.3.0

# Utilities
python-dotenv>=1.0.0
pathlib2>=2.3.7
pynvml>=11.5.0  # NVIDIA GPU monitoring

# Testing (for health checks)
pytest>=7.4.0
requests>=2.31.0

# production/monitoring/prometheus.yml
"""
Prometheus configuration for ML Trading System
"""

global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'ml-trading-app'
    static_configs:
      - targets: ['ml-trading-app:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'celery-workers'
    static_configs:
      - targets: ['celery-worker-cpu:8001', 'celery-worker-gpu:8001']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'redis-cluster'
    static_configs:
      - targets: ['redis-node-1:6379', 'redis-node-2:6379', 'redis-node-3:6379']
    metrics_path: '/metrics'
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

# production/monitoring/grafana/dashboards/ml-trading-dashboard.json
"""
Grafana Dashboard Configuration for ML Trading System
"""

{
  "dashboard": {
    "id": null,
    "title": "ML Trading System Dashboard",
    "tags": ["ml", "trading", "performance"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Training Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_training_requests_total[5m])",
            "legendFormat": "Training Requests/sec"
          }
        ]
      },
      {
        "title": "Training Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_training_duration_seconds",
            "legendFormat": "Duration ({{model_type}})"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_system_cpu_usage_percent",
            "legendFormat": "CPU Usage %"
          },
          {
            "expr": "ml_system_memory_usage_bytes / 1024 / 1024 / 1024",
            "legendFormat": "Memory Usage GB"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_gpu_utilization_percent",
            "legendFormat": "GPU {{device}}"
          }
        ]
      },
      {
        "title": "Active Celery Tasks",
        "type": "singlestat",
        "targets": [
          {
            "expr": "sum(ml_celery_active_tasks)",
            "legendFormat": "Active Tasks"
          }
        ]
      },
      {
        "title": "Cache Hit Ratio",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_cache_hit_ratio",
            "legendFormat": "Hit Ratio ({{cache_type}})"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  }
}

# production/scripts/setup_redis_cluster.sh
#!/bin/bash
"""
Script to initialize Redis Cluster
"""

set -e

echo "Setting up Redis Cluster..."

# Wait for all Redis nodes to be ready
echo "Waiting for Redis nodes to start..."
sleep 30

# Create the cluster
docker exec redis-node-1 redis-cli --cluster create \
  redis-node-1:6379 \
  redis-node-2:6379 \
  redis-node-3:6379 \
  redis-node-4:6379 \
  redis-node-5:6379 \
  redis-node-6:6379 \
  --cluster-replicas 1 \
  --cluster-yes

echo "Redis Cluster setup completed!"

# production/scripts/deploy.sh
#!/bin/bash
"""
Production deployment script
"""

set -e

echo "Starting ML Trading System deployment..."

# Build and start services
docker-compose -f production/docker-compose.yml build
docker-compose -f production/docker-compose.yml up -d

# Wait for Redis cluster to be ready
sleep 60

# Initialize Redis cluster
./production/scripts/setup_redis_cluster.sh

# Wait for application to be ready
echo "Waiting for application to start..."
sleep 30

# Health check
curl -f http://localhost:5000/health || {
    echo "Health check failed!"
    docker-compose -f production/docker-compose.yml logs ml-trading-app
    exit 1
}

echo "ML Trading System deployed successfully!"
echo "Application: http://localhost:5000"
echo "Flower (Celery monitoring): http://localhost:5555"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/admin123)"