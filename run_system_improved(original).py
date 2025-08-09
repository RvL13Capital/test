# run_system.py
"""
Enhanced ML Trading System Main Entry Point
Provides robust system initialization, monitoring, and management
"""

import os
import sys
import asyncio
import logging
import signal
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import psutil
import aiofiles

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add project to path
sys.path.insert(0, '.')


class SystemState(Enum):
    """System state enumeration"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ComponentStatus:
    """Status of a system component"""
    name: str
    state: str
    healthy: bool
    last_check: datetime
    details: Dict[str, Any]
    error: Optional[str] = None


class SystemMetrics:
    """System metrics collector"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.metrics = {
            'requests': 0,
            'errors': 0,
            'optimizations': 0,
            'predictions': 0,
            'data_points': 0
        }
        self.component_health = {}
    
    def increment(self, metric: str, value: int = 1):
        """Increment a metric"""
        if metric in self.metrics:
            self.metrics[metric] += value
    
    def get_uptime(self) -> timedelta:
        """Get system uptime"""
        return datetime.now() - self.start_time
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        return {
            'uptime': str(self.get_uptime()),
            'metrics': self.metrics,
            'component_health': self.component_health
        }


class SystemManager:
    """Enhanced system manager with better error handling and monitoring"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = self._setup_logging()
        self.components = {}
        self.shutdown_event = asyncio.Event()
        self.state = SystemState.INITIALIZING
        self.metrics = SystemMetrics()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._health_check_interval = 60  # seconds
        self._optimization_interval = 3600 * 24  # daily
        
    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging"""
        log_level = os.getenv('LOG_LEVEL', self.args.log_level).upper()
        
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'logs/system_{datetime.now().strftime("%Y%m%d")}.log', 'a'),
                logging.handlers.RotatingFileHandler(
                    'logs/system.log',
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                )
            ]
        )
        
        # Configure specific loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('google').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured at {log_level} level")
        
        return logger
    
    def validate_environment(self) -> bool:
        """Enhanced environment validation"""
        self.logger.info("Validating environment configuration...")
        
        validations = {
            'core': self._validate_core_env(),
            'data_providers': self._validate_data_providers(),
            'security': self._validate_security(),
            'performance': self._validate_performance_settings()
        }
        
        # Log validation results
        for category, result in validations.items():
            if result['valid']:
                self.logger.info(f"✅ {category}: {result['message']}")
            else:
                self.logger.error(f"❌ {category}: {result['message']}")
                if result.get('details'):
                    for detail in result['details']:
                        self.logger.error(f"  - {detail}")
        
        # Overall validation
        all_valid = all(v['valid'] for v in validations.values())
        
        if not all_valid and not self.args.force:
            self.logger.error("Environment validation failed. Use --force to override.")
            return False
        elif not all_valid and self.args.force:
            self.logger.warning("Environment validation failed but --force specified. Continuing...")
        
        return True
    
    def _validate_core_env(self) -> Dict:
        """Validate core environment variables"""
        required_vars = {
            'SECRET_KEY': 'Flask secret key',
            'JWT_SECRET_KEY': 'JWT authentication key',
            'GCS_BUCKET_NAME': 'Google Cloud Storage bucket',
            'CELERY_BROKER_URL': 'Message broker URL'
        }
        
        missing = []
        for var, desc in required_vars.items():
            if not os.getenv(var):
                missing.append(f"{var} ({desc})")
        
        if missing:
            return {
                'valid': False,
                'message': 'Missing required variables',
                'details': missing
            }
        
        return {
            'valid': True,
            'message': 'All core variables configured'
        }
    
    def _validate_data_providers(self) -> Dict:
        """Validate data provider configuration"""
        providers = [
            'POLYGON_API_KEY',
            'ALPHA_VANTAGE_API_KEY',
            'IEX_CLOUD_API_KEY',
            'TIINGO_API_KEY'
        ]
        
        configured = [p for p in providers if os.getenv(p)]
        
        if not configured:
            return {
                'valid': False,
                'message': 'No data providers configured'
            }
        
        return {
            'valid': True,
            'message': f'{len(configured)} data provider(s) configured'
        }
    
    def _validate_security(self) -> Dict:
        """Validate security settings"""
        issues = []
        
        # Check secret strength
        for var in ['SECRET_KEY', 'JWT_SECRET_KEY']:
            value = os.getenv(var, '')
            if len(value) < 32:
                issues.append(f"{var} is too short (min 32 chars)")
            if value.startswith('your_'):
                issues.append(f"{var} is using default value")
        
        # Check HTTPS enforcement
        if os.getenv('FLASK_ENV') == 'production' and not os.getenv('ENFORCE_HTTPS'):
            issues.append("HTTPS not enforced in production")
        
        if issues:
            return {
                'valid': False,
                'message': 'Security issues detected',
                'details': issues
            }
        
        return {
            'valid': True,
            'message': 'Security properly configured'
        }
    
    def _validate_performance_settings(self) -> Dict:
        """Validate performance settings"""
        # Check system resources
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if cpu_count < 2 or memory_gb < 4:
            return {
                'valid': False,
                'message': f'Insufficient resources: {cpu_count} CPUs, {memory_gb:.1f}GB RAM'
            }
        
        return {
            'valid': True,
            'message': f'Adequate resources: {cpu_count} CPUs, {memory_gb:.1f}GB RAM'
        }
    
    async def initialize_components(self):
        """Initialize all system components with error recovery"""
        self.logger.info("🚀 Initializing ML Trading System components...")
        
        initialization_steps = [
            ('storage', self._init_storage),
            ('market_data', self._init_market_data),
            ('redis', self._init_redis),
            ('optimizer', self._init_optimizer),
            ('monitoring', self._init_monitoring),
            ('web_server', self._init_web_server)
        ]
        
        for component_name, init_func in initialization_steps:
            try:
                self.logger.info(f"Initializing {component_name}...")
                await init_func()
                
                self.metrics.component_health[component_name] = ComponentStatus(
                    name=component_name,
                    state='initialized',
                    healthy=True,
                    last_check=datetime.now(),
                    details={}
                )
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {component_name}: {e}")
                
                self.metrics.component_health[component_name] = ComponentStatus(
                    name=component_name,
                    state='error',
                    healthy=False,
                    last_check=datetime.now(),
                    details={},
                    error=str(e)
                )
                
                if component_name in ['storage', 'web_server'] and not self.args.partial:
                    raise RuntimeError(f"Critical component {component_name} failed to initialize")
                elif self.args.partial:
                    self.logger.warning(f"Running in partial mode without {component_name}")
                    self.state = SystemState.DEGRADED
        
        # Check overall system health
        healthy_components = sum(1 for c in self.metrics.component_health.values() if c.healthy)
        total_components = len(self.metrics.component_health)
        
        if healthy_components == total_components:
            self.state = SystemState.RUNNING
            self.logger.info("✅ All components initialized successfully")
        elif healthy_components > 0:
            self.state = SystemState.DEGRADED
            self.logger.warning(f"⚠️  System running in degraded mode ({healthy_components}/{total_components} components)")
        else:
            self.state = SystemState.ERROR
            raise RuntimeError("No components could be initialized")
    
    async def _init_storage(self):
        """Initialize storage component"""
        from project.storage import get_gcs_storage
        
        storage = get_gcs_storage()
        if not storage.client:
            raise RuntimeError("GCS client initialization failed")
        
        # Test connection
        storage.client.list_buckets(max_results=1)
        
        self.components['storage'] = storage
        self.logger.info("✅ Storage initialized (GCS)")
    
    async def _init_market_data(self):
        """Initialize market data component"""
        from project.market_data import get_market_data_manager
        
        manager = await get_market_data_manager()
        
        # Test with a simple request
        test_info = await manager.get_stock_info('AAPL')
        if not test_info:
            self.logger.warning("Market data test request failed")
        
        self.components['market_data'] = manager
        self.logger.info("✅ Market data initialized")
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        import redis.asyncio as redis
        
        redis_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
        
        try:
            self.components['redis'] = await redis.from_url(redis_url)
            await self.components['redis'].ping()
            self.logger.info("✅ Redis connection established")
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            if not self.args.partial:
                raise
    
    async def _init_optimizer(self):
        """Initialize auto-optimizer"""
        if self.args.no_optimizer:
            self.logger.info("Optimizer disabled by --no-optimizer flag")
            return
        
        from project.auto_optimizer_working import WorkingAutoOptimizer
        
        optimizer = WorkingAutoOptimizer()
        await optimizer.initialize()
        
        self.components['optimizer'] = optimizer
        self.logger.info("✅ Auto-optimizer initialized")
    
    async def _init_monitoring(self):
        """Initialize monitoring"""
        if self.args.no_monitoring:
            self.logger.info("Monitoring disabled by --no-monitoring flag")
            return
        
        from project.monitoring import start_monitoring
        
        start_monitoring()
        self.components['monitoring'] = True
        self.logger.info("✅ Monitoring initialized")
    
    async def _init_web_server(self):
        """Initialize web server"""
        from project.server_integration import create_integrated_app
        
        app = create_integrated_app()
        self.components['app'] = app
        self.logger.info("✅ Web server initialized")
    
    async def start_background_services(self):
        """Start all background services"""
        self.logger.info("🚀 Starting background services...")
        
        services = []
        
        # Health monitoring
        services.append(asyncio.create_task(
            self._health_monitor(),
            name="health_monitor"
        ))
        
        # Optimization loop
        if self.components.get('optimizer') and not self.args.no_optimizer:
            services.append(asyncio.create_task(
                self._optimization_loop(),
                name="optimization_loop"
            ))
        
        # Data collection
        if os.getenv('AUTO_DATA_COLLECTION', 'false').lower() == 'true':
            services.append(asyncio.create_task(
                self._data_collection_loop(),
                name="data_collection"
            ))
        
        # Metrics reporter
        services.append(asyncio.create_task(
            self._metrics_reporter(),
            name="metrics_reporter"
        ))
        
        # Cleanup monitor
        services.append(asyncio.create_task(
            self._cleanup_monitor(),
            name="cleanup_monitor"
        ))
        
        self.logger.info(f"✅ Started {len(services)} background services")
        
        # Store tasks for cleanup
        self.components['background_tasks'] = services
    
    async def _health_monitor(self):
        """Enhanced health monitoring with auto-recovery"""
        self.logger.info("Starting health monitor...")
        
        consecutive_failures = {}
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self._health_check_interval)
                
                if self.shutdown_event.is_set():
                    break
                
                # Check each component
                for component_name, component in self.components.items():
                    if component_name in ['app', 'background_tasks']:
                        continue
                    
                    try:
                        # Component-specific health checks
                        is_healthy = await self._check_component_health(component_name, component)
                        
                        if is_healthy:
                            consecutive_failures[component_name] = 0
                            self.metrics.component_health[component_name] = ComponentStatus(
                                name=component_name,
                                state='healthy',
                                healthy=True,
                                last_check=datetime.now(),
                                details={}
                            )
                        else:
                            consecutive_failures[component_name] = consecutive_failures.get(component_name, 0) + 1
                            
                            if consecutive_failures[component_name] >= 3:
                                self.logger.error(f"{component_name} failed {consecutive_failures[component_name]} consecutive health checks")
                                await self._attempt_component_recovery(component_name)
                    
                    except Exception as e:
                        self.logger.error(f"Health check failed for {component_name}: {e}")
                        consecutive_failures[component_name] = consecutive_failures.get(component_name, 0) + 1
                
                # Update overall system state
                self._update_system_state()
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _check_component_health(self, name: str, component: Any) -> bool:
        """Check health of a specific component"""
        try:
            if name == 'storage':
                # Test GCS connection
                list(component.client.list_buckets(max_results=1))
                return True
            
            elif name == 'market_data':
                # Test market data
                info = await component.get_stock_info('AAPL')
                return info is not None
            
            elif name == 'redis':
                # Test Redis connection
                await component.ping()
                return True
            
            elif name == 'optimizer':
                # Check optimizer state
                return hasattr(component, 'initialized') and component.initialized
            
            else:
                return True
                
        except Exception:
            return False
    
    async def _attempt_component_recovery(self, component_name: str):
        """Attempt to recover a failed component"""
        self.logger.warning(f"Attempting to recover {component_name}...")
        
        try:
            if component_name == 'storage':
                await self._init_storage()
            elif component_name == 'market_data':
                await self._init_market_data()
            elif component_name == 'redis':
                await self._init_redis()
            elif component_name == 'optimizer':
                await self._init_optimizer()
            
            self.logger.info(f"✅ Successfully recovered {component_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to recover {component_name}: {e}")
            self.state = SystemState.DEGRADED
    
    def _update_system_state(self):
        """Update overall system state based on component health"""
        healthy_count = sum(1 for c in self.metrics.component_health.values() if c.healthy)
        total_count = len(self.metrics.component_health)
        
        if healthy_count == total_count:
            if self.state != SystemState.RUNNING:
                self.logger.info("System state: RUNNING (all components healthy)")
            self.state = SystemState.RUNNING
        elif healthy_count > total_count * 0.5:
            if self.state != SystemState.DEGRADED:
                self.logger.warning(f"System state: DEGRADED ({healthy_count}/{total_count} healthy)")
            self.state = SystemState.DEGRADED
        else:
            if self.state != SystemState.ERROR:
                self.logger.error(f"System state: ERROR ({healthy_count}/{total_count} healthy)")
            self.state = SystemState.ERROR
    
    async def _optimization_loop(self):
        """Background optimization with error recovery"""
        self.logger.info("Starting optimization loop...")
        
        optimizer = self.components.get('optimizer')
        if not optimizer:
            self.logger.warning("Optimizer not available")
            return
        
        # Initial optimization if configured
        if self.args.initial_optimization:
            try:
                self.logger.info("Running initial optimization...")
                await optimizer.run_optimization()
                self.metrics.increment('optimizations')
                self.logger.info("✅ Initial optimization completed")
            except Exception as e:
                self.logger.error(f"Initial optimization failed: {e}")
        
        # Periodic optimization
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self._optimization_interval)
                
                if self.shutdown_event.is_set():
                    break
                
                # Check if optimization should run
                if self._should_run_optimization():
                    self.logger.info("Starting scheduled optimization...")
                    
                    start_time = datetime.now()
                    await optimizer.run_optimization()
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    self.metrics.increment('optimizations')
                    self.logger.info(f"✅ Optimization completed in {duration:.1f} seconds")
                    
                    # Save optimization record
                    await self._save_optimization_record(duration)
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                self.metrics.increment('errors')
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    def _should_run_optimization(self) -> bool:
        """Determine if optimization should run"""
        # Check time window
        current_hour = datetime.now().hour
        start_hour = int(os.getenv('OPTIMIZATION_START_HOUR', '2'))
        end_hour = int(os.getenv('OPTIMIZATION_END_HOUR', '6'))
        
        if start_hour <= current_hour < end_hour:
            return True
        
        # Check if forced
        if os.path.exists('force_optimization.flag'):
            os.remove('force_optimization.flag')
            return True
        
        return False
    
    async def _save_optimization_record(self, duration: float):
        """Save optimization record"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'state': self.state.value,
            'metrics': self.metrics.get_summary()
        }
        
        # Save to file
        async with aiofiles.open('logs/optimization_history.jsonl', 'a') as f:
            await f.write(json.dumps(record) + '\n')
    
    async def _data_collection_loop(self):
        """Periodic data collection with rate limiting"""
        self.logger.info("Starting data collection loop...")
        
        collection_interval = int(os.getenv('DATA_COLLECTION_INTERVAL', '86400'))  # Daily default
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(collection_interval)
                
                if self.shutdown_event.is_set():
                    break
                
                optimizer = self.components.get('optimizer')
                if optimizer:
                    self.logger.info("Collecting training data...")
                    
                    days_back = int(os.getenv('DATA_COLLECTION_DAYS', '30'))
                    data = await optimizer.collect_training_data(days_back=days_back)
                    
                    self.metrics.increment('data_points', len(data))
                    self.logger.info(f"✅ Collected {len(data)} data points")
                
            except Exception as e:
                self.logger.error(f"Data collection error: {e}")
                self.metrics.increment('errors')
                await asyncio.sleep(3600)
    
    async def _metrics_reporter(self):
        """Periodic metrics reporting"""
        report_interval = int(os.getenv('METRICS_REPORT_INTERVAL', '3600'))  # Hourly
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(report_interval)
                
                if self.shutdown_event.is_set():
                    break
                
                # Generate report
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'state': self.state.value,
                    'uptime': str(self.metrics.get_uptime()),
                    'metrics': self.metrics.metrics,
                    'component_health': {
                        name: {
                            'healthy': status.healthy,
                            'state': status.state,
                            'last_check': status.last_check.isoformat()
                        }
                        for name, status in self.metrics.component_health.items()
                    },
                    'system_resources': {
                        'cpu_percent': psutil.cpu_percent(interval=1),
                        'memory_percent': psutil.virtual_memory().percent,
                        'disk_percent': psutil.disk_usage('/').percent
                    }
                }
                
                # Log summary
                self.logger.info(f"System metrics: {report['metrics']}")
                
                # Save detailed report
                async with aiofiles.open('logs/metrics_history.jsonl', 'a') as f:
                    await f.write(json.dumps(report) + '\n')
                
            except Exception as e:
                self.logger.error(f"Metrics reporter error: {e}")
    
    async def _cleanup_monitor(self):
        """Monitor and cleanup old files"""
        cleanup_interval = 3600 * 24  # Daily
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(cleanup_interval)
                
                if self.shutdown_event.is_set():
                    break
                
                # Cleanup old logs
                await self._cleanup_old_logs()
                
                # Cleanup temporary files
                await self._cleanup_temp_files()
                
            except Exception as e:
                self.logger.error(f"Cleanup monitor error: {e}")
    
    async def _cleanup_old_logs(self):
        """Cleanup old log files"""
        log_retention_days = int(os.getenv('LOG_RETENTION_DAYS', '30'))
        cutoff_date = datetime.now() - timedelta(days=log_retention_days)
        
        logs_dir = Path('logs')
        cleaned_count = 0
        
        for log_file in logs_dir.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink()
                cleaned_count += 1
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old log files")
    
    async def _cleanup_temp_files(self):
        """Cleanup temporary files"""
        temp_dir = Path('temp')
        if temp_dir.exists():
            for temp_file in temp_dir.glob('*'):
                if temp_file.is_file():
                    temp_file.unlink()
    
    async def run_web_server(self):
        """Run web server with graceful shutdown"""
        if 'app' not in self.components:
            self.logger.error("Web server not initialized")
            return
        
        if self.args.no_web:
            self.logger.info("Web server disabled by --no-web flag")
            return
        
        # Use Hypercorn for async support
        from hypercorn.asyncio import serve
        from hypercorn.config import Config as HypercornConfig
        from asgiref.wsgi import WsgiToAsgi
        
        # Convert Flask to ASGI
        asgi_app = WsgiToAsgi(self.components['app'])
        
        # Configure Hypercorn
        config = HypercornConfig()
        config.bind = [f"{self.args.host}:{self.args.port}"]
        config.workers = self.args.workers
        config.accesslog = "logs/access.log" if not self.args.no_access_log else None
        config.errorlog = "logs/error.log"
        config.use_reloader = self.args.reload
        
        # SSL configuration
        if self.args.ssl_cert and self.args.ssl_key:
            config.certfile = self.args.ssl_cert
            config.keyfile = self.args.ssl_key
            self.logger.info(f"🔒 HTTPS enabled with certificate: {self.args.ssl_cert}")
        
        self.logger.info(f"🌐 Starting web server on {config.bind[0]}")
        
        try:
            await serve(asgi_app, config, shutdown_trigger=self.shutdown_event.wait)
        except Exception as e:
            self.logger.error(f"Web server error: {e}")
            raise
    
    async def start(self):
        """Start the complete system"""
        try:
            self.logger.info("🚀 ML Trading System starting...")
            self.logger.info(f"Version: {os.getenv('SYSTEM_VERSION', '2025.1.0')}")
            self.logger.info(f"Environment: {os.getenv('FLASK_ENV', 'production')}")
            
            # Initialize components
            await self.initialize_components()
            
            # Start background services
            await self.start_background_services()
            
            # Start web server
            await self.run_web_server()
            
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            self.state = SystemState.ERROR
            raise
    
    async def shutdown(self):
        """Graceful shutdown with cleanup"""
        self.logger.info("🛑 Initiating graceful shutdown...")
        self.state = SystemState.STOPPING
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Stop background tasks
        if 'background_tasks' in self.components:
            self.logger.info("Stopping background tasks...")
            for task in self.components['background_tasks']:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.components['background_tasks'], return_exceptions=True)
        
        # Close components
        if self.components.get('market_data'):
            self.logger.info("Closing market data connections...")
            await self.components['market_data'].close()
        
        if self.components.get('redis'):
            self.logger.info("Closing Redis connection...")
            await self.components['redis'].close()
        
        if self.components.get('monitoring'):
            self.logger.info("Stopping monitoring...")
            from project.monitoring import stop_monitoring
            stop_monitoring()
        
        # Save final metrics
        await self._save_final_metrics()
        
        # Cleanup
        self.executor.shutdown(wait=True)
        
        self.state = SystemState.STOPPED
        self.logger.info("✅ Shutdown complete")
    
    async def _save_final_metrics(self):
        """Save final system metrics"""
        final_report = {
            'shutdown_time': datetime.now().isoformat(),
            'total_uptime': str(self.metrics.get_uptime()),
            'final_metrics': self.metrics.get_summary(),
            'final_state': self.state.value
        }
        
        report_path = Path(f"logs/shutdown_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_path.write_text(json.dumps(final_report, indent=2))
        self.logger.info(f"Final metrics saved to: {report_path}")


# Global system manager for signal handling
system_manager: Optional[SystemManager] = None


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    signal_name = signal.Signals(signum).name
    print(f"\n🛑 Received {signal_name}, initiating graceful shutdown...")
    
    if system_manager and system_manager.state == SystemState.RUNNING:
        asyncio.create_task(system_manager.shutdown())
    else:
        sys.exit(0)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, signal_handler)
    
    # Windows compatibility
    if sys.platform == 'win32':
        signal.signal(signal.SIGBREAK, signal_handler)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='ML Trading System - Advanced Trading Analytics Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_system.py                    # Run with defaults
  python run_system.py --dev             # Run in development mode
  python run_system.py --port 8080       # Run on port 8080
  python run_system.py --workers 4       # Run with 4 workers
  python run_system.py --no-optimizer    # Run without auto-optimizer
  python run_system.py --partial         # Run even if some components fail
        """
    )
    
    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--dev', '--development', action='store_true',
                           help='Run in development mode')
    mode_group.add_argument('--prod', '--production', action='store_true',
                           help='Run in production mode (default)')
    
    # Server arguments
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=None,
                       help='Port to bind to (default: from env or 5000)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes (default: 1)')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload on code changes')
    
    # SSL arguments
    parser.add_argument('--ssl-cert', help='Path to SSL certificate')
    parser.add_argument('--ssl-key', help='Path to SSL private key')
    
    # Component arguments
    parser.add_argument('--no-optimizer', action='store_true',
                       help='Disable auto-optimizer')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Disable system monitoring')
    parser.add_argument('--no-web', action='store_true',
                       help='Run without web server (background services only)')
    
    # Behavior arguments
    parser.add_argument('--force', action='store_true',
                       help='Force start even with validation errors')
    parser.add_argument('--partial', action='store_true',
                       help='Allow partial startup if some components fail')
    parser.add_argument('--initial-optimization', action='store_true',
                       help='Run optimization immediately on startup')
    
    # Logging arguments
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO',
                       help='Set logging level (default: INFO)')
    parser.add_argument('--no-access-log', action='store_true',
                       help='Disable HTTP access logging')
    
    # Other arguments
    parser.add_argument('--version', action='version',
                       version=f"ML Trading System {os.getenv('SYSTEM_VERSION', '2025.1.0')}")
    
    args = parser.parse_args()
    
    # Apply development mode settings
    if args.dev:
        os.environ['FLASK_ENV'] = 'development'
        args.reload = True
        args.log_level = 'DEBUG'
        if args.port is None:
            args.port = 5000
    else:
        os.environ['FLASK_ENV'] = 'production'
        if args.port is None:
            args.port = int(os.getenv('PORT', '8000'))
    
    return args


async def main():
    """Main entry point"""
    global system_manager
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup
    setup_signal_handlers()
    
    # Create system manager
    system_manager = SystemManager(args)
    
    # Validate environment
    if not system_manager.validate_environment():
        sys.exit(1)
    
    try:
        # Start system
        await system_manager.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
    except Exception as e:
        logging.error(f"❌ System error: {e}")
        if args.log_level == 'DEBUG':
            logging.exception("Full traceback:")
        sys.exit(1)
    finally:
        # Ensure cleanup
        if system_manager and system_manager.state != SystemState.STOPPED:
            await system_manager.shutdown()


def run_simple_dev():
    """Run simple development server (Flask built-in)"""
    print("🔧 Running in simple development mode...")
    
    from project.server import app
    
    # Setup breakout system if enabled
    if os.getenv('BREAKOUT_SYSTEM_ENABLED', 'true').lower() == 'true':
        from project.server_integration import setup_breakout_system
        setup_breakout_system(app)
    
    # Run Flask development server
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        use_reloader=True
    )


if __name__ == "__main__":
    # Check for simple dev mode
    if '--simple' in sys.argv:
        run_simple_dev()
    else:
        # Run full async system
        try:
            if sys.platform == 'win32':
                # Windows event loop policy
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
