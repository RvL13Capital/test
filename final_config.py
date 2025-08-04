# project/config_extended.py
"""
Extended configuration with real data provider keys
"""

import os
from .config import Config

class ExtendedConfig(Config):
    """Extended configuration for production-ready system"""
    
    # Data Provider API Keys
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY')
    TIINGO_API_KEY = os.getenv('TIINGO_API_KEY')
    
    # Data Provider Priority
    PRIMARY_DATA_PROVIDER = os.getenv('PRIMARY_DATA_PROVIDER', 'polygon')
    BACKUP_DATA_PROVIDERS = os.getenv('BACKUP_DATA_PROVIDERS', 'alpha_vantage,iex').split(',')
    
    # Model Configuration
    USE_GPU_FOR_INFERENCE = os.getenv('USE_GPU_FOR_INFERENCE', 'false').lower() == 'true'
    MODEL_CACHE_SIZE = int(os.getenv('MODEL_CACHE_SIZE', 5))
    MODEL_CACHE_TTL = int(os.getenv('MODEL_CACHE_TTL', 3600))  # 1 hour
    
    # Training Configuration
    TRAINING_SCHEDULE = os.getenv('TRAINING_SCHEDULE', '02:00')  # 2 AM
    TRAINING_DATA_DAYS = int(os.getenv('TRAINING_DATA_DAYS', 365))
    TRAINING_MIN_SAMPLES = int(os.getenv('TRAINING_MIN_SAMPLES', 10000))
    TRAINING_VALIDATION_SPLIT = float(os.getenv('TRAINING_VALIDATION_SPLIT', 0.2))
    
    # Adaptive Thresholds (initial values, will be learned)
    INITIAL_PHASE_TRANSITION_THRESHOLD = float(os.getenv('INITIAL_PHASE_TRANSITION_THRESHOLD', 0.65))
    INITIAL_NETWORK_DENSITY_THRESHOLD = float(os.getenv('INITIAL_NETWORK_DENSITY_THRESHOLD', 0.68))
    INITIAL_CLUSTERING_THRESHOLD = float(os.getenv('INITIAL_CLUSTERING_THRESHOLD', 0.45))
    
    # Performance Monitoring
    PERFORMANCE_LOG_INTERVAL = int(os.getenv('PERFORMANCE_LOG_INTERVAL', 3600))  # 1 hour
    MIN_TRADES_FOR_EVALUATION = int(os.getenv('MIN_TRADES_FOR_EVALUATION', 20))
    TARGET_WIN_RATE = float(os.getenv('TARGET_WIN_RATE', 0.65))
    TARGET_PROFIT_FACTOR = float(os.getenv('TARGET_PROFIT_FACTOR', 1.8))
    
    # Resource Management
    MAX_CONCURRENT_ANALYSES = int(os.getenv('MAX_CONCURRENT_ANALYSES', 10))
    ANALYSIS_TIMEOUT = int(os.getenv('ANALYSIS_TIMEOUT', 30))  # seconds
    SCREENING_BATCH_SIZE = int(os.getenv('SCREENING_BATCH_SIZE', 10))
    
    # Caching Configuration
    MARKET_DATA_CACHE_TTL = int(os.getenv('MARKET_DATA_CACHE_TTL', 300))  # 5 minutes
    FEATURE_CACHE_TTL = int(os.getenv('FEATURE_CACHE_TTL', 900))  # 15 minutes
    PREDICTION_CACHE_TTL = int(os.getenv('PREDICTION_CACHE_TTL', 60))  # 1 minute
    
    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        super().validate_config()
        
        # Check for at least one data provider
        if not any([cls.POLYGON_API_KEY, cls.ALPHA_VANTAGE_API_KEY, 
                   cls.IEX_CLOUD_API_KEY, cls.TIINGO_API_KEY]):
            raise ValueError("At least one data provider API key must be configured")
        
        # Validate thresholds
        assert 0 < cls.INITIAL_PHASE_TRANSITION_THRESHOLD < 1
        assert 0 < cls.INITIAL_NETWORK_DENSITY_THRESHOLD < 1
        assert 0 < cls.INITIAL_CLUSTERING_THRESHOLD < 1
        
        logger.info("Extended configuration validated successfully")

# ---
# project/server_integration.py
"""
Integration of the working system into main server
"""

# Add to your existing server.py imports:
from .breakout_api_working import breakout_api as breakout_api_v2
from .config_extended import ExtendedConfig
from .auto_optimizer_working import WorkingAutoOptimizer
from .market_data import create_market_data_manager
import asyncio
import threading

# Replace Config with ExtendedConfig
Config = ExtendedConfig

# In the main server setup:
def setup_breakout_system(app):
    """Setup the working breakout system"""
    
    # Validate configuration
    ExtendedConfig.validate_config()
    
    # Register improved API blueprint
    app.register_blueprint(breakout_api_v2)
    logger.info("Breakout system v2 API registered")
    
    # Initialize market data manager
    market_config = {
        'POLYGON_API_KEY': ExtendedConfig.POLYGON_API_KEY,
        'ALPHA_VANTAGE_API_KEY': ExtendedConfig.ALPHA_VANTAGE_API_KEY,
        'PRIMARY_PROVIDER': ExtendedConfig.PRIMARY_DATA_PROVIDER
    }
    
    market_manager = create_market_data_manager(market_config)
    app.market_manager = market_manager
    
    # Start background optimizer if not in training mode
    if not os.getenv('TRAINING_MODE'):
        start_background_optimizer()
    
    logger.info("Breakout system initialized with real data providers")

def start_background_optimizer():
    """Start optimizer in background thread"""
    def run_optimizer():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        optimizer = WorkingAutoOptimizer()
        
        async def initialize_and_monitor():
            await optimizer.initialize()
            logger.info("Background optimizer initialized")
            
            # Continuous monitoring loop
            while True:
                try:
                    # Check if optimization is needed
                    if len(optimizer.performance_history) >= ExtendedConfig.MIN_TRADES_FOR_EVALUATION:
                        recent_performance = optimizer.performance_history[-ExtendedConfig.MIN_TRADES_FOR_EVALUATION:]
                        win_rate = sum(1 for p in recent_performance if p.get('profitable', False)) / len(recent_performance)
                        
                        if win_rate < ExtendedConfig.TARGET_WIN_RATE:
                            logger.info(f"Win rate {win_rate:.2%} below target, triggering optimization")
                            await optimizer.run_optimization()
                    
                    # Sleep before next check
                    await asyncio.sleep(ExtendedConfig.PERFORMANCE_LOG_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"Background optimizer error: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes on error
        
        loop.run_until_complete(initialize_and_monitor())
    
    optimizer_thread = threading.Thread(target=run_optimizer, daemon=True, name="BackgroundOptimizer")
    optimizer_thread.start()

# Add to Flask app initialization:
if ExtendedConfig.BREAKOUT_SYSTEM_ENABLED:
    setup_breakout_system(app)

# Enhanced health check
@app.route('/api/v2/health', methods=['GET'])
def health_check_v2():
    """Enhanced health check with real system status"""
    try:
        health = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2.0.0',
            'services': {}
        }
        
        # Check data providers
        if hasattr(app, 'market_manager'):
            provider_status = 'healthy'
            # Could add actual provider health checks here
            health['services']['data_providers'] = {
                'status': provider_status,
                'primary': ExtendedConfig.PRIMARY_DATA_PROVIDER,
                'configured': [
                    p for p in ['polygon', 'alpha_vantage', 'iex', 'tiingo']
                    if getattr(ExtendedConfig, f'{p.upper()}_API_KEY', None)
                ]
            }
        
        # Check model status
        try:
            from .auto_optimizer_working import _optimizer
            if _optimizer and _optimizer.current_model:
                health['services']['model'] = {
                    'status': 'loaded',
                    'adaptive_thresholds': _optimizer.adaptive_thresholds
                }
            else:
                health['services']['model'] = {'status': 'not_loaded'}
        except:
            health['services']['model'] = {'status': 'error'}
        
        # Overall status
        if any(s.get('status') != 'healthy' for s in health['services'].values()):
            health['status'] = 'degraded'
        
        return jsonify(health)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

# ---
# .env.production.example
"""
# Data Provider API Keys (at least one required)
POLYGON_API_KEY=your_polygon_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
IEX_CLOUD_API_KEY=your_iex_cloud_key_here
TIINGO_API_KEY=your_tiingo_key_here

# Primary data provider
PRIMARY_DATA_PROVIDER=polygon
BACKUP_DATA_PROVIDERS=alpha_vantage,iex

# Security Keys
SECRET_KEY=generate_with_openssl_rand_hex_32
JWT_SECRET_KEY=generate_with_openssl_rand_hex_32
MODEL_ENCRYPTION_KEY=generate_with_fernet

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=breakout_system
POSTGRES_USER=breakout_user
POSTGRES_PASSWORD=secure_password

# Redis
REDIS_PASSWORD=secure_redis_password

# Google Cloud Storage
GCS_BUCKET_NAME=your-breakout-models
GCS_PROJECT_ID=your-gcp-project
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/gcs-key.json

# Training Configuration
TRAINING_SCHEDULE=02:00
USE_GPU_FOR_INFERENCE=false
MODEL_CACHE_SIZE=5

# Performance Targets
TARGET_WIN_RATE=0.65
TARGET_PROFIT_FACTOR=1.8

# Resource Limits
MAX_CONCURRENT_ANALYSES=10
SCREENING_BATCH_SIZE=10
"""