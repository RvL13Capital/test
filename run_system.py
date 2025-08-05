# run_system.py
"""
Main entry point for the ML Trading System
This script initializes and starts all components
"""

import os
import sys
import asyncio
import logging
import signal
from datetime import datetime
from dotenv import load_dotenv
import multiprocessing as mp

# Load environment variables first
load_dotenv()

# Add project to path
sys.path.insert(0, '.')

def setup_logging():
    """Configure logging for the entire system"""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/system.log', 'a')
        ]
    )
    
    # Suppress overly verbose loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)

def validate_environment():
    """Validate all required environment variables"""
    required_vars = [
        'SECRET_KEY',
        'JWT_SECRET_KEY',
        'GCS_BUCKET_NAME',
        'CELERY_BROKER_URL'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Missing required environment variables: {missing}")
        print("Run setup_gcs.py first to validate your configuration")
        return False
    
    # Check at least one data provider
    data_providers = [
        'POLYGON_API_KEY',
        'ALPHA_VANTAGE_API_KEY', 
        'IEX_CLOUD_API_KEY',
        'TIINGO_API_KEY'
    ]
    
    if not any(os.getenv(var) for var in data_providers):
        print("❌ No data provider API keys configured")
        print("At least one is required for the system to work")
        return False
    
    return True

class SystemManager:
    """Manages the lifecycle of all system components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.components = {}
        self.shutdown_event = asyncio.Event()
        
    async def initialize_components(self):
        """Initialize all system components"""
        self.logger.info("Initializing ML Trading System components...")
        
        try:
            # 1. Initialize storage
            self.logger.info("🗄️  Initializing storage...")
            from project.storage import get_gcs_storage
            self.components['storage'] = get_gcs_storage()
            
            if not self.components['storage'].client:
                raise RuntimeError("Failed to initialize GCS storage")
            
            # 2. Initialize market data manager
            self.logger.info("📊 Initializing market data...")
            from project.market_data import get_market_data_manager
            self.components['market_data'] = await get_market_data_manager()
            
            # 3. Initialize auto-optimizer
            self.logger.info("🤖 Initializing auto-optimizer...")
            from project.auto_optimizer_working import WorkingAutoOptimizer
            self.components['optimizer'] = WorkingAutoOptimizer()
            await self.components['optimizer'].initialize()
            
            # 4. Initialize monitoring
            self.logger.info("📈 Initializing monitoring...")
            from project.monitoring import start_monitoring
            start_monitoring()
            self.components['monitoring'] = True
            
            # 5. Initialize Flask app
            self.logger.info("🌐 Initializing web server...")
            from project.server_integration import create_integrated_app
            self.components['app'] = create_integrated_app()
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    async def start_background_services(self):
        """Start background services"""
        self.logger.info("🚀 Starting background services...")
        
        # Start optimization loop
        if self.components.get('optimizer'):
            asyncio.create_task(self._optimization_loop())
        
        # Start health monitoring
        asyncio.create_task(self._health_monitor())
        
        # Start data collection (if enabled)
        if os.getenv('AUTO_DATA_COLLECTION', 'false').lower() == 'true':
            asyncio.create_task(self._data_collection_loop())
    
    async def _optimization_loop(self):
        """Background optimization loop"""
        self.logger.info("Starting optimization loop...")
        
        optimizer = self.components['optimizer']
        
        # Initial training if no model exists
        try:
            await optimizer.run_optimization()
            self.logger.info("✅ Initial optimization completed")
        except Exception as e:
            self.logger.warning(f"Initial optimization failed: {e}")
        
        # Periodic optimization
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(3600 * 24)  # Daily optimization
                
                if not self.shutdown_event.is_set():
                    self.logger.info("Running scheduled optimization...")
                    await optimizer.run_optimization()
                    
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _health_monitor(self):
        """Monitor system health"""
        while not self.shutdown_event.is_set():
            try:
                # Check GCS connection
                if self.components.get('storage'):
                    try:
                        models = self.components['storage'].list_models(prefix="test/")
                        self.logger.debug("✅ GCS connection healthy")
                    except Exception as e:
                        self.logger.warning(f"GCS connection issue: {e}")
                
                # Check market data
                if self.components.get('market_data'):
                    try:
                        # Test with a simple stock
                        info = await self.components['market_data'].get_stock_info('AAPL')
                        if info:
                            self.logger.debug("✅ Market data connection healthy")
                    except Exception as e:
                        self.logger.warning(f"Market data issue: {e}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _data_collection_loop(self):
        """Periodic data collection for training"""
        self.logger.info("Starting data collection loop...")
        
        while not self.shutdown_event.is_set():
            try:
                # Collect fresh training data daily
                await asyncio.sleep(3600 * 24)  # Daily
                
                if self.components.get('optimizer'):
                    self.logger.info("Collecting fresh training data...")
                    data = await self.components['optimizer'].collect_training_data(days_back=30)
                    self.logger.info(f"Collected {len(data)} rows of training data")
                
            except Exception as e:
                self.logger.error(f"Data collection error: {e}")
                await asyncio.sleep(3600)
    
    async def run_flask_app(self):
        """Run Flask app with Hypercorn (async ASGI server)"""
        if 'app' not in self.components:
            self.logger.error("Flask app not initialized")
            return
        
        # Convert Flask app to ASGI
        from asgiref.wsgi import WsgiToAsgi
        asgi_app = WsgiToAsgi(self.components['app'])
        
        # Run with Hypercorn
        from hypercorn.asyncio import serve
        from hypercorn.config import Config as HypercornConfig
        
        config = HypercornConfig()
        config.bind = [f"0.0.0.0:{os.getenv('PORT', 5000)}"]
        config.accesslog = "logs/access.log"
        config.errorlog = "logs/error.log"
        
        self.logger.info(f"🌐 Starting web server on {config.bind[0]}")
        
        await serve(asgi_app, config, shutdown_trigger=self.shutdown_event.wait)
    
    async def start(self):
        """Start the complete system"""
        self.logger.info("🚀 Starting ML Trading System...")
        
        # Initialize all components
        await self.initialize_components()
        
        # Start background services
        await self.start_background_services()
        
        # Start web server
        await self.run_flask_app()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("🛑 Shutting down ML Trading System...")
        self.shutdown_event.set()
        
        # Close components
        if self.components.get('market_data'):
            asyncio.create_task(self.components['market_data'].close())
        
        if self.components.get('monitoring'):
            from project.monitoring import stop_monitoring
            stop_monitoring()

# Signal handlers for graceful shutdown
system_manager = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    if system_manager:
        system_manager.shutdown()

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point"""
    global system_manager
    
    print("🚀 ML Trading System Starting...")
    print("=" * 50)
    
    # Setup
    setup_logging()
    setup_signal_handlers()
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Create and start system
    system_manager = SystemManager()
    
    try:
        await system_manager.start()
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
    except Exception as e:
        logging.error(f"❌ System startup failed: {e}")
        sys.exit(1)
    finally:
        if system_manager:
            system_manager.shutdown()

def run_development():
    """Run in development mode with simpler setup"""
    print("🔧 Running in development mode...")
    
    # Simple Flask development server
    from project.server import app
    
    if os.getenv('BREAKOUT_SYSTEM_ENABLED', 'true').lower() == 'true':
        from project.server_integration import setup_breakout_system
        setup_breakout_system(app)
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=True
    )

if __name__ == "__main__":
    # Check if we're in development mode
    if os.getenv('FLASK_ENV') == 'development' and '--simple' in sys.argv:
        run_development()
    else:
        # Full async production mode
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
