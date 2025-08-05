# project/server_integration.py
"""
Integration module that combines all components into a working Flask app
"""

import os
import logging
from flask import Flask
from datetime import datetime

logger = logging.getLogger(__name__)

def create_integrated_app():
    """Create fully integrated Flask application"""
    
    # Import config first
    from .config import Config
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize extensions
    _setup_extensions(app)
    
    # Register main API
    _register_main_api(app)
    
    # Register breakout system if enabled
    if os.getenv('BREAKOUT_SYSTEM_ENABLED', 'true').lower() == 'true':
        _register_breakout_system(app)
    
    # Setup monitoring
    _setup_monitoring(app)
    
    # Setup error handlers
    _setup_error_handlers(app)
    
    logger.info("✅ Integrated Flask application created")
    return app

def _setup_extensions(app):
    """Setup Flask extensions"""
    from flask_jwt_extended import JWTManager
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    from flask_cors import CORS
    
    # JWT
    jwt = JWTManager(app)
    
    # Rate limiting
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        storage_uri=app.config.get('RATELIMIT_STORAGE_URL', 'memory://'),
        default_limits=["200/hour"]
    )
    
    # CORS
    CORS(app, origins=os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(','))
    
    # Store extensions for access
    app.jwt = jwt
    app.limiter = limiter

def _register_main_api(app):
    """Register main trading API"""
    try:
        # Import and register main server routes
        from .server import (
            login, logout, refresh, train_model, get_task_status, 
            cancel_task, list_models, get_model_metadata, delete_model,
            health_check, system_info
        )
        
        # These routes are already decorated and will be registered automatically
        # when the module is imported
        
        logger.info("✅ Main API routes registered")
        
    except ImportError as e:
        logger.error(f"Failed to import main API: {e}")

def _register_breakout_system(app):
    """Register breakout prediction system"""
    try:
        from .working_api import breakout_api
        app.register_blueprint(breakout_api)
        
        # Update user permissions for breakout features
        _update_user_permissions()
        
        logger.info("✅ Breakout system registered")
        
    except ImportError as e:
        logger.warning(f"Breakout system not available: {e}")
        logger.info("System will run without breakout features")

def _update_user_permissions():
    """Update user permissions to include breakout features"""
    from .config import Config
    
    breakout_permissions = [
        'analyze_stocks',
        'screen_market', 
        'predict_breakouts',
        'view_candidates'
    ]
    
    admin_permissions = breakout_permissions + ['manage_optimization']
    
    # Update default users
    for user_data in Config.DEFAULT_USERS.values():
        if user_data.get('role') == 'admin':
            user_data.setdefault('permissions', []).extend(admin_permissions)
        elif user_data.get('role') == 'trader':
            user_data.setdefault('permissions', []).extend(breakout_permissions)

def _setup_monitoring(app):
    """Setup monitoring and metrics"""
    try:
        from .monitoring import setup_flask_monitoring
        setup_flask_monitoring(app)
        
        logger.info("✅ Monitoring setup complete")
        
    except ImportError as e:
        logger.warning(f"Monitoring not available: {e}")

def _setup_error_handlers(app):
    """Setup global error handlers"""
    from flask import jsonify
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.error(f"Unhandled exception: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

def setup_breakout_system(app):
    """Legacy function for backwards compatibility"""
    _register_breakout_system(app)

# Health check endpoint
def create_health_endpoint(app):
    """Create comprehensive health check endpoint"""
    
    @app.route('/api/health/comprehensive', methods=['GET'])
    def comprehensive_health():
        from flask import jsonify
        
        health = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2025.1.0',
            'components': {}
        }
        
        # Check storage
        try:
            from .storage import get_gcs_storage
            gcs = get_gcs_storage()
            health['components']['storage'] = {
                'status': 'healthy' if gcs.client else 'unhealthy',
                'type': 'GCS'
            }
        except Exception as e:
            health['components']['storage'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Check breakout system
        if os.getenv('BREAKOUT_SYSTEM_ENABLED', 'true').lower() == 'true':
            try:
                health['components']['breakout_system'] = {
                    'status': 'enabled',
                    'features': ['analysis', 'screening', 'prediction']
                }
            except:
                health['components']['breakout_system'] = {
                    'status': 'error'
                }
        
        # Overall status
        component_statuses = [c.get('status') for c in health['components'].values()]
        if 'unhealthy' in component_statuses or 'error' in component_statuses:
            health['status'] = 'degraded'
        
        status_code = 200 if health['status'] != 'unhealthy' else 503
        return jsonify(health), status_code

# Quick test endpoint
def create_test_endpoints(app):
    """Create test endpoints for validation"""
    
    @app.route('/api/test/ping', methods=['GET'])
    def ping():
        from flask import jsonify
        return jsonify({
            'message': 'pong',
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'ML Trading System'
        })
    
    @app.route('/api/test/config', methods=['GET'])
    def test_config():
        from flask import jsonify
        from .config import Config
        
        config_info = {
            'breakout_enabled': os.getenv('BREAKOUT_SYSTEM_ENABLED', 'false').lower() == 'true',
            'data_providers': [
                provider for provider in ['polygon', 'alpha_vantage', 'iex', 'tiingo']
                if os.getenv(f'{provider.upper()}_API_KEY')
            ],
            'storage_configured': bool(os.getenv('GCS_BUCKET_NAME')),
            'flask_env': os.getenv('FLASK_ENV', 'production')
        }
        
        return jsonify(config_info)

def create_production_app():
    """Create production-ready application"""
    app = create_integrated_app()
    
    # Add production-specific endpoints
    create_health_endpoint(app)
    create_test_endpoints(app)
    
    return app

if __name__ == "__main__":
    # Test the integration
    app = create_production_app()
    app.run(debug=True)
