# project/server_integration.py
"""
Enhanced Flask Application Integration Module
Provides comprehensive component integration with advanced features
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from pathlib import Path
import json

from flask import Flask, Blueprint, jsonify, request, g
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from flask_compress import Compress
from flask_caching import Cache
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

logger = logging.getLogger(__name__)


class IntegrationConfig:
    """Configuration manager for integration"""
    
    def __init__(self):
        self.breakout_enabled = os.getenv('BREAKOUT_SYSTEM_ENABLED', 'true').lower() == 'true'
        self.monitoring_enabled = os.getenv('MONITORING_ENABLED', 'true').lower() == 'true'
        self.cache_enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        self.compression_enabled = os.getenv('COMPRESSION_ENABLED', 'true').lower() == 'true'
        self.rate_limiting_enabled = os.getenv('RATE_LIMITING_ENABLED', 'true').lower() == 'true'
        
        # Performance settings
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))
        self.max_content_length = int(os.getenv('MAX_CONTENT_LENGTH', '16')) * 1024 * 1024  # MB
        
        # Security settings
        self.enforce_https = os.getenv('ENFORCE_HTTPS', 'false').lower() == 'true'
        self.trusted_proxies = int(os.getenv('TRUSTED_PROXIES', '1'))


class PerformanceMonitor:
    """Monitor request performance"""
    
    def __init__(self):
        self.metrics = {
            'requests': 0,
            'errors': 0,
            'total_time': 0,
            'endpoint_metrics': {}
        }
    
    def record_request(self, endpoint: str, duration: float, status_code: int):
        """Record request metrics"""
        self.metrics['requests'] += 1
        self.metrics['total_time'] += duration
        
        if status_code >= 400:
            self.metrics['errors'] += 1
        
        if endpoint not in self.metrics['endpoint_metrics']:
            self.metrics['endpoint_metrics'][endpoint] = {
                'count': 0,
                'total_time': 0,
                'errors': 0
            }
        
        endpoint_metric = self.metrics['endpoint_metrics'][endpoint]
        endpoint_metric['count'] += 1
        endpoint_metric['total_time'] += duration
        if status_code >= 400:
            endpoint_metric['errors'] += 1
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        avg_time = self.metrics['total_time'] / max(self.metrics['requests'], 1)
        
        return {
            'total_requests': self.metrics['requests'],
            'total_errors': self.metrics['errors'],
            'average_response_time': round(avg_time * 1000, 2),  # ms
            'error_rate': round(self.metrics['errors'] / max(self.metrics['requests'], 1) * 100, 2),
            'endpoints': {
                endpoint: {
                    'requests': data['count'],
                    'avg_time': round(data['total_time'] / max(data['count'], 1) * 1000, 2),
                    'error_rate': round(data['errors'] / max(data['count'], 1) * 100, 2)
                }
                for endpoint, data in self.metrics['endpoint_metrics'].items()
            }
        }


def create_integrated_app(config: Optional[IntegrationConfig] = None) -> Flask:
    """Create fully integrated Flask application with all features"""
    
    if config is None:
        config = IntegrationConfig()
    
    # Import Flask config
    from .config import Config
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Apply additional configuration
    app.config['MAX_CONTENT_LENGTH'] = config.max_content_length
    app.config['REQUEST_TIMEOUT'] = config.request_timeout
    
    # Setup proxy support
    if config.trusted_proxies > 0:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=config.trusted_proxies)
    
    # Initialize performance monitor
    app.performance_monitor = PerformanceMonitor()
    
    # Setup extensions
    _setup_extensions(app, config)
    
    # Setup middleware
    _setup_middleware(app, config)
    
    # Register error handlers
    _setup_error_handlers(app)
    
    # Register main API
    _register_main_api(app)
    
    # Register optional components
    if config.breakout_enabled:
        _register_breakout_system(app)
    
    if config.monitoring_enabled:
        _setup_monitoring(app)
    
    # Register health and utility endpoints
    _register_utility_endpoints(app)
    
    # Setup request hooks
    _setup_request_hooks(app, config)
    
    logger.info("✅ Integrated Flask application created")
    logger.info(f"Features enabled: Breakout={config.breakout_enabled}, "
                f"Monitoring={config.monitoring_enabled}, "
                f"Cache={config.cache_enabled}, "
                f"Compression={config.compression_enabled}")
    
    return app


def _setup_extensions(app: Flask, config: IntegrationConfig):
    """Setup Flask extensions with configuration"""
    
    # JWT authentication
    jwt = JWTManager(app)
    app.jwt = jwt
    
    # Configure JWT callbacks
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({'error': 'Token has expired'}), 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({'error': 'Invalid token'}), 401
    
    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({'error': 'Authorization required'}), 401
    
    # Rate limiting
    if config.rate_limiting_enabled:
        storage_uri = app.config.get('RATELIMIT_STORAGE_URL', 'memory://')
        
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            storage_uri=storage_uri,
            default_limits=["200/hour", "50/minute"],
            headers_enabled=True,
            swallow_errors=True,
            in_memory_fallback_enabled=True
        )
        app.limiter = limiter
        
        # Configure rate limit error handler
        @app.errorhandler(429)
        def ratelimit_handler(e):
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': str(e.description)
            }), 429
    
    # CORS
    allowed_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')
    CORS(app, 
         origins=allowed_origins,
         supports_credentials=True,
         expose_headers=['X-Total-Count', 'X-Page', 'X-Per-Page'])
    
    # Compression
    if config.compression_enabled:
        Compress(app)
        app.config['COMPRESS_MIMETYPES'] = [
            'text/html', 'text/css', 'text/xml', 'text/javascript',
            'application/json', 'application/javascript'
        ]
    
    # Caching
    if config.cache_enabled:
        cache_config = {
            'CACHE_TYPE': os.getenv('CACHE_TYPE', 'simple'),
            'CACHE_DEFAULT_TIMEOUT': int(os.getenv('CACHE_TIMEOUT', '300'))
        }
        
        if cache_config['CACHE_TYPE'] == 'redis':
            cache_config['CACHE_REDIS_URL'] = os.getenv('CACHE_REDIS_URL', 
                                                        os.getenv('CELERY_BROKER_URL'))
        
        cache = Cache(app, config=cache_config)
        app.cache = cache
    
    logger.info("✅ Flask extensions configured")


def _setup_middleware(app: Flask, config: IntegrationConfig):
    """Setup custom middleware"""
    
    @app.before_request
    def enforce_https():
        """Enforce HTTPS in production"""
        if config.enforce_https and not request.is_secure:
            return jsonify({'error': 'HTTPS required'}), 403
    
    @app.before_request
    def check_content_type():
        """Validate content type for POST/PUT requests"""
        if request.method in ['POST', 'PUT'] and request.content_length:
            content_type = request.headers.get('Content-Type', '')
            if not content_type.startswith(('application/json', 'multipart/form-data')):
                return jsonify({'error': 'Invalid content type'}), 415
    
    @app.after_request
    def add_security_headers(response):
        """Add security headers to all responses"""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        if config.enforce_https:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response


def _setup_request_hooks(app: Flask, config: IntegrationConfig):
    """Setup request lifecycle hooks"""
    
    @app.before_request
    def before_request():
        """Pre-request setup"""
        g.start_time = time.time()
        g.request_id = request.headers.get('X-Request-ID', 
                                         f"{int(time.time())}-{os.urandom(4).hex()}")
    
    @app.after_request
    def after_request(response):
        """Post-request processing"""
        # Add request ID to response
        response.headers['X-Request-ID'] = g.get('request_id', 'unknown')
        
        # Record performance metrics
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            endpoint = request.endpoint or 'unknown'
            app.performance_monitor.record_request(
                endpoint,
                duration,
                response.status_code
            )
            
            # Add timing header
            response.headers['X-Response-Time'] = f"{duration * 1000:.2f}ms"
        
        return response
    
    @app.teardown_request
    def teardown_request(exception):
        """Cleanup after request"""
        if exception:
            logger.error(f"Request {g.get('request_id')} failed: {exception}")


def _register_main_api(app: Flask):
    """Register main trading API endpoints"""
    try:
        # Import main server module
        from . import server
        
        # The routes are already registered via decorators when the module is imported
        # Just verify they're available
        rules = [rule.endpoint for rule in app.url_map.iter_rules()]
        
        api_endpoints = [
            'login', 'logout', 'refresh', 'train_model', 
            'get_task_status', 'cancel_task', 'list_models',
            'get_model_metadata', 'delete_model', 'health_check'
        ]
        
        registered = sum(1 for endpoint in api_endpoints if endpoint in rules)
        logger.info(f"✅ Main API registered ({registered} endpoints)")
        
    except ImportError as e:
        logger.error(f"Failed to import main API: {e}")
        raise


def _register_breakout_system(app: Flask):
    """Register breakout prediction system"""
    try:
        from .working_api import create_breakout_blueprint
        
        # Create and register blueprint
        breakout_bp = create_breakout_blueprint()
        app.register_blueprint(breakout_bp, url_prefix='/api/breakout')
        
        # Update user permissions
        _update_user_permissions()
        
        # Register breakout-specific error handlers
        @breakout_bp.errorhandler(ValueError)
        def handle_value_error(e):
            return jsonify({'error': str(e)}), 400
        
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
        'view_candidates',
        'manage_alerts'
    ]
    
    admin_permissions = breakout_permissions + [
        'manage_optimization',
        'view_system_metrics',
        'manage_users'
    ]
    
    # Update default users
    for username, user_data in Config.DEFAULT_USERS.items():
        current_permissions = user_data.get('permissions', [])
        
        if user_data.get('role') == 'admin':
            new_permissions = list(set(current_permissions + admin_permissions))
        elif user_data.get('role') == 'trader':
            new_permissions = list(set(current_permissions + breakout_permissions))
        else:
            new_permissions = current_permissions
        
        user_data['permissions'] = new_permissions


def _setup_monitoring(app: Flask):
    """Setup monitoring and metrics collection"""
    try:
        from .monitoring import setup_flask_monitoring, create_metrics_blueprint
        
        # Setup monitoring
        setup_flask_monitoring(app)
        
        # Register metrics blueprint
        metrics_bp = create_metrics_blueprint()
        app.register_blueprint(metrics_bp, url_prefix='/api/metrics')
        
        logger.info("✅ Monitoring setup complete")
        
    except ImportError as e:
        logger.warning(f"Monitoring not available: {e}")


def _register_utility_endpoints(app: Flask):
    """Register utility and operational endpoints"""
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Basic health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': os.getenv('SYSTEM_VERSION', '2025.1.0')
        })
    
    @app.route('/api/health/detailed', methods=['GET'])
    @jwt_required(optional=True)
    def detailed_health():
        """Detailed health check with component status"""
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': os.getenv('SYSTEM_VERSION', '2025.1.0'),
            'environment': os.getenv('FLASK_ENV', 'production'),
            'components': {}
        }
        
        # Check storage
        try:
            from .storage import get_gcs_storage
            storage = get_gcs_storage()
            storage.client.list_buckets(max_results=1)
            health_data['components']['storage'] = {
                'status': 'healthy',
                'type': 'GCS',
                'bucket': os.getenv('GCS_BUCKET_NAME')
            }
        except Exception as e:
            health_data['components']['storage'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_data['status'] = 'degraded'
        
        # Check Redis
        try:
            import redis
            r = redis.from_url(os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'))
            r.ping()
            health_data['components']['redis'] = {
                'status': 'healthy',
                'info': 'Connected'
            }
        except Exception as e:
            health_data['components']['redis'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_data['status'] = 'degraded'
        
        # Check features
        config = IntegrationConfig()
        health_data['features'] = {
            'breakout_system': config.breakout_enabled,
            'monitoring': config.monitoring_enabled,
            'caching': config.cache_enabled,
            'rate_limiting': config.rate_limiting_enabled
        }
        
        # Add performance metrics if available
        if hasattr(app, 'performance_monitor'):
            health_data['performance'] = app.performance_monitor.get_metrics()
        
        status_code = 200 if health_data['status'] == 'healthy' else 503
        return jsonify(health_data), status_code
    
    @app.route('/api/info', methods=['GET'])
    def system_info():
        """System information endpoint"""
        import platform
        import psutil
        
        info = {
            'system': {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            },
            'application': {
                'version': os.getenv('SYSTEM_VERSION', '2025.1.0'),
                'environment': os.getenv('FLASK_ENV', 'production'),
                'debug': app.debug,
                'testing': app.testing
            },
            'configuration': {
                'max_content_length_mb': app.config.get('MAX_CONTENT_LENGTH', 0) / (1024*1024),
                'request_timeout': app.config.get('REQUEST_TIMEOUT', 30)
            }
        }
        
        return jsonify(info)
    
    @app.route('/api/ping', methods=['GET'])
    def ping():
        """Simple ping endpoint for monitoring"""
        return jsonify({
            'pong': True,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    @app.route('/api/ready', methods=['GET'])
    def readiness():
        """Kubernetes readiness probe endpoint"""
        # Check if all critical components are ready
        try:
            # Quick checks for critical components
            from .storage import get_gcs_storage
            storage = get_gcs_storage()
            
            if not storage.client:
                return jsonify({'ready': False, 'reason': 'Storage not initialized'}), 503
            
            return jsonify({'ready': True})
            
        except Exception as e:
            return jsonify({'ready': False, 'reason': str(e)}), 503
    
    @app.route('/api/live', methods=['GET'])
    def liveness():
        """Kubernetes liveness probe endpoint"""
        # Simple check that the app is responsive
        return jsonify({'alive': True})


def _setup_error_handlers(app: Flask):
    """Setup comprehensive error handlers"""
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad request',
            'message': str(error.description) if hasattr(error, 'description') else 'Invalid request'
        }), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({
            'error': 'Unauthorized',
            'message': 'Authentication required'
        }), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({
            'error': 'Forbidden',
            'message': 'Insufficient permissions'
        }), 403
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource was not found'
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            'error': 'Method not allowed',
            'message': f'The {request.method} method is not allowed for this endpoint'
        }), 405
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            'error': 'Request too large',
            'message': f'Maximum content length is {app.config.get("MAX_CONTENT_LENGTH", 0) / (1024*1024):.1f}MB'
        }), 413
    
    @app.errorhandler(415)
    def unsupported_media_type(error):
        return jsonify({
            'error': 'Unsupported media type',
            'message': 'Content-Type must be application/json'
        }), 415
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred',
            'request_id': g.get('request_id', 'unknown')
        }), 500
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        return jsonify({
            'error': e.name,
            'message': e.description,
            'code': e.code
        }), e.code
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        
        # Don't expose internal errors in production
        if app.debug:
            return jsonify({
                'error': 'Internal server error',
                'message': str(e),
                'type': type(e).__name__
            }), 500
        else:
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred',
                'request_id': g.get('request_id', 'unknown')
            }), 500


def create_production_app() -> Flask:
    """Create production-ready application with all optimizations"""
    config = IntegrationConfig()
    
    # Enable production features
    config.enforce_https = True
    config.compression_enabled = True
    config.cache_enabled = True
    config.rate_limiting_enabled = True
    
    app = create_integrated_app(config)
    
    # Additional production configuration
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
    
    # Disable debug features
    app.config['PROPAGATE_EXCEPTIONS'] = False
    app.config['TRAP_HTTP_EXCEPTIONS'] = False
    
    return app


def create_development_app() -> Flask:
    """Create development application with debugging features"""
    config = IntegrationConfig()
    
    # Development features
    config.enforce_https = False
    config.compression_enabled = False
    
    app = create_integrated_app(config)
    
    # Enable debugging
    app.config['DEBUG'] = True
    app.config['TESTING'] = False
    
    # Development routes
    @app.route('/api/debug/routes')
    def debug_routes():
        """List all registered routes"""
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append({
                'endpoint': rule.endpoint,
                'methods': list(rule.methods),
                'path': str(rule)
            })
        return jsonify(routes)
    
    @app.route('/api/debug/config')
    def debug_config():
        """Show non-sensitive configuration"""
        safe_config = {}
        for key, value in app.config.items():
            if 'SECRET' not in key and 'KEY' not in key and 'PASSWORD' not in key:
                safe_config[key] = str(value)
        return jsonify(safe_config)
    
    return app


# Legacy compatibility functions
def setup_breakout_system(app: Flask):
    """Legacy function for backwards compatibility"""
    logger.warning("setup_breakout_system is deprecated. Use create_integrated_app instead.")
    _register_breakout_system(app)


if __name__ == "__main__":
    # Test the integration
    import sys
    
    if '--production' in sys.argv:
        app = create_production_app()
        print("Created production app")
    else:
        app = create_development_app()
        print("Created development app")
    
    # List routes
    print("\nRegistered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.endpoint}: {rule}")
    
    # Run if requested
    if '--run' in sys.argv:
        app.run(debug=True)
