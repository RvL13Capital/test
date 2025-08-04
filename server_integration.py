# project/server_breakout_integration.py
"""
Integration of the breakout system into the main Flask server
Add this to your existing server.py file
"""

# Add to imports section of server.py:
from .breakout_api import breakout_bp
from .breakout_config import BreakoutConfig
from .auto_optimizer import start_auto_optimization
import asyncio
import threading

# Add after creating Flask app in server.py:

# Register breakout blueprint if enabled
if BreakoutConfig.BREAKOUT_SYSTEM_ENABLED:
    app.register_blueprint(breakout_bp)
    logger.info("Breakout prediction system enabled")
    
    # Update allowed tickers to include breakout universe
    if BreakoutConfig.BREAKOUT_STOCK_UNIVERSE:
        Config.ALLOWED_TICKERS.update(BreakoutConfig.BREAKOUT_STOCK_UNIVERSE)
    
    # Add breakout-specific permissions
    for user in Config.DEFAULT_USERS.values():
        if user.get('role') == 'admin':
            user['permissions'].extend([
                'analyze_stocks',
                'screen_market',
                'predict_breakouts',
                'view_candidates',
                'manage_optimization'
            ])
        elif user.get('role') == 'trader':
            user['permissions'].extend([
                'analyze_stocks',
                'screen_market',
                'predict_breakouts',
                'view_candidates'
            ])

# Add to the end of server.py, before if __name__ == '__main__':

def start_background_services():
    """Start background services for the breakout system"""
    if BreakoutConfig.BREAKOUT_SYSTEM_ENABLED:
        # Start auto-optimization in background thread
        def run_auto_optimizer():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            
            try:
                logger.info("Starting auto-optimization service...")
                loop.run_until_complete(start_auto_optimization())
            except Exception as e:
                logger.error(f"Auto-optimization service failed: {e}")
            finally:
                loop.close()
        
        optimizer_thread = threading.Thread(
            target=run_auto_optimizer,
            daemon=True,
            name="AutoOptimizer"
        )
        optimizer_thread.start()
        logger.info("Background services started")

# Add new health check endpoint that includes breakout system status
@app.route('/api/health/detailed', methods=['GET'])
@limiter.limit("60 per minute")
def detailed_health_check():
    """Enhanced health check including breakout system"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2025.1.0',
            'services': {}
        }
        
        # Check Celery
        try:
            celery_inspect = celery.control.inspect()
            active_tasks = celery_inspect.active()
            health_status['services']['celery'] = {
                'status': 'healthy' if active_tasks is not None else 'unhealthy',
                'active_tasks': len(active_tasks) if active_tasks else 0
            }
        except:
            health_status['services']['celery'] = {'status': 'unhealthy'}
        
        # Check GCS
        try:
            gcs = get_gcs_storage()
            health_status['services']['gcs'] = {
                'status': 'healthy' if gcs.client else 'unhealthy'
            }
        except:
            health_status['services']['gcs'] = {'status': 'unhealthy'}
        
        # Check Breakout System
        if BreakoutConfig.BREAKOUT_SYSTEM_ENABLED:
            try:
                from .auto_optimizer import get_auto_optimizer
                optimizer = get_auto_optimizer()
                optimizer_status = optimizer.get_system_status()
                
                health_status['services']['breakout_system'] = {
                    'status': optimizer_status.get('status', 'unknown'),
                    'current_model': optimizer_status.get('current_model'),
                    'last_optimization': optimizer_status.get('last_optimization'),
                    'performance': optimizer_status.get('recent_performance')
                }
            except Exception as e:
                health_status['services']['breakout_system'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        # Determine overall status
        service_statuses = [s.get('status') for s in health_status['services'].values()]
        if 'unhealthy' in service_statuses:
            health_status['status'] = 'degraded'
        
        status_code = 200 if health_status['status'] in ['healthy', 'degraded'] else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': 'Health check failed'
        }), 503

# Add to system info endpoint
@app.route('/api/system/info', methods=['GET'])
@jwt_required()
@limiter.limit("10 per minute")
def enhanced_system_info():
    """Get enhanced system information including breakout config"""
    try:
        current_user = get_jwt_identity()
        user_claims = get_jwt()
        
        if 'admin' not in user_claims.get('permissions', []):
            return jsonify({'error': 'Admin access required'}), 403
        
        info = {
            'version': '2025.1.0',
            'environment': Config.FLASK_ENV,
            'config': {
                'max_dataframe_size': Config.MAX_DATAFRAME_SIZE,
                'allowed_tickers': list(Config.ALLOWED_TICKERS),
                'training_rate_limit': Config.TRAINING_RATE_LIMIT,
                'optuna_n_trials': Config.OPTUNA_N_TRIALS,
                'optuna_timeout': Config.OPTUNA_TIMEOUT
            },
            'security': {
                'jwt_expires': Config.JWT_ACCESS_TOKEN_EXPIRES,
                'encryption_enabled': Config.MODEL_ENCRYPTION_KEY is not None,
                'audit_log_enabled': Config.AUDIT_LOG_ENABLED
            }
        }
        
        # Add breakout system info if enabled
        if BreakoutConfig.BREAKOUT_SYSTEM_ENABLED:
            info['breakout_system'] = {
                'enabled': True,
                'market_cap_ranges': {
                    'nano_cap_max': BreakoutConfig.NANO_CAP_MAX,
                    'micro_cap_max': BreakoutConfig.MICRO_CAP_MAX,
                    'small_cap_max': BreakoutConfig.SMALL_CAP_MAX
                },
                'consolidation_params': {
                    'min_days': BreakoutConfig.MIN_CONSOLIDATION_DAYS,
                    'max_days': BreakoutConfig.MAX_CONSOLIDATION_DAYS,
                    'price_tolerance': BreakoutConfig.CONSOLIDATION_PRICE_TOLERANCE
                },
                'trading_params': {
                    'max_positions': BreakoutConfig.MAX_POSITIONS,
                    'position_size': BreakoutConfig.POSITION_SIZE_PCT,
                    'stop_loss': BreakoutConfig.STOP_LOSS_PCT
                },
                'optimization': {
                    'interval_hours': BreakoutConfig.OPTIMIZATION_INTERVAL_HOURS,
                    'performance_threshold': BreakoutConfig.PERFORMANCE_THRESHOLD
                }
            }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"System info request failed: {e}")
        return jsonify({'error': 'Failed to get system info'}), 500

# Initialize background services when running directly
if __name__ == '__main__':
    # Start background services
    start_background_services()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=Config.FLASK_ENV == 'development'
    )

# For production with Gunicorn, add to your gunicorn_config.py:
def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Server is ready")
    
    # Import here to avoid circular imports
    from project.server import start_background_services
    
    # Start background services
    if os.getenv('BREAKOUT_SYSTEM_ENABLED', 'true').lower() == 'true':
        start_background_services()
        server.log.info("Breakout system background services started")