# project/server.py
import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, g
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, 
    get_jwt_identity, get_jwt
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
from marshmallow import Schema, fields, validate, ValidationError
import pandas as pd

# Import project modules
from .config import Config, get_default_hparams, estimate_training_time
from .tasks import tune_and_train_async, train_lstm_async, train_xgboost_async
from .storage import get_gcs_storage
from .extensions import celery

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
jwt = JWTManager(app)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    storage_uri=Config.RATELIMIT_STORAGE_URL,
    default_limits=[Config.API_RATE_LIMIT]
)

# Configure CORS with security
CORS(app, 
     origins=os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(','),
     supports_credentials=True)

# Input validation schemas
class LoginSchema(Schema):
    username = fields.Str(required=True, validate=validate.Length(min=3, max=50))
    password = fields.Str(required=True, validate=validate.Length(min=6, max=100))

class TrainingRequestSchema(Schema):
    ticker = fields.Str(
        required=True,
        validate=validate.OneOf(list(Config.ALLOWED_TICKERS)),
        error_messages={'validator_failed': 'Invalid ticker symbol'}
    )
    model_type = fields.Str(
        required=True,
        validate=validate.OneOf(['lstm', 'xgboost']),
        error_messages={'validator_failed': 'Model type must be lstm or xgboost'}
    )
    data = fields.Raw(required=True)
    tune_hyperparams = fields.Bool(missing=False)
    custom_hyperparams = fields.Dict(missing=None)

class TaskStatusSchema(Schema):
    task_id = fields.Str(required=True, validate=validate.Length(min=10, max=100))

# Initialize schemas
login_schema = LoginSchema()
training_schema = TrainingRequestSchema()
task_status_schema = TaskStatusSchema()

# Security middleware
@app.before_request
def security_middleware():
    """Apply security headers and validations"""
    # Content length check
    if request.content_length and request.content_length > Config.MAX_REQUEST_SIZE:
        return jsonify({'error': 'Request too large'}), 413
    
    # Set security headers
    g.security_headers = Config.SECURITY_HEADERS.copy()

@app.after_request
def apply_security_headers(response):
    """Apply security headers to all responses"""
    headers = getattr(g, 'security_headers', {})
    for header, value in headers.items():
        response.headers[header] = value
    
    # Add timestamp for debugging
    response.headers['X-Response-Time'] = datetime.utcnow().isoformat()
    return response

# JWT token blacklist (in production, use Redis or database)
blacklisted_tokens = set()

@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    """Check if JWT token is revoked"""
    return jwt_payload['jti'] in blacklisted_tokens

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    """Handle expired tokens"""
    return jsonify({'error': 'Token has expired'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    """Handle invalid tokens"""
    return jsonify({'error': 'Invalid token'}), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    """Handle missing tokens"""
    return jsonify({'error': 'Authentication token required'}), 401

# User management (simplified for demo - use proper database in production)
def verify_user(username: str, password: str) -> bool:
    """Verify user credentials"""
    users = Config.DEFAULT_USERS
    
    if username not in users:
        return False
    
    # In production, use proper password hashing
    # For demo, we'll just check if password is "password123"
    return password == "password123"

def get_user_info(username: str) -> dict:
    """Get user information"""
    users = Config.DEFAULT_USERS
    return users.get(username, {})

# Authentication endpoints
@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    """Secure authentication endpoint"""
    try:
        data = login_schema.load(request.get_json() or {})
        username = data['username']
        password = data['password']
        
        if verify_user(username, password):
            user_info = get_user_info(username)
            
            access_token = create_access_token(
                identity=username,
                additional_claims={
                    'role': user_info.get('role', 'user'),
                    'permissions': user_info.get('permissions', [])
                },
                expires_delta=timedelta(seconds=Config.JWT_ACCESS_TOKEN_EXPIRES)
            )
            
            # Audit log
            logger.info(f"AUDIT: Successful login for user {username}")
            
            return jsonify({
                'access_token': access_token,
                'user': {
                    'username': username,
                    'role': user_info.get('role', 'user'),
                    'permissions': user_info.get('permissions', [])
                },
                'expires_in': Config.JWT_ACCESS_TOKEN_EXPIRES
            })
        
        # Audit log for failed login
        logger.warning(f"AUDIT: Failed login attempt for user {username}")
        return jsonify({'error': 'Invalid credentials'}), 401
        
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/auth/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout endpoint - blacklist token"""
    try:
        jti = get_jwt()['jti']
        blacklisted_tokens.add(jti)
        
        username = get_jwt_identity()
        logger.info(f"AUDIT: User {username} logged out")
        
        return jsonify({'message': 'Successfully logged out'})
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'error': 'Logout failed'}), 500

@app.route('/api/auth/refresh', methods=['POST'])
@jwt_required()
def refresh():
    """Refresh access token"""
    try:
        current_user = get_jwt_identity()
        user_info = get_user_info(current_user)
        
        new_token = create_access_token(
            identity=current_user,
            additional_claims={
                'role': user_info.get('role', 'user'),
                'permissions': user_info.get('permissions', [])
            }
        )
        
        return jsonify({'access_token': new_token})
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        return jsonify({'error': 'Token refresh failed'}), 500

# Training endpoints
@app.route('/api/train', methods=['POST'])
@jwt_required()
@limiter.limit(Config.TRAINING_RATE_LIMIT)
def train_model():
    """Secure model training endpoint"""
    try:
        # Get user info
        current_user = get_jwt_identity()
        user_claims = get_jwt()
        user_permissions = user_claims.get('permissions', [])
        
        # Validate input
        data = training_schema.load(request.get_json() or {})
        
        # Check permissions
        required_permission = 'tune_hyperparams' if data['tune_hyperparams'] else 'train_model'
        
        if required_permission not in user_permissions:
            logger.warning(f"AUDIT: Permission denied for user {current_user}, operation: {required_permission}")
            return jsonify({'error': f'Permission denied: {required_permission} required'}), 403
        
        # Validate DataFrame
        try:
            df = pd.DataFrame(data['data'])
        except Exception as e:
            return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
        
        if df.empty:
            return jsonify({'error': 'DataFrame is empty'}), 400
        
        if len(df) > Config.MAX_DATAFRAME_SIZE:
            return jsonify({'error': f'DataFrame too large: {len(df)} rows > {Config.MAX_DATAFRAME_SIZE}'}), 400
        
        if len(df) < Config.MIN_DATAFRAME_SIZE:
            return jsonify({'error': f'DataFrame too small: {len(df)} rows < {Config.MIN_DATAFRAME_SIZE}'}), 400
        
        # Sanitize inputs
        ticker = data['ticker'].upper().strip()
        model_type = data['model_type'].lower().strip()
        
        # Create task data
        df_json = df.to_json(orient='split')
        
        # Handle hyperparameters
        if data['custom_hyperparams']:
            try:
                Config.validate_hyperparams(model_type, data['custom_hyperparams'])
                hparams = data['custom_hyperparams']
            except ValueError as e:
                return jsonify({'error': f'Invalid hyperparameters: {str(e)}'}), 400
        else:
            hparams = get_default_hparams(model_type)
        
        # Submit appropriate task
        if data['tune_hyperparams']:
            task = tune_and_train_async.delay(
                df_json=df_json,
                model_type=model_type,
                ticker=ticker,
                user_id=current_user
            )
        else:
            if model_type == 'lstm':
                task = train_lstm_async.delay(
                    df_json=df_json,
                    hparams=hparams,
                    ticker=ticker,
                    user_id=current_user
                )
            else:  # xgboost
                task = train_xgboost_async.delay(
                    df_json=df_json,
                    hparams=hparams,
                    ticker=ticker,
                    user_id=current_user
                )
        
        # Audit log
        logger.info(f"AUDIT: Training task submitted by {current_user} - "
                   f"ticker: {ticker}, model: {model_type}, tune: {data['tune_hyperparams']}")
        
        response_data = {
            'task_id': task.id,
            'status': 'submitted',
            'ticker': ticker,
            'model_type': model_type,
            'tune_hyperparams': data['tune_hyperparams'],
            'estimated_time_seconds': estimate_training_time(model_type, data['tune_hyperparams']),
            'user_id': current_user
        }
        
        return jsonify(response_data), 202
        
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    except Exception as e:
        logger.error(f"Training request failed for user {get_jwt_identity()}: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/tasks/<task_id>/status', methods=['GET'])
@jwt_required()
@limiter.limit("30 per minute")
def get_task_status(task_id: str):
    """Get training task status"""
    try:
        current_user = get_jwt_identity()
        
        # Validate task ID format (basic validation)
        if not task_id or len(task_id) < 10:
            return jsonify({'error': 'Invalid task ID'}), 400
        
        # Get task result
        task = celery.AsyncResult(task_id)
        
        response_data = {
            'task_id': task_id,
            'status': task.status,
            'user_id': current_user
        }
        
        if task.status == 'PENDING':
            response_data['message'] = 'Task is waiting to be processed'
        elif task.status == 'PROGRESS':
            response_data['current'] = task.info.get('progress', 0)
            response_data['message'] = task.info.get('status', 'Processing...')
        elif task.status == 'SUCCESS':
            response_data.update(task.result)
        elif task.status == 'FAILURE':
            response_data['error'] = str(task.info)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Task status check failed: {e}")
        return jsonify({'error': 'Failed to get task status'}), 500

@app.route('/api/tasks/<task_id>/cancel', methods=['POST'])
@jwt_required()
@limiter.limit("10 per minute")
def cancel_task(task_id: str):
    """Cancel a running task"""
    try:
        current_user = get_jwt_identity()
        user_claims = get_jwt()
        
        # Only admins or task owners can cancel tasks
        if 'admin' not in user_claims.get('permissions', []):
            # In production, check if user owns the task
            pass
        
        task = celery.AsyncResult(task_id)
        task.revoke(terminate=True)
        
        logger.info(f"AUDIT: Task {task_id} cancelled by user {current_user}")
        
        return jsonify({
            'message': 'Task cancelled successfully',
            'task_id': task_id
        })
        
    except Exception as e:
        logger.error(f"Task cancellation failed: {e}")
        return jsonify({'error': 'Failed to cancel task'}), 500

# Model management endpoints
@app.route('/api/models', methods=['GET'])
@jwt_required()
@limiter.limit("20 per minute")
def list_models():
    """List available models for the user"""
    try:
        current_user = get_jwt_identity()
        user_claims = get_jwt()
        
        # Get query parameters
        ticker = request.args.get('ticker')
        model_type = request.args.get('model_type')
        
        # Build prefix for filtering
        prefix = "models/"
        if ticker:
            if ticker.upper() not in Config.ALLOWED_TICKERS:
                return jsonify({'error': 'Invalid ticker'}), 400
            prefix += f"{ticker.upper()}/"
        
        # Get storage client
        gcs = get_gcs_storage()
        if not gcs.client:
            return jsonify({'error': 'Storage service unavailable'}), 503
        
        # List models
        models = gcs.list_models(prefix=prefix, user_id=current_user)
        
        # Filter by model type if specified
        if model_type:
            if model_type not in ['lstm', 'xgboost']:
                return jsonify({'error': 'Invalid model type'}), 400
            models = [m for m in models if model_type in m['name']]
        
        # Filter by user permissions (non-admins see only their models)
        if 'view_all_models' not in user_claims.get('permissions', []):
            models = [m for m in models 
                     if m.get('metadata', {}).get('user_id') == current_user]
        
        return jsonify({
            'models': models,
            'count': len(models),
            'user_id': current_user
        })
        
    except Exception as e:
        logger.error(f"Model listing failed: {e}")
        return jsonify({'error': 'Failed to list models'}), 500

@app.route('/api/models/<path:model_path>/metadata', methods=['GET'])
@jwt_required()
@limiter.limit("20 per minute")
def get_model_metadata(model_path: str):
    """Get model metadata"""
    try:
        current_user = get_jwt_identity()
        
        # Basic path validation
        if not model_path.startswith('models/'):
            return jsonify({'error': 'Invalid model path'}), 400
        
        gcs = get_gcs_storage()
        if not gcs.client:
            return jsonify({'error': 'Storage service unavailable'}), 503
        
        metadata = gcs.get_model_metadata(model_path, user_id=current_user)
        
        return jsonify({
            'metadata': metadata,
            'user_id': current_user
        })
        
    except Exception as e:
        logger.error(f"Model metadata retrieval failed: {e}")
        return jsonify({'error': 'Failed to get model metadata'}), 500

@app.route('/api/models/<path:model_path>', methods=['DELETE'])
@jwt_required()
@limiter.limit("5 per minute")
def delete_model(model_path: str):
    """Delete a model (admin only or model owner)"""
    try:
        current_user = get_jwt_identity()
        user_claims = get_jwt()
        
        # Basic path validation
        if not model_path.startswith('models/'):
            return jsonify({'error': 'Invalid model path'}), 400
        
        gcs = get_gcs_storage()
        if not gcs.client:
            return jsonify({'error': 'Storage service unavailable'}), 503
        
        # Check permissions
        if 'admin' not in user_claims.get('permissions', []):
            # Check if user owns the model
            try:
                metadata = gcs.get_model_metadata(model_path, user_id=current_user)
                if metadata.get('metadata', {}).get('user_id') != current_user:
                    return jsonify({'error': 'Permission denied'}), 403
            except:
                return jsonify({'error': 'Permission denied'}), 403
        
        # Delete the model
        gcs.delete_model(model_path, user_id=current_user)
        
        logger.info(f"AUDIT: Model {model_path} deleted by user {current_user}")
        
        return jsonify({
            'message': 'Model deleted successfully',
            'model_path': model_path
        })
        
    except Exception as e:
        logger.error(f"Model deletion failed: {e}")
        return jsonify({'error': 'Failed to delete model'}), 500

# Health and system endpoints
@app.route('/api/health', methods=['GET'])
@limiter.limit("60 per minute")
def health_check():
    """System health check"""
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

@app.route('/api/system/info', methods=['GET'])
@jwt_required()
@limiter.limit("10 per minute")
def system_info():
    """Get system information (admin only)"""
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
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"System info request failed: {e}")
        return jsonify({'error': 'Failed to get system info'}), 500

# Error handlers
@app.errorhandler(ValidationError)
def handle_validation_error(e):
    """Handle Marshmallow validation errors"""
    return jsonify({'error': 'Validation failed', 'details': e.messages}), 400

@app.errorhandler(429)
def handle_rate_limit_exceeded(e):
    """Handle rate limit exceeded"""
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please try again later.'
    }), 429

@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

# Request logging middleware
@app.before_request
def log_request_info():
    """Log request information for audit"""
    if Config.AUDIT_LOG_ENABLED and request.endpoint:
        logger.info(f"REQUEST: {request.method} {request.path} from {request.remote_addr}")

if __name__ == '__main__':
    # Development server (use gunicorn in production)
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=Config.FLASK_ENV == 'development'
    )