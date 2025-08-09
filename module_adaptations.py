# ======================== ANPASSUNG 1: secure_tasks.py ========================
# Ersetzen Sie die bestehenden Task-Implementierungen durch diese:

# project/tasks_integrated.py
"""
Updated Celery tasks that use the integrated system
Replace the content of secure_tasks.py with this
"""

from celery import Task
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

from .extensions import celery
from .integrated_system import IntegratedMLTradingSystem, IntegratedSystemConfig

logger = logging.getLogger(__name__)

class IntegratedTask(Task):
    """Base task class for integrated system"""
    
    _system = None
    
    @property
    def system(self):
        """Lazy load integrated system"""
        if IntegratedTask._system is None:
            config = IntegratedSystemConfig()
            IntegratedTask._system = IntegratedMLTradingSystem(config)
        return IntegratedTask._system
    
    def _audit_log(self, event: str, user_id: str, details: Dict, success: bool = True):
        """Audit logging"""
        logger.info(f"AUDIT: {event} by {user_id}: {details} - Success: {success}")

@celery.task(bind=True, base=IntegratedTask, name='tasks.train_model')
def train_model_integrated(self, ticker: str, model_type: str, user_id: str,
                          custom_params: Optional[Dict] = None):
    """
    Simplified training task using integrated system
    """
    try:
        self.update_state(state='PROGRESS', meta={
            'status': 'Initializing integrated pipeline...',
            'progress': 5
        })
        
        # Run through integrated system
        result = asyncio.run(
            self.system.run_complete_training_pipeline(ticker, model_type)
        )
        
        self._audit_log('training_completed', user_id, {
            'ticker': ticker,
            'model_type': model_type,
            'model_id': result.get('model_id')
        })
        
        return result
        
    except Exception as e:
        self._audit_log('training_failed', user_id, {
            'ticker': ticker,
            'error': str(e)
        }, success=False)
        raise

@celery.task(bind=True, base=IntegratedTask, name='tasks.update_eod_data')
def update_eod_data(self, tickers: list, user_id: str):
    """
    Update EOD data through integrated pipeline
    """
    try:
        result = asyncio.run(
            self.system.eod_pipeline.run_daily_update(tickers)
        )
        
        self._audit_log('eod_update_completed', user_id, {
            'tickers': tickers,
            'records': result.get('total_records', 0)
        })
        
        return result
        
    except Exception as e:
        self._audit_log('eod_update_failed', user_id, {
            'error': str(e)
        }, success=False)
        raise

# ======================== ANPASSUNG 2: server.py ========================
# Aktualisieren Sie die API-Endpunkte:

# project/server_integrated.py
"""
Updated Flask server that uses integrated system
Add this to your existing server.py
"""

from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import asyncio

from .integrated_system import IntegratedMLTradingSystem, IntegratedSystemConfig
from .tasks_integrated import train_model_integrated, update_eod_data

# Initialize integrated system
_integrated_system = None

def get_integrated_system():
    global _integrated_system
    if _integrated_system is None:
        config = IntegratedSystemConfig()
        _integrated_system = IntegratedMLTradingSystem(config)
    return _integrated_system

# Replace existing /api/train endpoint
@app.route('/api/v2/train', methods=['POST'])
@jwt_required()
def train_with_integrated_system():
    """
    New training endpoint using integrated system
    """
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'AAPL').upper()
        model_type = data.get('model_type', 'lstm').lower()
        user_id = get_jwt_identity()
        
        # Validate
        if ticker not in Config.ALLOWED_TICKERS:
            return jsonify({'error': f'Invalid ticker: {ticker}'}), 400
        
        if model_type not in ['lstm', 'xgboost']:
            return jsonify({'error': f'Invalid model type: {model_type}'}), 400
        
        # Submit integrated task
        task = train_model_integrated.delay(ticker, model_type, user_id)
        
        return jsonify({
            'task_id': task.id,
            'status': 'submitted',
            'ticker': ticker,
            'model_type': model_type,
            'pipeline': 'integrated',
            'message': 'Training initiated through integrated ML pipeline'
        }), 202
        
    except Exception as e:
        logger.error(f"Training request failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v2/predict/<ticker>', methods=['POST'])
@jwt_required()
def predict_with_registry():
    """
    New prediction endpoint that ONLY uses production models from registry
    """
    try:
        system = get_integrated_system()
        model_type = request.args.get('model_type', 'lstm')
        
        # Get production model from registry (NOT from file!)
        model_info = asyncio.run(
            system.get_production_model_for_prediction(ticker, model_type)
        )
        
        if not model_info:
            return jsonify({
                'error': f'No production model available for {ticker}',
                'suggestion': 'Please train a model first'
            }), 404
        
        # Get data from request or fetch latest
        data = request.get_json()
        if data and 'features' in data:
            # Use provided features
            features = data['features']
        else:
            # Get latest features from EOD pipeline
            df = asyncio.run(
                system._get_training_data_from_eod(ticker, days_back=60)
            )
            features = asyncio.run(
                system._compute_features_from_eod(ticker, df)
            )
        
        # Make prediction using the model
        # ... implement your prediction logic ...
        
        return jsonify({
            'ticker': ticker,
            'model_id': model_info['model_id'],
            'model_version': model_info['version'],
            'prediction': 'placeholder',  # Add actual prediction
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v2/models/production', methods=['GET'])
@jwt_required()
def list_production_models():
    """
    List all production models from registry
    """
    try:
        system = get_integrated_system()
        ticker = request.args.get('ticker')
        
        if ticker:
            # Get specific ticker's production models
            lstm_model = system.model_registry.get_production_model(f"lstm_{ticker}", ticker)
            xgb_model = system.model_registry.get_production_model(f"xgboost_{ticker}", ticker)
            
            models = []
            if lstm_model:
                models.append({
                    'id': lstm_model.id,
                    'type': 'lstm',
                    'ticker': ticker,
                    'version': lstm_model.version,
                    'training_date': lstm_model.training_date.isoformat()
                })
            if xgb_model:
                models.append({
                    'id': xgb_model.id,
                    'type': 'xgboost',
                    'ticker': ticker,
                    'version': xgb_model.version,
                    'training_date': xgb_model.training_date.isoformat()
                })
            
            return jsonify({'models': models})
        else:
            # List all production models
            # ... implement listing logic ...
            return jsonify({'message': 'Implement full listing'})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ======================== ANPASSUNG 3: auto_optimizer.py ========================
# Update der collect_training_data Methode:

# In auto_optimizer.py, ersetzen Sie die _collect_training_data Methode:

async def _collect_training_data(self, days_back: int = 365) -> List[pd.DataFrame]:
    """
    NEW: Collect training data from EOD Pipeline instead of yfinance
    """
    logger.info("Collecting training data from EOD database...")
    
    # Get integrated system
    from .integrated_system import IntegratedMLTradingSystem, IntegratedSystemConfig
    config = IntegratedSystemConfig()
    system = IntegratedMLTradingSystem(config)
    
    # Get list of stocks with good data
    candidates = await self._find_historical_breakouts(days_back)
    
    training_data = []
    
    for ticker in candidates[:50]:
        try:
            # Get from EOD pipeline
            df = await system._get_training_data_from_eod(ticker, days_back)
            
            if len(df) > 100:
                # Get features from feature store
                df_features = await system._compute_features_from_eod(ticker, df)
                
                # Add labels
                df_features['future_return_5d'] = df_features['close'].pct_change(5).shift(-5)
                df_features['future_return_20d'] = df_features['close'].pct_change(20).shift(-20)
                df_features['breakout_occurred'] = (df_features['future_return_20d'] > 0.3).astype(int)
                
                training_data.append(df_features)
                
        except Exception as e:
            logger.error(f"Error collecting data for {ticker}: {e}")
            continue
    
    logger.info(f"Collected training data for {len(training_data)} stocks from EOD database")
    return training_data

# ======================== ANPASSUNG 4: working_api.py ========================
# Update für Model Loading:

# In working_api.py, ersetzen Sie die _load_latest_model Funktion:

async def _load_latest_model():
    """
    NEW: Load the latest production model from Model Registry
    """
    from .integrated_system import IntegratedMLTradingSystem, IntegratedSystemConfig
    
    config = IntegratedSystemConfig()
    system = IntegratedMLTradingSystem(config)
    
    # Get production model for a default ticker (e.g., market composite)
    try:
        model_info = await system.get_production_model_for_prediction('SPY', 'lstm')
        
        if model_info:
            # Update optimizer with production model
            optimizer = await get_optimizer()
            optimizer.current_model = model_info['model']
            optimizer.current_scaler = model_info.get('scaler')
            
            logger.info(f"Loaded production model version {model_info['version']}")
        else:
            logger.warning("No production model found in registry")
            
    except Exception as e:
        logger.error(f"Failed to load production model: {e}")

# ======================== ANPASSUNG 5: run_system_improved.py ========================
# Hauptsystem-Start anpassen:

# In run_system_improved.py, ersetzen Sie die initialize_components Methode:

async def initialize_components(self):
    """
    NEW: Initialize with integrated system
    """
    self.logger.info("🚀 Initializing Integrated ML Trading System...")
    
    from .integrated_system import IntegratedMLTradingSystem, IntegratedSystemConfig
    
    # Create integrated system config
    config = IntegratedSystemConfig(
        environment=os.getenv('ENVIRONMENT', 'production'),
        db_host=os.getenv('DB_HOST', 'localhost'),
        db_port=int(os.getenv('DB_PORT', '5432')),
        db_name=os.getenv('DB_NAME', 'trading_eod_db'),
        db_user=os.getenv('DB_USER', 'postgres'),
        db_password=os.getenv('DB_PASSWORD'),
        mlflow_uri=os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001'),
        redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
        gcs_bucket=os.getenv('GCS_BUCKET_NAME', 'ml-trading-models'),
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        optimization_interval_hours=24,
        auto_deploy=os.getenv('AUTO_DEPLOY', 'false').lower() == 'true'
    )
    
    # Create integrated system
    self.integrated_system = IntegratedMLTradingSystem(config)
    self.components['integrated_system'] = self.integrated_system
    
    # Initialize EOD pipeline
    await self.integrated_system.eod_pipeline.initialize()
    self.logger.info("✅ EOD Pipeline initialized")
    
    # Schedule daily pipeline
    import schedule
    schedule.every().day.at("17:30").do(
        lambda: asyncio.run(self.integrated_system.run_daily_pipeline())
    )
    self.logger.info("✅ Daily pipeline scheduled")
    
    self.state = SystemState.RUNNING
    self.logger.info("✅ Integrated system initialized successfully")

# ======================== DEPLOYMENT SCRIPT ========================

# deploy_integrated.sh
#!/bin/bash
"""
Deployment script for integrated system
"""

echo "Deploying Integrated ML Trading System..."

# 1. Database setup
echo "Setting up databases..."
psql -U postgres << EOF
CREATE DATABASE trading_eod_db;
CREATE DATABASE model_registry;
EOF

# 2. Initialize TimescaleDB
psql -U postgres -d trading_eod_db << EOF
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
EOF

# 3. Start MLflow server
echo "Starting MLflow server..."
mlflow server \
    --backend-store-uri postgresql://postgres:password@localhost/mlflow \
    --default-artifact-root gs://ml-trading-models/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5001 &

# 4. Start Redis
echo "Starting Redis..."
redis-server &

# 5. Start Celery workers
echo "Starting Celery workers..."
celery -A project.extensions worker --loglevel=info --concurrency=4 &

# 6. Start integrated system
echo "Starting integrated system..."
python -m project.integrated_system

# ======================== ENVIRONMENT VARIABLES ========================

# .env.integrated
"""
Environment variables for integrated system
"""

# Environment
ENVIRONMENT=production
FLASK_ENV=production

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_eod_db
DB_USER=postgres
DB_PASSWORD=your_secure_password

# Model Registry Database  
REGISTRY_DB_URL=postgresql://postgres:your_secure_password@localhost:5432/model_registry

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5001

# Redis
REDIS_URL=redis://localhost:6379
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# GCS
GCS_BUCKET_NAME=ml-trading-models
GCS_PROJECT_ID=your-project-id

# Trading Configuration
ALLOWED_TICKERS=AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,JPM,V,JNJ

# Security
SECRET_KEY=your_very_secure_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here

# Optimization
AUTO_DEPLOY=false
OPTIMIZATION_START_HOUR=2
OPTIMIZATION_END_HOUR=6

# Data Providers
POLYGON_API_KEY=your_polygon_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
