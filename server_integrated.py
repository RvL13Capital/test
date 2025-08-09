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
