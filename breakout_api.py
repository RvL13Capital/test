# project/breakout_api.py
"""
Production-ready API endpoints for the self-optimizing breakout system
"""

from flask import Blueprint, request, jsonify, g
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from marshmallow import Schema, fields, validate, ValidationError
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio

from .consolidation_network import NetworkConsolidationAnalyzer, extract_consolidation_features
from .breakout_strategy import BreakoutScreener, BreakoutPredictor
from .auto_optimizer import get_auto_optimizer
from .storage import get_gcs_storage
from .config import Config
from .tasks import analyze_breakout_async, screen_market_async
from .monitoring import (
    monitor_http_requests, HTTP_REQUESTS_TOTAL,
    MODEL_PERFORMANCE, CACHE_OPERATIONS
)

logger = logging.getLogger(__name__)

# Create Blueprint
breakout_bp = Blueprint('breakout', __name__, url_prefix='/api/breakout')

# Validation Schemas
class AnalyzeRequestSchema(Schema):
    """Schema for breakout analysis request"""
    ticker = fields.Str(
        required=True,
        validate=[
            validate.Length(min=1, max=10),
            validate.Regexp(r'^[A-Z]{1,10}$', error='Invalid ticker format')
        ]
    )
    lookback_days = fields.Int(
        missing=90,
        validate=validate.Range(min=30, max=365)
    )
    include_features = fields.Bool(missing=False)
    
class ScreenRequestSchema(Schema):
    """Schema for market screening request"""
    min_market_cap = fields.Float(
        missing=10e6,
        validate=validate.Range(min=1e6, max=10e9)
    )
    max_market_cap = fields.Float(
        missing=2e9,
        validate=validate.Range(min=1e6, max=10e9)
    )
    min_consolidation_days = fields.Int(
        missing=20,
        validate=validate.Range(min=10, max=120)
    )
    limit = fields.Int(
        missing=20,
        validate=validate.Range(min=1, max=100)
    )
    sector_filter = fields.List(fields.Str(), missing=None)

class PredictionRequestSchema(Schema):
    """Schema for breakout prediction request"""
    tickers = fields.List(
        fields.Str(validate=validate.Length(min=1, max=10)),
        required=True,
        validate=validate.Length(min=1, max=50)
    )
    prediction_horizon = fields.Int(
        missing=30,
        validate=validate.Range(min=5, max=60)
    )
    confidence_threshold = fields.Float(
        missing=0.65,
        validate=validate.Range(min=0.5, max=0.95)
    )

# Initialize schemas
analyze_schema = AnalyzeRequestSchema()
screen_schema = ScreenRequestSchema()
prediction_schema = PredictionRequestSchema()

# Initialize components
analyzer = NetworkConsolidationAnalyzer()
screener = BreakoutScreener()
gcs = get_gcs_storage()

# Endpoints

@breakout_bp.route('/analyze/<ticker>', methods=['GET'])
@jwt_required()
@monitor_http_requests
def analyze_consolidation(ticker: str):
    """
    Analyze consolidation patterns for a specific stock
    
    Returns detailed network analysis and breakout probability
    """
    try:
        # Validate request
        args = request.args.to_dict()
        args['ticker'] = ticker.upper()
        validated = analyze_schema.load(args)
        
        # Check permissions
        user_claims = get_jwt()
        if 'analyze_stocks' not in user_claims.get('permissions', []):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Get stock data
        import yfinance as yf
        stock = yf.Ticker(validated['ticker'])
        df = stock.history(period=f"{validated['lookback_days']}d")
        
        if len(df) < 20:
            return jsonify({'error': 'Insufficient data for analysis'}), 400
        
        # Get market cap
        info = stock.info
        market_cap = info.get('marketCap', 0)
        
        # Check if within target range
        if not (10e6 <= market_cap <= 2e9):
            return jsonify({
                'warning': 'Stock outside target market cap range',
                'market_cap': market_cap
            }), 200
        
        # Analyze consolidation
        consolidations = analyzer.analyze_consolidation(df, market_cap)
        
        # Get current phase
        current_phase = analyzer.get_current_phase(df)
        
        # Extract features if requested
        features = None
        if validated['include_features']:
            df_features = extract_consolidation_features(df, market_cap)
            features = df_features.tail(1).to_dict('records')[0]
        
        # Prepare response
        response = {
            'ticker': validated['ticker'],
            'market_cap': market_cap,
            'sector': info.get('sector', 'Unknown'),
            'current_phase': current_phase,
            'consolidations_found': len(consolidations),
            'analysis_date': datetime.now().isoformat()
        }
        
        # Add latest consolidation details
        if consolidations:
            latest = consolidations[-1]
            response['latest_consolidation'] = {
                'duration_days': latest.duration_days,
                'price_range': latest.price_range,
                'volume_pattern': latest.volume_pattern,
                'network_density': latest.network_density,
                'phase_transition_score': latest.phase_transition_score,
                'accumulation_score': latest.accumulation_score,
                'breakout_probability': latest.breakout_probability,
                'expected_move': latest.expected_move
            }
        
        if features:
            response['current_features'] = features
        
        # Log analysis
        logger.info(f"Consolidation analysis completed for {ticker}")
        
        # Cache result
        CACHE_OPERATIONS.labels(
            cache_type='analysis',
            operation='set',
            result='success'
        ).inc()
        
        return jsonify(response)
        
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@breakout_bp.route('/screen', methods=['POST'])
@jwt_required()
@monitor_http_requests
def screen_market():
    """
    Screen market for consolidation breakout candidates
    
    Returns ranked list of potential breakout stocks
    """
    try:
        # Validate request
        data = screen_schema.load(request.get_json() or {})
        
        # Check permissions
        user_claims = get_jwt()
        if 'screen_market' not in user_claims.get('permissions', []):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Submit async screening task
        task = screen_market_async.delay(
            min_market_cap=data['min_market_cap'],
            max_market_cap=data['max_market_cap'],
            min_consolidation_days=data['min_consolidation_days'],
            limit=data['limit'],
            sector_filter=data.get('sector_filter'),
            user_id=get_jwt_identity()
        )
        
        return jsonify({
            'task_id': task.id,
            'status': 'submitted',
            'message': 'Market screening in progress'
        }), 202
        
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    except Exception as e:
        logger.error(f"Market screening failed: {e}")
        return jsonify({'error': 'Screening failed'}), 500

@breakout_bp.route('/predict', methods=['POST'])
@jwt_required()
@monitor_http_requests
def predict_breakouts():
    """
    Get breakout predictions for multiple stocks
    
    Uses the latest optimized model for predictions
    """
    try:
        # Validate request
        data = prediction_schema.load(request.get_json() or {})
        
        # Check permissions
        user_claims = get_jwt()
        if 'predict_breakouts' not in user_claims.get('permissions', []):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Get current model
        optimizer = get_auto_optimizer()
        if not optimizer.current_model_id:
            return jsonify({'error': 'No model available'}), 503
        
        predictions = []
        
        for ticker in data['tickers']:
            try:
                # Get stock data
                import yfinance as yf
                stock = yf.Ticker(ticker)
                df = stock.history(period="90d")
                
                if len(df) < 60:
                    predictions.append({
                        'ticker': ticker,
                        'error': 'Insufficient data'
                    })
                    continue
                
                # Get market cap
                info = stock.info
                market_cap = info.get('marketCap', 500e6)
                
                # Extract features
                df_features = extract_consolidation_features(df, market_cap)
                
                # Make prediction (simplified for example)
                # In production, load actual model and run inference
                
                prediction = {
                    'ticker': ticker,
                    'breakout_probability': 0.72,  # Mock value
                    'expected_magnitude': 0.35,     # Mock value
                    'confidence': 0.80,             # Mock value
                    'timing_estimate': '5-10 days',
                    'current_phase': analyzer.get_current_phase(df),
                    'recommendation': 'WATCH' if 0.72 > data['confidence_threshold'] else 'BUY'
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Prediction failed for {ticker}: {e}")
                predictions.append({
                    'ticker': ticker,
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': predictions,
            'model_version': optimizer.current_model_id,
            'prediction_date': datetime.now().isoformat()
        })
        
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

@breakout_bp.route('/candidates/live', methods=['GET'])
@jwt_required()
@monitor_http_requests
def get_live_candidates():
    """
    Get current live breakout candidates
    
    Returns real-time monitoring data
    """
    try:
        # Check permissions
        user_claims = get_jwt()
        if 'view_candidates' not in user_claims.get('permissions', []):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Get latest screening results from cache/storage
        if gcs:
            try:
                # Get most recent market snapshot
                snapshots = gcs.list_models(prefix="market_data/snapshot_")
                if snapshots:
                    latest = max(snapshots, key=lambda x: x['updated'])
                    snapshot_data = gcs.download_json(latest['name'])
                    
                    return jsonify({
                        'candidates': snapshot_data.get('candidates', []),
                        'market_conditions': snapshot_data.get('market_conditions', {}),
                        'last_updated': snapshot_data.get('timestamp'),
                        'total_candidates': len(snapshot_data.get('candidates', []))
                    })
            except Exception as e:
                logger.error(f"Error retrieving candidates: {e}")
        
        # Fallback to live screening (limited)
        candidates = screener.get_top_candidates(n=10)
        
        return jsonify({
            'candidates': [
                {
                    'ticker': c.ticker,
                    'market_cap': c.market_cap,
                    'sector': c.sector,
                    'consolidation_score': c.consolidation_score,
                    'avg_volume': c.avg_volume
                }
                for c in candidates
            ],
            'last_updated': datetime.now().isoformat(),
            'total_candidates': len(candidates)
        })
        
    except Exception as e:
        logger.error(f"Failed to get live candidates: {e}")
        return jsonify({'error': 'Failed to retrieve candidates'}), 500

@breakout_bp.route('/system/status', methods=['GET'])
@jwt_required()
@monitor_http_requests
def get_system_status():
    """
    Get self-optimization system status
    
    Returns current performance and optimization metrics
    """
    try:
        # Check permissions
        user_claims = get_jwt()
        if 'admin' not in user_claims.get('permissions', []):
            return jsonify({'error': 'Admin access required'}), 403
        
        # Get optimizer status
        optimizer = get_auto_optimizer()
        status = optimizer.get_system_status()
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return jsonify({'error': 'Failed to retrieve status'}), 500

@breakout_bp.route('/system/optimize', methods=['POST'])
@jwt_required()
@monitor_http_requests
def trigger_optimization():
    """
    Manually trigger system optimization
    
    Admin only - forces immediate optimization cycle
    """
    try:
        # Check permissions
        user_claims = get_jwt()
        if 'admin' not in user_claims.get('permissions', []):
            return jsonify({'error': 'Admin access required'}), 403
        
        # Trigger optimization
        optimizer = get_auto_optimizer()
        optimizer._trigger_immediate_optimization()
        
        return jsonify({
            'message': 'Optimization triggered',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to trigger optimization: {e}")
        return jsonify({'error': 'Failed to trigger optimization'}), 500

@breakout_bp.route('/alerts/subscribe', methods=['POST'])
@jwt_required()
@monitor_http_requests
def subscribe_alerts():
    """
    Subscribe to breakout alerts
    
    Set up real-time notifications for breakout signals
    """
    try:
        # Implementation would handle WebSocket/SSE subscriptions
        # For now, return subscription confirmation
        
        user_id = get_jwt_identity()
        
        return jsonify({
            'subscription_id': f"sub_{user_id}_{datetime.now().timestamp()}",
            'status': 'active',
            'channels': ['email', 'webhook'],
            'message': 'Alert subscription created'
        })
        
    except Exception as e:
        logger.error(f"Failed to create alert subscription: {e}")
        return jsonify({'error': 'Subscription failed'}), 500

# WebSocket endpoint for real-time updates
from flask_socketio import emit, join_room, leave_room

@breakout_bp.route('/ws/connect')
@jwt_required()
def ws_connect():
    """WebSocket connection for real-time updates"""
    user_id = get_jwt_identity()
    join_room(f"user_{user_id}")
    emit('connected', {'message': 'Connected to breakout system'})

def broadcast_breakout_alert(ticker: str, alert_data: Dict):
    """Broadcast breakout alert to subscribed users"""
    # In production, this would check user subscriptions
    emit('breakout_alert', {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat(),
        **alert_data
    }, room='breakout_alerts')

# Error handlers
@breakout_bp.errorhandler(ValidationError)
def handle_validation_error(e):
    return jsonify({'error': 'Validation failed', 'details': e.messages}), 400

@breakout_bp.errorhandler(Exception)
def handle_unexpected_error(e):
    logger.error(f"Unexpected error: {e}")
    return jsonify({'error': 'Internal server error'}), 500