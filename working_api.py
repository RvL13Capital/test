# project/breakout_api_working.py
"""
Working API implementation with real model predictions
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
import pandas as pd
import torch
import numpy as np
import logging
from datetime import datetime
import asyncio
from typing import Dict, List

from .market_data import get_market_data_manager
from .auto_optimizer_working import WorkingAutoOptimizer
from .consolidation_network import NetworkConsolidationAnalyzer, extract_consolidation_features
from .storage import get_gcs_storage
from .monitoring import monitor_http_requests

logger = logging.getLogger(__name__)

# Create Blueprint
breakout_api = Blueprint('breakout_api', __name__, url_prefix='/api/v2/breakout')

# Initialize components
_optimizer = None
_model_cache = {}

async def get_optimizer():
    """Get or create optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = WorkingAutoOptimizer()
        await _optimizer.initialize()
        
        # Load latest model if available
        try:
            await _load_latest_model()
        except Exception as e:
            logger.warning(f"No existing model found: {e}")
    
    return _optimizer

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

@breakout_api.route('/analyze/<ticker>', methods=['GET'])
@jwt_required()
@monitor_http_requests
async def analyze_stock_real(ticker: str):
    """Analyze a stock with real data and predictions"""
    try:
        # Validate permissions
        user_claims = get_jwt()
        if 'analyze_stocks' not in user_claims.get('permissions', []):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Get market data manager
        market_manager = await get_market_data_manager()
        
        # Get stock info
        stock_info = await market_manager.get_stock_info(ticker.upper())
        if not stock_info:
            return jsonify({'error': f'Stock {ticker} not found'}), 404
        
        # Check market cap range
        if not (10e6 <= stock_info.market_cap <= 2e9):
            return jsonify({
                'warning': 'Stock outside target market cap range',
                'ticker': ticker,
                'market_cap': stock_info.market_cap,
                'recommendation': 'This system is optimized for nano/small cap stocks'
            }), 200
        
        # Get historical data
        lookback_days = int(request.args.get('lookback_days', 90))
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=lookback_days)
        
        df = await market_manager.get_historical_data(ticker, start_date, end_date)
        if df is None or len(df) < 30:
            return jsonify({'error': 'Insufficient historical data'}), 400
        
        # Get relative strength
        relative_strength = await market_manager.calculate_relative_strength(ticker)
        
        # Extract features
        df_features = extract_consolidation_features(df, stock_info.market_cap)
        
        # Get consolidation analysis
        optimizer = await get_optimizer()
        analyzer = optimizer.get_adaptive_analyzer()
        consolidations = analyzer.analyze_consolidation(df, stock_info.market_cap)
        current_phase = analyzer.get_current_phase(df)
        
        # Make prediction if model is available
        prediction = None
        if optimizer.current_model is not None:
            prediction = await _make_real_prediction(
                optimizer.current_model,
                optimizer.current_scaler,
                df_features,
                stock_info
            )
        
        # Prepare response
        response = {
            'ticker': ticker.upper(),
            'company_name': stock_info.company_name if hasattr(stock_info, 'company_name') else ticker,
            'market_cap': stock_info.market_cap,
            'sector': stock_info.sector,
            'current_price': stock_info.last_price,
            'avg_volume_30d': stock_info.avg_volume_30d,
            'relative_strength': relative_strength,
            'current_phase': current_phase,
            'consolidations_found': len(consolidations),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Add latest consolidation if found
        if consolidations:
            latest = consolidations[-1]
            response['latest_consolidation'] = {
                'duration_days': latest.duration_days,
                'price_range': latest.price_range,
                'volume_pattern': latest.volume_pattern,
                'network_density': latest.network_density,
                'phase_transition_score': latest.phase_transition_score,
                'accumulation_score': latest.accumulation_score,
                'mispricing_indicator': latest.mispricing_indicator
            }
        
        # Add prediction if available
        if prediction:
            response['prediction'] = prediction
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}")
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500

async def _make_real_prediction(model, scaler, df_features: pd.DataFrame, 
                               stock_info) -> Dict:
    """Make real prediction using the trained model"""
    try:
        device = next(model.parameters()).device
        
        # Prepare sequence (last 60 days)
        window_size = 60
        if len(df_features) < window_size:
            return None
        
        # Get feature columns (must match training)
        feature_cols = [col for col in df_features.columns if col not in [
            'datetime', 'ticker', 'market_cap'
        ]]
        
        # Get last window
        sequence = df_features[feature_cols].tail(window_size).values
        
        # Normalize
        sequence_flat = sequence.reshape(-1, len(feature_cols))
        sequence_normalized = scaler.transform(sequence_flat).reshape(1, window_size, -1)
        
        # Convert to tensor
        input_tensor = torch.tensor(sequence_normalized, dtype=torch.float32).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Extract predictions
        breakout_prob = outputs['breakout_probability'].cpu().item()
        expected_magnitude = outputs['expected_magnitude'].cpu().item()
        timing_dist = outputs['timing_distribution'].cpu().numpy()[0]
        attention_weights = outputs['attention_weights'].cpu().numpy()[0]
        feature_importance = outputs['feature_importance'].cpu().numpy()[0]
        
        # Determine timing estimate
        timing_categories = ['1-5 days', '6-10 days', '11-15 days', '16-20 days', '21-30 days']
        most_likely_timing = timing_categories[np.argmax(timing_dist)]
        
        # Calculate confidence (based on prediction certainty)
        confidence = max(breakout_prob, 1 - breakout_prob)
        
        # Determine recommendation
        if breakout_prob > 0.7 and expected_magnitude > 0.25:
            recommendation = 'STRONG BUY'
        elif breakout_prob > 0.6 and expected_magnitude > 0.20:
            recommendation = 'BUY'
        elif breakout_prob > 0.5:
            recommendation = 'WATCH'
        else:
            recommendation = 'AVOID'
        
        # Find most important features
        top_features_idx = np.argsort(feature_importance)[-5:]
        top_features = [feature_cols[i] for i in top_features_idx]
        
        return {
            'breakout_probability': float(breakout_prob),
            'expected_magnitude': float(expected_magnitude),
            'confidence': float(confidence),
            'timing_estimate': most_likely_timing,
            'timing_distribution': {
                cat: float(prob) for cat, prob in zip(timing_categories, timing_dist)
            },
            'recommendation': recommendation,
            'important_features': top_features,
            'model_version': model.__class__.__name__,
            'prediction_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None

@breakout_api.route('/screen', methods=['POST'])
@jwt_required()
@monitor_http_requests
async def screen_market_real():
    """Screen market with real data and predictions"""
    try:
        # Validate permissions
        user_claims = get_jwt()
        if 'screen_market' not in user_claims.get('permissions', []):
            return jsonify({'error': 'Insufficient permissions'}), 403
        
        # Get parameters
        data = request.get_json() or {}
        min_market_cap = data.get('min_market_cap', 10e6)
        max_market_cap = data.get('max_market_cap', 2e9)
        limit = min(data.get('limit', 20), 50)  # Cap at 50 for performance
        
        # Get optimizer and ensure model is loaded
        optimizer = await get_optimizer()
        if optimizer.current_model is None:
            return jsonify({
                'error': 'No trained model available',
                'message': 'Please wait for the system to complete initial training'
            }), 503
        
        # Get universe of stocks
        universe = await optimizer.get_universe_stocks(min_market_cap, max_market_cap)
        
        if not universe:
            return jsonify({
                'error': 'No stocks found in specified market cap range'
            }), 404
        
        # Screen stocks
        candidates = []
        market_manager = await get_market_data_manager()
        analyzer = optimizer.get_adaptive_analyzer()
        
        # Process in batches for efficiency
        batch_size = 10
        for i in range(0, min(len(universe), limit * 2), batch_size):
            batch = universe[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [_screen_single_stock(
                ticker, market_manager, analyzer, optimizer
            ) for ticker in batch]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and result.get('score', 0) > 0:
                    candidates.append(result)
        
        # Sort by score and limit
        candidates.sort(key=lambda x: x['score'], reverse=True)
        candidates = candidates[:limit]
        
        return jsonify({
            'candidates': candidates,
            'total_screened': min(len(universe), limit * 2),
            'screening_criteria': {
                'min_market_cap': min_market_cap,
                'max_market_cap': max_market_cap,
                'adaptive_thresholds': optimizer.adaptive_thresholds
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Market screening failed: {e}")
        return jsonify({'error': 'Screening failed', 'details': str(e)}), 500

async def _screen_single_stock(ticker: str, market_manager, analyzer, optimizer) -> Optional[Dict]:
    """Screen a single stock"""
    try:
        # Get stock info
        info = await market_manager.get_stock_info(ticker)
        if not info:
            return None
        
        # Get recent data
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=90)
        df = await market_manager.get_historical_data(ticker, start_date, end_date)
        
        if df is None or len(df) < 60:
            return None
        
        # Check if in consolidation
        current_phase = analyzer.get_current_phase(df)
        if current_phase['status'] != 'consolidation':
            return None
        
        # Extract features and make prediction
        df_features = extract_consolidation_features(df, info.market_cap)
        prediction = await _make_real_prediction(
            optimizer.current_model,
            optimizer.current_scaler,
            df_features,
            info
        )
        
        if not prediction:
            return None
        
        # Calculate score
        score = (
            prediction['breakout_probability'] * 
            prediction['expected_magnitude'] * 
            prediction['confidence']
        )
        
        return {
            'ticker': ticker,
            'market_cap': info.market_cap,
            'sector': info.sector,
            'current_price': info.last_price,
            'consolidation_days': current_phase.get('duration', 0),
            'phase_transition_score': current_phase.get('phase_transition_score', 0),
            'breakout_probability': prediction['breakout_probability'],
            'expected_magnitude': prediction['expected_magnitude'],
            'timing_estimate': prediction['timing_estimate'],
            'recommendation': prediction['recommendation'],
            'score': score
        }
        
    except Exception as e:
        logger.debug(f"Error screening {ticker}: {e}")
        return None

@breakout_api.route('/train', methods=['POST'])
@jwt_required()
@monitor_http_requests
async def trigger_training():
    """Trigger model training with real data"""
    try:
        # Admin only
        user_claims = get_jwt()
        if 'admin' not in user_claims.get('permissions', []):
            return jsonify({'error': 'Admin access required'}), 403
        
        # Get optimizer
        optimizer = await get_optimizer()
        
        # Run optimization in background (would use Celery in production)
        asyncio.create_task(optimizer.run_optimization())
        
        return jsonify({
            'message': 'Training initiated',
            'status': 'STARTED',
            'timestamp': datetime.now().isoformat()
        }), 202
        
    except Exception as e:
        logger.error(f"Failed to trigger training: {e}")
        return jsonify({'error': 'Training failed', 'details': str(e)}), 500
