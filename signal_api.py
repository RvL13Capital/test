# project/signal_api.py
"""
REST API endpoints for signal generation and analysis
Replacement for autonomous trading endpoints
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from marshmallow import Schema, fields, validate, ValidationError
import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from .signal_intelligence_hub import (
    SignalIntelligenceHub, TradingSignal, 
    SignalStrength, SignalType, TimeHorizon
)
from .signal_performance_tracker import (
    SignalPerformanceTracker, UserAction, 
    SignalOutcome, SignalPerformanceRecord
)
from .decision_dashboard import DecisionDashboard
from .alert_system import AlertSystem
from .risk_assessment_engine import RiskAssessmentEngine
from .storage import get_gcs_storage
from .config import Config
from .monitoring import SIGNAL_GENERATION, SIGNAL_FEEDBACK

logger = logging.getLogger(__name__)

# Create Blueprint
signal_bp = Blueprint('signals', __name__, url_prefix='/api/signals')

# Validation Schemas
class SignalGenerationSchema(Schema):
    """Schema for signal generation request"""
    tickers = fields.List(
        fields.Str(validate=validate.Length(min=1, max=10)),
        missing=None
    )
    min_confidence = fields.Float(
        missing=0.65,
        validate=validate.Range(min=0.0, max=1.0)
    )
    max_risk_percent = fields.Float(
        missing=5.0,
        validate=validate.Range(min=1.0, max=10.0)
    )
    min_risk_reward = fields.Float(
        missing=2.0,
        validate=validate.Range(min=1.0, max=10.0)
    )
    signal_types = fields.List(
        fields.Str(validate=validate.OneOf([t.value for t in SignalType])),
        missing=None
    )
    time_horizons = fields.List(
        fields.Str(validate=validate.OneOf([t.value for t in TimeHorizon])),
        missing=None
    )
    market_cap_min = fields.Float(missing=10e6)
    market_cap_max = fields.Float(missing=2e9)
    sectors = fields.List(fields.Str(), missing=None)
    limit = fields.Int(
        missing=20,
        validate=validate.Range(min=1, max=100)
    )

class SignalFeedbackSchema(Schema):
    """Schema for signal feedback"""
    action = fields.Str(
        required=True,
        validate=validate.OneOf([a.value for a in UserAction])
    )
    rating = fields.Int(
        missing=None,
        validate=validate.Range(min=1, max=5)
    )
    usefulness = fields.Bool(missing=None)
    accuracy = fields.Bool(missing=None)
    notes = fields.Str(
        missing=None,
        validate=validate.Length(max=1000)
    )
    entry_price = fields.Float(missing=None)
    exit_price = fields.Float(missing=None)
    position_size = fields.Float(missing=None)

class SignalOutcomeSchema(Schema):
    """Schema for signal outcome reporting"""
    exit_price = fields.Float(required=True)
    exit_time = fields.DateTime(missing=None)
    hit_target = fields.Int(
        missing=None,
        validate=validate.Range(min=1, max=3)
    )
    stopped_out = fields.Bool(missing=False)
    max_favorable_excursion = fields.Float(missing=None)
    max_adverse_excursion = fields.Float(missing=None)
    notes = fields.Str(missing=None)

class AlertPreferencesSchema(Schema):
    """Schema for alert preferences"""
    channels = fields.List(
        fields.Str(validate=validate.OneOf(['email', 'slack', 'webhook', 'desktop'])),
        required=True
    )
    min_strength = fields.Str(
        missing='MODERATE',
        validate=validate.OneOf([s.value for s in SignalStrength])
    )
    tickers = fields.List(fields.Str(), missing=None)
    immediate_notify = fields.Bool(missing=True)

# Initialize components
signal_hub = SignalIntelligenceHub()
performance_tracker = SignalPerformanceTracker()
alert_system = AlertSystem()
risk_engine = RiskAssessmentEngine()
dashboard = DecisionDashboard(signal_hub)

# Initialize schemas
generation_schema = SignalGenerationSchema()
feedback_schema = SignalFeedbackSchema()
outcome_schema = SignalOutcomeSchema()
alert_schema = AlertPreferencesSchema()

# Signal Generation Endpoints

@signal_bp.route('/generate', methods=['POST'])
@jwt_required()
def generate_signals():
    """
    Generate trading signals based on user criteria
    """
    try:
        # Validate input
        data = generation_schema.load(request.get_json() or {})
        user_id = get_jwt_identity()
        
        logger.info(f"Signal generation request from {user_id}")
        
        # Prepare filters
        filters = {
            'min_confidence': data['min_confidence'],
            'max_risk_percent': data['max_risk_percent'],
            'min_risk_reward': data['min_risk_reward'],
            'signal_types': data.get('signal_types'),
            'time_horizons': data.get('time_horizons'),
            'market_cap_range': (data['market_cap_min'], data['market_cap_max']),
            'sectors': data.get('sectors')
        }
        
        # Generate signals
        signals = asyncio.run(
            signal_hub.generate_comprehensive_signals(
                candidates=data.get('tickers'),
                filters=filters
            )
        )
        
        # Track signals for user
        for signal in signals:
            performance_tracker.track_signal_issued(signal, user_id)
        
        # Limit to requested number
        signals = signals[:data['limit']]
        
        # Convert to response format
        response_signals = []
        for signal in signals:
            response_signals.append({
                'signal_id': signal.signal_id,
                'ticker': signal.ticker,
                'timestamp': signal.timestamp.isoformat(),
                'signal_type': signal.signal_type.value,
                'strength': signal.strength.value,
                'confidence': signal.confidence,
                'recommendation': signal.recommendation,
                'targets': {
                    'entry': signal.targets.entry,
                    'stop_loss': signal.targets.stop_loss,
                    'target_1': signal.targets.target_1,
                    'target_2': signal.targets.target_2,
                    'target_3': signal.targets.target_3,
                    'risk_percent': signal.targets.risk_percent,
                    'risk_reward_ratio': signal.risk_reward_ratio
                },
                'time_horizon': signal.time_horizon.value,
                'expected_days': signal.expected_breakout_days,
                'action_items': signal.action_items,
                'watch_conditions': signal.watch_conditions,
                'quality_score': signal.signal_quality_score
            })
        
        # Update metrics
        SIGNAL_GENERATION.labels(status='success').inc()
        
        response = {
            'status': 'success',
            'generated_at': datetime.now().isoformat(),
            'total_signals': len(response_signals),
            'signals': response_signals,
            'filters_applied': filters,
            'user_id': user_id
        }
        
        # Send alerts if configured
        asyncio.run(alert_system.check_and_send_alerts(signals, user_id))
        
        return jsonify(response)
        
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        SIGNAL_GENERATION.labels(status='failed').inc()
        return jsonify({'error': 'Signal generation failed'}), 500

@signal_bp.route('/active', methods=['GET'])
@jwt_required()
def get_active_signals():
    """
    Get currently active signals
    """
    try:
        ticker = request.args.get('ticker')
        
        signals = signal_hub.get_active_signals(ticker)
        
        response_signals = []
        for signal in signals:
            response_signals.append({
                'signal_id': signal.signal_id,
                'ticker': signal.ticker,
                'timestamp': signal.timestamp.isoformat(),
                'signal_type': signal.signal_type.value,
                'strength': signal.strength.value,
                'confidence': signal.confidence,
                'recommendation': signal.recommendation,
                'expiry': signal.signal_expiry.isoformat(),
                'quality_score': signal.signal_quality_score
            })
        
        return jsonify({
            'total': len(response_signals),
            'signals': response_signals,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get active signals: {e}")
        return jsonify({'error': 'Failed to retrieve signals'}), 500

@signal_bp.route('/<signal_id>', methods=['GET'])
@jwt_required()
def get_signal_details(signal_id: str):
    """
    Get detailed information for a specific signal
    """
    try:
        signal = signal_hub.get_signal_by_id(signal_id)
        
        if not signal:
            return jsonify({'error': 'Signal not found'}), 404
        
        # Create detailed response
        response = dashboard.create_signal_card(signal)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Failed to get signal details: {e}")
        return jsonify({'error': 'Failed to retrieve signal'}), 500

# Deep Analysis Endpoints

@signal_bp.route('/analyze/<ticker>', methods=['GET'])
@jwt_required()
def deep_analysis(ticker: str):
    """
    Provide deep-dive analysis for a ticker
    """
    try:
        ticker = ticker.upper()
        
        # Run comprehensive analysis
        analysis = asyncio.run(signal_hub._analyze_ticker(ticker))
        
        if not analysis:
            return jsonify({'error': 'Analysis failed or ticker not eligible'}), 400
        
        # Risk assessment
        risk_assessment = risk_engine.assess_ticker(ticker, analysis)
        
        # Historical patterns
        similar_patterns = performance_tracker._analyze_by_ticker(
            [r for r in performance_tracker.performance_records if r.ticker == ticker]
        )
        
        response = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'current_price': analysis['current_price'],
            'market_cap': analysis['market_cap'],
            'sector': analysis['sector'],
            'analysis': {
                'consolidation': analysis.get('consolidation', {}),
                'technical': analysis.get('technical', {}),
                'volume': analysis.get('volume', {}),
                'patterns': analysis.get('patterns', {}),
                'ml_predictions': analysis.get('ml_predictions', {}),
                'market_context': analysis.get('market_regime', {})
            },
            'risk_assessment': risk_assessment,
            'historical_performance': similar_patterns,
            'signal_potential': {
                'ready_for_signal': analysis.get('should_signal', False),
                'confidence': analysis.get('confidence', 0),
                'missing_factors': []
            }
        }
        
        # Identify missing factors
        if not analysis.get('should_signal', False):
            missing = []
            if not analysis.get('consolidation', {}).get('in_consolidation'):
                missing.append('Not in consolidation phase')
            if analysis.get('volume', {}).get('volume_ratio', 1) < 1.5:
                missing.append('Insufficient volume')
            if analysis.get('ml_predictions', {}).get('breakout_probability', 0) < 0.5:
                missing.append('Low ML confidence')
            
            response['signal_potential']['missing_factors'] = missing
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Deep analysis failed for {ticker}: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

# Feedback and Performance Endpoints

@signal_bp.route('/<signal_id>/feedback', methods=['POST'])
@jwt_required()
def submit_feedback(signal_id: str):
    """
    Submit feedback on a signal
    """
    try:
        data = feedback_schema.load(request.get_json() or {})
        user_id = get_jwt_identity()
        
        # Record user action
        action = UserAction[data['action']]
        
        action_details = {
            'entry_price': data.get('entry_price'),
            'entry_time': datetime.now() if data.get('entry_price') else None,
            'position_size': data.get('position_size'),
            'notes': data.get('notes')
        }
        
        performance_tracker.record_user_action(
            signal_id, user_id, action, action_details
        )
        
        # Record feedback
        if data.get('rating') is not None:
            feedback = {
                'rating': data['rating'],
                'usefulness': data.get('usefulness'),
                'accuracy': data.get('accuracy'),
                'notes': data.get('notes'),
                'would_follow_again': data.get('rating', 0) >= 4
            }
            
            performance_tracker.record_user_feedback(signal_id, user_id, feedback)
        
        # Check if should trigger learning
        if performance_tracker.should_trigger_learning():
            # Trigger async model update
            from .tasks import tune_and_train_async
            tune_and_train_async.delay(
                feedback_data=performance_tracker.get_recent_feedback()
            )
        
        # Update metrics
        SIGNAL_FEEDBACK.labels(action=action.value).inc()
        
        return jsonify({
            'status': 'Feedback recorded',
            'signal_id': signal_id,
            'action': action.value
        })
        
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        return jsonify({'error': 'Failed to record feedback'}), 500

@signal_bp.route('/<signal_id>/outcome', methods=['POST'])
@jwt_required()
def report_outcome(signal_id: str):
    """
    Report the outcome of a signal
    """
    try:
        data = outcome_schema.load(request.get_json() or {})
        
        # Record outcome
        outcome_data = {
            'exit_price': data['exit_price'],
            'exit_time': data.get('exit_time', datetime.now()),
            'hit_target': data.get('hit_target'),
            'stopped_out': data.get('stopped_out', False),
            'expired': False,
            'max_favorable_excursion': data.get('max_favorable_excursion'),
            'max_adverse_excursion': data.get('max_adverse_excursion')
        }
        
        record = performance_tracker.record_signal_outcome(signal_id, outcome_data)
        
        if not record:
            return jsonify({'error': 'Signal not found'}), 404
        
        return jsonify({
            'status': 'Outcome recorded',
            'signal_id': signal_id,
            'outcome': record.outcome.value,
            'actual_return': f"{record.actual_return_percent:.2f}%"
        })
        
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    except Exception as e:
        logger.error(f"Failed to record outcome: {e}")
        return jsonify({'error': 'Failed to record outcome'}), 500

@signal_bp.route('/performance/report', methods=['GET'])
@jwt_required()
def get_performance_report():
    """
    Get performance report for signals
    """
    try:
        period_days = request.args.get('period_days', 30, type=int)
        
        report = performance_tracker.generate_performance_report(period_days)
        
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"Failed to generate performance report: {e}")
        return jsonify({'error': 'Failed to generate report'}), 500

@signal_bp.route('/performance/export', methods=['GET'])
@jwt_required()
def export_performance():
    """
    Export performance data
    """
    try:
        format = request.args.get('format', 'json')
        
        data = performance_tracker.export_performance_data(format)
        
        if format == 'csv':
            from flask import Response
            return Response(
                data,
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=performance.csv'}
            )
        else:
            return jsonify(json.loads(data) if isinstance(data, str) else data)
        
    except Exception as e:
        logger.error(f"Failed to export performance: {e}")
        return jsonify({'error': 'Failed to export data'}), 500

# Alert Management Endpoints

@signal_bp.route('/alerts/preferences', methods=['GET'])
@jwt_required()
def get_alert_preferences():
    """
    Get user alert preferences
    """
    try:
        user_id = get_jwt_identity()
        preferences = alert_system.get_user_preferences(user_id)
        
        return jsonify(preferences)
        
    except Exception as e:
        logger.error(f"Failed to get alert preferences: {e}")
        return jsonify({'error': 'Failed to get preferences'}), 500

@signal_bp.route('/alerts/preferences', methods=['POST'])
@jwt_required()
def update_alert_preferences():
    """
    Update user alert preferences
    """
    try:
        data = alert_schema.load(request.get_json() or {})
        user_id = get_jwt_identity()
        
        alert_system.update_user_preferences(user_id, data)
        
        return jsonify({'status': 'Preferences updated'})
        
    except ValidationError as e:
        return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
    except Exception as e:
        logger.error(f"Failed to update alert preferences: {e}")
        return jsonify({'error': 'Failed to update preferences'}), 500

@signal_bp.route('/alerts/test', methods=['POST'])
@jwt_required()
def test_alert():
    """
    Send test alert to verify configuration
    """
    try:
        user_id = get_jwt_identity()
        channel = request.get_json().get('channel', 'email')
        
        # Create test signal
        from .signal_intelligence_hub import TradingSignal, PriceTargets
        
        test_signal = TradingSignal(
            signal_id='TEST_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
            ticker='TEST',
            timestamp=datetime.now(),
            signal_type=SignalType.BREAKOUT_IMMINENT,
            strength=SignalStrength.STRONG,
            confidence=0.85,
            targets=PriceTargets(
                entry=100,
                stop_loss=95,
                target_1=110,
                target_2=120,
                risk_percent=5
            ),
            risk_reward_ratio=2.0,
            expected_return=0.15,
            time_horizon=TimeHorizon.SHORT_TERM,
            expected_breakout_days=3,
            signal_expiry=datetime.now() + timedelta(days=7),
            recommendation="This is a test signal",
            signal_quality_score=0.85
        )
        
        # Send test alert
        success = asyncio.run(
            alert_system.send_test_alert(test_signal, user_id, channel)
        )
        
        if success:
            return jsonify({'status': 'Test alert sent successfully'})
        else:
            return jsonify({'error': 'Failed to send test alert'}), 500
        
    except Exception as e:
        logger.error(f"Failed to send test alert: {e}")
        return jsonify({'error': 'Failed to send test alert'}), 500

# Position Sizing and Risk Management

@signal_bp.route('/position-calculator', methods=['POST'])
@jwt_required()
def calculate_position():
    """
    Calculate optimal position size for a signal
    """
    try:
        data = request.get_json()
        
        signal_id = data.get('signal_id')
        account_size = data.get('account_size', 10000)
        risk_per_trade = data.get('risk_per_trade', 0.02)  # 2% default
        
        signal = signal_hub.get_signal_by_id(signal_id)
        
        if not signal:
            return jsonify({'error': 'Signal not found'}), 404
        
        # Calculate position
        risk_amount = account_size * risk_per_trade
        stop_distance = signal.targets.entry - signal.targets.stop_loss
        
        if stop_distance <= 0:
            return jsonify({'error': 'Invalid stop loss'}), 400
        
        shares = int(risk_amount / stop_distance)
        position_value = shares * signal.targets.entry
        
        # Kelly Criterion (simplified)
        win_prob = signal.confidence
        win_amount = signal.targets.target_1 - signal.targets.entry
        loss_amount = stop_distance
        
        kelly_fraction = (win_prob * win_amount - (1 - win_prob) * loss_amount) / win_amount
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        kelly_position = int((account_size * kelly_fraction) / signal.targets.entry)
        
        response = {
            'signal_id': signal_id,
            'ticker': signal.ticker,
            'account_size': account_size,
            'risk_per_trade': risk_per_trade,
            'calculations': {
                'risk_amount': risk_amount,
                'stop_distance': stop_distance,
                'standard_position': {
                    'shares': shares,
                    'position_value': position_value,
                    'position_percent': (position_value / account_size) * 100
                },
                'kelly_position': {
                    'shares': kelly_position,
                    'position_value': kelly_position * signal.targets.entry,
                    'kelly_fraction': kelly_fraction
                },
                'recommended_shares': min(shares, kelly_position)
            },
            'risk_metrics': {
                'max_loss': shares * stop_distance,
                'target_1_profit': shares * (signal.targets.target_1 - signal.targets.entry),
                'target_2_profit': shares * (signal.targets.target_2 - signal.targets.entry)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Position calculation failed: {e}")
        return jsonify({'error': 'Calculation failed'}), 500

# Market Overview

@signal_bp.route('/market/overview', methods=['GET'])
@jwt_required()
def market_overview():
    """
    Get market overview and signal statistics
    """
    try:
        overview = dashboard.get_market_overview()
        
        # Add performance metrics
        overview['performance'] = performance_tracker.performance_metrics
        
        return jsonify(overview)
        
    except Exception as e:
        logger.error(f"Failed to get market overview: {e}")
        return jsonify({'error': 'Failed to get overview'}), 500

# WebSocket support for real-time updates
from flask_socketio import emit

def broadcast_new_signal(signal: TradingSignal):
    """Broadcast new signal to connected clients"""
    dashboard.broadcast_signal_update(signal)
