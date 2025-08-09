# project/decision_dashboard.py
"""
Interactive Decision Support Dashboard
Real-time visualization and analysis interface
"""

from flask import Blueprint, render_template, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_socketio import SocketIO, emit, join_room, leave_room
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

from .signal_intelligence_hub import SignalIntelligenceHub, TradingSignal
from .signal_performance_tracker import SignalPerformanceTracker
from .risk_assessment_engine import RiskAssessmentEngine
from .storage import get_gcs_storage
from .config import Config

import logging
logger = logging.getLogger(__name__)

# Create Blueprint
dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/api/dashboard')

class DecisionDashboard:
    """
    Interactive dashboard for signal visualization and decision support
    """
    
    def __init__(self, signal_hub: SignalIntelligenceHub):
        self.signal_hub = signal_hub
        self.performance_tracker = SignalPerformanceTracker()
        self.risk_engine = RiskAssessmentEngine()
        self.socketio = None
        self.active_views = {}
        self.user_preferences = {}
        
        # Cache for frequently accessed data
        self.cache = {
            'market_overview': None,
            'last_update': None
        }
        
        logger.info("Decision Dashboard initialized")
    
    def init_socketio(self, socketio: SocketIO):
        """Initialize WebSocket connection"""
        self.socketio = socketio
        self._register_socket_handlers()
    
    def _register_socket_handlers(self):
        """Register WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            emit('connected', {'status': 'Connected to Decision Dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe_signals')
        def handle_subscribe(data):
            room = data.get('room', 'signals')
            join_room(room)
            emit('subscribed', {'room': room})
        
        @self.socketio.on('request_signal_update')
        def handle_signal_update(data):
            signal_id = data.get('signal_id')
            if signal_id:
                signal = self.signal_hub.get_signal_by_id(signal_id)
                if signal:
                    emit('signal_update', self.create_signal_card(signal))
    
    def create_signal_card(self, signal: TradingSignal) -> Dict:
        """Create comprehensive signal card for display"""
        
        try:
            # Create visualizations
            chart_data = self._create_signal_chart(signal)
            risk_gauge = self._create_risk_gauge(signal)
            confidence_meter = self._create_confidence_visualization(signal)
            target_visualization = self._create_target_visualization(signal)
            
            # Format metrics
            metrics = self._format_key_metrics(signal)
            factors = self._format_supporting_factors(signal)
            
            # Create action items
            actions = self._create_action_buttons(signal)
            
            return {
                'signal_id': signal.signal_id,
                'ticker': signal.ticker,
                'timestamp': signal.timestamp.isoformat(),
                'signal_type': signal.signal_type.value,
                'signal_strength': signal.strength.value,
                'confidence': signal.confidence,
                'recommendation': signal.recommendation,
                'visualizations': {
                    'price_chart': chart_data,
                    'risk_gauge': risk_gauge,
                    'confidence_meter': confidence_meter,
                    'targets': target_visualization
                },
                'metrics': metrics,
                'supporting_factors': factors,
                'action_items': signal.action_items,
                'watch_conditions': signal.watch_conditions,
                'actions': actions,
                'quality_score': signal.signal_quality_score,
                'time_horizon': signal.time_horizon.value,
                'expiry': signal.signal_expiry.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create signal card: {e}")
            return {'error': str(e)}
    
    def _create_signal_chart(self, signal: TradingSignal) -> str:
        """Create interactive price chart with signal annotations"""
        
        try:
            # Get price data
            df = self._get_price_data(signal.ticker, days=60)
            
            if df.empty:
                return json.dumps({'error': 'No price data available'})
            
            # Create candlestick chart
            candlestick = go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            )
            
            # Add volume bars
            volume_trace = go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                yaxis='y2',
                marker_color='lightblue',
                opacity=0.3
            )
            
            # Create figure with subplots
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=('Price Action', 'Volume')
            )
            
            fig.add_trace(candlestick, row=1, col=1)
            fig.add_trace(volume_trace, row=2, col=1)
            
            # Add signal annotations
            annotations = []
            shapes = []
            
            # Entry line
            shapes.append(dict(
                type='line',
                xref='paper', x0=0, x1=1,
                yref='y', y0=signal.targets.entry, y1=signal.targets.entry,
                line=dict(color='blue', width=2, dash='dash'),
            ))
            
            annotations.append(dict(
                x=df.index[-1], y=signal.targets.entry,
                text=f"Entry: ${signal.targets.entry:.2f}",
                showarrow=True, arrowhead=2,
                bgcolor='blue', font=dict(color='white')
            ))
            
            # Stop loss line
            shapes.append(dict(
                type='line',
                xref='paper', x0=0, x1=1,
                yref='y', y0=signal.targets.stop_loss, y1=signal.targets.stop_loss,
                line=dict(color='red', width=2, dash='dash'),
            ))
            
            annotations.append(dict(
                x=df.index[-1], y=signal.targets.stop_loss,
                text=f"Stop: ${signal.targets.stop_loss:.2f}",
                showarrow=True, arrowhead=2,
                bgcolor='red', font=dict(color='white')
            ))
            
            # Target lines
            for i, target in enumerate([signal.targets.target_1, signal.targets.target_2, signal.targets.target_3]):
                if target:
                    shapes.append(dict(
                        type='line',
                        xref='paper', x0=0, x1=1,
                        yref='y', y0=target, y1=target,
                        line=dict(color='green', width=1, dash='dot'),
                    ))
                    
                    annotations.append(dict(
                        x=df.index[-1], y=target,
                        text=f"T{i+1}: ${target:.2f}",
                        showarrow=False,
                        bgcolor='green', font=dict(color='white', size=10)
                    ))
            
            # Add moving averages
            if len(df) >= 20:
                ma20 = go.Scatter(
                    x=df.index,
                    y=df['Close'].rolling(20).mean(),
                    name='MA20',
                    line=dict(color='orange', width=1)
                )
                fig.add_trace(ma20, row=1, col=1)
            
            if len(df) >= 50:
                ma50 = go.Scatter(
                    x=df.index,
                    y=df['Close'].rolling(50).mean(),
                    name='MA50',
                    line=dict(color='purple', width=1)
                )
                fig.add_trace(ma50, row=1, col=1)
            
            # Update layout
            fig.update_layout(
                title=f"{signal.ticker} - {signal.signal_type.value}",
                xaxis_title="Date",
                yaxis_title="Price",
                shapes=shapes,
                annotations=annotations,
                hovermode='x unified',
                template='plotly_dark',
                height=600,
                showlegend=True,
                legend=dict(x=0, y=1)
            )
            
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Failed to create chart: {e}")
            return json.dumps({'error': str(e)})
    
    def _create_risk_gauge(self, signal: TradingSignal) -> str:
        """Create risk assessment gauge visualization"""
        
        try:
            risk_percent = signal.targets.risk_percent
            
            # Determine risk level
            if risk_percent < 3:
                risk_level = "Low"
                color = "green"
            elif risk_percent < 5:
                risk_level = "Moderate"
                color = "yellow"
            elif risk_percent < 8:
                risk_level = "Elevated"
                color = "orange"
            else:
                risk_level = "High"
                color = "red"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_percent,
                title={'text': f"Risk Level: {risk_level}"},
                delta={'reference': 5, 'position': "top"},
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [None, 10], 'tickwidth': 1},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 3], 'color': "lightgreen"},
                        {'range': [3, 5], 'color': "lightyellow"},
                        {'range': [5, 8], 'color': "lightcoral"},
                        {'range': [8, 10], 'color': "lightpink"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 8
                    }
                }
            ))
            
            fig.update_layout(height=250, template='plotly_dark')
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Failed to create risk gauge: {e}")
            return json.dumps({'error': str(e)})
    
    def _create_confidence_visualization(self, signal: TradingSignal) -> str:
        """Create confidence level visualization"""
        
        try:
            confidence = signal.confidence * 100
            
            # Create horizontal bar chart
            fig = go.Figure(go.Bar(
                x=[confidence],
                y=['Confidence'],
                orientation='h',
                marker=dict(
                    color=confidence,
                    colorscale=[
                        [0, 'red'],
                        [0.5, 'yellow'],
                        [0.75, 'lightgreen'],
                        [1, 'green']
                    ],
                    cmin=0,
                    cmax=100,
                    colorbar=dict(
                        title="Confidence %",
                        thickness=15,
                        len=0.7
                    )
                ),
                text=f"{confidence:.1f}%",
                textposition='inside'
            ))
            
            # Add reference lines
            fig.add_vline(x=65, line_dash="dash", line_color="yellow", 
                         annotation_text="Min Threshold")
            fig.add_vline(x=75, line_dash="dash", line_color="green", 
                         annotation_text="Strong Signal")
            
            fig.update_layout(
                title="Signal Confidence Level",
                xaxis_title="Confidence %",
                xaxis=dict(range=[0, 100]),
                height=150,
                template='plotly_dark',
                showlegend=False
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Failed to create confidence visualization: {e}")
            return json.dumps({'error': str(e)})
    
    def _create_target_visualization(self, signal: TradingSignal) -> str:
        """Create target levels visualization"""
        
        try:
            targets = signal.targets
            current = targets.entry
            
            # Calculate percentage moves
            stop_pct = ((targets.stop_loss - current) / current) * 100
            t1_pct = ((targets.target_1 - current) / current) * 100
            t2_pct = ((targets.target_2 - current) / current) * 100
            t3_pct = ((targets.target_3 - current) / current * 100) if targets.target_3 else None
            
            # Create waterfall chart
            fig = go.Figure(go.Waterfall(
                name="Price Targets",
                orientation="v",
                measure=["relative", "relative", "relative", "relative"] if t3_pct else ["relative", "relative", "relative"],
                x=["Stop Loss", "Target 1", "Target 2", "Target 3"] if t3_pct else ["Stop Loss", "Target 1", "Target 2"],
                textposition="outside",
                text=[f"{stop_pct:.1f}%", f"+{t1_pct:.1f}%", f"+{t2_pct:.1f}%", f"+{t3_pct:.1f}%"] if t3_pct else [f"{stop_pct:.1f}%", f"+{t1_pct:.1f}%", f"+{t2_pct:.1f}%"],
                y=[stop_pct, t1_pct, t2_pct, t3_pct] if t3_pct else [stop_pct, t1_pct, t2_pct],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "green"}},
                decreasing={"marker": {"color": "red"}}
            ))
            
            fig.update_layout(
                title="Risk/Reward Profile",
                yaxis_title="% from Entry",
                height=300,
                template='plotly_dark',
                showlegend=False
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Failed to create target visualization: {e}")
            return json.dumps({'error': str(e)})
    
    def _format_key_metrics(self, signal: TradingSignal) -> Dict:
        """Format key metrics for display"""
        
        return {
            'entry_price': f"${signal.targets.entry:.2f}",
            'stop_loss': f"${signal.targets.stop_loss:.2f}",
            'risk_amount': f"${signal.targets.risk_amount:.2f}",
            'risk_percent': f"{signal.targets.risk_percent:.1f}%",
            'target_1': f"${signal.targets.target_1:.2f} ({signal.targets.prob_target_1:.0%})",
            'target_2': f"${signal.targets.target_2:.2f} ({signal.targets.prob_target_2:.0%})",
            'target_3': f"${signal.targets.target_3:.2f} ({signal.targets.prob_target_3:.0%})" if signal.targets.target_3 else "N/A",
            'risk_reward_ratio': f"{signal.risk_reward_ratio:.2f}:1",
            'expected_return': f"{signal.expected_return * 100:.1f}%",
            'time_horizon': signal.time_horizon.value,
            'expected_days': f"{signal.expected_breakout_days} days",
            'quality_score': f"{signal.signal_quality_score:.2f}/1.00"
        }
    
    def _format_supporting_factors(self, signal: TradingSignal) -> Dict:
        """Format supporting factors for display"""
        
        factors = {
            'technical': {},
            'ml_predictions': {},
            'volume': {},
            'consolidation': {},
            'market_context': {}
        }
        
        # Technical factors
        if signal.technical_factors:
            factors['technical'] = {
                'RSI': f"{signal.technical_factors.get('rsi', 0):.1f}",
                'ATR': f"${signal.technical_factors.get('atr', 0):.2f}",
                'Trend': signal.technical_factors.get('trend', 'Unknown'),
                'Volatility': signal.technical_factors.get('volatility_regime', 'Normal')
            }
        
        # ML predictions
        if signal.ml_predictions:
            factors['ml_predictions'] = {
                'Breakout Probability': f"{signal.ml_predictions.get('breakout_probability', 0) * 100:.1f}%",
                'Expected Move': f"{signal.ml_predictions.get('expected_magnitude', 0) * 100:.1f}%",
                'Model Confidence': f"{signal.ml_predictions.get('model_confidence', 0) * 100:.1f}%"
            }
        
        # Volume analysis
        if signal.volume_analysis:
            factors['volume'] = {
                'Volume Ratio': f"{signal.volume_analysis.get('volume_ratio', 1):.2f}x",
                'Volume Trend': 'Increasing' if signal.volume_analysis.get('volume_increasing', False) else 'Stable',
                'OBV Trend': signal.volume_analysis.get('obv_trend', 'Neutral')
            }
        
        # Consolidation data
        if signal.consolidation_data:
            factors['consolidation'] = {
                'Duration': f"{signal.consolidation_data.get('duration_days', 0)} days",
                'Phase Score': f"{signal.consolidation_data.get('phase_transition_score', 0):.2f}",
                'Accumulation': f"{signal.consolidation_data.get('accumulation_score', 0):.2f}"
            }
        
        # Market context
        if signal.market_context:
            factors['market_context'] = {
                'Market Regime': signal.market_context.get('regime', 'Unknown'),
                'VIX Level': f"{signal.market_context.get('vix_level', 20):.1f}",
                'Market Breadth': signal.market_context.get('market_breadth', 'Neutral')
            }
        
        return factors
    
    def _create_action_buttons(self, signal: TradingSignal) -> List[Dict]:
        """Create action buttons for signal"""
        
        actions = [
            {
                'id': 'view_details',
                'label': 'View Details',
                'icon': 'info',
                'action': 'show_signal_details',
                'params': {'signal_id': signal.signal_id}
            },
            {
                'id': 'set_alerts',
                'label': 'Set Alerts',
                'icon': 'bell',
                'action': 'configure_alerts',
                'params': {'ticker': signal.ticker, 'levels': signal.key_levels}
            },
            {
                'id': 'position_size',
                'label': 'Calculate Position',
                'icon': 'calculator',
                'action': 'open_position_calculator',
                'params': {'signal': signal.signal_id}
            },
            {
                'id': 'save_signal',
                'label': 'Save Signal',
                'icon': 'bookmark',
                'action': 'save_to_watchlist',
                'params': {'signal_id': signal.signal_id}
            },
            {
                'id': 'share_signal',
                'label': 'Share',
                'icon': 'share',
                'action': 'share_signal',
                'params': {'signal_id': signal.signal_id}
            }
        ]
        
        return actions
    
    def _get_price_data(self, ticker: str, days: int = 60) -> pd.DataFrame:
        """Get price data for ticker"""
        
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            df = stock.history(period=f"{days}d")
            return df
        except Exception as e:
            logger.error(f"Failed to get price data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_market_overview(self) -> Dict:
        """Get market overview data"""
        
        # Check cache
        if self.cache['market_overview'] and self.cache['last_update']:
            if datetime.now() - self.cache['last_update'] < timedelta(minutes=5):
                return self.cache['market_overview']
        
        try:
            # Get active signals
            active_signals = self.signal_hub.get_active_signals()
            
            # Group by strength
            strength_distribution = {}
            for signal in active_signals:
                strength = signal.strength.value
                strength_distribution[strength] = strength_distribution.get(strength, 0) + 1
            
            # Group by sector
            sector_distribution = {}
            for signal in active_signals:
                sector = signal.market_context.get('sector', 'Unknown')
                sector_distribution[sector] = sector_distribution.get(sector, 0) + 1
            
            # Calculate averages
            avg_confidence = np.mean([s.confidence for s in active_signals]) if active_signals else 0
            avg_rr_ratio = np.mean([s.risk_reward_ratio for s in active_signals]) if active_signals else 0
            
            overview = {
                'total_signals': len(active_signals),
                'strength_distribution': strength_distribution,
                'sector_distribution': sector_distribution,
                'average_confidence': avg_confidence,
                'average_risk_reward': avg_rr_ratio,
                'top_signals': [self.create_signal_card(s) for s in active_signals[:5]],
                'last_updated': datetime.now().isoformat()
            }
            
            # Update cache
            self.cache['market_overview'] = overview
            self.cache['last_update'] = datetime.now()
            
            return overview
            
        except Exception as e:
            logger.error(f"Failed to get market overview: {e}")
            return {'error': str(e)}
    
    def broadcast_signal_update(self, signal: TradingSignal):
        """Broadcast signal update to connected clients"""
        
        if self.socketio:
            signal_card = self.create_signal_card(signal)
            self.socketio.emit('signal_update', signal_card, room='signals')
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user dashboard preferences"""
        
        if user_id not in self.user_preferences:
            # Default preferences
            self.user_preferences[user_id] = {
                'theme': 'dark',
                'default_view': 'grid',
                'chart_type': 'candlestick',
                'indicators': ['MA20', 'MA50', 'Volume'],
                'refresh_interval': 60,  # seconds
                'notifications': {
                    'very_strong_signals': True,
                    'strong_signals': True,
                    'moderate_signals': False,
                    'weak_signals': False
                }
            }
        
        return self.user_preferences[user_id]
    
    def update_user_preferences(self, user_id: str, preferences: Dict):
        """Update user dashboard preferences"""
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        self.user_preferences[user_id].update(preferences)
        
        # Store to persistent storage if available
        if hasattr(self, 'gcs'):
            try:
                path = f"user_preferences/{user_id}.json"
                self.gcs.upload_json(self.user_preferences[user_id], path)
            except Exception as e:
                logger.error(f"Failed to save user preferences: {e}")

# Dashboard API endpoints
dashboard_instance = None

def get_dashboard() -> DecisionDashboard:
    """Get or create dashboard instance"""
    global dashboard_instance
    if dashboard_instance is None:
        hub = SignalIntelligenceHub()
        dashboard_instance = DecisionDashboard(hub)
    return dashboard_instance

@dashboard_bp.route('/overview', methods=['GET'])
@jwt_required()
def get_dashboard_overview():
    """Get dashboard overview data"""
    
    dashboard = get_dashboard()
    overview = dashboard.get_market_overview()
    
    return jsonify(overview)

@dashboard_bp.route('/signals/<signal_id>', methods=['GET'])
@jwt_required()
def get_signal_details(signal_id: str):
    """Get detailed signal information"""
    
    dashboard = get_dashboard()
    signal = dashboard.signal_hub.get_signal_by_id(signal_id)
    
    if not signal:
        return jsonify({'error': 'Signal not found'}), 404
    
    signal_card = dashboard.create_signal_card(signal)
    
    return jsonify(signal_card)

@dashboard_bp.route('/preferences', methods=['GET'])
@jwt_required()
def get_preferences():
    """Get user dashboard preferences"""
    
    user_id = get_jwt_identity()
    dashboard = get_dashboard()
    preferences = dashboard.get_user_preferences(user_id)
    
    return jsonify(preferences)

@dashboard_bp.route('/preferences', methods=['POST'])
@jwt_required()
def update_preferences():
    """Update user dashboard preferences"""
    
    user_id = get_jwt_identity()
    dashboard = get_dashboard()
    
    data = request.get_json()
    dashboard.update_user_preferences(user_id, data)
    
    return jsonify({'status': 'Preferences updated'})
