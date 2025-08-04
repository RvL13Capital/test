# project/breakout_strategy.py
"""
Self-Optimizing Breakout Trading Strategy for Nano/Small Cap Stocks
Specializes in explosive moves from consolidation phases
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import backtrader as bt
import yfinance as yf
from dataclasses import dataclass
import json

from .consolidation_network import NetworkConsolidationAnalyzer, extract_consolidation_features
from .config import Config
from .storage import get_gcs_storage
from .features_optimized import OptimizedFeatureEngine
from .monitoring import MODEL_PERFORMANCE, TRAINING_REQUESTS_TOTAL

logger = logging.getLogger(__name__)

@dataclass
class StockCandidate:
    """Candidate stock for breakout trading"""
    ticker: str
    market_cap: float
    sector: str
    avg_volume: float
    consolidation_score: float
    last_updated: datetime

class BreakoutPredictor(nn.Module):
    """
    Enhanced LSTM model specifically designed for breakout prediction
    Incorporates attention mechanism for consolidation patterns
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        
        # Multi-layer LSTM with larger hidden dimension for complex patterns
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism for important time points
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Feature importance network
        self.feature_importance = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Output layers for different predictions
        self.breakout_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary: breakout or not
        )
        
        self.magnitude_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Regression: expected move size
            nn.Sigmoid()  # Output between 0 and 1 (multiply by 100 for percentage)
        )
        
        self.timing_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5)  # Classify: 1-5 days, 6-10 days, etc.
        )
    
    def forward(self, x):
        # Apply feature importance weighting
        feature_weights = self.feature_importance(x)
        x_weighted = x * feature_weights
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x_weighted)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Predictions
        breakout_logits = self.breakout_classifier(context)
        magnitude = self.magnitude_predictor(context)
        timing_logits = self.timing_predictor(context)
        
        return {
            'breakout_probability': torch.softmax(breakout_logits, dim=-1)[:, 1],
            'expected_magnitude': magnitude.squeeze(),
            'timing_distribution': torch.softmax(timing_logits, dim=-1),
            'attention_weights': attention_weights.squeeze(),
            'feature_importance': feature_weights.mean(dim=1)  # Average over time
        }

class SelfOptimizingBreakoutStrategy(bt.Strategy):
    """
    Self-optimizing strategy that continuously learns and adapts
    """
    
    params = (
        ('predictor_model', None),
        ('consolidation_analyzer', None),
        ('min_market_cap', 10e6),  # $10M minimum
        ('max_market_cap', 2e9),   # $2B maximum (small cap)
        ('min_breakout_probability', 0.65),
        ('min_expected_move', 0.25),  # 25% minimum
        ('position_size_pct', 0.05),  # 5% per position
        ('max_positions', 10),
        ('stop_loss_pct', 0.08),  # 8% stop loss
        ('learning_enabled', True),
        ('rebalance_days', 5),
        ('gcs_storage', None)
    )
    
    def __init__(self):
        self.predictor = self.params.predictor_model
        self.analyzer = self.params.consolidation_analyzer or NetworkConsolidationAnalyzer()
        self.gcs = self.params.gcs_storage or get_gcs_storage()
        
        # Track positions and performance
        self.active_positions = {}
        self.position_history = []
        self.learning_buffer = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_return = 0
        
        # Self-optimization state
        self.last_optimization = datetime.now()
        self.optimization_frequency = timedelta(days=7)
        self.performance_threshold = 0.6  # 60% win rate target
        
        logger.info("SelfOptimizingBreakoutStrategy initialized")
    
    def next(self):
        current_date = self.datas[0].datetime.datetime(0)
        
        # Periodic self-optimization
        if self.params.learning_enabled and (current_date - self.last_optimization) > self.optimization_frequency:
            self._self_optimize()
        
        # Scan for breakout candidates
        for data in self.datas:
            ticker = data._name
            
            if ticker not in self.active_positions:
                self._evaluate_entry(data, ticker, current_date)
            else:
                self._manage_position(data, ticker, current_date)
    
    def _evaluate_entry(self, data, ticker: str, current_date: datetime):
        """Evaluate potential entry for a stock"""
        try:
            # Get recent price history
            lookback = 60
            if len(data) < lookback:
                return
            
            # Create DataFrame for analysis
            df = self._create_dataframe(data, lookback)
            
            # Get market cap (would need real data source in production)
            market_cap = self._estimate_market_cap(ticker, df)
            
            # Check market cap criteria
            if not (self.params.min_market_cap <= market_cap <= self.params.max_market_cap):
                return
            
            # Extract consolidation features
            df_features = extract_consolidation_features(df, market_cap)
            
            # Get current phase
            current_phase = self.analyzer.get_current_phase(df, lookback)
            
            # Only proceed if in consolidation with high transition score
            if (current_phase['status'] == 'consolidation' and 
                current_phase.get('phase_transition_score', 0) > 0.6):
                
                # Prepare features for model
                features = self._prepare_model_features(df_features)
                
                # Get predictions
                with torch.no_grad():
                    predictions = self.predictor(features.unsqueeze(0))
                
                breakout_prob = predictions['breakout_probability'].item()
                expected_move = predictions['expected_magnitude'].item()
                
                # Check entry criteria
                if (breakout_prob >= self.params.min_breakout_probability and
                    expected_move >= self.params.min_expected_move and
                    len(self.active_positions) < self.params.max_positions):
                    
                    # Calculate position size
                    position_size = self._calculate_position_size(
                        breakout_prob, expected_move
                    )
                    
                    # Enter position
                    self.buy(data=data, size=position_size)
                    
                    # Track position
                    self.active_positions[ticker] = {
                        'entry_date': current_date,
                        'entry_price': data.close[0],
                        'breakout_probability': breakout_prob,
                        'expected_move': expected_move,
                        'consolidation_days': current_phase.get('duration', 0),
                        'phase_score': current_phase.get('phase_transition_score', 0),
                        'stop_loss': data.close[0] * (1 - self.params.stop_loss_pct)
                    }
                    
                    logger.info(f"Entered {ticker}: prob={breakout_prob:.2%}, "
                              f"expected={expected_move:.2%}, size={position_size}")
                    
        except Exception as e:
            logger.error(f"Error evaluating {ticker}: {e}")
    
    def _manage_position(self, data, ticker: str, current_date: datetime):
        """Manage existing position"""
        position = self.active_positions[ticker]
        current_price = data.close[0]
        entry_price = position['entry_price']
        
        # Calculate return
        position_return = (current_price - entry_price) / entry_price
        
        # Stop loss
        if current_price <= position['stop_loss']:
            self.close(data=data)
            self._record_trade(ticker, position, position_return, 'stop_loss')
            del self.active_positions[ticker]
            return
        
        # Dynamic exit based on magnitude achievement
        expected_move = position['expected_move']
        
        if position_return >= expected_move * 0.8:  # Achieved 80% of expected move
            self.close(data=data)
            self._record_trade(ticker, position, position_return, 'target_reached')
            del self.active_positions[ticker]
            return
        
        # Time-based exit (if move doesn't materialize)
        days_held = (current_date - position['entry_date']).days
        
        if days_held > 30 and position_return < 0.1:  # Flat after 30 days
            self.close(data=data)
            self._record_trade(ticker, position, position_return, 'time_exit')
            del self.active_positions[ticker]
            return
        
        # Trailing stop for winners
        if position_return > 0.15:  # 15% gain
            new_stop = current_price * 0.9  # 10% trailing stop
            position['stop_loss'] = max(position['stop_loss'], new_stop)
    
    def _self_optimize(self):
        """Self-optimization routine"""
        if not self.learning_buffer:
            return
        
        logger.info("Starting self-optimization cycle...")
        
        # Calculate recent performance
        recent_trades = self.learning_buffer[-50:]  # Last 50 trades
        if len(recent_trades) < 20:
            return
        
        win_rate = sum(1 for t in recent_trades if t['return'] > 0) / len(recent_trades)
        avg_win = np.mean([t['return'] for t in recent_trades if t['return'] > 0])
        avg_loss = np.mean([t['return'] for t in recent_trades if t['return'] <= 0])
        
        logger.info(f"Recent performance: win_rate={win_rate:.2%}, "
                   f"avg_win={avg_win:.2%}, avg_loss={avg_loss:.2%}")
        
        # Adjust strategy parameters based on performance
        if win_rate < self.performance_threshold:
            # Increase selectivity
            self.params.min_breakout_probability = min(
                self.params.min_breakout_probability * 1.05, 0.85
            )
            self.params.min_expected_move = min(
                self.params.min_expected_move * 1.05, 0.40
            )
            logger.info("Increased selectivity due to low win rate")
        
        elif win_rate > self.performance_threshold + 0.1:
            # Decrease selectivity slightly
            self.params.min_breakout_probability = max(
                self.params.min_breakout_probability * 0.98, 0.60
            )
            logger.info("Decreased selectivity due to high win rate")
        
        # Update model if performance is consistently poor
        if win_rate < 0.5 and len(recent_trades) >= 30:
            self._trigger_model_retraining()
        
        # Save optimization state
        self._save_optimization_state()
        
        self.last_optimization = datetime.now()
    
    def _trigger_model_retraining(self):
        """Trigger model retraining based on recent performance"""
        logger.info("Triggering model retraining due to poor performance...")
        
        # Prepare training data from recent trades
        training_data = []
        
        for trade in self.learning_buffer[-100:]:
            training_data.append({
                'features': trade['features'],
                'actual_return': trade['return'],
                'holding_period': trade['holding_days'],
                'exit_reason': trade['exit_reason']
            })
        
        # Save training data for offline retraining
        if self.gcs:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            training_path = f"retraining_data/breakout_strategy_{timestamp}.json"
            
            self.gcs.upload_json(training_data, training_path)
            
            # Log retraining request
            TRAINING_REQUESTS_TOTAL.labels(
                model_type='breakout_predictor',
                status='requested'
            ).inc()
    
    def _record_trade(self, ticker: str, position: Dict, return_pct: float, exit_reason: str):
        """Record trade for learning"""
        trade_record = {
            'ticker': ticker,
            'entry_date': position['entry_date'],
            'exit_date': datetime.now(),
            'holding_days': (datetime.now() - position['entry_date']).days,
            'return': return_pct,
            'expected_move': position['expected_move'],
            'breakout_probability': position['breakout_probability'],
            'exit_reason': exit_reason,
            'features': position.get('features', {})
        }
        
        self.learning_buffer.append(trade_record)
        self.position_history.append(trade_record)
        
        # Update metrics
        self.total_trades += 1
        if return_pct > 0:
            self.winning_trades += 1
        self.total_return += return_pct
        
        # Log performance
        MODEL_PERFORMANCE.labels(
            model_type='breakout_strategy',
            ticker=ticker,
            metric_type='trade_return'
        ).set(return_pct)
        
        logger.info(f"Trade closed: {ticker}, return={return_pct:.2%}, reason={exit_reason}")
    
    def _create_dataframe(self, data, lookback: int) -> pd.DataFrame:
        """Create DataFrame from backtrader data"""
        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        for i in range(-lookback + 1, 1):
            dates.append(data.datetime.datetime(i))
            opens.append(data.open[i])
            highs.append(data.high[i])
            lows.append(data.low[i])
            closes.append(data.close[i])
            volumes.append(data.volume[i])
        
        return pd.DataFrame({
            'datetime': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }).set_index('datetime')
    
    def _prepare_model_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare features for model input"""
        # Select relevant features
        feature_cols = [
            'close', 'volume', 'high', 'low',
            'in_consolidation', 'consolidation_days',
            'phase_transition_score', 'accumulation_score',
            'breakout_probability', 'expected_move',
            'network_density', 'volume_surge',
            'price_squeeze_ratio', 'relative_strength'
        ]
        
        # Ensure all features exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Normalize features
        features = df[feature_cols].values
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32)
    
    def _calculate_position_size(self, breakout_prob: float, expected_move: float) -> int:
        """Calculate position size based on conviction and Kelly Criterion"""
        # Kelly Criterion simplified
        win_prob = breakout_prob
        win_amount = expected_move
        loss_amount = self.params.stop_loss_pct
        
        kelly_fraction = (win_prob * win_amount - (1 - win_prob) * loss_amount) / win_amount
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Adjust by base position size
        position_value = self.broker.get_value() * self.params.position_size_pct * kelly_fraction
        
        # Calculate shares (would need current price in production)
        current_price = self.datas[0].close[0]
        shares = int(position_value / current_price)
        
        return shares
    
    def _estimate_market_cap(self, ticker: str, df: pd.DataFrame) -> float:
        """Estimate market cap (in production, use real data)"""
        # Simplified estimation based on price and volume
        avg_price = df['close'].mean()
        avg_volume = df['volume'].mean()
        
        # Very rough estimation (in production, use real data APIs)
        if avg_volume < 100000:
            return 50e6  # Nano cap estimate
        elif avg_volume < 1000000:
            return 300e6  # Micro cap estimate
        else:
            return 1e9  # Small cap estimate
    
    def _save_optimization_state(self):
        """Save current optimization state"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'min_breakout_probability': self.params.min_breakout_probability,
                'min_expected_move': self.params.min_expected_move,
                'stop_loss_pct': self.params.stop_loss_pct
            },
            'performance': {
                'total_trades': self.total_trades,
                'win_rate': self.winning_trades / max(self.total_trades, 1),
                'total_return': self.total_return
            },
            'recent_trades': self.learning_buffer[-20:]
        }
        
        if self.gcs:
            self.gcs.upload_json(
                state,
                f"optimization_state/state_{datetime.now().strftime('%Y%m%d')}.json"
            )


class BreakoutScreener:
    """
    Screens market for potential breakout candidates
    Focuses on nano and small cap stocks
    """
    
    def __init__(self, min_market_cap: float = 10e6, max_market_cap: float = 2e9):
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.analyzer = NetworkConsolidationAnalyzer()
        
    def scan_market(self, tickers: List[str], days_back: int = 90) -> List[StockCandidate]:
        """Scan market for consolidation candidates"""
        candidates = []
        
        for ticker in tickers:
            try:
                # Get stock data
                stock = yf.Ticker(ticker)
                df = stock.history(period=f"{days_back}d")
                
                if len(df) < 20:
                    continue
                
                # Get market cap
                info = stock.info
                market_cap = info.get('marketCap', 0)
                
                if not (self.min_market_cap <= market_cap <= self.max_market_cap):
                    continue
                
                # Check if in consolidation
                current_phase = self.analyzer.get_current_phase(df)
                
                if current_phase['status'] == 'consolidation':
                    candidate = StockCandidate(
                        ticker=ticker,
                        market_cap=market_cap,
                        sector=info.get('sector', 'Unknown'),
                        avg_volume=df['Volume'].mean(),
                        consolidation_score=current_phase.get('phase_transition_score', 0),
                        last_updated=datetime.now()
                    )
                    
                    candidates.append(candidate)
                    
                    logger.info(f"Found candidate: {ticker}, "
                              f"score={candidate.consolidation_score:.2f}")
                    
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        # Sort by consolidation score
        candidates.sort(key=lambda x: x.consolidation_score, reverse=True)
        
        return candidates
    
    def get_top_candidates(self, n: int = 20) -> List[StockCandidate]:
        """Get top N breakout candidates from pre-screened list"""
        # In production, this would query a database of pre-screened stocks
        # For now, use a predefined list of small caps
        
        nano_small_caps = [
            'BNGO', 'GEVO', 'SENS', 'PROG', 'ATOS', 'XELA', 'BBIG', 'CEI',
            'MULN', 'NILE', 'INDO', 'IMPP', 'COSM', 'RELI', 'BKTI', 'APRN',
            'ALLK', 'PRPO', 'CLOV', 'WISH', 'MILE', 'GOEV', 'RIDE', 'WKHS'
        ]
        
        return self.scan_market(nano_small_caps[:n], days_back=60)