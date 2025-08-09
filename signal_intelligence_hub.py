# project/signal_intelligence_hub.py
"""
Zentrale Signal-Generierung und Analyse-Engine
Transformation von autonomem Trading zu Decision Support
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import json
from scipy import stats
import yfinance as yf

from .consolidation_network import NetworkConsolidationAnalyzer
from .breakout_strategy import BreakoutPredictor, BreakoutScreener
from .features_optimized import OptimizedFeatureEngine
from .storage import get_gcs_storage
from .config import Config
from .monitoring import SIGNAL_GENERATION, SIGNAL_CONFIDENCE

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength categories based on confidence"""
    VERY_STRONG = "VERY_STRONG"     # > 90% confidence
    STRONG = "STRONG"               # 75-90% confidence  
    MODERATE = "MODERATE"           # 60-75% confidence
    WEAK = "WEAK"                   # 45-60% confidence
    NEUTRAL = "NEUTRAL"             # < 45% confidence

class SignalType(Enum):
    """Types of trading signals"""
    BREAKOUT_IMMINENT = "BREAKOUT_IMMINENT"
    CONSOLIDATION_COMPLETE = "CONSOLIDATION_COMPLETE"
    VOLUME_SURGE = "VOLUME_SURGE"
    MOMENTUM_SHIFT = "MOMENTUM_SHIFT"
    RISK_WARNING = "RISK_WARNING"
    PATTERN_DETECTED = "PATTERN_DETECTED"
    ML_PREDICTION = "ML_PREDICTION"

class TimeHorizon(Enum):
    """Expected time horizon for signal"""
    INTRADAY = "INTRADAY"           # < 1 day
    SHORT_TERM = "SHORT_TERM"       # 1-5 days
    MEDIUM_TERM = "MEDIUM_TERM"     # 5-20 days
    LONG_TERM = "LONG_TERM"         # > 20 days

@dataclass
class PriceTargets:
    """Price targets with probability estimates"""
    entry: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: Optional[float] = None
    
    # Probability of reaching each target
    prob_target_1: float = 0.0
    prob_target_2: float = 0.0
    prob_target_3: float = 0.0
    
    # Risk metrics
    risk_amount: float = 0.0
    risk_percent: float = 0.0
    
    # Reward metrics
    reward_1: float = 0.0
    reward_2: float = 0.0
    reward_3: float = 0.0

@dataclass
class TradingSignal:
    """Comprehensive trading signal with all supporting data"""
    # Core identification
    signal_id: str
    ticker: str
    timestamp: datetime
    signal_type: SignalType
    strength: SignalStrength
    confidence: float
    
    # Price targets and risk management
    targets: PriceTargets
    risk_reward_ratio: float
    expected_return: float
    
    # Time expectations
    time_horizon: TimeHorizon
    expected_breakout_days: int
    signal_expiry: datetime
    
    # Supporting evidence
    technical_factors: Dict = field(default_factory=dict)
    ml_predictions: Dict = field(default_factory=dict)
    market_context: Dict = field(default_factory=dict)
    consolidation_data: Dict = field(default_factory=dict)
    volume_analysis: Dict = field(default_factory=dict)
    
    # Actionable insights
    recommendation: str = ""
    action_items: List[str] = field(default_factory=list)
    key_levels: List[float] = field(default_factory=list)
    watch_conditions: List[str] = field(default_factory=list)
    
    # Quality metrics
    signal_quality_score: float = 0.0
    historical_accuracy: Optional[float] = None
    similar_patterns_performance: Optional[Dict] = None
    
    # Meta information
    model_version: str = ""
    optimization_cycle: int = 0
    generated_by: str = "signal_intelligence_hub"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['signal_expiry'] = self.signal_expiry.isoformat()
        data['signal_type'] = self.signal_type.value
        data['strength'] = self.strength.value
        data['time_horizon'] = self.time_horizon.value
        return data

class SignalIntelligenceHub:
    """
    Central hub for intelligent signal generation and analysis
    Replaces autonomous trading with decision support
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Core components
        self.analyzer = NetworkConsolidationAnalyzer()
        self.screener = BreakoutScreener()
        self.feature_engine = OptimizedFeatureEngine()
        self.predictor = None  # Will be loaded on demand
        
        # Signal management
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history: List[TradingSignal] = []
        self.signal_performance: Dict[str, float] = {}
        
        # Thresholds and parameters
        self.min_confidence = self.config.get('min_confidence', 0.65)
        self.min_consolidation_days = self.config.get('min_consolidation_days', 20)
        self.max_active_signals = self.config.get('max_active_signals', 50)
        
        # Storage
        self.gcs = get_gcs_storage()
        
        logger.info("Signal Intelligence Hub initialized")
    
    async def generate_comprehensive_signals(self, 
                                            candidates: Optional[List[str]] = None,
                                            filters: Optional[Dict] = None) -> List[TradingSignal]:
        """
        Generate comprehensive trading signals with full analysis
        
        Args:
            candidates: List of tickers to analyze (None = scan market)
            filters: Optional filters for signal generation
        
        Returns:
            List of high-quality trading signals
        """
        try:
            # Get candidates if not provided
            if candidates is None:
                candidates = await self._get_market_candidates(filters)
            
            logger.info(f"Analyzing {len(candidates)} candidates for signals")
            
            # Parallel analysis of all candidates
            tasks = [self._analyze_ticker(ticker) for ticker in candidates]
            analyses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Generate signals from analyses
            signals = []
            for analysis in analyses:
                if isinstance(analysis, Exception):
                    logger.error(f"Analysis failed: {analysis}")
                    continue
                    
                if analysis and analysis.get('should_signal', False):
                    signal = self._create_comprehensive_signal(analysis)
                    if signal:
                        signals.append(signal)
            
            # Enhance signals with additional context
            signals = await self._enhance_signals(signals)
            
            # Rank and filter signals
            signals = self._rank_and_filter_signals(signals, filters)
            
            # Store signals
            self._store_signals(signals)
            
            # Update metrics
            SIGNAL_GENERATION.labels(status='success').inc()
            
            logger.info(f"Generated {len(signals)} high-quality signals")
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            SIGNAL_GENERATION.labels(status='failed').inc()
            return []
    
    async def _analyze_ticker(self, ticker: str) -> Optional[Dict]:
        """
        Comprehensive analysis of a single ticker
        """
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            df = stock.history(period="90d")
            
            if len(df) < 60:
                return None
            
            info = stock.info
            market_cap = info.get('marketCap', 0)
            
            # Skip if outside target range
            if not (10e6 <= market_cap <= 2e9):
                return None
            
            # Comprehensive analysis
            analysis = {
                'ticker': ticker,
                'market_cap': market_cap,
                'sector': info.get('sector', 'Unknown'),
                'current_price': df['Close'].iloc[-1],
                'df': df
            }
            
            # 1. Consolidation analysis
            analysis['consolidation'] = await self._analyze_consolidation(ticker, df, market_cap)
            
            # 2. ML predictions
            analysis['ml_predictions'] = await self._get_ml_predictions(ticker, df)
            
            # 3. Technical analysis
            analysis['technical'] = await self._technical_analysis(df)
            
            # 4. Volume analysis
            analysis['volume'] = await self._volume_analysis(df)
            
            # 5. Market regime context
            analysis['market_regime'] = await self._get_market_regime()
            
            # 6. Pattern recognition
            analysis['patterns'] = await self._detect_patterns(df)
            
            # Decision logic
            analysis['should_signal'] = self._evaluate_signal_criteria(analysis)
            analysis['confidence'] = self._calculate_comprehensive_confidence(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")
            return None
    
    async def _analyze_consolidation(self, ticker: str, df: pd.DataFrame, 
                                    market_cap: float) -> Dict:
        """Analyze consolidation patterns"""
        try:
            # Extract consolidation features
            features = self.feature_engine.create_features_optimized(df)
            
            # Analyze with network model
            consolidations = self.analyzer.analyze_consolidation(df, market_cap)
            
            # Get current phase
            current_phase = self.analyzer.get_current_phase(df)
            
            if not consolidations:
                return {
                    'in_consolidation': False,
                    'phase': current_phase,
                    'score': 0.0
                }
            
            latest = consolidations[-1]
            
            return {
                'in_consolidation': True,
                'duration_days': latest.duration_days,
                'price_range': latest.price_range,
                'volume_pattern': latest.volume_pattern,
                'network_density': latest.network_density,
                'phase_transition_score': latest.phase_transition_score,
                'accumulation_score': latest.accumulation_score,
                'breakout_probability': latest.breakout_probability,
                'expected_move': latest.expected_move,
                'current_phase': current_phase,
                'consolidation_quality': self._assess_consolidation_quality(latest)
            }
            
        except Exception as e:
            logger.error(f"Consolidation analysis failed: {e}")
            return {'in_consolidation': False, 'score': 0.0}
    
    async def _get_ml_predictions(self, ticker: str, df: pd.DataFrame) -> Dict:
        """Get ML model predictions"""
        try:
            # Load model if needed
            if self.predictor is None:
                self._load_predictor()
            
            if self.predictor is None:
                return {'available': False}
            
            # Prepare features
            features = self.feature_engine.create_features_optimized(df)
            
            # Get predictions (simplified - in production use actual model)
            # This is a placeholder for the actual ML prediction logic
            predictions = {
                'breakout_probability': np.random.uniform(0.5, 0.95),
                'expected_magnitude': np.random.uniform(0.15, 0.50),
                'timing_days': np.random.randint(2, 10),
                'confidence_interval': (0.10, 0.40),
                'model_confidence': np.random.uniform(0.60, 0.90)
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return {'available': False}
    
    async def _technical_analysis(self, df: pd.DataFrame) -> Dict:
        """Comprehensive technical analysis"""
        try:
            current_price = df['Close'].iloc[-1]
            
            # Basic metrics
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
            
            # ATR for volatility
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Key support/resistance levels
            key_levels = self._calculate_key_levels(df)
            
            # Trend analysis
            trend = self._analyze_trend(df)
            
            return {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'atr': atr,
                'atr_percent': (atr / current_price) * 100,
                'rsi': rsi,
                'key_levels': key_levels,
                'support': key_levels[0] if key_levels else current_price * 0.95,
                'resistance': key_levels[-1] if key_levels else current_price * 1.05,
                'trend': trend,
                'volatility_regime': self._classify_volatility(atr / current_price)
            }
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {}
    
    async def _volume_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        try:
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            
            # Volume trend
            volume_sma_5 = df['Volume'].rolling(5).mean().iloc[-1]
            volume_increasing = volume_sma_5 > avg_volume
            
            # Volume spikes
            volume_std = df['Volume'].rolling(20).std().iloc[-1]
            volume_zscore = (current_volume - avg_volume) / volume_std if volume_std > 0 else 0
            
            # On-Balance Volume
            obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
            obv_trend = 'bullish' if obv.iloc[-1] > obv.iloc[-5] else 'bearish'
            
            # Volume profile
            volume_profile = self._calculate_volume_profile(df)
            
            return {
                'current_volume': current_volume,
                'avg_volume_20d': avg_volume,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
                'volume_increasing': volume_increasing,
                'volume_zscore': volume_zscore,
                'volume_spike': volume_zscore > 2,
                'obv_trend': obv_trend,
                'volume_profile': volume_profile,
                'accumulation_distribution': self._calculate_accumulation_distribution(df)
            }
            
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            return {}
    
    async def _get_market_regime(self) -> Dict:
        """Get current market regime context"""
        try:
            # Get market indices
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="30d")
            
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="5d")
            
            # Market metrics
            spy_return_5d = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-6] - 1)
            spy_return_20d = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-21] - 1)
            vix_level = vix_data['Close'].iloc[-1] if len(vix_data) > 0 else 20
            
            # Classify regime
            if vix_level > 30:
                regime = "high_volatility"
            elif vix_level < 15:
                regime = "low_volatility"
            else:
                regime = "normal"
            
            # Market breadth (simplified)
            if spy_return_5d > 0.02:
                breadth = "bullish"
            elif spy_return_5d < -0.02:
                breadth = "bearish"
            else:
                breadth = "neutral"
            
            return {
                'regime': regime,
                'vix_level': vix_level,
                'spy_return_5d': spy_return_5d,
                'spy_return_20d': spy_return_20d,
                'market_breadth': breadth,
                'risk_on': vix_level < 20 and spy_return_5d > 0,
                'favorable_for_breakouts': regime != "high_volatility" and breadth != "bearish"
            }
            
        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            return {'regime': 'unknown', 'favorable_for_breakouts': True}
    
    async def _detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect chart patterns"""
        patterns = {
            'cup_and_handle': self._detect_cup_and_handle(df),
            'ascending_triangle': self._detect_ascending_triangle(df),
            'bull_flag': self._detect_bull_flag(df),
            'volume_dry_up': self._detect_volume_dry_up(df)
        }
        
        patterns['patterns_detected'] = [k for k, v in patterns.items() if v]
        patterns['pattern_count'] = len(patterns['patterns_detected'])
        
        return patterns
    
    def _evaluate_signal_criteria(self, analysis: Dict) -> bool:
        """Evaluate if analysis meets signal criteria"""
        
        # Must be in consolidation
        if not analysis.get('consolidation', {}).get('in_consolidation', False):
            return False
        
        # Minimum consolidation duration
        if analysis['consolidation'].get('duration_days', 0) < self.min_consolidation_days:
            return False
        
        # Minimum phase transition score
        if analysis['consolidation'].get('phase_transition_score', 0) < 0.6:
            return False
        
        # ML prediction threshold
        ml_prob = analysis.get('ml_predictions', {}).get('breakout_probability', 0)
        if ml_prob < 0.5:
            return False
        
        # Volume confirmation
        if not analysis.get('volume', {}).get('volume_increasing', False):
            return False
        
        # Market regime check
        if not analysis.get('market_regime', {}).get('favorable_for_breakouts', True):
            return False
        
        return True
    
    def _calculate_comprehensive_confidence(self, analysis: Dict) -> float:
        """Calculate overall signal confidence"""
        
        weights = {
            'consolidation': 0.30,
            'ml_prediction': 0.25,
            'technical': 0.15,
            'volume': 0.15,
            'patterns': 0.10,
            'market_regime': 0.05
        }
        
        scores = {}
        
        # Consolidation score
        cons = analysis.get('consolidation', {})
        scores['consolidation'] = cons.get('breakout_probability', 0)
        
        # ML prediction score
        ml = analysis.get('ml_predictions', {})
        scores['ml_prediction'] = ml.get('breakout_probability', 0) * ml.get('model_confidence', 1)
        
        # Technical score
        tech = analysis.get('technical', {})
        rsi = tech.get('rsi', 50)
        scores['technical'] = 0.5 + (0.5 if 40 <= rsi <= 60 else 0)
        
        # Volume score
        vol = analysis.get('volume', {})
        scores['volume'] = min(vol.get('volume_ratio', 1), 2) / 2
        
        # Pattern score
        patterns = analysis.get('patterns', {})
        scores['patterns'] = min(patterns.get('pattern_count', 0) / 3, 1)
        
        # Market regime score
        regime = analysis.get('market_regime', {})
        scores['market_regime'] = 1.0 if regime.get('favorable_for_breakouts', False) else 0.5
        
        # Weighted average
        confidence = sum(scores[k] * weights[k] for k in weights.keys())
        
        return min(max(confidence, 0), 1)
    
    def _create_comprehensive_signal(self, analysis: Dict) -> Optional[TradingSignal]:
        """Create comprehensive trading signal from analysis"""
        try:
            ticker = analysis['ticker']
            current_price = analysis['current_price']
            
            # Calculate targets and stops
            targets = self._calculate_intelligent_targets(analysis)
            
            # Determine signal characteristics
            signal_type = self._determine_signal_type(analysis)
            confidence = analysis['confidence']
            strength = self._determine_strength(confidence)
            time_horizon = self._determine_time_horizon(analysis)
            
            # Generate recommendations
            recommendation = self._generate_recommendation(analysis, targets)
            action_items = self._generate_action_items(analysis, targets)
            watch_conditions = self._generate_watch_conditions(analysis)
            
            # Calculate signal quality
            quality_score = self._calculate_signal_quality(analysis)
            
            # Create signal
            signal = TradingSignal(
                signal_id=f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                ticker=ticker,
                timestamp=datetime.now(),
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                targets=targets,
                risk_reward_ratio=self._calculate_risk_reward(targets),
                expected_return=analysis['ml_predictions'].get('expected_magnitude', 0.2),
                time_horizon=time_horizon,
                expected_breakout_days=analysis['ml_predictions'].get('timing_days', 5),
                signal_expiry=datetime.now() + timedelta(days=10),
                technical_factors=analysis.get('technical', {}),
                ml_predictions=analysis.get('ml_predictions', {}),
                market_context=analysis.get('market_regime', {}),
                consolidation_data=analysis.get('consolidation', {}),
                volume_analysis=analysis.get('volume', {}),
                recommendation=recommendation,
                action_items=action_items,
                key_levels=analysis['technical'].get('key_levels', []),
                watch_conditions=watch_conditions,
                signal_quality_score=quality_score,
                model_version="v2025.1.0",
                optimization_cycle=self.config.get('optimization_cycle', 0)
            )
            
            # Update confidence metric
            SIGNAL_CONFIDENCE.labels(
                ticker=ticker,
                signal_type=signal_type.value
            ).set(confidence)
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to create signal: {e}")
            return None
    
    def _calculate_intelligent_targets(self, analysis: Dict) -> PriceTargets:
        """Calculate intelligent price targets"""
        
        current_price = analysis['current_price']
        atr = analysis['technical']['atr']
        expected_move = analysis['consolidation'].get('expected_move', 0.25)
        
        # Stop loss based on ATR and support
        support = analysis['technical'].get('support', current_price * 0.95)
        stop_loss = max(
            support - (0.5 * atr),  # Below support
            current_price - (2 * atr),  # 2 ATR stop
            current_price * 0.92  # Maximum 8% stop
        )
        
        # Targets based on expected move and resistance levels
        resistance = analysis['technical'].get('resistance', current_price * 1.1)
        
        target_1 = min(
            current_price * (1 + expected_move * 0.5),  # Conservative target
            resistance
        )
        
        target_2 = current_price * (1 + expected_move)  # Expected move target
        
        target_3 = current_price * (1 + expected_move * 1.5) if expected_move > 0.3 else None
        
        # Calculate probabilities (simplified)
        ml_conf = analysis['ml_predictions'].get('model_confidence', 0.7)
        
        targets = PriceTargets(
            entry=current_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            target_3=target_3,
            prob_target_1=min(ml_conf + 0.2, 0.95),
            prob_target_2=ml_conf,
            prob_target_3=max(ml_conf - 0.2, 0.3) if target_3 else 0,
            risk_amount=current_price - stop_loss,
            risk_percent=((current_price - stop_loss) / current_price) * 100,
            reward_1=target_1 - current_price,
            reward_2=target_2 - current_price,
            reward_3=(target_3 - current_price) if target_3 else 0
        )
        
        return targets
    
    def _determine_signal_type(self, analysis: Dict) -> SignalType:
        """Determine primary signal type"""
        
        # Check for volume surge
        if analysis['volume'].get('volume_spike', False):
            return SignalType.VOLUME_SURGE
        
        # Check for pattern completion
        if analysis['patterns'].get('pattern_count', 0) > 0:
            return SignalType.PATTERN_DETECTED
        
        # Check consolidation completion
        if analysis['consolidation'].get('phase_transition_score', 0) > 0.8:
            return SignalType.CONSOLIDATION_COMPLETE
        
        # Check for breakout imminent
        if analysis['ml_predictions'].get('timing_days', 10) < 3:
            return SignalType.BREAKOUT_IMMINENT
        
        # Default to ML prediction
        return SignalType.ML_PREDICTION
    
    def _determine_strength(self, confidence: float) -> SignalStrength:
        """Determine signal strength from confidence"""
        
        if confidence >= 0.90:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.75:
            return SignalStrength.STRONG
        elif confidence >= 0.60:
            return SignalStrength.MODERATE
        elif confidence >= 0.45:
            return SignalStrength.WEAK
        else:
            return SignalStrength.NEUTRAL
    
    def _determine_time_horizon(self, analysis: Dict) -> TimeHorizon:
        """Determine expected time horizon"""
        
        expected_days = analysis['ml_predictions'].get('timing_days', 5)
        
        if expected_days <= 1:
            return TimeHorizon.INTRADAY
        elif expected_days <= 5:
            return TimeHorizon.SHORT_TERM
        elif expected_days <= 20:
            return TimeHorizon.MEDIUM_TERM
        else:
            return TimeHorizon.LONG_TERM
    
    def _generate_recommendation(self, analysis: Dict, targets: PriceTargets) -> str:
        """Generate human-readable recommendation"""
        
        confidence = analysis['confidence']
        rr_ratio = self._calculate_risk_reward(targets)
        signal_type = self._determine_signal_type(analysis)
        ticker = analysis['ticker']
        
        # Strong buy signal
        if confidence > 0.85 and rr_ratio > 3:
            return (f"🟢 STRONG BUY SIGNAL for {ticker}: High-confidence {signal_type.value} setup "
                   f"with excellent risk/reward ratio of {rr_ratio:.1f}:1. "
                   f"Entry near ${targets.entry:.2f} with stop at ${targets.stop_loss:.2f}. "
                   f"Initial target ${targets.target_1:.2f} ({targets.prob_target_1:.0%} probability). "
                   "Consider full position size with disciplined stop-loss management.")
        
        # Moderate buy signal
        elif confidence > 0.70 and rr_ratio > 2:
            return (f"🟡 MODERATE BUY SIGNAL for {ticker}: Good {signal_type.value} setup "
                   f"with favorable risk/reward of {rr_ratio:.1f}:1. "
                   f"Entry near ${targets.entry:.2f}, stop at ${targets.stop_loss:.2f}. "
                   f"Target ${targets.target_1:.2f} likely ({targets.prob_target_1:.0%}). "
                   "Consider 50-75% position size, scale in if confirmation appears.")
        
        # Watch signal
        elif confidence > 0.60:
            return (f"🔵 WATCH CLOSELY - {ticker}: Potential {signal_type.value} developing. "
                   f"Monitor for entry near ${targets.entry:.2f}. Key resistance at ${targets.target_1:.2f}. "
                   "Wait for volume confirmation or price action above key levels before entry.")
        
        # Weak signal
        else:
            return (f"⚪ MONITORING - {ticker}: Early-stage setup detected. "
                   "Insufficient conviction for immediate action. Continue observation for better entry.")
    
    def _generate_action_items(self, analysis: Dict, targets: PriceTargets) -> List[str]:
        """Generate specific action items"""
        
        actions = []
        confidence = analysis['confidence']
        ticker = analysis['ticker']
        
        if confidence > 0.75:
            actions.append(f"Set BUY alert at ${targets.entry:.2f}")
            actions.append(f"Place stop-loss order at ${targets.stop_loss:.2f} after entry")
            actions.append(f"Set profit target alerts at ${targets.target_1:.2f} and ${targets.target_2:.2f}")
        
        if analysis['volume'].get('volume_spike', False):
            actions.append("Monitor for sustained volume increase")
        
        if analysis['consolidation'].get('phase_transition_score', 0) > 0.7:
            actions.append("Watch for breakout confirmation above resistance")
        
        if analysis['patterns'].get('pattern_count', 0) > 0:
            patterns = analysis['patterns'].get('patterns_detected', [])
            actions.append(f"Monitor pattern completion: {', '.join(patterns)}")
        
        return actions
    
    def _generate_watch_conditions(self, analysis: Dict) -> List[str]:
        """Generate conditions to watch for"""
        
        conditions = []
        
        # Volume conditions
        if not analysis['volume'].get('volume_spike', False):
            conditions.append("Volume surge above 2x average")
        
        # Price conditions
        resistance = analysis['technical'].get('resistance', 0)
        if resistance:
            conditions.append(f"Price break above ${resistance:.2f}")
        
        # Technical conditions
        rsi = analysis['technical'].get('rsi', 50)
        if rsi < 40:
            conditions.append("RSI recovery above 40")
        elif rsi > 70:
            conditions.append("RSI cooling below 70")
        
        # Market conditions
        if not analysis['market_regime'].get('favorable_for_breakouts', True):
            conditions.append("Market regime improvement (VIX < 20)")
        
        return conditions
    
    def _calculate_risk_reward(self, targets: PriceTargets) -> float:
        """Calculate risk/reward ratio"""
        
        risk = targets.risk_amount
        reward = targets.reward_1  # Use first target for conservative RR
        
        if risk > 0:
            return reward / risk
        return 0
    
    def _calculate_signal_quality(self, analysis: Dict) -> float:
        """Calculate overall signal quality score"""
        
        factors = {
            'consolidation_quality': analysis['consolidation'].get('consolidation_quality', 0.5),
            'ml_confidence': analysis['ml_predictions'].get('model_confidence', 0.5),
            'volume_confirmation': 1.0 if analysis['volume'].get('volume_spike', False) else 0.5,
            'pattern_strength': min(analysis['patterns'].get('pattern_count', 0) / 2, 1),
            'market_alignment': 1.0 if analysis['market_regime'].get('favorable_for_breakouts', False) else 0.3
        }
        
        # Weighted average
        weights = {'consolidation_quality': 0.3, 'ml_confidence': 0.25, 
                  'volume_confirmation': 0.2, 'pattern_strength': 0.15, 
                  'market_alignment': 0.1}
        
        quality = sum(factors[k] * weights[k] for k in weights.keys())
        
        return min(max(quality, 0), 1)
    
    def _assess_consolidation_quality(self, consolidation) -> float:
        """Assess quality of consolidation pattern"""
        
        quality_factors = {
            'duration': min(consolidation.duration_days / 30, 1),  # Longer is better up to 30 days
            'tightness': 1 - min(consolidation.price_range / 0.15, 1),  # Tighter is better
            'volume_pattern': 1.0 if consolidation.volume_pattern == 'declining' else 0.5,
            'accumulation': consolidation.accumulation_score,
            'network_density': consolidation.network_density
        }
        
        return np.mean(list(quality_factors.values()))
    
    def _calculate_key_levels(self, df: pd.DataFrame) -> List[float]:
        """Calculate key support/resistance levels"""
        
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find local maxima and minima
        from scipy.signal import argrelextrema
        
        local_maxima = argrelextrema(highs, np.greater, order=5)[0]
        local_minima = argrelextrema(lows, np.less, order=5)[0]
        
        key_levels = []
        
        if len(local_maxima) > 0:
            key_levels.extend(highs[local_maxima[-3:]])  # Last 3 resistance levels
        
        if len(local_minima) > 0:
            key_levels.extend(lows[local_minima[-3:]])  # Last 3 support levels
        
        # Add psychological levels
        current_price = closes[-1]
        key_levels.extend([
            round(current_price / 5) * 5,  # Nearest $5
            round(current_price / 10) * 10  # Nearest $10
        ])
        
        # Sort and remove duplicates
        key_levels = sorted(list(set(key_levels)))
        
        return key_levels
    
    def _analyze_trend(self, df: pd.DataFrame) -> str:
        """Analyze price trend"""
        
        closes = df['Close'].values
        
        # Simple trend classification
        sma_20 = df['Close'].rolling(20).mean().values
        sma_50 = df['Close'].rolling(50).mean().values if len(df) >= 50 else sma_20
        
        if closes[-1] > sma_20[-1] > sma_50[-1]:
            return "strong_uptrend"
        elif closes[-1] > sma_20[-1]:
            return "uptrend"
        elif closes[-1] < sma_20[-1] < sma_50[-1]:
            return "strong_downtrend"
        elif closes[-1] < sma_20[-1]:
            return "downtrend"
        else:
            return "sideways"
    
    def _classify_volatility(self, atr_percent: float) -> str:
        """Classify volatility regime"""
        
        if atr_percent < 2:
            return "low"
        elif atr_percent < 4:
            return "normal"
        elif atr_percent < 6:
            return "elevated"
        else:
            return "high"
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Calculate volume profile"""
        
        # Simplified volume profile
        price_bins = pd.qcut(df['Close'], q=10, duplicates='drop')
        volume_by_price = df.groupby(price_bins)['Volume'].sum()
        
        # Find high volume node (price with most volume)
        hvn = volume_by_price.idxmax()
        
        return {
            'high_volume_node': float(hvn.mid) if hasattr(hvn, 'mid') else df['Close'].median(),
            'volume_distribution': 'normal'  # Simplified
        }
    
    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> float:
        """Calculate Accumulation/Distribution indicator"""
        
        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mfm = mfm.fillna(0)
        
        ad = (mfm * df['Volume']).cumsum()
        
        # Trend of A/D line
        if len(ad) >= 5:
            recent_trend = 1 if ad.iloc[-1] > ad.iloc[-5] else -1
        else:
            recent_trend = 0
        
        return recent_trend
    
    def _detect_cup_and_handle(self, df: pd.DataFrame) -> bool:
        """Detect cup and handle pattern"""
        # Simplified detection
        return False
    
    def _detect_ascending_triangle(self, df: pd.DataFrame) -> bool:
        """Detect ascending triangle pattern"""
        # Simplified detection
        return False
    
    def _detect_bull_flag(self, df: pd.DataFrame) -> bool:
        """Detect bull flag pattern"""
        # Simplified detection
        return False
    
    def _detect_volume_dry_up(self, df: pd.DataFrame) -> bool:
        """Detect volume dry-up pattern"""
        
        recent_volume = df['Volume'].iloc[-5:].mean()
        prior_volume = df['Volume'].iloc[-20:-5].mean()
        
        return recent_volume < prior_volume * 0.5
    
    async def _get_market_candidates(self, filters: Optional[Dict] = None) -> List[str]:
        """Get market candidates for signal generation"""
        
        # Use screener to get candidates
        candidates = self.screener.get_top_candidates(n=100)
        
        # Extract tickers
        tickers = [c.ticker for c in candidates]
        
        # Apply filters if provided
        if filters:
            # Filter by market cap
            if 'min_market_cap' in filters:
                tickers = [t for t in tickers 
                          if self._get_market_cap(t) >= filters['min_market_cap']]
            
            if 'max_market_cap' in filters:
                tickers = [t for t in tickers 
                          if self._get_market_cap(t) <= filters['max_market_cap']]
            
            # Filter by sector
            if 'sectors' in filters and filters['sectors']:
                tickers = [t for t in tickers 
                          if self._get_sector(t) in filters['sectors']]
        
        return tickers[:50]  # Limit to top 50
    
    def _get_market_cap(self, ticker: str) -> float:
        """Get market cap for ticker (cached)"""
        # In production, implement caching
        try:
            info = yf.Ticker(ticker).info
            return info.get('marketCap', 0)
        except:
            return 0
    
    def _get_sector(self, ticker: str) -> str:
        """Get sector for ticker (cached)"""
        # In production, implement caching
        try:
            info = yf.Ticker(ticker).info
            return info.get('sector', 'Unknown')
        except:
            return 'Unknown'
    
    async def _enhance_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Enhance signals with additional context"""
        
        for signal in signals:
            # Add historical performance of similar signals
            signal.historical_accuracy = await self._get_historical_accuracy(signal)
            
            # Add similar pattern performance
            signal.similar_patterns_performance = await self._get_similar_patterns_performance(signal)
        
        return signals
    
    async def _get_historical_accuracy(self, signal: TradingSignal) -> float:
        """Get historical accuracy for similar signals"""
        # Simplified - in production query historical database
        return np.random.uniform(0.6, 0.85)
    
    async def _get_similar_patterns_performance(self, signal: TradingSignal) -> Dict:
        """Get performance of similar historical patterns"""
        # Simplified - in production query pattern database
        return {
            'similar_patterns_found': np.random.randint(5, 20),
            'average_return': np.random.uniform(0.15, 0.35),
            'win_rate': np.random.uniform(0.60, 0.80)
        }
    
    def _rank_and_filter_signals(self, signals: List[TradingSignal], 
                                filters: Optional[Dict] = None) -> List[TradingSignal]:
        """Rank and filter signals"""
        
        # Apply filters
        if filters:
            if 'min_confidence' in filters:
                signals = [s for s in signals if s.confidence >= filters['min_confidence']]
            
            if 'min_risk_reward' in filters:
                signals = [s for s in signals if s.risk_reward_ratio >= filters['min_risk_reward']]
            
            if 'signal_types' in filters:
                signals = [s for s in signals if s.signal_type in filters['signal_types']]
        
        # Rank by composite score
        def signal_score(signal):
            return (
                signal.confidence * 0.3 +
                min(signal.risk_reward_ratio / 5, 1) * 0.3 +
                signal.signal_quality_score * 0.2 +
                (signal.historical_accuracy or 0.5) * 0.2
            )
        
        signals.sort(key=signal_score, reverse=True)
        
        # Limit to max active signals
        return signals[:self.max_active_signals]
    
    def _store_signals(self, signals: List[TradingSignal]):
        """Store signals for tracking"""
        
        timestamp = datetime.now()
        
        for signal in signals:
            # Add to active signals
            self.active_signals[signal.signal_id] = signal
            
            # Add to history
            self.signal_history.append(signal)
            
            # Store to GCS if available
            if self.gcs:
                try:
                    signal_data = signal.to_dict()
                    path = f"signals/{timestamp.strftime('%Y%m%d')}/{signal.signal_id}.json"
                    self.gcs.upload_json(signal_data, path)
                except Exception as e:
                    logger.error(f"Failed to store signal to GCS: {e}")
        
        # Cleanup old signals
        self._cleanup_old_signals()
    
    def _cleanup_old_signals(self):
        """Remove expired signals"""
        
        now = datetime.now()
        
        # Remove expired active signals
        expired = [sid for sid, signal in self.active_signals.items() 
                  if signal.signal_expiry < now]
        
        for sid in expired:
            del self.active_signals[sid]
        
        # Limit history size
        if len(self.signal_history) > 10000:
            self.signal_history = self.signal_history[-10000:]
    
    def _load_predictor(self):
        """Load ML predictor model"""
        try:
            # In production, load actual trained model
            logger.info("Loading ML predictor model...")
            # self.predictor = load_model_from_storage()
            pass
        except Exception as e:
            logger.error(f"Failed to load predictor: {e}")
    
    def get_active_signals(self, ticker: Optional[str] = None) -> List[TradingSignal]:
        """Get currently active signals"""
        
        if ticker:
            return [s for s in self.active_signals.values() if s.ticker == ticker]
        
        return list(self.active_signals.values())
    
    def get_signal_by_id(self, signal_id: str) -> Optional[TradingSignal]:
        """Get specific signal by ID"""
        return self.active_signals.get(signal_id)
    
    def invalidate_signal(self, signal_id: str, reason: str):
        """Invalidate a signal"""
        
        if signal_id in self.active_signals:
            signal = self.active_signals[signal_id]
            signal.signal_expiry = datetime.now()
            del self.active_signals[signal_id]
            
            logger.info(f"Signal {signal_id} invalidated: {reason}")
