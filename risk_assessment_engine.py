# project/risk_assessment_engine.py
"""
Risk assessment engine for signal evaluation
Provides comprehensive risk analysis for decision support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level categories"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment"""
    
    overall_risk_level: RiskLevel
    risk_score: float  # 0-100
    
    # Risk components
    market_risk: float
    liquidity_risk: float
    volatility_risk: float
    concentration_risk: float
    timing_risk: float
    
    # Risk factors
    risk_factors: List[str]
    mitigating_factors: List[str]
    
    # Recommendations
    position_size_adjustment: float  # Multiplier for position size
    recommended_stop_adjustment: float  # Adjustment to stop loss
    additional_considerations: List[str]

class RiskAssessmentEngine:
    """
    Engine for comprehensive risk assessment
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'low': 25,
            'moderate': 50,
            'elevated': 70,
            'high': 85,
            'extreme': 95
        }
        
        self.risk_weights = {
            'market_risk': 0.25,
            'liquidity_risk': 0.20,
            'volatility_risk': 0.20,
            'concentration_risk': 0.15,
            'timing_risk': 0.20
        }
    
    def assess_signal(self, signal: Any) -> RiskAssessment:
        """Assess risk for a trading signal"""
        
        # Calculate risk components
        market_risk = self._assess_market_risk(signal.market_context)
        liquidity_risk = self._assess_liquidity_risk(signal)
        volatility_risk = self._assess_volatility_risk(signal)
        concentration_risk = self._assess_concentration_risk(signal)
        timing_risk = self._assess_timing_risk(signal)
        
        # Calculate overall risk score
        risk_score = (
            market_risk * self.risk_weights['market_risk'] +
            liquidity_risk * self.risk_weights['liquidity_risk'] +
            volatility_risk * self.risk_weights['volatility_risk'] +
            concentration_risk * self.risk_weights['concentration_risk'] +
            timing_risk * self.risk_weights['timing_risk']
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            market_risk, liquidity_risk, volatility_risk,
            concentration_risk, timing_risk, signal
        )
        
        # Identify mitigating factors
        mitigating_factors = self._identify_mitigating_factors(signal)
        
        # Calculate adjustments
        position_adjustment = self._calculate_position_adjustment(risk_score)
        stop_adjustment = self._calculate_stop_adjustment(risk_score, volatility_risk)
        
        # Generate recommendations
        considerations = self._generate_considerations(
            risk_level, risk_factors, signal
        )
        
        return RiskAssessment(
            overall_risk_level=risk_level,
            risk_score=risk_score,
            market_risk=market_risk,
            liquidity_risk=liquidity_risk,
            volatility_risk=volatility_risk,
            concentration_risk=concentration_risk,
            timing_risk=timing_risk,
            risk_factors=risk_factors,
            mitigating_factors=mitigating_factors,
            position_size_adjustment=position_adjustment,
            recommended_stop_adjustment=stop_adjustment,
            additional_considerations=considerations
        )
    
    def assess_ticker(self, ticker: str, analysis: Dict) -> Dict:
        """Assess risk for a ticker based on analysis"""
        
        risk_assessment = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'risk_components': {},
            'overall_assessment': {}
        }
        
        # Market cap risk
        market_cap = analysis.get('market_cap', 0)
        if market_cap < 100e6:
            risk_assessment['risk_components']['market_cap'] = {
                'level': 'HIGH',
                'score': 80,
                'reason': 'Micro-cap stock with high volatility risk'
            }
        elif market_cap < 500e6:
            risk_assessment['risk_components']['market_cap'] = {
                'level': 'ELEVATED',
                'score': 60,
                'reason': 'Small-cap stock with elevated risk'
            }
        else:
            risk_assessment['risk_components']['market_cap'] = {
                'level': 'MODERATE',
                'score': 40,
                'reason': 'Acceptable market cap range'
            }
        
        # Volume risk
        volume_data = analysis.get('volume', {})
        if volume_data.get('avg_volume_20d', 0) < 100000:
            risk_assessment['risk_components']['liquidity'] = {
                'level': 'HIGH',
                'score': 75,
                'reason': 'Low average volume may cause slippage'
            }
        else:
            risk_assessment['risk_components']['liquidity'] = {
                'level': 'LOW',
                'score': 25,
                'reason': 'Adequate liquidity'
            }
        
        # Technical risk
        technical = analysis.get('technical', {})
        if technical.get('volatility_regime') == 'high':
            risk_assessment['risk_components']['volatility'] = {
                'level': 'HIGH',
                'score': 70,
                'reason': 'High volatility environment'
            }
        else:
            risk_assessment['risk_components']['volatility'] = {
                'level': 'MODERATE',
                'score': 40,
                'reason': 'Normal volatility'
            }
        
        # Calculate overall risk
        risk_scores = [c['score'] for c in risk_assessment['risk_components'].values()]
        overall_score = np.mean(risk_scores)
        
        risk_assessment['overall_assessment'] = {
            'risk_score': overall_score,
            'risk_level': self._score_to_level(overall_score),
            'tradeable': overall_score < 70,
            'position_size_recommendation': self._get_position_recommendation(overall_score)
        }
        
        return risk_assessment
    
    def _assess_market_risk(self, market_context: Dict) -> float:
        """Assess market-wide risk"""
        
        risk_score = 0
        
        # VIX level
        vix = market_context.get('vix_level', 20)
        if vix > 30:
            risk_score += 30
        elif vix > 25:
            risk_score += 20
        elif vix > 20:
            risk_score += 10
        
        # Market regime
        regime = market_context.get('regime', 'normal')
        if regime == 'high_volatility':
            risk_score += 25
        elif regime == 'low_volatility':
            risk_score -= 10
        
        # Market breadth
        breadth = market_context.get('market_breadth', 'neutral')
        if breadth == 'bearish':
            risk_score += 20
        elif breadth == 'bullish':
            risk_score -= 10
        
        # SPY performance
        spy_return = market_context.get('spy_return_5d', 0)
        if spy_return < -0.05:
            risk_score += 15
        elif spy_return > 0.03:
            risk_score -= 5
        
        return max(0, min(100, risk_score))
    
    def _assess_liquidity_risk(self, signal: Any) -> float:
        """Assess liquidity risk"""
        
        risk_score = 0
        
        # Volume analysis
        volume_data = signal.volume_analysis
        avg_volume = volume_data.get('avg_volume_20d', 0)
        
        if avg_volume < 50000:
            risk_score += 40
        elif avg_volume < 100000:
            risk_score += 25
        elif avg_volume < 500000:
            risk_score += 10
        
        # Volume trend
        if not volume_data.get('volume_increasing', False):
            risk_score += 15
        
        # Market cap impact
        if hasattr(signal, 'market_cap'):
            if signal.market_cap < 100e6:
                risk_score += 20
            elif signal.market_cap < 300e6:
                risk_score += 10
        
        return min(100, risk_score)
    
    def _assess_volatility_risk(self, signal: Any) -> float:
        """Assess volatility risk"""
        
        risk_score = 0
        
        # ATR-based volatility
        technical = signal.technical_factors
        atr_percent = technical.get('atr_percent', 3)
        
        if atr_percent > 6:
            risk_score += 35
        elif atr_percent > 4:
            risk_score += 20
        elif atr_percent > 2:
            risk_score += 10
        
        # Volatility regime
        vol_regime = technical.get('volatility_regime', 'normal')
        if vol_regime == 'high':
            risk_score += 25
        elif vol_regime == 'elevated':
            risk_score += 15
        
        # Risk percentage
        if signal.targets.risk_percent > 8:
            risk_score += 20
        elif signal.targets.risk_percent > 5:
            risk_score += 10
        
        return min(100, risk_score)
    
    def _assess_concentration_risk(self, signal: Any) -> float:
        """Assess concentration risk"""
        
        # Simplified - in production would check portfolio concentration
        risk_score = 20  # Base concentration risk
        
        # Sector concentration (would check portfolio in production)
        # For now, just flag certain sectors as higher risk
        risky_sectors = ['Biotechnology', 'Pharmaceuticals', 'Energy']
        if signal.market_context.get('sector') in risky_sectors:
            risk_score += 20
        
        return min(100, risk_score)
    
    def _assess_timing_risk(self, signal: Any) -> float:
        """Assess timing risk"""
        
        risk_score = 0
        
        # Time to expiry
        time_to_expiry = (signal.signal_expiry - datetime.now()).days
        if time_to_expiry < 3:
            risk_score += 25
        elif time_to_expiry < 7:
            risk_score += 15
        
        # Expected breakout timing
        if signal.expected_breakout_days > 10:
            risk_score += 20
        elif signal.expected_breakout_days > 5:
            risk_score += 10
        
        # Market timing
        if not signal.market_context.get('favorable_for_breakouts', True):
            risk_score += 25
        
        return min(100, risk_score)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        
        if risk_score < self.risk_thresholds['low']:
            return RiskLevel.LOW
        elif risk_score < self.risk_thresholds['moderate']:
            return RiskLevel.MODERATE
        elif risk_score < self.risk_thresholds['elevated']:
            return RiskLevel.ELEVATED
        elif risk_score < self.risk_thresholds['high']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    def _identify_risk_factors(self, market_risk: float, liquidity_risk: float,
                              volatility_risk: float, concentration_risk: float,
                              timing_risk: float, signal: Any) -> List[str]:
        """Identify specific risk factors"""
        
        risk_factors = []
        
        if market_risk > 50:
            risk_factors.append("Unfavorable market conditions")
        
        if liquidity_risk > 50:
            risk_factors.append("Low liquidity may cause slippage")
        
        if volatility_risk > 50:
            risk_factors.append("High volatility environment")
        
        if concentration_risk > 50:
            risk_factors.append("Portfolio concentration risk")
        
        if timing_risk > 50:
            risk_factors.append("Timing uncertainty")
        
        if signal.targets.risk_percent > 7:
            risk_factors.append(f"Large stop loss distance ({signal.targets.risk_percent:.1f}%)")
        
        if signal.confidence < 0.7:
            risk_factors.append(f"Moderate confidence level ({signal.confidence:.1%})")
        
        return risk_factors
    
    def _identify_mitigating_factors(self, signal: Any) -> List[str]:
        """Identify mitigating factors"""
        
        mitigating_factors = []
        
        if signal.risk_reward_ratio > 3:
            mitigating_factors.append(f"Excellent risk/reward ratio ({signal.risk_reward_ratio:.1f}:1)")
        
        if signal.confidence > 0.85:
            mitigating_factors.append(f"High confidence signal ({signal.confidence:.1%})")
        
        if signal.signal_quality_score > 0.8:
            mitigating_factors.append(f"High quality signal ({signal.signal_quality_score:.2f})")
        
        if signal.consolidation_data.get('duration_days', 0) > 30:
            mitigating_factors.append("Extended consolidation period")
        
        if signal.volume_analysis.get('volume_spike', False):
            mitigating_factors.append("Volume confirmation present")
        
        if signal.historical_accuracy and signal.historical_accuracy > 0.75:
            mitigating_factors.append(f"Strong historical accuracy ({signal.historical_accuracy:.1%})")
        
        return mitigating_factors
    
    def _calculate_position_adjustment(self, risk_score: float) -> float:
        """Calculate position size adjustment based on risk"""
        
        if risk_score < 25:
            return 1.0  # Full position
        elif risk_score < 50:
            return 0.75  # 75% position
        elif risk_score < 70:
            return 0.5  # 50% position
        elif risk_score < 85:
            return 0.25  # 25% position
        else:
            return 0.1  # Minimal position
    
    def _calculate_stop_adjustment(self, risk_score: float, 
                                  volatility_risk: float) -> float:
        """Calculate stop loss adjustment"""
        
        # In high risk, widen stops to avoid premature exit
        if risk_score > 70 or volatility_risk > 70:
            return 1.2  # Widen stop by 20%
        elif risk_score > 50 or volatility_risk > 50:
            return 1.1  # Widen stop by 10%
        else:
            return 1.0  # Normal stop
    
    def _generate_considerations(self, risk_level: RiskLevel, 
                                risk_factors: List[str], signal: Any) -> List[str]:
        """Generate additional considerations"""
        
        considerations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME]:
            considerations.append("Consider paper trading or very small position")
            considerations.append("Set wider stops to account for volatility")
            considerations.append("Monitor position closely")
        
        elif risk_level == RiskLevel.ELEVATED:
            considerations.append("Use reduced position size")
            considerations.append("Consider scaling into position")
        
        if 'Low liquidity' in ' '.join(risk_factors):
            considerations.append("Use limit orders to control entry")
            considerations.append("Plan exit strategy in advance")
        
        if signal.time_horizon.value == 'INTRADAY':
            considerations.append("Requires active monitoring")
            considerations.append("Have clear exit plan for end of day")
        
        return considerations
    
    def _score_to_level(self, score: float) -> str:
        """Convert risk score to level string"""
        
        if score < 25:
            return "LOW"
        elif score < 50:
            return "MODERATE"
        elif score < 70:
            return "ELEVATED"
        elif score < 85:
            return "HIGH"
        else:
            return "EXTREME"
    
    def _get_position_recommendation(self, risk_score: float) -> str:
        """Get position size recommendation"""
        
        if risk_score < 25:
            return "Full position (100%)"
        elif risk_score < 50:
            return "Moderate position (50-75%)"
        elif risk_score < 70:
            return "Small position (25-50%)"
        else:
            return "Minimal or no position"
