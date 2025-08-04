# risk_management_regime.py
"""
Risk Management System mit Regime Detection und Emergency Response
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime, timedelta
from scipy import stats
import warnings

class MarketRegime(Enum):
    NORMAL = "NORMAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    PANIC = "PANIC"
    CORRELATION_BREAKDOWN = "CORRELATION_BREAKDOWN"
    LIQUIDITY_CRISIS = "LIQUIDITY_CRISIS"
    FLASH_CRASH = "FLASH_CRASH"
    REGIME_CHANGE = "REGIME_CHANGE"

class RiskAction(Enum):
    CONTINUE = "CONTINUE"
    REDUCE_POSITION = "REDUCE_POSITION"
    HEDGE = "HEDGE"
    LIQUIDATE_PARTIAL = "LIQUIDATE_PARTIAL"
    LIQUIDATE_ALL = "LIQUIDATE_ALL"
    HALT_TRADING = "HALT_TRADING"

@dataclass
class MarketData:
    timestamp: datetime
    vix: float
    sp500_return: float
    volume_ratio: float  # Current vol / 20day avg
    spread_percentiles: Dict[str, float]
    correlation_matrix: pd.DataFrame
    circuit_breaker_level: int  # 0=none, 1=level1, 2=level2, 3=level3
    
@dataclass
class RiskMetrics:
    portfolio_var: float
    portfolio_cvar: float
    max_drawdown: float
    current_drawdown: float
    leverage: float
    concentration_risk: float
    correlation_risk: float
    liquidity_score: float

class RegimeDetector:
    """Erkennt Marktregime und strukturelle Veränderungen"""
    
    def __init__(self):
        self.regime_history = []
        self.regime_thresholds = {
            "vix_panic": 40,
            "vix_high_vol": 25,
            "correlation_breakdown": 0.85,
            "liquidity_crisis_spread": 3.0,  # 3x normal spread
            "flash_crash_move": -0.03,  # 3% in minutes
            "volume_spike": 5.0  # 5x normal volume
        }
        
        # Hidden Markov Model für Regime Detection
        self.hmm_model = self.initialize_hmm()
        
    def initialize_hmm(self):
        """Initialisiert Hidden Markov Model für Regime Detection"""
        # Simplified HMM setup - in production würde man hmmlearn verwenden
        return {
            "states": ["normal", "stressed", "crisis"],
            "transition_matrix": np.array([
                [0.95, 0.04, 0.01],  # normal -> normal, stressed, crisis
                [0.10, 0.80, 0.10],  # stressed -> normal, stressed, crisis
                [0.05, 0.15, 0.80]   # crisis -> normal, stressed, crisis
            ]),
            "current_state": "normal"
        }
        
    def detect_regime(self, market_data: MarketData, 
                     historical_data: pd.DataFrame) -> MarketRegime:
        """Hauptmethode zur Regime-Erkennung"""
        
        # Quick checks für akute Krisen
        if market_data.circuit_breaker_level > 0:
            return MarketRegime.FLASH_CRASH
            
        if market_data.vix > self.regime_thresholds["vix_panic"]:
            return MarketRegime.PANIC
            
        # Correlation breakdown check
        avg_correlation = self.calculate_average_correlation(market_data.correlation_matrix)
        if avg_correlation > self.regime_thresholds["correlation_breakdown"]:
            return MarketRegime.CORRELATION_BREAKDOWN
            
        # Liquidity crisis check
        if self.check_liquidity_crisis(market_data):
            return MarketRegime.LIQUIDITY_CRISIS
            
        # High volatility regime
        if market_data.vix > self.regime_thresholds["vix_high_vol"]:
            return MarketRegime.HIGH_VOLATILITY
            
        # Structural regime change detection
        if self.detect_structural_break(historical_data):
            return MarketRegime.REGIME_CHANGE
            
        return MarketRegime.NORMAL
        
    def calculate_average_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """Berechnet durchschnittliche Korrelation (ohne Diagonale)"""
        mask = np.ones_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, 0)
        return corr_matrix.where(mask).stack().mean()
        
    def check_liquidity_crisis(self, market_data: MarketData) -> bool:
        """Prüft auf Liquiditätskrise"""
        spread_ratio = market_data.spread_percentiles['95th'] / market_data.spread_percentiles['median']
        volume_spike = market_data.volume_ratio > self.regime_thresholds["volume_spike"]
        
        return spread_ratio > self.regime_thresholds["liquidity_crisis_spread"] and not volume_spike
        
    def detect_structural_break(self, historical_data: pd.DataFrame, 
                               window: int = 60) -> bool:
        """Erkennt strukturelle Brüche in Zeitreihen"""
        if len(historical_data) < window * 2:
            return False
            
        # Chow Test für strukturellen Bruch
        recent_data = historical_data.iloc[-window:]
        older_data = historical_data.iloc[-2*window:-window]
        
        # Simple version - in production würde man statsmodels verwenden
        recent_mean = recent_data['returns'].mean()
        older_mean = older_data['returns'].mean()
        recent_std = recent_data['returns'].std()
        older_std = older_data['returns'].std()
        
        # Z-score für Mittelwertdifferenz
        z_score = abs(recent_mean - older_mean) / np.sqrt(recent_std**2/window + older_std**2/window)
        
        # F-test für Varianzdifferenz
        f_stat = recent_std**2 / older_std**2
        
        return z_score > 3 or f_stat > 2 or f_stat < 0.5
        
    def calculate_regime_probability(self, features: np.ndarray) -> Dict[str, float]:
        """Berechnet Wahrscheinlichkeiten für verschiedene Regime"""
        # Simplified - in production würde man ein trainiertes ML-Modell verwenden
        vix_level = features[0]
        correlation = features[1]
        
        if vix_level < 15 and correlation < 0.5:
            return {"normal": 0.8, "stressed": 0.15, "crisis": 0.05}
        elif vix_level < 25:
            return {"normal": 0.3, "stressed": 0.6, "crisis": 0.1}
        else:
            return {"normal": 0.1, "stressed": 0.3, "crisis": 0.6}

class KillSwitch:
    """Notfall-Abschaltung für extreme Situationen"""
    
    def __init__(self):
        self.is_active = False
        self.activation_time = None
        self.activation_reason = None
        self.auto_reactivation_delay = timedelta(hours=1)
        
    def activate(self, reason: str, auto_reactivate: bool = True):
        """Aktiviert Kill Switch"""
        self.is_active = True
        self.activation_time = datetime.now()
        self.activation_reason = reason
        
        print(f"🚨 KILL SWITCH ACTIVATED: {reason}")
        
        if auto_reactivate:
            asyncio.create_task(self.auto_reactivate())
            
    def deactivate(self, manual: bool = False):
        """Deaktiviert Kill Switch"""
        if manual or self.can_reactivate():
            self.is_active = False
            print("✅ Kill Switch deactivated")
        else:
            remaining_time = self.get_remaining_lockout_time()
            print(f"⏰ Cannot reactivate yet. {remaining_time} remaining")
            
    def can_reactivate(self) -> bool:
        """Prüft ob Reaktivierung möglich ist"""
        if not self.activation_time:
            return True
            
        elapsed = datetime.now() - self.activation_time
        return elapsed > self.auto_reactivation_delay
        
    def get_remaining_lockout_time(self) -> timedelta:
        """Gibt verbleibende Sperrzeit zurück"""
        if not self.activation_time:
            return timedelta(0)
            
        elapsed = datetime.now() - self.activation_time
        remaining = self.auto_reactivation_delay - elapsed
        return max(remaining, timedelta(0))
        
    async def auto_reactivate(self):
        """Automatische Reaktivierung nach Delay"""
        await asyncio.sleep(self.auto_reactivation_delay.total_seconds())
        self.deactivate()

class RiskManagementSystem:
    """Hauptklasse für Risk Management und Emergency Response"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.regime_detector = RegimeDetector()
        self.kill_switch = KillSwitch()
        self.current_regime = MarketRegime.NORMAL
        self.risk_history = []
        
        # Risk limits
        self.risk_limits = {
            "max_var": config.get("max_var", 0.02),  # 2% VaR
            "max_leverage": config.get("max_leverage", 2.0),
            "max_drawdown": config.get("max_drawdown", 0.15),  # 15%
            "max_concentration": config.get("max_concentration", 0.1),  # 10% per position
            "min_liquidity_score": config.get("min_liquidity_score", 0.7)
        }
        
        # Regime-specific adjustments
        self.regime_adjustments = {
            MarketRegime.NORMAL: {"leverage": 1.0, "var_limit": 1.0},
            MarketRegime.HIGH_VOLATILITY: {"leverage": 0.5, "var_limit": 0.7},
            MarketRegime.PANIC: {"leverage": 0.2, "var_limit": 0.5},
            MarketRegime.CORRELATION_BREAKDOWN: {"leverage": 0.3, "var_limit": 0.5},
            MarketRegime.LIQUIDITY_CRISIS: {"leverage": 0.4, "var_limit": 0.6},
            MarketRegime.FLASH_CRASH: {"leverage": 0.0, "var_limit": 0.0},
            MarketRegime.REGIME_CHANGE: {"leverage": 0.5, "var_limit": 0.8}
        }
        
    async def evaluate_risk(self, portfolio: Any, 
                          market_data: MarketData,
                          historical_data: pd.DataFrame) -> Tuple[RiskAction, Dict[str, Any]]:
        """Hauptmethode zur Risikobewertung"""
        
        # Check Kill Switch first
        if self.kill_switch.is_active:
            return RiskAction.HALT_TRADING, {"reason": "Kill switch active"}
            
        # Detect current regime
        new_regime = self.regime_detector.detect_regime(market_data, historical_data)
        
        # Handle regime change
        if new_regime != self.current_regime:
            await self.handle_regime_change(self.current_regime, new_regime)
            self.current_regime = new_regime
            
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(portfolio, market_data)
        
        # Determine required action
        action, details = self.determine_risk_action(risk_metrics, new_regime)
        
        # Log risk assessment
        self.log_risk_assessment(risk_metrics, new_regime, action)
        
        return action, details
        
    def calculate_risk_metrics(self, portfolio: Any, 
                             market_data: MarketData) -> RiskMetrics:
        """Berechnet aktuelle Risikometriken"""
        # Portfolio VaR calculation
        returns = self.get_portfolio_returns(portfolio)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Concentration risk
        position_weights = self.get_position_weights(portfolio)
        concentration_risk = position_weights.max()
        
        # Correlation risk
        correlation_risk = self.calculate_correlation_risk(portfolio, market_data)
        
        # Liquidity score
        liquidity_score = self.calculate_liquidity_score(portfolio, market_data)
        
        return RiskMetrics(
            portfolio_var=abs(var_95),
            portfolio_cvar=abs(cvar_95),
            max_drawdown=drawdown.min(),
            current_drawdown=drawdown.iloc[-1],
            leverage=portfolio.leverage,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk,
            liquidity_score=liquidity_score
        )
        
    def determine_risk_action(self, metrics: RiskMetrics, 
                            regime: MarketRegime) -> Tuple[RiskAction, Dict[str, Any]]:
        """Bestimmt erforderliche Risikomaßnahmen"""
        
        # Get regime-adjusted limits
        adjustments = self.regime_adjustments[regime]
        adjusted_var_limit = self.risk_limits["max_var"] * adjustments["var_limit"]
        adjusted_leverage_limit = self.risk_limits["max_leverage"] * adjustments["leverage"]
        
        # Critical checks - immediate action required
        if regime == MarketRegime.FLASH_CRASH:
            return RiskAction.LIQUIDATE_ALL, {"reason": "Flash crash detected"}
            
        if metrics.current_drawdown < -self.risk_limits["max_drawdown"]:
            return RiskAction.LIQUIDATE_PARTIAL, {
                "reason": "Maximum drawdown exceeded",
                "target_reduction": 0.5
            }
            
        if metrics.portfolio_var > adjusted_var_limit * 1.5:
            return RiskAction.LIQUIDATE_PARTIAL, {
                "reason": "VaR limit severely exceeded",
                "target_reduction": 0.7
            }
            
        # Warning level checks - reduce risk
        if metrics.portfolio_var > adjusted_var_limit:
            return RiskAction.REDUCE_POSITION, {
                "reason": "VaR limit exceeded",
                "target_var": adjusted_var_limit
            }
            
        if metrics.leverage > adjusted_leverage_limit:
            return RiskAction.REDUCE_POSITION, {
                "reason": "Leverage limit exceeded",
                "target_leverage": adjusted_leverage_limit
            }
            
        if metrics.liquidity_score < self.risk_limits["min_liquidity_score"]:
            return RiskAction.HEDGE, {
                "reason": "Low liquidity score",
                "hedge_type": "index_futures"
            }
            
        if metrics.concentration_risk > self.risk_limits["max_concentration"]:
            return RiskAction.REDUCE_POSITION, {
                "reason": "Concentration limit exceeded",
                "target_positions": self.get_concentrated_positions(metrics)
            }
            
        return RiskAction.CONTINUE, {"status": "All risk metrics within limits"}
        
    async def handle_regime_change(self, old_regime: MarketRegime, 
                                 new_regime: MarketRegime):
        """Behandelt Regime-Wechsel"""
        print(f"🔄 Regime change detected: {old_regime.value} -> {new_regime.value}")
        
        # Emergency regimes trigger immediate action
        if new_regime in [MarketRegime.PANIC, MarketRegime.FLASH_CRASH]:
            self.kill_switch.activate(f"Emergency regime: {new_regime.value}")
            
        # Notify risk management team
        await self.send_risk_alert({
            "type": "regime_change",
            "old_regime": old_regime.value,
            "new_regime": new_regime.value,
            "timestamp": datetime.now(),
            "recommended_actions": self.get_regime_recommendations(new_regime)
        })
        
    def get_regime_recommendations(self, regime: MarketRegime) -> List[str]:
        """Gibt Empfehlungen für spezifisches Regime"""
        recommendations = {
            MarketRegime.NORMAL: [
                "Resume normal trading operations",
                "Review and reset risk limits to baseline"
            ],
            MarketRegime.HIGH_VOLATILITY: [
                "Reduce position sizes by 50%",
                "Increase cash buffer to 30%",
                "Tighten stop-losses"
            ],
            MarketRegime.PANIC: [
                "Halt new position entries",
                "Liquidate all leveraged positions",
                "Move to defensive assets"
            ],
            MarketRegime.CORRELATION_BREAKDOWN: [
                "Reduce all correlated positions",
                "Increase diversification requirements",
                "Monitor sector exposures closely"
            ],
            MarketRegime.LIQUIDITY_CRISIS: [
                "Exit all small-cap positions",
                "Increase position in liquid assets",
                "Widen bid-ask spread assumptions"
            ],
            MarketRegime.FLASH_CRASH: [
                "Immediate trading halt",
                "Cancel all open orders",
                "Await market stabilization"
            ],
            MarketRegime.REGIME_CHANGE: [
                "Re-evaluate all model assumptions",
                "Reduce reliance on historical patterns",
                "Increase monitoring frequency"
            ]
        }
        
        return recommendations.get(regime, ["Monitor situation closely"])
        
    def emergency_liquidation(self, urgency: str = "HIGH") -> Dict[str, Any]:
        """Notfall-Liquidierung von Positionen"""
        liquidation_plan = {
            "timestamp": datetime.now(),
            "urgency": urgency,
            "steps": []
        }
        
        if urgency == "IMMEDIATE":
            # Market orders for everything
            liquidation_plan["steps"] = [
                {"action": "CANCEL_ALL_ORDERS"},
                {"action": "LIQUIDATE_ALL_MARKET"},
                {"action": "MOVE_TO_CASH"}
            ]
        elif urgency == "HIGH":
            # Orderly liquidation over minutes
            liquidation_plan["steps"] = [
                {"action": "CANCEL_ALL_ORDERS"},
                {"action": "LIQUIDATE_LEVERAGED"},
                {"action": "LIQUIDATE_ILLIQUID"},
                {"action": "LIQUIDATE_REMAINING_VWAP"}
            ]
        else:
            # Orderly liquidation over hours/days
            liquidation_plan["steps"] = [
                {"action": "HALT_NEW_ENTRIES"},
                {"action": "LIQUIDATE_GRADUAL"},
                {"action": "MONITOR_IMPACT"}
            ]
            
        return liquidation_plan
        
    async def send_risk_alert(self, alert_data: Dict[str, Any]):
        """Sendet Risikowarnungen an Team"""
        # In production: Send to Slack, Email, SMS, etc.
        print(f"🚨 RISK ALERT: {alert_data}")
        
    def log_risk_assessment(self, metrics: RiskMetrics, 
                          regime: MarketRegime, 
                          action: RiskAction):
        """Protokolliert Risikobewertung für Audit Trail"""
        assessment = {
            "timestamp": datetime.now(),
            "regime": regime.value,
            "action": action.value,
            "metrics": {
                "var": metrics.portfolio_var,
                "cvar": metrics.portfolio_cvar,
                "drawdown": metrics.current_drawdown,
                "leverage": metrics.leverage,
                "concentration": metrics.concentration_risk,
                "liquidity": metrics.liquidity_score
            }
        }
        
        self.risk_history.append(assessment)
        
        # Keep only last 10000 assessments
        if len(self.risk_history) > 10000:
            self.risk_history = self.risk_history[-10000:]
            
    # Helper methods (simplified implementations)
    def get_portfolio_returns(self, portfolio) -> pd.Series:
        """Mock implementation"""
        return pd.Series(np.random.normal(0.001, 0.02, 252))
        
    def get_position_weights(self, portfolio) -> pd.Series:
        """Mock implementation"""
        return pd.Series(np.random.dirichlet(np.ones(10), 1)[0])
        
    def calculate_correlation_risk(self, portfolio, market_data) -> float:
        """Mock implementation"""
        return 0.6
        
    def calculate_liquidity_score(self, portfolio, market_data) -> float:
        """Mock implementation"""
        return 0.8
        
    def get_concentrated_positions(self, metrics) -> List[str]:
        """Mock implementation"""
        return ["AAPL", "MSFT"]