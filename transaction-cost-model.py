# transaction_cost_model.py
"""
Realistisches Transaktionskosten-Modell mit Slippage, Market Impact und Bid-Ask Spreads
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class MarketCapCategory(Enum):
    MICRO = "MICRO"  # < $300M
    SMALL = "SMALL"  # $300M - $2B
    MID = "MID"      # $2B - $10B
    LARGE = "LARGE"  # > $10B

@dataclass
class Stock:
    symbol: str
    market_cap: float
    avg_volume_20d: float
    avg_spread_bps: float
    volatility_20d: float
    
    @property
    def market_cap_category(self) -> MarketCapCategory:
        if self.market_cap < 300_000_000:
            return MarketCapCategory.MICRO
        elif self.market_cap < 2_000_000_000:
            return MarketCapCategory.SMALL
        elif self.market_cap < 10_000_000_000:
            return MarketCapCategory.MID
        else:
            return MarketCapCategory.LARGE

@dataclass
class Trade:
    stock: Stock
    size: int  # Number of shares
    price: float
    order_type: OrderType
    urgency: float  # 0 to 1, affects execution strategy
    
    @property
    def value(self) -> float:
        return self.size * self.price
        
    @property
    def participation_rate(self) -> float:
        """Anteil am durchschnittlichen Tagesvolumen"""
        return self.size / self.stock.avg_volume_20d

class MarketImpactModel:
    """Almgren-Chriss Market Impact Model"""
    
    def __init__(self):
        # Model parameters calibrated by market cap
        self.impact_params = {
            MarketCapCategory.MICRO: {
                "temporary_impact_factor": 0.3,
                "permanent_impact_factor": 0.5,
                "nonlinear_exponent": 0.6
            },
            MarketCapCategory.SMALL: {
                "temporary_impact_factor": 0.2,
                "permanent_impact_factor": 0.3,
                "nonlinear_exponent": 0.5
            },
            MarketCapCategory.MID: {
                "temporary_impact_factor": 0.1,
                "permanent_impact_factor": 0.15,
                "nonlinear_exponent": 0.4
            },
            MarketCapCategory.LARGE: {
                "temporary_impact_factor": 0.05,
                "permanent_impact_factor": 0.05,
                "nonlinear_exponent": 0.3
            }
        }
        
    def calculate_temporary_impact(self, trade: Trade, 
                                  execution_time_minutes: float = 30) -> float:
        """Berechnet temporären Market Impact (verschwindet nach Ausführung)"""
        params = self.impact_params[trade.stock.market_cap_category]
        
        # Participation rate adjusted for execution time
        time_factor = execution_time_minutes / 390  # Trading minutes per day
        adjusted_participation = trade.participation_rate / time_factor
        
        # Temporary impact formula
        temp_impact = params["temporary_impact_factor"] * \
                     trade.stock.volatility_20d * \
                     np.power(adjusted_participation, params["nonlinear_exponent"])
                     
        # Urgency multiplier
        urgency_multiplier = 1 + trade.urgency
        
        return temp_impact * urgency_multiplier
        
    def calculate_permanent_impact(self, trade: Trade) -> float:
        """Berechnet permanenten Market Impact (Information Leakage)"""
        params = self.impact_params[trade.stock.market_cap_category]
        
        # Permanent impact is proportional to size
        perm_impact = params["permanent_impact_factor"] * \
                     trade.participation_rate * \
                     trade.stock.volatility_20d
                     
        return perm_impact

class BidAskSpreadModel:
    """Modelliert Bid-Ask Spreads basierend auf Marktbedingungen"""
    
    def __init__(self):
        self.base_spreads = {
            MarketCapCategory.MICRO: 0.0075,  # 75 bps
            MarketCapCategory.SMALL: 0.0050,  # 50 bps
            MarketCapCategory.MID: 0.0020,    # 20 bps
            MarketCapCategory.LARGE: 0.0005   # 5 bps
        }
        
    def get_effective_spread(self, stock: Stock, 
                           market_conditions: Dict[str, float]) -> float:
        """Berechnet effektiven Spread unter aktuellen Marktbedingungen"""
        base_spread = self.base_spreads[stock.market_cap_category]
        
        # Adjust for volatility
        vol_multiplier = stock.volatility_20d / 0.02  # Normalized to 2% daily vol
        
        # Adjust for market stress (VIX)
        vix_level = market_conditions.get("vix", 20)
        stress_multiplier = np.sqrt(vix_level / 20)
        
        # Time of day adjustment (spreads wider at open/close)
        time_multiplier = market_conditions.get("time_of_day_factor", 1.0)
        
        effective_spread = base_spread * vol_multiplier * stress_multiplier * time_multiplier
        
        return effective_spread
        
    def calculate_spread_cost(self, trade: Trade, 
                            market_conditions: Dict[str, float]) -> float:
        """Berechnet Kosten durch Bid-Ask Spread"""
        effective_spread = self.get_effective_spread(trade.stock, market_conditions)
        
        # For market orders, pay half spread
        # For aggressive limit orders, might pay full spread
        if trade.order_type == OrderType.MARKET:
            spread_cost = 0.5 * effective_spread
        elif trade.order_type == OrderType.LIMIT and trade.urgency > 0.7:
            spread_cost = 0.75 * effective_spread
        else:
            spread_cost = 0.25 * effective_spread
            
        return spread_cost

class SlippageModel:
    """Modelliert Slippage basierend auf Ordersize und Markttiefe"""
    
    def __init__(self):
        self.liquidity_profiles = self.load_liquidity_profiles()
        
    def load_liquidity_profiles(self) -> Dict[MarketCapCategory, Dict]:
        """Lädt typische Liquiditätsprofile nach Market Cap"""
        return {
            MarketCapCategory.MICRO: {
                "avg_depth_bps": 10,  # Avg market depth in bps from mid
                "depth_decay_rate": 0.5  # How fast liquidity decreases
            },
            MarketCapCategory.SMALL: {
                "avg_depth_bps": 25,
                "depth_decay_rate": 0.3
            },
            MarketCapCategory.MID: {
                "avg_depth_bps": 50,
                "depth_decay_rate": 0.2
            },
            MarketCapCategory.LARGE: {
                "avg_depth_bps": 100,
                "depth_decay_rate": 0.1
            }
        }
        
    def estimate_slippage(self, trade: Trade) -> float:
        """Schätzt Slippage basierend auf Orderbook-Tiefe"""
        profile = self.liquidity_profiles[trade.stock.market_cap_category]
        
        # Calculate how deep into the book we need to go
        market_depth_consumed = trade.participation_rate * 100  # in bps
        
        # Slippage increases non-linearly with depth
        if market_depth_consumed <= profile["avg_depth_bps"]:
            # Within normal depth
            slippage_bps = market_depth_consumed * 0.5
        else:
            # Beyond normal depth - exponential increase
            excess = market_depth_consumed - profile["avg_depth_bps"]
            base_slippage = profile["avg_depth_bps"] * 0.5
            excess_slippage = excess * np.exp(profile["depth_decay_rate"] * excess / 100)
            slippage_bps = base_slippage + excess_slippage
            
        return slippage_bps / 10000  # Convert to decimal

class OpportunityCostModel:
    """Modelliert Opportunity Costs für nicht ausgeführte Orders"""
    
    def __init__(self):
        self.fill_probability_model = self.build_fill_probability_model()
        
    def build_fill_probability_model(self) -> callable:
        """Baut Modell für Fill-Wahrscheinlichkeiten"""
        def fill_probability(limit_distance_bps: float, 
                           volatility: float,
                           time_horizon_minutes: float) -> float:
            """
            Berechnet Wahrscheinlichkeit eines Fills basierend auf:
            - Distanz des Limit Preises zum aktuellen Preis
            - Volatilität
            - Zeithorizont
            """
            # Normalize distance by volatility
            normalized_distance = limit_distance_bps / (volatility * 10000)
            
            # Time adjustment
            time_factor = np.sqrt(time_horizon_minutes / 390)
            
            # Probability using normal distribution
            fill_prob = stats.norm.cdf(-normalized_distance / time_factor)
            
            return fill_prob
            
        return fill_probability
        
    def calculate_opportunity_cost(self, trade: Trade, 
                                 expected_alpha: float,
                                 limit_distance_bps: float = 10) -> float:
        """Berechnet Opportunity Cost für Limit Orders"""
        if trade.order_type != OrderType.LIMIT:
            return 0.0
            
        fill_prob = self.fill_probability_model(
            limit_distance_bps,
            trade.stock.volatility_20d,
            time_horizon_minutes=60  # 1 hour horizon
        )
        
        # Opportunity cost is the expected alpha we miss if order doesn't fill
        opportunity_cost = (1 - fill_prob) * expected_alpha
        
        return opportunity_cost

class RealisticTransactionCostModel:
    """Hauptklasse für realistische Transaktionskosten-Berechnung"""
    
    def __init__(self):
        self.commission_rate = 0.001  # 10 bps base commission
        self.market_impact_model = MarketImpactModel()
        self.spread_model = BidAskSpreadModel()
        self.slippage_model = SlippageModel()
        self.opportunity_cost_model = OpportunityCostModel()
        
        # Cost breakdown tracking
        self.cost_breakdown = []
        
    def calculate_total_cost(self, trade: Trade,
                           market_conditions: Dict[str, float],
                           expected_alpha: float = 0.0,
                           execution_strategy: str = "VWAP") -> Dict[str, float]:
        """Berechnet gesamte Transaktionskosten mit detailliertem Breakdown"""
        
        # 1. Commission
        commission = self.commission_rate * trade.value
        
        # 2. Spread costs
        spread_cost = self.spread_model.calculate_spread_cost(
            trade, market_conditions
        ) * trade.value
        
        # 3. Market impact
        temp_impact = self.market_impact_model.calculate_temporary_impact(trade)
        perm_impact = self.market_impact_model.calculate_permanent_impact(trade)
        market_impact_cost = (temp_impact + perm_impact) * trade.value
        
        # 4. Slippage
        slippage = self.slippage_model.estimate_slippage(trade)
        slippage_cost = slippage * trade.value
        
        # 5. Opportunity cost (for limit orders)
        opportunity_cost = self.opportunity_cost_model.calculate_opportunity_cost(
            trade, expected_alpha
        ) * trade.value
        
        # Total costs
        total_cost = commission + spread_cost + market_impact_cost + \
                    slippage_cost + opportunity_cost
                    
        # Store breakdown
        breakdown = {
            "timestamp": pd.Timestamp.now(),
            "trade": trade,
            "commission": commission,
            "spread_cost": spread_cost,
            "temp_impact": temp_impact * trade.value,
            "perm_impact": perm_impact * trade.value,
            "slippage_cost": slippage_cost,
            "opportunity_cost": opportunity_cost,
            "total_cost": total_cost,
            "total_cost_bps": (total_cost / trade.value) * 10000
        }
        
        self.cost_breakdown.append(breakdown)
        
        return breakdown
        
    def optimize_execution_strategy(self, trade: Trade,
                                  market_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Optimiert Execution Strategy basierend auf Kosten"""
        strategies = ["TWAP", "VWAP", "POV", "AGGRESSIVE", "PASSIVE"]
        strategy_costs = {}
        
        for strategy in strategies:
            # Adjust parameters based on strategy
            adjusted_trade = self.adjust_trade_for_strategy(trade, strategy)
            
            # Calculate costs
            costs = self.calculate_total_cost(
                adjusted_trade, 
                market_conditions
            )
            
            strategy_costs[strategy] = costs["total_cost_bps"]
            
        # Find optimal strategy
        optimal_strategy = min(strategy_costs, key=strategy_costs.get)
        
        return {
            "optimal_strategy": optimal_strategy,
            "expected_cost_bps": strategy_costs[optimal_strategy],
            "all_strategies": strategy_costs
        }
        
    def adjust_trade_for_strategy(self, trade: Trade, strategy: str) -> Trade:
        """Passt Trade-Parameter basierend auf Execution Strategy an"""
        adjusted_trade = Trade(
            stock=trade.stock,
            size=trade.size,
            price=trade.price,
            order_type=trade.order_type,
            urgency=trade.urgency
        )
        
        if strategy == "AGGRESSIVE":
            adjusted_trade.urgency = 0.9
            adjusted_trade.order_type = OrderType.MARKET
        elif strategy == "PASSIVE":
            adjusted_trade.urgency = 0.1
            adjusted_trade.order_type = OrderType.LIMIT
        elif strategy == "TWAP":
            adjusted_trade.urgency = 0.5
        elif strategy == "VWAP":
            adjusted_trade.urgency = 0.4
        elif strategy == "POV":  # Percentage of Volume
            adjusted_trade.urgency = 0.3
            
        return adjusted_trade
        
    def generate_cost_report(self, start_date: pd.Timestamp = None,
                           end_date: pd.Timestamp = None) -> pd.DataFrame:
        """Generiert detaillierten Transaktionskosten-Report"""
        if not self.cost_breakdown:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.cost_breakdown)
        
        if start_date:
            df = df[df["timestamp"] >= start_date]
        if end_date:
            df = df[df["timestamp"] <= end_date]
            
        # Add summary statistics
        summary = {
            "total_trades": len(df),
            "total_value": df["trade"].apply(lambda t: t.value).sum(),
            "avg_cost_bps": df["total_cost_bps"].mean(),
            "commission_bps": (df["commission"].sum() / df["trade"].apply(lambda t: t.value).sum()) * 10000,
            "spread_cost_bps": (df["spread_cost"].sum() / df["trade"].apply(lambda t: t.value).sum()) * 10000,
            "impact_cost_bps": ((df["temp_impact"] + df["perm_impact"]).sum() / df["trade"].apply(lambda t: t.value).sum()) * 10000,
            "slippage_cost_bps": (df["slippage_cost"].sum() / df["trade"].apply(lambda t: t.value).sum()) * 10000
        }
        
        return df, summary

class TransactionCostAnalyzer:
    """Analysiert historische Transaktionskosten für Verbesserungen"""
    
    def __init__(self, cost_model: RealisticTransactionCostModel):
        self.cost_model = cost_model
        
    def analyze_cost_drivers(self, period_days: int = 30) -> Dict[str, Any]:
        """Analysiert Hauptkostentreiber"""
        df, summary = self.cost_model.generate_cost_report(
            start_date=pd.Timestamp.now() - pd.Timedelta(days=period_days)
        )
        
        if df.empty:
            return {}
            
        analysis = {
            "period_days": period_days,
            "total_trades": len(df),
            "cost_breakdown": {
                "commission": summary["commission_bps"],
                "spread": summary["spread_cost_bps"],
                "market_impact": summary["impact_cost_bps"],
                "slippage": summary["slippage_cost_bps"]
            },
            "cost_by_market_cap": {},
            "cost_by_urgency": {},
            "recommendations": []
        }
        
        # Analyze by market cap
        for category in MarketCapCategory:
            category_trades = df[df["trade"].apply(
                lambda t: t.stock.market_cap_category == category
            )]
            if not category_trades.empty:
                analysis["cost_by_market_cap"][category.value] = {
                    "avg_cost_bps": category_trades["total_cost_bps"].mean(),
                    "trade_count": len(category_trades)
                }
                
        # Generate recommendations
        if summary["avg_cost_bps"] > 50:
            analysis["recommendations"].append(
                "High average costs detected. Consider more passive execution strategies."
            )
            
        if analysis["cost_breakdown"]["market_impact"] > 20:
            analysis["recommendations"].append(
                "High market impact costs. Consider splitting large orders or extending execution time."
            )
            
        return analysis
        
    def backtest_alternative_strategies(self, 
                                      historical_trades: List[Trade]) -> pd.DataFrame:
        """Backtestet alternative Execution Strategies"""
        results = []
        
        for trade in historical_trades:
            # Get actual costs
            actual_costs = self.cost_model.calculate_total_cost(
                trade, 
                {"vix": 20, "time_of_day_factor": 1.0}
            )
            
            # Test alternative strategies
            alternatives = self.cost_model.optimize_execution_strategy(
                trade,
                {"vix": 20, "time_of_day_factor": 1.0}
            )
            
            results.append({
                "trade_id": f"{trade.stock.symbol}_{trade.size}",
                "actual_cost_bps": actual_costs["total_cost_bps"],
                "optimal_strategy": alternatives["optimal_strategy"],
                "optimal_cost_bps": alternatives["expected_cost_bps"],
                "potential_savings_bps": actual_costs["total_cost_bps"] - alternatives["expected_cost_bps"]
            })
            
        return pd.DataFrame(results)