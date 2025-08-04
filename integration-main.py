# main_integration.py
"""
Hauptintegrations-Modul das alle Komponenten zusammenführt
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import yaml
import pandas as pd

# Import all components
from staging_environment import (
    StagingEnvironment, ConfigurationManager, 
    DeploymentStage, ModelVersion
)
from transaction_cost_model import (
    RealisticTransactionCostModel, Stock, Trade,
    OrderType, MarketCapCategory, TransactionCostAnalyzer
)
from risk_management_regime import (
    RiskManagementSystem, RegimeDetector, KillSwitch,
    MarketData, MarketRegime, RiskAction
)
from comprehensive_monitoring import (
    MonitoringDashboard, AlertSeverity
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionTradingSystem:
    """Haupt-Produktionssystem das alle Komponenten integriert"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        
        # Initialize all components
        logger.info("Initializing Production Trading System...")
        
        # Staging & Configuration
        self.staging_env = StagingEnvironment(config_path)
        self.config_manager = ConfigurationManager()
        
        # Transaction Costs
        self.transaction_cost_model = RealisticTransactionCostModel()
        self.cost_analyzer = TransactionCostAnalyzer(self.transaction_cost_model)
        
        # Risk Management
        self.risk_management = RiskManagementSystem(self.config['risk_management'])
        self.kill_switch = KillSwitch()
        
        # Monitoring
        self.monitoring = MonitoringDashboard(self.config['monitoring'])
        
        # System state
        self.is_running = False
        self.current_regime = MarketRegime.NORMAL
        self.active_positions = {}
        
        logger.info("System initialization complete")
        
    def load_config(self) -> Dict[str, Any]:
        """Lädt Systemkonfiguration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    async def start(self):
        """Startet das Produktionssystem"""
        logger.info("Starting Production Trading System...")
        
        # Pre-flight checks
        if not await self.perform_preflight_checks():
            logger.error("Pre-flight checks failed. System start aborted.")
            return
            
        self.is_running = True
        
        # Start all components
        tasks = [
            asyncio.create_task(self.monitoring.start_monitoring()),
            asyncio.create_task(self.risk_management_loop()),
            asyncio.create_task(self.trading_loop()),
            asyncio.create_task(self.optimization_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"System error: {e}")
            await self.emergency_shutdown()
            
    async def perform_preflight_checks(self) -> bool:
        """Führt Pre-Flight Checks durch"""
        logger.info("Performing pre-flight checks...")
        
        checks = {
            "api_connectivity": await self.check_api_connectivity(),
            "data_integrity": await self.check_data_integrity(),
            "model_readiness": await self.check_model_readiness(),
            "risk_limits": self.check_risk_limits(),
            "staging_validation": await self.validate_staging_environment()
        }
        
        failed_checks = [name for name, passed in checks.items() if not passed]
        
        if failed_checks:
            logger.error(f"Pre-flight checks failed: {failed_checks}")
            return False
            
        logger.info("All pre-flight checks passed ✓")
        return True
        
    async def risk_management_loop(self):
        """Kontinuierliche Risikoüberwachung"""
        while self.is_running:
            try:
                # Get current market data
                market_data = await self.get_market_data()
                historical_data = await self.get_historical_data()
                
                # Evaluate risk
                action, details = await self.risk_management.evaluate_risk(
                    self.active_positions,
                    market_data,
                    historical_data
                )
                
                # Execute risk action
                await self.execute_risk_action(action, details)
                
                # Update monitoring
                self.monitoring.metrics.var_utilization.labels(
                    portfolio="main"
                ).set(details.get("var_utilization", 0))
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Risk management error: {e}")
                await asyncio.sleep(30)
                
    async def trading_loop(self):
        """Haupt-Trading-Loop"""
        while self.is_running:
            try:
                # Check if trading is allowed
                if self.kill_switch.is_active:
                    logger.warning("Trading halted - Kill switch active")
                    await asyncio.sleep(60)
                    continue
                    
                # Get trading signals
                signals = await self.get_trading_signals()
                
                # Calculate optimal execution with transaction costs
                for signal in signals:
                    optimal_execution = await self.optimize_execution(signal)
                    
                    # Execute trade if profitable after costs
                    if optimal_execution['expected_profit_after_costs'] > 0:
                        await self.execute_trade(optimal_execution)
                        
                await asyncio.sleep(1)  # Main trading loop frequency
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5)
                
    async def optimize_execution(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Optimiert Trade-Ausführung unter Berücksichtigung von Kosten"""
        stock = Stock(
            symbol=signal['symbol'],
            market_cap=signal['market_cap'],
            avg_volume_20d=signal['avg_volume'],
            avg_spread_bps=signal['avg_spread_bps'],
            volatility_20d=signal['volatility']
        )
        
        trade = Trade(
            stock=stock,
            size=signal['target_size'],
            price=signal['current_price'],
            order_type=OrderType.LIMIT,
            urgency=signal['urgency']
        )
        
        # Get current market conditions
        market_conditions = {
            "vix": await self.get_vix_level(),
            "time_of_day_factor": self.get_time_of_day_factor()
        }
        
        # Calculate costs for different strategies
        optimal_strategy = self.transaction_cost_model.optimize_execution_strategy(
            trade, market_conditions
        )
        
        # Calculate expected profit after costs
        cost_breakdown = self.transaction_cost_model.calculate_total_cost(
            trade, market_conditions, signal['expected_alpha']
        )
        
        expected_profit = signal['expected_alpha'] * trade.value
        expected_profit_after_costs = expected_profit - cost_breakdown['total_cost']
        
        return {
            'signal': signal,
            'trade': trade,
            'optimal_strategy': optimal_strategy['optimal_strategy'],
            'expected_cost_bps': optimal_strategy['expected_cost_bps'],
            'expected_profit_after_costs': expected_profit_after_costs,
            'cost_breakdown': cost_breakdown
        }
        
    async def optimization_loop(self):
        """Kontinuierliche Systemoptimierung"""
        while self.is_running:
            try:
                # Run optimization less frequently
                await asyncio.sleep(3600)  # Every hour
                
                # Analyze transaction costs
                cost_analysis = self.cost_analyzer.analyze_cost_drivers(period_days=7)
                
                if cost_analysis.get('recommendations'):
                    logger.info(f"Cost optimization recommendations: {cost_analysis['recommendations']}")
                    
                # Check for model drift
                for model_id in self.get_active_models():
                    drift = self.monitoring.drift_monitor.detect_prediction_drift(model_id)
                    
                    if drift and drift.is_drifting:
                        await self.handle_model_drift(model_id, drift)
                        
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                
    async def execute_risk_action(self, action: RiskAction, details: Dict[str, Any]):
        """Führt Risikomanagement-Aktionen aus"""
        logger.info(f"Executing risk action: {action.value} - {details}")
        
        if action == RiskAction.HALT_TRADING:
            self.kill_switch.activate(details.get('reason', 'Risk limit exceeded'))
            
        elif action == RiskAction.LIQUIDATE_ALL:
            liquidation_plan = self.risk_management.emergency_liquidation("IMMEDIATE")
            await self.execute_liquidation_plan(liquidation_plan)
            
        elif action == RiskAction.LIQUIDATE_PARTIAL:
            target_reduction = details.get('target_reduction', 0.5)
            await self.reduce_positions(target_reduction)
            
        elif action == RiskAction.REDUCE_POSITION:
            await self.adjust_position_sizes(details)
            
        elif action == RiskAction.HEDGE:
            await self.implement_hedging(details.get('hedge_type'))
            
    async def handle_model_drift(self, model_id: str, drift_detection):
        """Behandelt erkannten Model Drift"""
        logger.warning(f"Model drift detected for {model_id}: {drift_detection.drift_score:.3f}")
        
        # Create canary deployment for new model version
        if drift_detection.drift_score > 0.2:
            logger.info(f"Initiating model retraining for {model_id}")
            
            # In production: Trigger retraining pipeline
            # For now, reduce reliance on drifting model
            await self.reduce_model_allocation(model_id, reduction=0.5)
            
    async def execute_trade(self, execution_plan: Dict[str, Any]):
        """Führt Trade aus mit optimalem Execution Plan"""
        trade = execution_plan['trade']
        strategy = execution_plan['optimal_strategy']
        
        logger.info(f"Executing trade: {trade.stock.symbol} "
                   f"size={trade.size} strategy={strategy}")
        
        # Record pre-trade metrics
        pre_trade_metrics = {
            'expected_cost_bps': execution_plan['expected_cost_bps'],
            'expected_profit': execution_plan['expected_profit_after_costs']
        }
        
        try:
            # Execute based on strategy
            if strategy == "AGGRESSIVE":
                result = await self.execute_aggressive_order(trade)
            elif strategy == "PASSIVE":
                result = await self.execute_passive_order(trade)
            elif strategy in ["TWAP", "VWAP"]:
                result = await self.execute_algo_order(trade, strategy)
            else:
                result = await self.execute_default_order(trade)
                
            # Record actual costs
            actual_costs = self.calculate_actual_costs(trade, result)
            
            # Update monitoring
            self.monitoring.metrics.slippage_bps.labels(
                strategy="main",
                order_type=trade.order_type.value
            ).observe(actual_costs['slippage_bps'])
            
            # Learn from execution
            self.record_execution_quality(pre_trade_metrics, actual_costs)
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            
    async def emergency_shutdown(self):
        """Notfall-Shutdown des Systems"""
        logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        # Activate kill switch
        self.kill_switch.activate("Emergency shutdown", auto_reactivate=False)
        
        # Cancel all open orders
        await self.cancel_all_open_orders()
        
        # Liquidate positions if configured
        if self.config.get('emergency_liquidate', False):
            plan = self.risk_management.emergency_liquidation("HIGH")
            await self.execute_liquidation_plan(plan)
            
        # Save system state
        await self.save_system_state()
        
        # Stop all components
        self.is_running = False
        
        logger.critical("Emergency shutdown completed")
        
    async def validate_staging_environment(self) -> bool:
        """Validiert Staging-Umgebung vor Production-Deployment"""
        try:
            # Test API compatibility
            for provider in self.config['data_providers']:
                result = await self.staging_env.validate_api_compatibility(provider)
                if not result['passed']:
                    logger.error(f"API validation failed for {provider}: {result['errors']}")
                    return False
                    
            # Test model deployments
            for model_id in self.get_active_models():
                model_version = self.get_model_version(model_id)
                validation = self.staging_env.validate_model_deployment(
                    model_id, model_version
                )
                
                if not validation['passed']:
                    logger.error(f"Model validation failed for {model_id}: {validation['errors']}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Staging validation error: {e}")
            return False
            
    # Notification and Alert Integration
    async def setup_notifications(self):
        """Konfiguriert Benachrichtigungskanäle"""
        # Slack integration
        async def slack_notifier(alert, resolved=False):
            status = "🟢 RESOLVED" if resolved else "🔴 ALERT"
            message = f"{status} - {alert.severity.value.upper()}: {alert.message}"
            # await send_slack_message(message)
            
        # Email integration  
        async def email_notifier(alert, resolved=False):
            subject = f"Trading System Alert: {alert.metric_name}"
            # await send_email(subject, alert.message)
            
        # Add notification channels
        self.monitoring.alerting.notification_channels['slack'] = slack_notifier
        self.monitoring.alerting.notification_channels['email'] = email_notifier
        
    # Helper methods (simplified implementations)
    async def check_api_connectivity(self) -> bool:
        """Mock implementation"""
        return True
        
    async def check_data_integrity(self) -> bool:
        """Mock implementation"""
        return True
        
    async def check_model_readiness(self) -> bool:
        """Mock implementation"""
        return True
        
    def check_risk_limits(self) -> bool:
        """Mock implementation"""
        return True
        
    async def get_market_data(self) -> MarketData:
        """Mock implementation"""
        return MarketData(
            timestamp=datetime.now(),
            vix=18.5,
            sp500_return=-0.002,
            volume_ratio=1.2,
            spread_percentiles={'median': 10, '95th': 25},
            correlation_matrix=pd.DataFrame(),
            circuit_breaker_level=0
        )
        
    async def get_historical_data(self) -> pd.DataFrame:
        """Mock implementation"""
        return pd.DataFrame()
        
    async def get_trading_signals(self) -> list:
        """Mock implementation"""
        return []
        
    async def get_vix_level(self) -> float:
        """Mock implementation"""
        return 18.5
        
    def get_time_of_day_factor(self) -> float:
        """Mock implementation"""
        hour = datetime.now().hour
        if hour < 10 or hour > 15:  # Market open/close
            return 1.5
        return 1.0
        
    def get_active_models(self) -> list:
        """Mock implementation"""
        return ["model_v1", "model_v2"]
        
    def get_model_version(self, model_id: str) -> ModelVersion:
        """Mock implementation"""
        return ModelVersion(
            model_id=model_id,
            version="1.0.0",
            training_date=datetime.now(),
            parameters={},
            performance_metrics={},
            status="production"
        )

# Example configuration file structure
EXAMPLE_CONFIG = """
# config.yaml
system:
  name: "Production Trading System"
  environment: "production"
  
risk_management:
  max_var: 0.02
  max_leverage: 2.0
  max_drawdown: 0.15
  max_concentration: 0.1
  min_liquidity_score: 0.7
  
monitoring:
  prometheus_endpoint: "http://localhost:9090"
  grafana_endpoint: "http://localhost:3000"
  alert_endpoints:
    slack_webhook: "${SLACK_WEBHOOK_URL}"
    email_smtp: "${EMAIL_SMTP_SERVER}"
    
data_providers:
  - alpaca
  - polygon
  - yahoo_finance
  
execution:
  brokers:
    - name: "alpaca"
      api_key: "${ALPACA_API_KEY}"
      api_secret: "${ALPACA_API_SECRET}"
      
emergency_liquidate: false
"""

# Main entry point
async def main():
    """Haupteinstiegspunkt für das System"""
    # Initialize system
    system = ProductionTradingSystem("config.yaml")
    
    # Setup notifications
    await system.setup_notifications()
    
    # Start system
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
        await system.emergency_shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await system.emergency_shutdown()

if __name__ == "__main__":
    asyncio.run(main())