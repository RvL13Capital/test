# project/auto_optimizer.py
"""
Automated Self-Optimization System
Continuously monitors, learns, and improves the breakout prediction system
"""

import pandas as pd
import numpy as np
import torch
import logging
import json
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
from dataclasses import dataclass, asdict
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
import optuna
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .consolidation_network import NetworkConsolidationAnalyzer, extract_consolidation_features
from .breakout_strategy import BreakoutPredictor, BreakoutScreener
from .storage import get_gcs_storage
from .training_optimized import training_manager
from .monitoring import (
    MODEL_PERFORMANCE, TRAINING_REQUESTS_TOTAL, 
    TRAINING_DURATION, get_metrics
)
from .tasks import tune_and_train_async

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    model_id: str
    created_date: datetime
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    profit_factor: float
    accuracy_score: float

@dataclass
class OptimizationResult:
    """Results from optimization cycle"""
    timestamp: datetime
    models_evaluated: int
    best_model_id: str
    improvement_pct: float
    new_hyperparams: Dict
    validation_metrics: Dict

class AutoOptimizer:
    """
    Main self-optimization engine that coordinates all components
    """
    
    def __init__(self, 
                 optimization_interval_hours: int = 24,
                 min_data_points: int = 1000,
                 performance_threshold: float = 0.65):
        
        self.optimization_interval = timedelta(hours=optimization_interval_hours)
        self.min_data_points = min_data_points
        self.performance_threshold = performance_threshold
        
        # Storage
        self.gcs = get_gcs_storage()
        
        # Components
        self.screener = BreakoutScreener()
        self.analyzer = NetworkConsolidationAnalyzer()
        
        # State tracking
        self.current_model_id = None
        self.model_performance_history = []
        self.optimization_history = []
        self.last_optimization = datetime.now() - self.optimization_interval
        
        # Performance tracking
        self.live_trades = []
        self.paper_trades = []
        
        # Async executor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("AutoOptimizer initialized")
    
    async def start(self):
        """Start the self-optimization loop"""
        logger.info("Starting automated self-optimization system...")
        
        # Load existing model or create initial
        await self._initialize_models()
        
        # Schedule tasks
        schedule.every(1).hours.do(self._collect_market_data)
        schedule.every(6).hours.do(self._evaluate_performance)
        schedule.every(self.optimization_interval.total_seconds() / 3600).hours.do(
            self._run_optimization_cycle
        )
        schedule.every(1).days.do(self._cleanup_old_data)
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                
                # Real-time monitoring
                await self._monitor_positions()
                
                # Sleep briefly
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _initialize_models(self):
        """Initialize or load existing models"""
        try:
            # Try to load best performing model
            models = self.gcs.list_models(prefix="models/breakout_predictor/")
            
            if models:
                # Load model with best historical performance
                best_model = max(models, key=lambda m: 
                    float(m.get('metadata', {}).get('validation_score', 0)))
                
                self.current_model_id = best_model['name']
                logger.info(f"Loaded existing model: {self.current_model_id}")
            else:
                # Create initial model
                await self._create_initial_model()
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            await self._create_initial_model()
    
    async def _create_initial_model(self):
        """Create initial breakout prediction model"""
        logger.info("Creating initial breakout prediction model...")
        
        # Get training data
        training_data = await self._collect_training_data()
        
        if not training_data:
            raise ValueError("No training data available")
        
        # Prepare features
        df_combined = pd.concat(training_data)
        
        # Initial hyperparameters
        initial_hparams = {
            'model_type': 'breakout_lstm',
            'hidden_dim': 128,
            'n_layers': 3,
            'dropout_prob': 0.3,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'window_size': 60,
            'prediction_horizon': 30
        }
        
        # Train model
        task = tune_and_train_async.delay(
            df_json=df_combined.to_json(orient='split'),
            model_type='lstm',
            ticker='BREAKOUT_COMPOSITE',
            user_id='auto_optimizer',
            custom_hparams=initial_hparams
        )
        
        # Wait for completion
        result = task.get(timeout=3600)
        
        if result['status'] == 'SUCCESS':
            self.current_model_id = result['model_paths']['model_path']
            logger.info(f"Initial model created: {self.current_model_id}")
        else:
            raise ValueError(f"Initial model training failed: {result}")
    
    async def _collect_training_data(self, days_back: int = 365) -> List[pd.DataFrame]:
    """
    NEW: Collect training data from EOD Pipeline instead of yfinance
    """
    logger.info("Collecting training data from EOD database...")
    
    # Get integrated system
    from .integrated_system import IntegratedMLTradingSystem, IntegratedSystemConfig
    config = IntegratedSystemConfig()
    system = IntegratedMLTradingSystem(config)
    
    # Get list of stocks with good data
    candidates = await self._find_historical_breakouts(days_back)
    
    training_data = []
    
    for ticker in candidates[:50]:
        try:
            # Get from EOD pipeline
            df = await system._get_training_data_from_eod(ticker, days_back)
            
            if len(df) > 100:
                # Get features from feature store
                df_features = await system._compute_features_from_eod(ticker, df)
                
                # Add labels
                df_features['future_return_5d'] = df_features['close'].pct_change(5).shift(-5)
                df_features['future_return_20d'] = df_features['close'].pct_change(20).shift(-20)
                df_features['breakout_occurred'] = (df_features['future_return_20d'] > 0.3).astype(int)
                
                training_data.append(df_features)
                
        except Exception as e:
            logger.error(f"Error collecting data for {ticker}: {e}")
            continue
    
    logger.info(f"Collected training data for {len(training_data)} stocks from EOD database")
    return training_data
        
        # Collect data in parallel
        tasks = [collect_stock_data(ticker) for ticker in candidates[:50]]
        results = await asyncio.gather(*tasks)
        
        training_data = [df for df in results if df is not None]
        
        logger.info(f"Collected training data for {len(training_data)} stocks")
        
        return training_data
    
    async def _find_historical_breakouts(self, days_back: int) -> List[str]:
        """Find stocks that had significant breakouts historically"""
        # In production, this would query a database of historical breakouts
        # For now, return known volatile small caps
        
        volatile_stocks = [
            'GME', 'AMC', 'BBBY', 'ATER', 'PROG', 'SPRT', 'IRNT', 'OPAD',
            'TMC', 'ANY', 'BGFV', 'BKKT', 'DWAC', 'PHUN', 'MARK', 'SGOC',
            'RDBX', 'SST', 'MULN', 'NILE', 'HYMC', 'INDO', 'IMPP', 'GFAI',
            'BBIG', 'ATER', 'COSM', 'RELI', 'NEGG', 'CARV', 'ISPC', 'LGVN'
        ]
        
        return volatile_stocks
    
    def _collect_market_data(self):
        """Collect current market data for monitoring"""
        logger.info("Collecting market data...")
        
        try:
            # Get current candidates
            candidates = self.screener.get_top_candidates(n=50)
            
            # Store for analysis
            market_snapshot = {
                'timestamp': datetime.now().isoformat(),
                'candidates': [asdict(c) for c in candidates],
                'market_conditions': self._analyze_market_conditions()
            }
            
            if self.gcs:
                self.gcs.upload_json(
                    market_snapshot,
                    f"market_data/snapshot_{datetime.now().strftime('%Y%m%d_%H')}.json"
                )
            
            logger.info(f"Market data collected: {len(candidates)} candidates")
            
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
    
    def _analyze_market_conditions(self) -> Dict:
        """Analyze overall market conditions"""
        try:
            # Get market indices
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="30d")
            
            iwm = yf.Ticker("IWM")  # Small cap index
            iwm_data = iwm.history(period="30d")
            
            # Calculate metrics
            spy_return = (spy_data['Close'][-1] / spy_data['Close'][0] - 1)
            iwm_return = (iwm_data['Close'][-1] / iwm_data['Close'][0] - 1)
            
            spy_volatility = spy_data['Close'].pct_change().std() * np.sqrt(252)
            iwm_volatility = iwm_data['Close'].pct_change().std() * np.sqrt(252)
            
            return {
                'spy_30d_return': spy_return,
                'iwm_30d_return': iwm_return,
                'spy_volatility': spy_volatility,
                'iwm_volatility': iwm_volatility,
                'small_cap_relative_strength': iwm_return - spy_return,
                'market_regime': self._classify_market_regime(spy_volatility, iwm_volatility)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {}
    
    def _classify_market_regime(self, spy_vol: float, iwm_vol: float) -> str:
        """Classify current market regime"""
        if spy_vol < 0.15:
            return "low_volatility"
        elif spy_vol < 0.25:
            if iwm_vol / spy_vol > 1.5:
                return "risk_on"
            else:
                return "normal"
        else:
            return "high_volatility"
    
    def _evaluate_performance(self):
        """Evaluate current model performance"""
        logger.info("Evaluating model performance...")
        
        try:
            # Get recent predictions vs outcomes
            recent_trades = self._get_recent_trades(days=30)
            
            if len(recent_trades) < 10:
                logger.warning("Insufficient trades for evaluation")
                return
            
            # Calculate metrics
            metrics = self._calculate_performance_metrics(recent_trades)
            
            # Create performance record
            performance = ModelPerformance(
                model_id=self.current_model_id,
                created_date=datetime.now(),
                win_rate=metrics['win_rate'],
                avg_return=metrics['avg_return'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                total_trades=len(recent_trades),
                profit_factor=metrics['profit_factor'],
                accuracy_score=metrics['accuracy_score']
            )
            
            self.model_performance_history.append(performance)
            
            # Log to monitoring
            MODEL_PERFORMANCE.labels(
                model_type='breakout_predictor',
                ticker='aggregate',
                metric_type='win_rate'
            ).set(performance.win_rate)
            
            MODEL_PERFORMANCE.labels(
                model_type='breakout_predictor',
                ticker='aggregate',
                metric_type='sharpe_ratio'
            ).set(performance.sharpe_ratio)
            
            # Check if optimization needed
            if performance.win_rate < self.performance_threshold:
                logger.warning(f"Performance below threshold: {performance.win_rate:.2%}")
                self._trigger_immediate_optimization()
            
            logger.info(f"Performance evaluation complete: win_rate={performance.win_rate:.2%}")
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
    
    def _run_optimization_cycle(self):
        """Run full optimization cycle"""
        if (datetime.now() - self.last_optimization) < self.optimization_interval:
            logger.info("Skipping optimization - too soon since last run")
            return
        
        logger.info("Starting optimization cycle...")
        
        try:
            # Collect recent data
            training_data = asyncio.run(self._collect_training_data(days_back=180))
            
            if not training_data:
                logger.error("No training data available for optimization")
                return
            
            # Prepare combined dataset
            df_combined = pd.concat(training_data)
            
            # Run hyperparameter optimization
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner()
            )
            
            study.optimize(
                lambda trial: self._optimization_objective(trial, df_combined),
                n_trials=50,
                timeout=7200  # 2 hours
            )
            
            # Get best parameters
            best_params = study.best_params
            best_score = study.best_value
            
            logger.info(f"Optimization complete: best_score={best_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            # Train new model with best parameters
            self._train_optimized_model(df_combined, best_params)
            
            # Record optimization
            result = OptimizationResult(
                timestamp=datetime.now(),
                models_evaluated=len(study.trials),
                best_model_id=self.current_model_id,
                improvement_pct=best_score - 0.65,  # vs baseline
                new_hyperparams=best_params,
                validation_metrics={'score': best_score}
            )
            
            self.optimization_history.append(result)
            self.last_optimization = datetime.now()
            
            # Save optimization history
            if self.gcs:
                self.gcs.upload_json(
                    asdict(result),
                    f"optimization_history/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
            
        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")
    
    def _optimization_objective(self, trial: optuna.Trial, df: pd.DataFrame) -> float:
        """Objective function for hyperparameter optimization"""
        # Suggest hyperparameters
        hparams = {
            'hidden_dim': trial.suggest_int('hidden_dim', 64, 256),
            'n_layers': trial.suggest_int('n_layers', 2, 4),
            'dropout_prob': trial.suggest_float('dropout_prob', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'window_size': trial.suggest_int('window_size', 30, 90),
            'min_consolidation_days': trial.suggest_int('min_consolidation_days', 15, 40),
            'phase_threshold': trial.suggest_float('phase_threshold', 0.5, 0.8)
        }
        
        # Train and evaluate model
        score = self._train_and_evaluate(df, hparams)
        
        return score
    
    def _train_and_evaluate(self, df: pd.DataFrame, hparams: Dict) -> float:
        """Train model and evaluate performance"""
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(df):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            # Train model (simplified for example)
            # In production, use full training pipeline
            
            # Evaluate on validation set
            predictions = self._make_predictions(val_df, hparams)
            score = self._calculate_prediction_score(predictions, val_df)
            scores.append(score)
        
        return np.mean(scores)
    
    def _make_predictions(self, df: pd.DataFrame, hparams: Dict) -> pd.DataFrame:
        """Make predictions using current model"""
        # Simplified prediction logic
        # In production, use actual model inference
        
        predictions = pd.DataFrame(index=df.index)
        predictions['breakout_probability'] = np.random.rand(len(df))
        predictions['expected_magnitude'] = np.random.rand(len(df)) * 0.5
        
        return predictions
    
    def _calculate_prediction_score(self, predictions: pd.DataFrame, 
                                  actual_df: pd.DataFrame) -> float:
        """Calculate prediction accuracy score"""
        # Compare predictions to actual outcomes
        actual_breakouts = (actual_df['future_return_20d'] > 0.3).astype(int)
        predicted_breakouts = (predictions['breakout_probability'] > 0.65).astype(int)
        
        # Calculate metrics
        accuracy = (predicted_breakouts == actual_breakouts).mean()
        
        # Weight by magnitude accuracy for true positives
        true_positives = (predicted_breakouts == 1) & (actual_breakouts == 1)
        if true_positives.sum() > 0:
            magnitude_error = abs(
                predictions.loc[true_positives, 'expected_magnitude'] - 
                actual_df.loc[true_positives, 'future_return_20d']
            ).mean()
            magnitude_score = 1 - magnitude_error
        else:
            magnitude_score = 0
        
        # Combined score
        return 0.7 * accuracy + 0.3 * magnitude_score
    
    def _train_optimized_model(self, df: pd.DataFrame, best_params: Dict):
        """Train new model with optimized parameters"""
        logger.info("Training optimized model...")
        
        # Submit training task
        task = tune_and_train_async.delay(
            df_json=df.to_json(orient='split'),
            model_type='lstm',
            ticker='BREAKOUT_OPTIMIZED',
            user_id='auto_optimizer',
            custom_hparams=best_params
        )
        
        # Wait for completion
        result = task.get(timeout=7200)
        
        if result['status'] == 'SUCCESS':
            self.current_model_id = result['model_paths']['model_path']
            logger.info(f"Optimized model trained: {self.current_model_id}")
        else:
            logger.error(f"Optimized model training failed: {result}")
    
    async def _monitor_positions(self):
        """Monitor current positions and predictions"""
        # This would integrate with live trading system
        # For now, just log status
        
        if len(self.live_trades) > 0:
            logger.debug(f"Monitoring {len(self.live_trades)} live positions")
    
    def _get_recent_trades(self, days: int) -> List[Dict]:
        """Get recent trade history"""
        # In production, query from database
        # For now, return mock data
        
        return self.live_trades[-100:]
    
    def _calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {
                'win_rate': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'accuracy_score': 0
            }
        
        returns = [t.get('return', 0) for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        
        win_rate = len(wins) / len(returns)
        avg_return = np.mean(returns)
        
        # Sharpe ratio (simplified)
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative_returns = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Profit factor
        if losses:
            profit_factor = abs(sum(wins) / sum(losses))
        else:
            profit_factor = float('inf') if wins else 0
        
        # Accuracy (for classification)
        correct_predictions = sum(
            1 for t in trades 
            if (t.get('predicted_breakout', False) and t.get('return', 0) > 0.2) or
               (not t.get('predicted_breakout', False) and t.get('return', 0) <= 0.2)
        )
        accuracy_score = correct_predictions / len(trades)
        
        return {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'accuracy_score': accuracy_score
        }
    
    def _trigger_immediate_optimization(self):
        """Trigger immediate optimization due to poor performance"""
        logger.warning("Triggering immediate optimization due to poor performance")
        
        # Reset optimization timer
        self.last_optimization = datetime.now() - self.optimization_interval
        
        # Run optimization
        self._run_optimization_cycle()
    
    def _cleanup_old_data(self):
        """Clean up old data to manage storage"""
        logger.info("Cleaning up old data...")
        
        try:
            # Remove old market snapshots
            cutoff_date = datetime.now() - timedelta(days=30)
            
            # In production, implement actual cleanup
            logger.info("Data cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        recent_performance = self.model_performance_history[-1] if self.model_performance_history else None
        
        return {
            'status': 'running',
            'current_model': self.current_model_id,
            'last_optimization': self.last_optimization.isoformat(),
            'recent_performance': asdict(recent_performance) if recent_performance else None,
            'total_optimizations': len(self.optimization_history),
            'monitoring_metrics': get_metrics()
        }


# Singleton instance
_auto_optimizer = None

def get_auto_optimizer() -> AutoOptimizer:
    """Get or create auto optimizer instance"""
    global _auto_optimizer
    if _auto_optimizer is None:
        _auto_optimizer = AutoOptimizer()
    return _auto_optimizer

# Start optimization system
async def start_auto_optimization():
    """Start the automated optimization system"""
    optimizer = get_auto_optimizer()
    await optimizer.start()
