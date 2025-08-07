"""
PERFORMANCE OPTIMIZATION: Pre-compute features before backtesting
This addresses the critical performance bottleneck in EnhancedModelBasedStrategy
"""

# 1. MODIFIED: integrated_backtesting.py - Key changes for performance
import backtrader as bt
import pandas as pd
import torch
import joblib
import numpy as np
import traceback
from project.config import Config
from project.features import prepare_sequences, get_feature_columns, _create_features, prepare_inference_from_df
from project.lstm_models import Seq2Seq, Encoder, Decoder
from project.signals import calculate_breakout_signal
from sklearn.model_selection import TimeSeriesSplit
from project.storage import get_model_registry
from project.features import select_features as features_select_features
from project.evaluation import TradingStrategyEvaluator, EvaluationConfig
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_features(train_df):
    """
    FIXED: Use the real feature selection logic from features.py
    This ensures consistency between backtesting and live training
    """
    return features_select_features(train_df)

def precompute_features_for_backtest(df, selected_features):
    """
    NEW: Pre-compute all features for the entire DataFrame before backtesting
    This eliminates the need to recalculate features at each time step
    
    Args:
        df: OHLCV DataFrame
        selected_features: List of selected feature columns
    
    Returns:
        DataFrame with all required features pre-computed
    """
    logger.info("Pre-computing features for backtest performance optimization...")
    
    # Create all features upfront
    df_with_features = _create_features(df.copy())
    
    # Ensure all selected features are present
    missing_features = set(selected_features) - set(df_with_features.columns)
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        # Add missing features as zeros (fallback)
        for feature in missing_features:
            df_with_features[feature] = 0.0
    
    # Add feature columns as additional data fields for backtrader
    feature_columns = {}
    for i, feature in enumerate(selected_features):
        if feature in df_with_features.columns:
            feature_columns[f'feature_{i}'] = df_with_features[feature]
    
    # Combine OHLCV with feature columns
    result_df = df.copy()
    for feature_name, feature_data in feature_columns.items():
        result_df[feature_name] = feature_data
    
    logger.info(f"Pre-computed {len(feature_columns)} features")
    return result_df, list(feature_columns.keys())

def walk_forward_validation(df, ticker, folds=5, benchmark_data=None, run_evaluation=True, 
                          export_results=True, output_dir="results"):
    """
    OPTIMIZED: Enhanced Walk-Forward-Validation with pre-computed features
    """
    if len(df) < Config.DATA_WINDOW_SIZE * 2:
        return {'error': 'Nicht genügend Daten für Walk-Forward-Validation'}
    
    tscv = TimeSeriesSplit(n_splits=folds)
    results = {
        'total_return': 0, 
        'trades': [], 
        'sharpe_ratio': 0,
        'fold_results': [],
        'evaluation_report': None
    }
    
    all_fold_returns = []
    logger.info(f"Starting Walk-Forward Validation for {ticker} with {folds} folds")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        logger.info(f"Processing fold {fold + 1}/{folds}")
        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        try:
            # Import from training_sync module
            from project.training_sync import train_models_sync
            lstm_model, scaler, xgboost_model, selected_features = train_models_sync(
                train_df, f"{ticker}_fold_{fold}"
            )
        except Exception as e:
            logger.error(f"Training fehlgeschlagen für Fold {fold}: {str(e)}")
            continue
        
        # OPTIMIZATION: Pre-compute features for the test set
        test_df_with_features, feature_column_names = precompute_features_for_backtest(
            test_df, selected_features
        )
        
        # Enhanced Backtest setup
        cerebro = bt.Cerebro()
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=Config.RISK_FREE_RATE)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # MODIFIED: Pass pre-computed features to strategy
        cerebro.addstrategy(
            OptimizedModelBasedStrategy,  # NEW: Using optimized strategy class
            lstm_model=lstm_model, 
            scaler=scaler, 
            xgboost_model=xgboost_model,
            selected_features=selected_features,
            feature_column_names=feature_column_names,  # NEW: Pre-computed feature column names
            fold_id=fold
        )
        
        # MODIFIED: Use DataFrame with pre-computed features
        data_feed = bt.feeds.PandasData(
            dataname=test_df_with_features,
            # Map additional feature columns to backtrader lines
            **{f'line{i}': f'feature_{i}' for i in range(len(feature_column_names))}
        )
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(Config.BACKTEST_INITIAL_CASH)
        cerebro.broker.setcommission(Config.BACKTEST_COMMISSION)
        cerebro.broker.set_slippage_perc(perc=0.0005)
        
        logger.info(f"Running optimized backtest for fold {fold + 1}")
        backtest_results = cerebro.run()
        
        # Process results (unchanged)
        strategy = backtest_results[0]
        time_return_analyzer = strategy.analyzers.time_return.get_analysis()
        fold_returns = pd.Series(time_return_analyzer, name=f'fold_{fold}')
        all_fold_returns.append(fold_returns)
        
        fold_result = {
            'fold': fold,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'final_value': cerebro.broker.getvalue(),
            'total_return': (cerebro.broker.getvalue() - Config.BACKTEST_INITIAL_CASH) / Config.BACKTEST_INITIAL_CASH,
            'sharpe_ratio': strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0),
            'max_drawdown': strategy.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0),
            'total_trades': strategy.analyzers.trades.get_analysis().get('total', {}).get('total', 0),
            'returns_series': fold_returns
        }
        
        results['fold_results'].append(fold_result)
        
        if hasattr(strategy, 'trade_log'):
            results['trades'].extend(strategy.trade_log)
        
        final_value = cerebro.broker.getvalue()
        fold_return = (final_value - Config.BACKTEST_INITIAL_CASH) / Config.BACKTEST_INITIAL_CASH
        results['total_return'] += fold_return
        
        logger.info(f"Fold {fold + 1} completed - Return: {fold_return:.2%}, "
                   f"Sharpe: {fold_result['sharpe_ratio']:.3f}")
    
    # Calculate final results (unchanged)
    if len(results['fold_results']) > 0:
        results['total_return'] /= len(results['fold_results'])
        avg_sharpe = np.mean([f['sharpe_ratio'] for f in results['fold_results']])
        results['sharpe_ratio'] = avg_sharpe
        
        logger.info(f"Walk-Forward Validation completed - "
                   f"Average Return: {results['total_return']:.2%}, "
                   f"Average Sharpe: {avg_sharpe:.3f}")
        
        if all_fold_returns:
            strategy_returns = pd.concat(all_fold_returns)
            strategy_returns.sort_index(inplace=True)
            
            try:
                validate_evaluation_inputs(strategy_returns, benchmark_data)
                
                if run_evaluation:
                    logger.info("Running comprehensive strategy evaluation...")
                    results['evaluation_report'] = run_comprehensive_evaluation(
                        strategy_returns=strategy_returns,
                        benchmark_returns=benchmark_data,
                        trades_data=results['trades'],
                        ticker=ticker,
                        export_results=export_results,
                        output_dir=output_dir
                    )
            except ValueError as e:
                logger.warning(f"Skipping evaluation due to invalid data: {e}")
                results['evaluation_report'] = {'error': str(e)}
    
    return results


class OptimizedModelBasedStrategy(bt.Strategy):
    """
    NEW: Performance-optimized strategy that uses pre-computed features
    This eliminates the major performance bottleneck of recalculating features
    """
    
    params = (
        ('lstm_model', None),
        ('scaler', None),
        ('xgboost_model', None),
        ('selected_features', None),
        ('feature_column_names', None),  # NEW: Pre-computed feature column names
        ('fold_id', None),
    )
    
    def __init__(self):
        self.lstm_model = self.params.lstm_model
        self.scaler = self.params.scaler
        self.xgboost_model = self.params.xgboost_model
        self.selected_features = self.params.selected_features
        self.feature_column_names = self.params.feature_column_names
        self.fold_id = self.params.fold_id
        
        self.trade_log = []
        self.position_size = 0
        self.last_signals = {'lstm': 0, 'xgboost': 0, 'breakout': 0}
        
        # Pre-map feature lines for fast access
        self.feature_lines = {}
        if self.feature_column_names:
            for i, feature_name in enumerate(self.feature_column_names):
                # Access backtrader lines for pre-computed features
                line_attr = f'line{i}'
                if hasattr(self.data, line_attr):
                    self.feature_lines[feature_name] = getattr(self.data, line_attr)
        
        logger.info(f"Optimized strategy initialized for fold {self.fold_id} with {len(self.feature_lines)} pre-computed features")
    
    def next(self):
        """
        OPTIMIZED: Use pre-computed features instead of recalculating
        """
        current_idx = len(self.data) - 1
        
        if current_idx < Config.DATA_WINDOW_SIZE:
            return
        
        try:
            # OPTIMIZATION: Use pre-computed features directly from data lines
            current_features = {}
            for i, feature_name in enumerate(self.feature_column_names):
                if feature_name in self.feature_lines:
                    current_features[self.selected_features[i]] = self.feature_lines[feature_name][0]
                else:
                    current_features[self.selected_features[i]] = 0.0
            
            # Create feature vector for models
            feature_vector = np.array([current_features.get(f, 0.0) for f in self.selected_features])
            
            # LSTM Prediction (using sequence of recent features)
            if current_idx >= Config.DATA_WINDOW_SIZE:
                # Build sequence from pre-computed features
                sequence_features = []
                for seq_idx in range(Config.DATA_WINDOW_SIZE):
                    step_features = {}
                    for i, feature_name in enumerate(self.feature_column_names):
                        if feature_name in self.feature_lines:
                            step_features[self.selected_features[i]] = self.feature_lines[feature_name][-seq_idx-1]
                        else:
                            step_features[self.selected_features[i]] = 0.0
                    
                    sequence_features.append([step_features.get(f, 0.0) for f in self.selected_features])
                
                sequence_array = np.array([sequence_features])
                sequence_scaled = self.scaler.transform(sequence_array.reshape(-1, len(self.selected_features)))
                sequence_scaled = sequence_scaled.reshape(sequence_array.shape)
                
                # LSTM prediction
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    src_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).to(device)
                    dummy_trg = torch.zeros(1, Config.DATA_PREDICTION_LENGTH, len(self.selected_features)).to(device)
                    lstm_output = self.lstm_model(src_tensor, dummy_trg, teacher_forcing_ratio=0.0)
                    
                    close_idx = self.selected_features.index('close') if 'close' in self.selected_features else 0
                    predicted_close = lstm_output[0, -1, close_idx].item()
                    
                    # Inverse transform prediction
                    dummy_features = np.zeros((1, len(self.selected_features)))
                    dummy_features[0, close_idx] = predicted_close
                    inverse_transformed = self.scaler.inverse_transform(dummy_features)
                    lstm_prediction = inverse_transformed[0, close_idx]
            else:
                lstm_prediction = self.data.close[0]
            
            # XGBoost Prediction
            xgb_prediction = self.xgboost_model.predict([feature_vector])[0]
            
            # Calculate signals
            current_price = self.data.close[0]
            lstm_signal = 1 if lstm_prediction > current_price * 1.001 else (-1 if lstm_prediction < current_price * 0.999 else 0)
            xgb_signal = 1 if xgb_prediction > current_price * 1.001 else (-1 if xgb_prediction < current_price * 0.999 else 0)
            
            # Breakout signal (using OHLC data directly)
            breakout_signal = calculate_breakout_signal(
                high=self.data.high[0],
                low=self.data.low[0],
                close=self.data.close[0],
                prev_high=self.data.high[-1] if len(self.data) > 1 else self.data.high[0],
                prev_low=self.data.low[-1] if len(self.data) > 1 else self.data.low[0],
                volume=self.data.volume[0]
            )
            
            # Combine signals
            combined_signal = (lstm_signal + xgb_signal + breakout_signal) / 3
            
            # Execute trades based on combined signal
            if combined_signal > 0.33 and self.position_size <= 0:
                size = int(self.broker.get_cash() * Config.RISK_PER_TRADE / current_price)
                if size > 0:
                    self.buy(size=size)
                    self.position_size = size
                    self.trade_log.append({
                        'date': self.data.datetime.datetime(0),
                        'action': 'BUY',
                        'price': current_price,
                        'size': size,
                        'lstm_signal': lstm_signal,
                        'xgb_signal': xgb_signal,
                        'breakout_signal': breakout_signal,
                        'combined_signal': combined_signal
                    })
            
            elif combined_signal < -0.33 and self.position_size >= 0:
                size = int(self.broker.get_cash() * Config.RISK_PER_TRADE / current_price)
                if size > 0:
                    self.sell(size=size)
                    self.position_size = -size
                    self.trade_log.append({
                        'date': self.data.datetime.datetime(0),
                        'action': 'SELL',
                        'price': current_price,
                        'size': size,
                        'lstm_signal': lstm_signal,
                        'xgb_signal': xgb_signal,
                        'breakout_signal': breakout_signal,
                        'combined_signal': combined_signal
                    })
            
            # Update last signals for monitoring
            self.last_signals = {
                'lstm': lstm_signal,
                'xgboost': xgb_signal,
                'breakout': breakout_signal
            }
        
        except Exception as e:
            logger.error(f"Error in optimized strategy next(): {e}")
            logger.error(traceback.format_exc())


def run_comprehensive_evaluation(strategy_returns, benchmark_returns=None, trades_data=None, 
                               ticker="STRATEGY", export_results=True, output_dir="results"):
    """Wrapper for backward compatibility"""
    from project.evaluation_integration import run_comprehensive_evaluation as _run_evaluation
    return _run_evaluation(strategy_returns, benchmark_returns, trades_data, ticker, export_results, output_dir)

def convert_backtrader_trades_to_df(trades_data):
    """Wrapper for backward compatibility"""
    from project.evaluation_integration import convert_backtrader_trades_to_df as _convert_trades
    return _convert_trades(trades_data)

def validate_evaluation_inputs(strategy_returns, benchmark_data):
    """Validate inputs before running evaluation"""
    if strategy_returns is None or len(strategy_returns) == 0:
        raise ValueError("Strategy returns are empty or None")
    
    if benchmark_data is not None and len(benchmark_data) == 0:
        raise ValueError("Benchmark data is empty")
    
    if strategy_returns.isna().all():
        raise ValueError("All strategy returns are NaN")

# Enhanced wrapper function (unchanged)
def run_enhanced_backtest(data_df, ticker, benchmark_df=None, config_overrides=None, 
                         output_dir="backtest_results"):
    """Convenience function to run enhanced backtest with comprehensive evaluation"""
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(Config, key):
                setattr(Config, key, value)
                logger.info(f"Config override: {key} = {value}")
    
    benchmark_returns = None
    if benchmark_df is not None:
        benchmark_returns = benchmark_df['close'].pct_change().dropna()
        benchmark_returns.name = 'benchmark'
        logger.info(f"Benchmark data prepared with {len(benchmark_returns)} observations")
    
    logger.info(f"Starting enhanced backtest for {ticker}")
    results = walk_forward_validation(
        df=data_df,
        ticker=ticker,
        folds=5,
        benchmark_data=benchmark_returns,
        run_evaluation=True,
        export_results=True,
        output_dir=output_dir
    )
    
    logger.info(f"Enhanced backtest completed for {ticker}")
    return results

# CLEANUP: Remove legacy classes and imports
# NOTE: AsymmetricWeightedMSELoss should be imported from lstm_models.py only

# 2. CORRECTED: training_sync.py - Fix Config import
"""
Training Synchronization Module - Contains training functions used by backtesting
Moved from backtesting.py for better separation of concerns
"""

import logging
# FIXED: Add missing Config import
from project.config import Config
from project.training import build_and_train_lstm, build_and_train_xgboost
from project.features import select_features

logger = logging.getLogger(__name__)

def train_models_sync(train_df, ticker_fold_id):
    """
    Synchronous Training von LSTM und XGBoost Modellen für Backtesting
    Verwendet die zentralisierten Trainingsfunktionen aus training.py
    
    Args:
        train_df: Training data DataFrame
        ticker_fold_id: Identifier for this training run (e.g., 'AAPL_fold_1')
    
    Returns:
        tuple: (lstm_model, scaler, xgboost_model, selected_features)
    
    Raises:
        Exception: If training fails
    """
    try:
        logger.info(f"Starting training for {ticker_fold_id}")
        
        # 1. Feature Selection - using real implementation from features.py
        selected_features = select_features(train_df)
        logger.info(f"Selected {len(selected_features)} features for {ticker_fold_id}")
        
        # 2. LSTM Training - verwendet zentrale Funktion
        logger.info(f"Training LSTM for {ticker_fold_id}")
        lstm_result = build_and_train_lstm(
            df=train_df,
            selected_features=selected_features,
            hparams=Config.LSTM_HPARAMS.copy(),
            validation_split=0.0,  # No validation split for backtesting
            update_callback=None   # No progress updates needed
        )
        
        lstm_model = lstm_result['model']
        scaler = lstm_result['scaler']
        logger.info(f"LSTM training completed for {ticker_fold_id} - Loss: {lstm_result['train_loss']:.4f}")
        
        # 3. XGBoost Training - verwendet zentrale Funktion
        logger.info(f"Training XGBoost for {ticker_fold_id}")
        xgboost_result = build_and_train_xgboost(
            df=train_df,
            hparams=Config.XGBOOST_HPARAMS.copy(),
            validation_split=0.0   # No validation split for backtesting
        )
        
        xgboost_model = xgboost_result['model']
        logger.info(f"XGBoost training completed for {ticker_fold_id} - Score: {xgboost_result['train_score']:.4f}")
        
        return lstm_model, scaler, xgboost_model, selected_features
        
    except Exception as e:
        logger.error(f"Training failed for {ticker_fold_id}: {str(e)}")
        raise Exception(f"Training failed for {ticker_fold_id}: {str(e)}")

# For backward compatibility
def get_trained_models(train_df, ticker_fold_id):
    """
    Legacy wrapper function for backward compatibility
    """
    return train_models_sync(train_df, ticker_fold_id)

if __name__ == "__main__":
    print("Optimized Backtesting System with Pre-computed Features")
    print("=" * 60)
    print("Performance optimization: Features are now pre-computed before backtesting")
    print("This eliminates the major bottleneck of recalculating features at each time step")
