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
# Import the real feature selection function from features.py
from project.features import select_features as features_select_features
# Import the enhanced evaluator
from project.evaluation import TradingStrategyEvaluator, EvaluationConfig
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_features(train_df):
    """
    FIXED: Use the real feature selection logic from features.py
    This ensures consistency between backtesting and live training
    """
    return features_select_features(train_df)

def train_models_sync(train_df, ticker_fold_id):
    """
    Synchrones Training von LSTM und XGBoost Modellen für Backtesting
    Verwendet die zentralisierten Trainingsfunktionen aus training.py
    """
    try:
        # 1. Feature Selection
        selected_features = select_features(train_df)
        
        # 2. LSTM Training - verwendet zentrale Funktion
        lstm_result = build_and_train_lstm(
            df=train_df,
            selected_features=selected_features,
            hparams=Config.LSTM_HPARAMS.copy(),
            validation_split=0.0,  # No validation split for backtesting
            update_callback=None   # No progress updates needed
        )
        
        lstm_model = lstm_result['model']
        scaler = lstm_result['scaler']
        
        # 3. XGBoost Training - verwendet zentrale Funktion
        xgboost_result = build_and_train_xgboost(
            df=train_df,
            hparams=Config.XGBOOST_HPARAMS.copy(),
            validation_split=0.0   # No validation split for backtesting
        )
        
        xgboost_model = xgboost_result['model']
        
        return lstm_model, scaler, xgboost_model, selected_features
        
    except Exception as e:
        print(f"Training failed for {ticker_fold_id}: {str(e)}")
        raise

def walk_forward_validation(df, ticker, folds=5, benchmark_data=None, run_evaluation=True, 
                          export_results=True, output_dir="results"):
    """
    Enhanced Walk-Forward-Validation mit integrierter TradingStrategyEvaluator Analyse
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Stock ticker symbol
        folds: Number of folds for time series cross-validation
        benchmark_data: Optional benchmark data (e.g., SPY returns)
        run_evaluation: Whether to run comprehensive evaluation
        export_results: Whether to export results to files
        output_dir: Directory for output files
    """
    if len(df) < Config.DATA_WINDOW_SIZE * 2:
        return {'error': 'Nicht genügend Daten für Walk-Forward-Validation'}
    
    tscv = TimeSeriesSplit(n_splits=folds)
    results = {
        'total_return': 0, 
        'trades': [], 
        'sharpe_ratio': 0,
        'fold_results': [],
        'evaluation_report': None  # NEW: Will store evaluation results
    }
    
    # FIXED: Proper container for fold returns that will be concatenated
    all_fold_returns = []
    
    logger.info(f"Starting Walk-Forward Validation for {ticker} with {folds} folds")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        logger.info(f"Processing fold {fold + 1}/{folds}")
        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        try:
            lstm_model, scaler, xgboost_model, selected_features = train_models_sync(
                train_df, f"{ticker}_fold_{fold}"
            )
        except Exception as e:
            logger.error(f"Training fehlgeschlagen für Fold {fold}: {str(e)}")
            continue
        
        # Enhanced Backtest setup with analyzers
        cerebro = bt.Cerebro()
        
        # Add analyzers to capture daily returns and performance metrics
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=Config.RISK_FREE_RATE)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Add strategy with enhanced parameters
        cerebro.addstrategy(
            EnhancedModelBasedStrategy, 
            lstm_model=lstm_model, 
            scaler=scaler, 
            xgboost_model=xgboost_model,
            full_test_df=test_df,
            selected_features=selected_features,
            fold_id=fold  # NEW: Track fold for logging
        )
        
        data_feed = bt.feeds.PandasData(dataname=test_df)
        cerebro.adddata(data_feed)
        cerebro.broker.setcash(Config.BACKTEST_INITIAL_CASH)
        cerebro.broker.setcommission(Config.BACKTEST_COMMISSION)
        cerebro.broker.set_slippage_perc(perc=0.0005)
        
        # Run backtest
        logger.info(f"Running backtest for fold {fold + 1}")
        backtest_results = cerebro.run()
        
        # Extract results from analyzers
        strategy = backtest_results[0]
        
        # FIXED: Get daily returns from TimeReturn analyzer and store for concatenation
        time_return_analyzer = strategy.analyzers.time_return.get_analysis()
        fold_returns = pd.Series(time_return_analyzer, name=f'fold_{fold}')
        all_fold_returns.append(fold_returns)  # Simply collect, don't manipulate
        
        # Store fold-specific results
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
        
        # Collect trade details from enhanced strategy
        if hasattr(strategy, 'trade_log'):
            results['trades'].extend(strategy.trade_log)
        
        # Update overall metrics
        final_value = cerebro.broker.getvalue()
        fold_return = (final_value - Config.BACKTEST_INITIAL_CASH) / Config.BACKTEST_INITIAL_CASH
        results['total_return'] += fold_return
        
        logger.info(f"Fold {fold + 1} completed - Return: {fold_return:.2%}, "
                   f"Sharpe: {fold_result['sharpe_ratio']:.3f}")
    
    # Calculate average results across folds
    if len(results['fold_results']) > 0:
        results['total_return'] /= len(results['fold_results'])
        avg_sharpe = np.mean([f['sharpe_ratio'] for f in results['fold_results']])
        results['sharpe_ratio'] = avg_sharpe
        
        logger.info(f"Walk-Forward Validation completed - "
                   f"Average Return: {results['total_return']:.2%}, "
                   f"Average Sharpe: {avg_sharpe:.3f}")
        
        # FIXED: Properly concatenate fold returns to create continuous time series
        if all_fold_returns:
            strategy_returns = pd.concat(all_fold_returns)
            strategy_returns.sort_index(inplace=True)  # Ensure chronological order
            
            # Validate inputs before evaluation
            try:
                validate_evaluation_inputs(strategy_returns, benchmark_data)
                
                # NEW: Run comprehensive evaluation if requested
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

def run_comprehensive_evaluation(strategy_returns, benchmark_returns=None, trades_data=None, 
                               ticker="STRATEGY", export_results=True, output_dir="results"):
    """
    MOVED: This function has been moved to evaluation_integration.py
    This is a wrapper for backward compatibility
    """
    from project.evaluation_integration import run_comprehensive_evaluation as _run_evaluation
    return _run_evaluation(strategy_returns, benchmark_returns, trades_data, ticker, export_results, output_dir)

def convert_backtrader_trades_to_df(trades_data):
    """
    MOVED: This function has been moved to evaluation_integration.py
    This is a wrapper for backward compatibility
    """
    from project.evaluation_integration import convert_backtrader_trades_to_df as _convert_trades
    return _convert_trades(trades_data)

class EnhancedModelBasedStrategy(bt.Strategy):
    """
    MOVED: This class has been moved to strategy.py
    This is kept for backward compatibility - imports from strategy module
    """
    def __init__(self):
        # Import and delegate to the actual implementation
        from project.strategy import EnhancedModelBasedStrategy as _Strategy
        # This is a compatibility shim - the real implementation is in strategy.py
        logger.warning("Using deprecated EnhancedModelBasedStrategy from backtesting.py. "
                      "Please import from project.strategy instead.")
        super().__init__()

# Enhanced wrapper function for easier usage
def run_enhanced_backtest(data_df, ticker, benchmark_df=None, config_overrides=None, 
                         output_dir="backtest_results"):
    """
    Convenience function to run enhanced backtest with comprehensive evaluation
    
    Args:
        data_df: OHLCV DataFrame
        ticker: Ticker symbol
        benchmark_df: Optional benchmark data
        config_overrides: Dict to override default config values
        output_dir: Output directory for results
    
    Returns:
        Dict with backtest and evaluation results
    """
    # Apply config overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(Config, key):
                setattr(Config, key, value)
                logger.info(f"Config override: {key} = {value}")
    
    # Prepare benchmark returns if provided
    benchmark_returns = None
    if benchmark_df is not None:
        benchmark_returns = benchmark_df['close'].pct_change().dropna()
        benchmark_returns.name = 'benchmark'
        logger.info(f"Benchmark data prepared with {len(benchmark_returns)} observations")
    
    # Run walk-forward validation with evaluation
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

# Legacy compatibility functions
def train_models_sync(train_df, ticker_fold_id):
    """
    MOVED: This function has been moved to training_sync.py
    This is a wrapper for backward compatibility
    """
    from project.training_sync import train_models_sync as _train_models
    return _train_models(train_df, ticker_fold_id)

# Legacy compatibility - kept for backward compatibility
class AsymmetricWeightedMSELoss(torch.nn.Module):
    """
    Legacy loss class - actual implementation now in lstm_models.py
    Kept here for backward compatibility
    """
    def __init__(self):
        super(AsymmetricWeightedMSELoss, self).__init__()
        print("Warning: Using legacy AsymmetricWeightedMSELoss. Consider importing from lstm_models.")
        
    def forward(self, predictions, targets, upper_bound, lower_bound):
        errors = predictions - targets
        weights = torch.where(
            (targets > upper_bound) | (targets < lower_bound),
            Config.ATR_MULTIPLIER,
            torch.ones_like(targets)
        )
        return torch.mean(weights * errors ** 2)

# Example usage
if __name__ == "__main__":
    # Example of how to use the enhanced backtesting system
    print("Enhanced Backtesting System with Comprehensive Evaluation")
    print("=" * 60)
    
    # This would typically be replaced with your actual data loading
    print("To use this system:")
    print("1. Load your OHLCV data into a pandas DataFrame")
    print("2. Optionally load benchmark data (e.g., SPY)")
    print("3. Call run_enhanced_backtest() with your data")
    print("4. Results will include both backtest metrics and comprehensive evaluation")
    
    print("\nExample usage:")
    print("""
    # Load data
    data_df = load_your_data('AAPL')  # Your data loading function
    benchmark_df = load_your_data('SPY')  # Optional benchmark
    
    # Run enhanced backtest
    results = run_enhanced_backtest(
        data_df=data_df,
        ticker='AAPL',
        benchmark_df=benchmark_df,
        config_overrides={'RISK_PER_TRADE': 0.02},  # 2% risk per trade
        output_dir='results/AAPL_backtest'
    )
    
    # Access results
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    
    # Comprehensive evaluation results
    if results['evaluation_report']:
        eval_results = results['evaluation_report']
        print("Comprehensive evaluation completed!")
        print("Check the output directory for detailed reports and visualizations.")
    """)
