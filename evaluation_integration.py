"""
Evaluation Integration Module - Contains comprehensive evaluation logic
Separated from backtesting orchestration for better maintainability
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from project.evaluation import TradingStrategyEvaluator, EvaluationConfig
from project.config import Config

logger = logging.getLogger(__name__)

def run_comprehensive_evaluation(strategy_returns, benchmark_returns=None, trades_data=None, 
                               ticker="STRATEGY", export_results=True, output_dir="results"):
    """
    Run comprehensive evaluation using TradingStrategyEvaluator
    
    Args:
        strategy_returns: pd.Series of daily strategy returns
        benchmark_returns: Optional benchmark returns
        trades_data: List of trade dictionaries from backtrader
        ticker: Strategy/ticker name for output files
        export_results: Whether to export results
        output_dir: Output directory
    
    Returns:
        Dictionary with comprehensive evaluation results
    """
    try:
        # Configure evaluation
        config = EvaluationConfig(
            risk_free_rate=getattr(Config, 'RISK_FREE_RATE', 0.02) / 252,
            monte_carlo_runs=1000,
            confidence_levels=[0.90, 0.95, 0.99],
            rolling_window_days=252
        )
        
        # Initialize evaluator
        evaluator = TradingStrategyEvaluator(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            config=config,
            transaction_costs=getattr(Config, 'BACKTEST_COMMISSION', 0.001)
        )
        
        # Convert trades data if available
        if trades_data and len(trades_data) > 0:
            trades_df = convert_backtrader_trades_to_df(trades_data)
            evaluator.set_trade_data(trades_df)
        
        # Run full evaluation
        logger.info("Running comprehensive evaluation analysis...")
        evaluation_results = evaluator.run_full_evaluation()
        
        # Export results if requested
        if export_results:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{ticker}_evaluation_{timestamp}"
            
            # Export JSON results
            json_path = output_path / f"{base_filename}.json"
            evaluator.export_results(str(json_path), format='json')
            
            # Save plots
            plot_path = output_path / f"{base_filename}_plots.png"
            evaluator.save_plots(str(plot_path), dpi=300)
            
            # Generate and save summary report
            summary_path = output_path / f"{base_filename}_summary.txt"
            summary_report = evaluator.generate_summary_report()
            with open(summary_path, 'w') as f:
                f.write(summary_report)
            
            logger.info(f"Evaluation results exported to {output_dir}")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {e}")
        return {'error': str(e)}

def convert_backtrader_trades_to_df(trades_data):
    """
    Convert backtrader trade data to DataFrame format expected by TradingStrategyEvaluator
    
    Args:
        trades_data: List of trade dictionaries from ModelBasedStrategy.trade_log
    
    Returns:
        pd.DataFrame with required columns for trade analysis
    """
    if not trades_data:
        return pd.DataFrame()
    
    trades_df = pd.DataFrame(trades_data)
    
    # Map to expected column names and add required fields
    required_columns = {
        'entry_price': 'entry_price',
        'exit_price': 'exit_price', 
        'pnl': 'pnl',
        'pnl_comm': 'pnl_comm',
        'size': 'size'
    }
    
    # Ensure all required columns exist
    for col in required_columns.values():
        if col not in trades_df.columns:
            if col == 'size':
                trades_df[col] = 1.0  # Default size
            else:
                logger.warning(f"Missing trade column: {col}")
    
    # Add synthetic dates if not present (backtrader doesn't always provide these)
    if 'entry_time' not in trades_df.columns:
        # Create synthetic entry/exit times based on trade sequence
        base_date = pd.Timestamp.now() - pd.Timedelta(days=len(trades_df)*2)
        trades_df['entry_time'] = [base_date + pd.Timedelta(days=i*2) for i in range(len(trades_df))]
        trades_df['exit_time'] = [base_date + pd.Timedelta(days=i*2+1) for i in range(len(trades_df))]
    
    # Add direction based on PnL and price movement
    if 'direction' not in trades_df.columns:
        price_change = trades_df['exit_price'] - trades_df['entry_price']
        trades_df['direction'] = np.where(
            (price_change > 0) & (trades_df['pnl'] > 0), 'long',
            np.where((price_change < 0) & (trades_df['pnl'] > 0), 'short', 'long')
        )
    
    return trades_df

def validate_evaluation_inputs(strategy_returns, benchmark_returns=None):
    """
    Validate inputs before running evaluation
    
    Args:
        strategy_returns: Strategy returns series
        benchmark_returns: Optional benchmark returns
    
    Returns:
        bool: True if valid, raises ValueError if not
    """
    if not isinstance(strategy_returns, pd.Series):
        raise ValueError("strategy_returns must be a pandas Series")
    
    if len(strategy_returns) < 30:
        raise ValueError("Insufficient data points for meaningful evaluation (minimum 30 required)")
    
    if strategy_returns.isnull().all():
        raise ValueError("strategy_returns contains only null values")
    
    if benchmark_returns is not None:
        if not isinstance(benchmark_returns, pd.Series):
            raise ValueError("benchmark_returns must be a pandas Series")
        
        if len(benchmark_returns) != len(strategy_returns):
            logger.warning("Strategy and benchmark returns have different lengths - will be aligned")
    
    return True
