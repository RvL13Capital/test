# project/breakout_config.py
"""
Configuration extensions for the self-optimizing breakout system
"""

import os
from datetime import timedelta
from typing import Set, Dict, List
from .config import Config

class BreakoutConfig(Config):
    """Extended configuration for breakout prediction system"""
    
    # Breakout System Settings
    BREAKOUT_SYSTEM_ENABLED = os.getenv('BREAKOUT_SYSTEM_ENABLED', 'true').lower() == 'true'
    
    # Market Cap Ranges (in USD)
    NANO_CAP_MAX = float(os.getenv('NANO_CAP_MAX', 50e6))      # $50M
    MICRO_CAP_MAX = float(os.getenv('MICRO_CAP_MAX', 300e6))   # $300M
    SMALL_CAP_MAX = float(os.getenv('SMALL_CAP_MAX', 2e9))     # $2B
    
    # Consolidation Parameters
    MIN_CONSOLIDATION_DAYS = int(os.getenv('MIN_CONSOLIDATION_DAYS', 20))
    MAX_CONSOLIDATION_DAYS = int(os.getenv('MAX_CONSOLIDATION_DAYS', 120))
    CONSOLIDATION_PRICE_TOLERANCE = float(os.getenv('CONSOLIDATION_PRICE_TOLERANCE', 0.15))  # 15%
    CONSOLIDATION_VOLUME_RATIO = float(os.getenv('CONSOLIDATION_VOLUME_RATIO', 0.7))
    
    # Network Analysis Parameters
    NETWORK_CORRELATION_WINDOW = int(os.getenv('NETWORK_CORRELATION_WINDOW', 10))
    NETWORK_CORRELATION_THRESHOLD = float(os.getenv('NETWORK_CORRELATION_THRESHOLD', 0.65))
    PERCOLATION_CRITICAL_DENSITY = float(os.getenv('PERCOLATION_CRITICAL_DENSITY', 0.68))
    CRITICAL_CLUSTERING_COEFFICIENT = float(os.getenv('CRITICAL_CLUSTERING_COEFFICIENT', 0.45))
    
    # Breakout Prediction Thresholds
    MIN_BREAKOUT_PROBABILITY = float(os.getenv('MIN_BREAKOUT_PROBABILITY', 0.65))
    MIN_EXPECTED_MOVE = float(os.getenv('MIN_EXPECTED_MOVE', 0.25))  # 25%
    MAX_EXPECTED_MOVE = float(os.getenv('MAX_EXPECTED_MOVE', 1.0))   # 100%
    
    # Trading Parameters
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 10))
    POSITION_SIZE_PCT = float(os.getenv('POSITION_SIZE_PCT', 0.05))  # 5% per position
    STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', 0.08))         # 8%
    TRAILING_STOP_ACTIVATION = float(os.getenv('TRAILING_STOP_ACTIVATION', 0.15))  # 15% gain
    TRAILING_STOP_DISTANCE = float(os.getenv('TRAILING_STOP_DISTANCE', 0.10))     # 10%
    
    # Self-Optimization Parameters
    OPTIMIZATION_INTERVAL_HOURS = int(os.getenv('OPTIMIZATION_INTERVAL_HOURS', 24))
    PERFORMANCE_THRESHOLD = float(os.getenv('PERFORMANCE_THRESHOLD', 0.65))  # 65% win rate
    MIN_TRADES_FOR_OPTIMIZATION = int(os.getenv('MIN_TRADES_FOR_OPTIMIZATION', 30))
    OPTIMIZATION_LOOKBACK_DAYS = int(os.getenv('OPTIMIZATION_LOOKBACK_DAYS', 180))
    
    # Model Training Parameters
    BREAKOUT_MODEL_EPOCHS = int(os.getenv('BREAKOUT_MODEL_EPOCHS', 100))
    BREAKOUT_MODEL_BATCH_SIZE = int(os.getenv('BREAKOUT_MODEL_BATCH_SIZE', 32))
    BREAKOUT_MODEL_LEARNING_RATE = float(os.getenv('BREAKOUT_MODEL_LEARNING_RATE', 0.001))
    BREAKOUT_MODEL_HIDDEN_DIM = int(os.getenv('BREAKOUT_MODEL_HIDDEN_DIM', 128))
    BREAKOUT_MODEL_LAYERS = int(os.getenv('BREAKOUT_MODEL_LAYERS', 3))
    BREAKOUT_MODEL_DROPOUT = float(os.getenv('BREAKOUT_MODEL_DROPOUT', 0.3))
    
    # Screening Parameters
    MAX_STOCKS_TO_SCREEN = int(os.getenv('MAX_STOCKS_TO_SCREEN', 500))
    SCREENING_BATCH_SIZE = int(os.getenv('SCREENING_BATCH_SIZE', 50))
    SCREENING_CACHE_TTL = int(os.getenv('SCREENING_CACHE_TTL', 3600))  # 1 hour
    
    # Alert System
    BREAKOUT_ALERT_THRESHOLD = float(os.getenv('BREAKOUT_ALERT_THRESHOLD', 0.05))  # 5% move
    VOLUME_SURGE_THRESHOLD = float(os.getenv('VOLUME_SURGE_THRESHOLD', 2.0))       # 2x average
    ALERT_COOLDOWN_MINUTES = int(os.getenv('ALERT_COOLDOWN_MINUTES', 60))
    
    # Data Sources
    MARKET_DATA_PROVIDER = os.getenv('MARKET_DATA_PROVIDER', 'yfinance')
    BACKUP_DATA_PROVIDERS = os.getenv('BACKUP_DATA_PROVIDERS', 'alphavantage,polygon').split(',')
    
    # Allowed Stock Universe (can be extended dynamically)
    BREAKOUT_STOCK_UNIVERSE = set(os.getenv('BREAKOUT_STOCK_UNIVERSE', '').split(',')) if os.getenv('BREAKOUT_STOCK_UNIVERSE') else set()
    
    # Sector Filters
    ALLOWED_SECTORS = set(os.getenv('ALLOWED_SECTORS', 'Technology,Healthcare,Consumer Cyclical,Energy,Financial Services').split(','))
    EXCLUDED_SECTORS = set(os.getenv('EXCLUDED_SECTORS', 'Utilities,Real Estate').split(','))
    
    # Risk Management
    MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', 0.20))  # 20% max drawdown
    MAX_CORRELATION_BETWEEN_POSITIONS = float(os.getenv('MAX_CORRELATION_BETWEEN_POSITIONS', 0.7))
    VOLATILITY_SCALING_ENABLED = os.getenv('VOLATILITY_SCALING_ENABLED', 'true').lower() == 'true'
    
    # Performance Metrics
    TARGET_SHARPE_RATIO = float(os.getenv('TARGET_SHARPE_RATIO', 1.5))
    MIN_PROFIT_FACTOR = float(os.getenv('MIN_PROFIT_FACTOR', 1.5))
    
    # Storage Paths
    BREAKOUT_MODELS_PATH = os.getenv('BREAKOUT_MODELS_PATH', 'models/breakout_predictor/')
    SCREENING_RESULTS_PATH = os.getenv('SCREENING_RESULTS_PATH', 'screening_results/')
    OPTIMIZATION_HISTORY_PATH = os.getenv('OPTIMIZATION_HISTORY_PATH', 'optimization_history/')
    ALERT_HISTORY_PATH = os.getenv('ALERT_HISTORY_PATH', 'alerts/')
    
    # API Rate Limits (specific to breakout endpoints)
    BREAKOUT_ANALYSIS_RATE_LIMIT = os.getenv('BREAKOUT_ANALYSIS_RATE_LIMIT', '100/hour')
    SCREENING_RATE_LIMIT = os.getenv('SCREENING_RATE_LIMIT', '20/hour')
    PREDICTION_RATE_LIMIT = os.getenv('PREDICTION_RATE_LIMIT', '200/hour')
    
    # Monitoring and Metrics
    BREAKOUT_METRICS_ENABLED = os.getenv('BREAKOUT_METRICS_ENABLED', 'true').lower() == 'true'
    TRACK_PAPER_TRADES = os.getenv('TRACK_PAPER_TRADES', 'true').lower() == 'true'
    
    # Feature Engineering
    TECHNICAL_INDICATORS = [
        'RSI', 'MACD', 'BB', 'ATR', 'OBV', 'ADX', 'CMF', 'MFI',
        'VWAP', 'PIVOT', 'STOCH', 'CCI', 'ROC', 'WILLIAMS_R'
    ]
    
    CONSOLIDATION_FEATURES = [
        'in_consolidation', 'consolidation_days', 'phase_transition_score',
        'accumulation_score', 'network_density', 'clustering_coefficient',
        'volume_pattern', 'price_efficiency', 'mispricing_indicator'
    ]
    
    MARKET_REGIME_FEATURES = [
        'market_volatility', 'sector_strength', 'small_cap_relative_performance',
        'breadth_indicators', 'sentiment_scores'
    ]
    
    @classmethod
    def validate_breakout_config(cls):
        """Validate breakout-specific configuration"""
        # Market cap validation
        assert cls.NANO_CAP_MAX < cls.MICRO_CAP_MAX < cls.SMALL_CAP_MAX, \
            "Invalid market cap ranges"
        
        # Consolidation parameters
        assert 0 < cls.CONSOLIDATION_PRICE_TOLERANCE < 1, \
            "Price tolerance must be between 0 and 1"
        
        assert cls.MIN_CONSOLIDATION_DAYS < cls.MAX_CONSOLIDATION_DAYS, \
            "Invalid consolidation day range"
        
        # Trading parameters
        assert 0 < cls.POSITION_SIZE_PCT <= 0.1, \
            "Position size must be between 0 and 10%"
        
        assert 0 < cls.STOP_LOSS_PCT < cls.TRAILING_STOP_ACTIVATION, \
            "Stop loss must be less than trailing stop activation"
        
        # Model parameters
        assert cls.BREAKOUT_MODEL_HIDDEN_DIM >= 32, \
            "Hidden dimension too small"
        
        assert 0 < cls.BREAKOUT_MODEL_DROPOUT < 1, \
            "Dropout must be between 0 and 1"
        
        logger.info("Breakout configuration validated successfully")
    
    @classmethod
    def get_breakout_hyperparams(cls, model_type: str = 'lstm') -> Dict:
        """Get default hyperparameters for breakout models"""
        if model_type == 'lstm':
            return {
                'hidden_dim': cls.BREAKOUT_MODEL_HIDDEN_DIM,
                'n_layers': cls.BREAKOUT_MODEL_LAYERS,
                'dropout_prob': cls.BREAKOUT_MODEL_DROPOUT,
                'learning_rate': cls.BREAKOUT_MODEL_LEARNING_RATE,
                'epochs': cls.BREAKOUT_MODEL_EPOCHS,
                'batch_size': cls.BREAKOUT_MODEL_BATCH_SIZE,
                'bidirectional': True,
                'attention_enabled': True
            }
        elif model_type == 'xgboost':
            return {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'early_stopping_rounds': 50
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @classmethod
    def get_trading_rules(cls) -> Dict:
        """Get trading rules for the breakout system"""
        return {
            'entry_rules': {
                'min_breakout_probability': cls.MIN_BREAKOUT_PROBABILITY,
                'min_expected_move': cls.MIN_EXPECTED_MOVE,
                'max_positions': cls.MAX_POSITIONS,
                'position_sizing': 'kelly_criterion',
                'max_position_size': cls.POSITION_SIZE_PCT
            },
            'exit_rules': {
                'stop_loss': cls.STOP_LOSS_PCT,
                'trailing_stop': {
                    'activation': cls.TRAILING_STOP_ACTIVATION,
                    'distance': cls.TRAILING_STOP_DISTANCE
                },
                'time_stop': 30,  # Exit after 30 days if no movement
                'target_multiplier': 0.8  # Exit at 80% of expected move
            },
            'risk_management': {
                'max_portfolio_risk': cls.MAX_PORTFOLIO_RISK,
                'max_correlation': cls.MAX_CORRELATION_BETWEEN_POSITIONS,
                'volatility_scaling': cls.VOLATILITY_SCALING_ENABLED,
                'position_rebalancing': 'weekly'
            }
        }
    
    @classmethod
    def get_screening_criteria(cls) -> Dict:
        """Get screening criteria for breakout candidates"""
        return {
            'market_cap_range': {
                'min': 10e6,  # $10M
                'max': cls.SMALL_CAP_MAX
            },
            'volume_criteria': {
                'min_avg_volume': 100000,
                'min_dollar_volume': 1e6  # $1M daily
            },
            'price_criteria': {
                'min_price': 0.50,  # Avoid penny stocks
                'max_price': 100    # Focus on lower-priced stocks
            },
            'consolidation_criteria': {
                'min_days': cls.MIN_CONSOLIDATION_DAYS,
                'max_days': cls.MAX_CONSOLIDATION_DAYS,
                'price_range': cls.CONSOLIDATION_PRICE_TOLERANCE,
                'volume_consistency': cls.CONSOLIDATION_VOLUME_RATIO
            },
            'sector_filters': {
                'allowed': list(cls.ALLOWED_SECTORS),
                'excluded': list(cls.EXCLUDED_SECTORS)
            }
        }

# Celery Beat Schedule for Breakout System
BREAKOUT_CELERY_BEAT_SCHEDULE = {
    'market-screening': {
        'task': 'project.breakout_tasks.scheduled_market_scan',
        'schedule': timedelta(hours=1),  # Every hour during market hours
        'options': {
            'queue': 'breakout_analysis',
            'priority': 5
        }
    },
    'system-optimization': {
        'task': 'project.breakout_tasks.scheduled_optimization',
        'schedule': timedelta(hours=BreakoutConfig.OPTIMIZATION_INTERVAL_HOURS),
        'options': {
            'queue': 'breakout_optimization',
            'priority': 3
        }
    },
    'breakout-monitoring': {
        'task': 'project.breakout_tasks.scheduled_monitoring',
        'schedule': timedelta(minutes=5),  # Every 5 minutes during market hours
        'options': {
            'queue': 'breakout_alerts',
            'priority': 10
        }
    },
    'performance-evaluation': {
        'task': 'project.breakout_tasks.evaluate_system_performance',
        'schedule': timedelta(hours=6),
        'options': {
            'queue': 'breakout_analysis',
            'priority': 4
        }
    }
}

# Update main Config Celery beat schedule
Config.CELERY_BEAT_SCHEDULE.update(BREAKOUT_CELERY_BEAT_SCHEDULE)

# Validate configuration on import
BreakoutConfig.validate_breakout_config()