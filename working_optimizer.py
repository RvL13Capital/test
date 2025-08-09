# project/auto_optimizer_working.py
"""
Working implementation of the auto-optimization system
Replaces placeholder code with actual functionality
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import asyncio

from .consolidation_network import NetworkConsolidationAnalyzer, extract_consolidation_features
from .breakout_strategy import BreakoutPredictor
from .market_data import get_market_data_manager, StockInfo
from .storage import get_gcs_storage
from .config import Config

logger = logging.getLogger(__name__)

class RealBreakoutDataset:
    """Dataset for real breakout training data"""
    
    def __init__(self, df: pd.DataFrame, window_size: int = 60, 
                 prediction_horizon: int = 20, breakout_threshold: float = 0.3):
        self.df = df
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.breakout_threshold = breakout_threshold
        
        # Calculate labels
        self._calculate_labels()
        
    def _calculate_labels(self):
        """Calculate actual breakout labels from price data"""
        # Future returns
        self.df['future_return'] = (
            self.df['close'].shift(-self.prediction_horizon) / self.df['close'] - 1
        )
        
        # Maximum return in prediction window
        self.df['max_future_return'] = self.df['close'].rolling(
            window=self.prediction_horizon, min_periods=1
        ).apply(lambda x: (x.max() / x.iloc[0] - 1) if len(x) > 0 else 0).shift(-self.prediction_horizon)
        
        # Breakout occurred
        self.df['breakout_occurred'] = (
            self.df['max_future_return'] >= self.breakout_threshold
        ).astype(int)
        
        # Breakout magnitude (normalized)
        self.df['breakout_magnitude'] = self.df['max_future_return'].clip(0, 1)
        
        # Days to breakout
        self.df['days_to_breakout'] = self.df['close'].rolling(
            window=self.prediction_horizon
        ).apply(
            lambda x: np.argmax(x.values) if (x.max() / x.iloc[0] - 1) >= self.breakout_threshold else -1
        ).shift(-self.prediction_horizon)
    
    def prepare_sequences(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare sequences for training"""
        sequences = []
        labels = []
        
        # Drop rows with NaN labels
        valid_df = self.df.dropna(subset=['breakout_occurred', 'breakout_magnitude'])
        
        feature_cols = [col for col in valid_df.columns if col not in [
            'breakout_occurred', 'breakout_magnitude', 'future_return', 
            'max_future_return', 'days_to_breakout'
        ]]
        
        for i in range(len(valid_df) - self.window_size):
            # Get window
            window = valid_df.iloc[i:i + self.window_size]
            
            # Features
            features = window[feature_cols].values
            sequences.append(features)
            
            # Labels
            label_idx = i + self.window_size - 1
            labels.append({
                'breakout': valid_df['breakout_occurred'].iloc[label_idx],
                'magnitude': valid_df['breakout_magnitude'].iloc[label_idx],
                'days': int(valid_df['days_to_breakout'].iloc[label_idx])
            })
        
        # Convert to arrays
        X = np.array(sequences, dtype=np.float32)
        y = {
            'breakout': np.array([l['breakout'] for l in labels], dtype=np.int64),
            'magnitude': np.array([l['magnitude'] for l in labels], dtype=np.float32),
            'days': np.array([l['days'] for l in labels], dtype=np.int64)
        }
        
        return X, y

class WorkingAutoOptimizer:
    """Actually working auto-optimization implementation"""
    
    def __init__(self):
        self.market_data_manager = None
        self.current_model = None
        self.current_scaler = StandardScaler()
        self.performance_history = []
        self.optimization_history = []
        
        # Learnable thresholds (not fixed)
        self.adaptive_thresholds = {
            'phase_transition_threshold': 0.65,
            'network_density_threshold': 0.68,
            'clustering_threshold': 0.45,
            'volume_surge_threshold': 2.0,
            'accumulation_threshold': 5.0
        }
    
    async def initialize(self):
        """Initialize with real data sources"""
        self.market_data_manager = await get_market_data_manager()
        logger.info("WorkingAutoOptimizer initialized with real data sources")
    
    async def get_universe_stocks(self, min_market_cap: float = 10e6, 
                                max_market_cap: float = 2e9) -> List[str]:
        """Get dynamic universe of stocks based on market cap"""
        from .stock_universe import get_universe_manager
        
        universe_manager = get_universe_manager()
        
        # Get all stocks in market cap range
        stocks = await universe_manager.get_stocks_by_market_cap(min_market_cap, max_market_cap)
        
        # Extract tickers
        universe = [stock.ticker for stock in stocks]
        
        logger.info(f"Universe contains {len(universe)} stocks in market cap range")
        return universe
    
    def create_model(self, input_dim: int, hparams: Dict) -> BreakoutPredictor:
        """Create actual model with given hyperparameters"""
        model = BreakoutPredictor(
            input_dim=input_dim,
            hidden_dim=hparams['hidden_dim'],
            n_layers=hparams['n_layers'],
            dropout=hparams['dropout']
        )
        return model
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, hparams: Dict) -> Dict[str, float]:
        """Actually train the model (not placeholder)"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Loss functions
        breakout_criterion = nn.CrossEntropyLoss()
        magnitude_criterion = nn.MSELoss()
        timing_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(hparams['epochs']):
            # Training
            model.train()
            train_losses = []
            
            for batch_idx, (sequences, labels) in enumerate(train_loader):
                sequences = sequences.to(device)
                breakout_labels = labels['breakout'].to(device)
                magnitude_labels = labels['magnitude'].to(device)
                timing_labels = labels['days'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(sequences)
                
                # Calculate losses
                breakout_loss = breakout_criterion(
                    outputs['breakout_probability'].unsqueeze(1).repeat(1, 2),
                    breakout_labels
                )
                magnitude_loss = magnitude_criterion(
                    outputs['expected_magnitude'],
                    magnitude_labels
                )
                
                # Convert days to categories (0-4: 1-5 days, 5-9: 6-10 days, etc.)
                timing_categories = torch.clamp(timing_labels // 5, 0, 4)
                timing_loss = timing_criterion(
                    outputs['timing_distribution'],
                    timing_categories
                )
                
                # Combined loss
                total_loss = breakout_loss + 0.5 * magnitude_loss + 0.3 * timing_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_losses.append(total_loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device)
                    breakout_labels = labels['breakout'].to(device)
                    
                    outputs = model(sequences)
                    
                    # Store predictions for metrics
                    predictions = (outputs['breakout_probability'] > 0.5).cpu().numpy()
                    val_predictions.extend(predictions)
                    val_labels.extend(breakout_labels.cpu().numpy())
                    
                    # Validation loss (simplified)
                    val_loss = breakout_criterion(
                        outputs['breakout_probability'].unsqueeze(1).repeat(1, 2),
                        breakout_labels
                    )
                    val_losses.append(val_loss.item())
            
            # Calculate metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            accuracy = accuracy_score(val_labels, val_predictions)
            precision = precision_score(val_labels, val_predictions, zero_division=0)
            recall = recall_score(val_labels, val_predictions, zero_division=0)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= 10:
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'val_loss': best_val_loss
        }
    
    async def collect_training_data(self, days_back: int = 365) -> pd.DataFrame:
        """Collect real training data from multiple sources"""
        all_data = []
        
        # Get universe of stocks
        universe = await self.get_universe_stocks()
        
        # Sample if universe is too large
        if len(universe) > 200:
            universe = np.random.choice(universe, 200, replace=False).tolist()
        
        # Collect data for each stock
        for ticker in universe:
            try:
                # Get historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                df = await self.market_data_manager.get_historical_data(
                    ticker, start_date, end_date
                )
                
                if df is not None and len(df) > 100:
                    # Get stock info for market cap
                    info = await self.market_data_manager.get_stock_info(ticker)
                    if info:
                        # Extract features
                        df_features = extract_consolidation_features(df, info.market_cap)
                        df_features['ticker'] = ticker
                        df_features['market_cap'] = info.market_cap
                        
                        all_data.append(df_features)
                
            except Exception as e:
                logger.debug(f"Error collecting data for {ticker}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No training data collected")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Collected {len(combined_df)} rows of training data from {len(all_data)} stocks")
        
        return combined_df
    
    def optimization_objective(self, trial: optuna.Trial, train_data: pd.DataFrame) -> float:
        """Real optimization objective function"""
        # Suggest hyperparameters
        hparams = {
            'hidden_dim': trial.suggest_int('hidden_dim', 32, 256, step=32),
            'n_layers': trial.suggest_int('n_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'epochs': 50  # Fixed for speed
        }
        
        # Also optimize thresholds
        self.adaptive_thresholds['phase_transition_threshold'] = trial.suggest_float(
            'phase_transition_threshold', 0.5, 0.8
        )
        self.adaptive_thresholds['network_density_threshold'] = trial.suggest_float(
            'network_density_threshold', 0.5, 0.8
        )
        
        # Prepare data
        dataset = RealBreakoutDataset(train_data)
        X, y = dataset.prepare_sequences()
        
        if len(X) < 100:
            return float('inf')  # Not enough data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train = {k: v[:split_idx] for k, v in y.items()}
        y_val = {k: v[split_idx:] for k, v in y.items()}
        
        # Normalize features
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            {k: torch.tensor(v) for k, v in y_train.items()}
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            {k: torch.tensor(v) for k, v in y_val.items()}
        )
        
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'])
        
        # Create and train model
        model = self.create_model(X_train.shape[-1], hparams)
        metrics = self.train_model(model, train_loader, val_loader, hparams)
        
        # Optimization metric: F1 score (harmonic mean of precision and recall)
        if metrics['precision'] + metrics['recall'] > 0:
            f1_score = 2 * (metrics['precision'] * metrics['recall']) / (
                metrics['precision'] + metrics['recall']
            )
        else:
            f1_score = 0
        
        return -f1_score  # Negative because Optuna minimizes
    
    async def run_optimization(self):
        """Run real optimization cycle"""
        logger.info("Starting real optimization cycle...")
        
        try:
            # Collect fresh training data
            train_data = await self.collect_training_data()
            
            # Create Optuna study
            study = optuna.create_study(
                direction='minimize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
            )
            
            # Run optimization
            study.optimize(
                lambda trial: self.optimization_objective(trial, train_data),
                n_trials=30,
                timeout=3600  # 1 hour timeout
            )
            
            # Get best parameters
            best_params = study.best_params
            best_score = -study.best_value  # Convert back to positive F1
            
            logger.info(f"Optimization complete. Best F1 score: {best_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            # Train final model with best parameters
            await self._train_final_model(train_data, best_params)
            
            # Update thresholds
            for key in self.adaptive_thresholds:
                if key in best_params:
                    self.adaptive_thresholds[key] = best_params[key]
            
            # Save optimization results
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'best_score': best_score,
                'best_params': best_params,
                'adaptive_thresholds': self.adaptive_thresholds.copy()
            })
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    async def _train_final_model(self, train_data: pd.DataFrame, best_params: Dict):
        """Train final model with best parameters and validate through backtesting"""
        # Prepare full dataset
        dataset = RealBreakoutDataset(train_data)
        X, y = dataset.prepare_sequences()
        
        # Normalize
        X_flat = X.reshape(-1, X.shape[-1])
        self.current_scaler = StandardScaler()
        X_scaled = self.current_scaler.fit_transform(X_flat).reshape(X.shape)
        
        # Create data loader
        full_dataset = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            {k: torch.tensor(v) for k, v in y.items()}
        )
        train_loader = DataLoader(
            full_dataset, 
            batch_size=best_params.get('batch_size', 32),
            shuffle=True
        )
        
        # Create model
        model_hparams = {k: v for k, v in best_params.items() 
                        if k in ['hidden_dim', 'n_layers', 'dropout', 'learning_rate']}
        model_hparams['epochs'] = 100  # More epochs for final model
        
        new_model = self.create_model(X.shape[-1], model_hparams)
        
        # Train
        metrics = self.train_model(new_model, train_loader, train_loader, model_hparams)
        
        logger.info(f"New model trained. Accuracy: {metrics['accuracy']:.4f}")
        
        # CRITICAL: Validate new model through backtesting before deployment
        should_deploy = await self._validate_model_performance(new_model, best_params)
        
        if should_deploy:
            logger.info("New model passed validation, deploying to production")
            self.current_model = new_model
            
            # Update adaptive thresholds
            for key in self.adaptive_thresholds:
                if key in best_params:
                    self.adaptive_thresholds[key] = best_params[key]
            
            # Save model
            if get_gcs_storage():
                self._save_model()
        else:
            logger.warning("New model failed validation, keeping current model")
    
    async def _validate_model_performance(self, new_model, params: Dict) -> bool:
        """Validate model performance through backtesting"""
        from .backtesting_integrated import get_backtest_engine
        
        logger.info("Validating new model through backtesting...")
        
        try:
            backtest_engine = await get_backtest_engine()
            
            # Prepare models for comparison
            models_to_test = []
            
            # New model
            models_to_test.append({
                'name': 'new_model',
                'model': new_model,
                'scaler': self.current_scaler,
                'analyzer': self.get_adaptive_analyzer(),
                'features': dataset.feature_cols if 'dataset' in locals() else None
            })
            
            # Current model (if exists)
            if self.current_model is not None:
                models_to_test.append({
                    'name': 'current_model',
                    'model': self.current_model,
                    'scaler': self.current_scaler,
                    'analyzer': self.get_adaptive_analyzer(),
                    'features': dataset.feature_cols if 'dataset' in locals() else None
                })
            
            # Run comparison backtest
            comparison_results = await backtest_engine.compare_models(
                models_to_test,
                test_period_days=90  # 3 months of out-of-sample testing
            )
            
            # Save results for audit
            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backtest_engine.save_backtest_results(comparison_results, model_id)
            
            # Decision logic
            new_model_result = comparison_results['results'].get('new_model')
            current_model_result = comparison_results['results'].get('current_model')
            
            if new_model_result is None:
                logger.error("New model backtest failed")
                return False
            
            # Validation criteria
            min_sharpe = 1.0
            min_win_rate = 0.55
            max_drawdown = 0.20
            min_profit_factor = 1.3
            
            # Check absolute performance
            if (new_model_result.sharpe_ratio < min_sharpe or
                new_model_result.win_rate < min_win_rate or
                new_model_result.max_drawdown > max_drawdown or
                new_model_result.profit_factor < min_profit_factor):
                
                logger.warning(f"New model doesn't meet minimum criteria: "
                             f"Sharpe={new_model_result.sharpe_ratio:.2f}, "
                             f"WinRate={new_model_result.win_rate:.2%}")
                return False
            
            # If we have a current model, compare performance
            if current_model_result:
                # New model should be at least 10% better in key metrics
                improvement_threshold = 1.1
                
                if (new_model_result.sharpe_ratio < current_model_result.sharpe_ratio * improvement_threshold and
                    new_model_result.total_return < current_model_result.total_return * improvement_threshold):
                    
                    logger.warning(f"New model not sufficiently better than current: "
                                 f"New Sharpe={new_model_result.sharpe_ratio:.2f} vs "
                                 f"Current={current_model_result.sharpe_ratio:.2f}")
                    return False
            
            logger.info(f"New model validated successfully: "
                       f"Sharpe={new_model_result.sharpe_ratio:.2f}, "
                       f"Return={new_model_result.total_return:.2%}, "
                       f"Trades={new_model_result.total_trades}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def _save_model(self):
        """Save current model to storage"""
        try:
            gcs = get_gcs_storage()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_path = f"models/breakout_predictor/model_{timestamp}.pth"
            gcs.upload_pytorch_model(
                self.current_model.state_dict(),
                model_path,
                model_info={
                    'adaptive_thresholds': self.adaptive_thresholds,
                    'optimization_history': len(self.optimization_history)
                }
            )
            
            # Save scaler
            scaler_path = f"models/breakout_predictor/scaler_{timestamp}.pkl"
            gcs.upload_joblib(self.current_scaler, scaler_path)
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def get_adaptive_analyzer(self) -> NetworkConsolidationAnalyzer:
        """Get analyzer with current adaptive thresholds"""
        analyzer = NetworkConsolidationAnalyzer()
        
        # Update with learned thresholds
        analyzer.critical_density = self.adaptive_thresholds['network_density_threshold']
        analyzer.critical_clustering = self.adaptive_thresholds['clustering_threshold']
        
        return analyzer