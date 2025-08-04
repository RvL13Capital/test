# project/training_optimized.py
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.model_selection import train_test_split
import logging
import traceback
import copy
import numpy as np
import time
import psutil
import gc
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from prometheus_client import Histogram, Counter, Gauge

# Import from other project modules
from .data_processing import data_quality_check
from .features_optimized import OptimizedFeatureEngine, select_features
from .models import Seq2Seq, Encoder, Decoder, get_device
from .config import Config

logger = logging.getLogger(__name__)

# Prometheus metrics for training monitoring
TRAINING_DURATION = Histogram('ml_training_duration_seconds', 
                             'Training duration', ['model_type', 'ticker'])
TRAINING_MEMORY_PEAK = Gauge('ml_training_memory_peak_mb', 
                            'Peak memory usage during training', ['model_type'])
TRAINING_LOSS = Gauge('ml_training_final_loss', 
                     'Final training loss', ['model_type', 'ticker'])
BATCH_PROCESSING_TIME = Histogram('ml_batch_processing_seconds',
                                 'Time to process one batch', ['model_type'])

@dataclass
class TrainingMetrics:
    """Training metrics collection"""
    train_losses: list
    val_losses: list
    epoch_times: list
    memory_usage: list
    batch_times: list
    
    def to_dict(self) -> Dict:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epoch_times': self.epoch_times,
            'memory_usage': self.memory_usage,
            'batch_times': self.batch_times,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
            'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'peak_memory_mb': max(self.memory_usage) if self.memory_usage else 0
        }

class MemoryEfficientDataLoader:
    """
    OPTIMIZATION: Memory-efficient data loader for large datasets
    Prevents OOM errors and optimizes GPU memory usage
    """
    
    def __init__(self, src_data: np.ndarray, trg_data: np.ndarray, 
                 batch_size: int = 32, shuffle: bool = True):
        self.src_data = src_data
        self.trg_data = trg_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(src_data)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        
        if shuffle:
            self.indices = np.random.permutation(self.n_samples)
        else:
            self.indices = np.arange(self.n_samples)
    
    def __iter__(self):
        for i in range(0, self.n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, self.n_samples)
            batch_indices = self.indices[i:end_idx]
            
            batch_src = self.src_data[batch_indices]
            batch_trg = self.trg_data[batch_indices]
            
            yield torch.tensor(batch_src, dtype=torch.float32), torch.tensor(batch_trg, dtype=torch.float32)
    
    def __len__(self):
        return self.n_batches

class OptimizedLSTMTrainer:
    """
    PERFORMANCE: Memory-efficient, GPU-optimized LSTM trainer
    Features:
    - Dynamic batch sizing based on available memory
    - Gradient accumulation for large effective batch sizes
    - Memory cleanup and monitoring
    - Mixed precision training (optional)
    """
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 use_mixed_precision: bool = False):
        self.model = model
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # Memory monitoring
        self.memory_monitor = psutil.Process()
        
    def _get_optimal_batch_size(self, data_size: int, max_memory_gb: float = 4.0) -> int:
        """
        SMART: Dynamically calculate optimal batch size based on available memory
        """
        if torch.cuda.is_available():
            # GPU memory calculation
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = gpu_memory * 0.8  # Use 80% of GPU memory
            
            # Estimate memory per sample (rough approximation)
            memory_per_sample = data_size * 4 * 8  # 4 bytes per float, 8x overhead
            optimal_batch_size = int(available_memory / memory_per_sample)
            
            # Clamp to reasonable range
            optimal_batch_size = max(8, min(optimal_batch_size, 128))
        else:
            # CPU memory calculation
            available_memory = max_memory_gb * 1024 * 1024 * 1024  # Convert to bytes
            memory_per_sample = data_size * 4 * 4  # Less overhead for CPU
            optimal_batch_size = int(available_memory / memory_per_sample)
            optimal_batch_size = max(16, min(optimal_batch_size, 64))
        
        logger.info(f"Optimal batch size calculated: {optimal_batch_size}")
        return optimal_batch_size
    
    def _memory_cleanup(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return self.memory_monitor.memory_info().rss / 1024 / 1024
    
    def train_epoch(self, data_loader: MemoryEfficientDataLoader, 
                   optimizer: torch.optim.Optimizer, 
                   criterion: nn.Module,
                   teacher_forcing_ratio: float = 0.5,
                   gradient_accumulation_steps: int = 1) -> Tuple[float, float]:
        """
        OPTIMIZED: Memory-efficient epoch training with monitoring
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        memory_peak = 0.0
        
        optimizer.zero_grad()
        
        for batch_idx, (batch_src, batch_trg) in enumerate(data_loader):
            batch_start_time = time.time()
            
            # Move to device
            batch_src = batch_src.to(self.device, non_blocking=True)
            batch_trg = batch_trg.to(self.device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(batch_src, batch_trg, teacher_forcing_ratio)
                    loss = criterion(output, batch_trg)
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard precision training
                output = self.model(batch_src, batch_trg, teacher_forcing_ratio)
                loss = criterion(output, batch_trg)
                loss = loss / gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            batch_count += 1
            
            # Memory monitoring
            current_memory = self._get_memory_usage_mb()
            memory_peak = max(memory_peak, current_memory)
            
            # Batch timing
            batch_time = time.time() - batch_start_time
            BATCH_PROCESSING_TIME.labels(model_type='lstm').observe(batch_time)
            
            # Memory cleanup every 10 batches
            if batch_idx % 10 == 0:
                self._memory_cleanup()
            
            # Progress logging for large datasets
            if batch_idx % 50 == 0:
                logger.debug(f"Batch {batch_idx}/{len(data_loader)}, "
                           f"Loss: {loss.item():.6f}, "
                           f"Memory: {current_memory:.1f}MB")
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        
        # Update metrics
        TRAINING_MEMORY_PEAK.labels(model_type='lstm').set(memory_peak)
        
        return avg_loss, memory_peak
    
    def validate_epoch(self, data_loader: MemoryEfficientDataLoader, 
                      criterion: nn.Module) -> float:
        """
        OPTIMIZED: Memory-efficient validation
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch_src, batch_trg in data_loader:
                batch_src = batch_src.to(self.device, non_blocking=True)
                batch_trg = batch_trg.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(batch_src, batch_trg, teacher_forcing_ratio=0.0)
                        loss = criterion(output, batch_trg)
                else:
                    output = self.model(batch_src, batch_trg, teacher_forcing_ratio=0.0)
                    loss = criterion(output, batch_trg)
                
                total_loss += loss.item()
                batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        return avg_loss

def build_and_train_lstm_optimized(df, selected_features, hparams, validation_split=0.2,
                                 progress_callback: Optional[Callable] = None) -> Dict:
    """
    MAIN OPTIMIZATION: Memory-efficient LSTM training with comprehensive monitoring
    Features:
    - Dynamic batch sizing
    - Memory monitoring and cleanup
    - Mixed precision training
    - Gradient accumulation
    - Comprehensive metrics collection
    """
    training_start_time = time.time()
    ticker = hparams.get('ticker', 'unknown')
    
    try:
        # Initialize feature engine
        feature_engine = OptimizedFeatureEngine()
        
        # Data quality check
        data_quality_check(df)
        
        # Progress update
        if progress_callback:
            progress_callback(10, "Preparing data and features...")
        
        # Train/validation split
        train_df, val_df = train_test_split(df, test_size=validation_split, shuffle=False)
        
        # Feature preparation with optimized engine
        from .features_optimized import prepare_sequences
        src, trg, scaler = prepare_sequences(train_df, Config.DATA_WINDOW_SIZE, 
                                           Config.DATA_PREDICTION_LENGTH, selected_features)
        
        if validation_split > 0:
            val_src, val_trg, _ = prepare_sequences(val_df, Config.DATA_WINDOW_SIZE, 
                                                  Config.DATA_PREDICTION_LENGTH, 
                                                  selected_features, scaler=scaler)
        else:
            val_src, val_trg = np.array([]), np.array([])
        
        if len(src) == 0:
            raise ValueError("Not enough data to create training sequences.")
        
        if progress_callback:
            progress_callback(25, f"Created {len(src)} training sequences...")
        
        # Device setup
        device = get_device()
        use_mixed_precision = torch.cuda.is_available() and hparams.get('use_mixed_precision', True)
        
        # Model initialization
        model = Seq2Seq(
            Encoder(src.shape[-1], hparams['hidden_dim'], hparams['n_layers'], hparams['dropout_prob']),
            Decoder(src.shape[-1], hparams['hidden_dim'], hparams['n_layers'], hparams['dropout_prob']),
            device
        ).to(device)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=hparams['learning_rate'], 
            weight_decay=hparams.get('weight_decay', 1e-5),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        criterion = nn.MSELoss()
        
        # Initialize trainer
        trainer = OptimizedLSTMTrainer(model, device, use_mixed_precision)
        
        # Dynamic batch size calculation
        optimal_batch_size = trainer._get_optimal_batch_size(
            src.shape[1] * src.shape[2], 
            max_memory_gb=hparams.get('max_memory_gb', 6.0)
        )
        batch_size = min(hparams.get('batch_size', optimal_batch_size), optimal_batch_size)
        
        # Create data loaders
        train_loader = MemoryEfficientDataLoader(src, trg, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if len(val_src) > 0:
            val_loader = MemoryEfficientDataLoader(val_src, val_trg, batch_size=batch_size, shuffle=False)
        
        if progress_callback:
            progress_callback(40, f"Starting training with batch size {batch_size}...")
        
        # Training metrics
        metrics = TrainingMetrics([], [], [], [], [])
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Gradient accumulation steps for effective larger batch size
        gradient_accumulation_steps = max(1, hparams.get('effective_batch_size', 64) // batch_size)
        
        logger.info(f"Training LSTM: {hparams['epochs']} epochs, batch_size={batch_size}, "
                   f"gradient_accumulation={gradient_accumulation_steps}, "
                   f"mixed_precision={use_mixed_precision}")
        
        # Training loop
        for epoch in range(hparams['epochs']):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, memory_peak = trainer.train_epoch(
                train_loader, optimizer, criterion,
                teacher_forcing_ratio=hparams.get('teacher_forcing_ratio', 0.5),
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            
            # Validation phase
            val_loss = float('inf')
            if val_loader is not None:
                val_loss = trainer.validate_epoch(val_loader, criterion)
                scheduler.step(val_loss)
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            memory_usage = trainer._get_memory_usage_mb()
            
            metrics.train_losses.append(train_loss)
            metrics.val_losses.append(val_loss)
            metrics.epoch_times.append(epoch_time)
            metrics.memory_usage.append(memory_usage)
            
            # Progress update
            if progress_callback:
                progress = 40 + int((epoch + 1) / hparams['epochs'] * 50)
                progress_callback(progress, 
                    f"Epoch {epoch+1}/{hparams['epochs']}: Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}")
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"LSTM Epoch {epoch+1}/{hparams['epochs']}: "
                           f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                           f"Time: {epoch_time:.2f}s, Memory: {memory_usage:.1f}MB")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Memory cleanup every few epochs
            if epoch % 5 == 0:
                trainer._memory_cleanup()
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Final cleanup
        trainer._memory_cleanup()
        
        # Record final metrics
        training_time = time.time() - training_start_time
        final_metrics = metrics.to_dict()
        
        # Prometheus metrics
        TRAINING_DURATION.labels(model_type='lstm', ticker=ticker).observe(training_time)
        TRAINING_LOSS.labels(model_type='lstm', ticker=ticker).set(best_val_loss)
        
        if progress_callback:
            progress_callback(95, "Training completed, preparing results...")
        
        result = {
            'model': model,
            'scaler': scaler,
            'train_loss': final_metrics['final_train_loss'],
            'val_loss': best_val_loss,
            'training_time': training_time,
            'metrics': final_metrics,
            'hyperparameters': hparams,
            'batch_size_used': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'mixed_precision_used': use_mixed_precision
        }
        
        logger.info(f"LSTM training completed: {training_time:.2f}s, "
                   f"Best Val Loss: {best_val_loss:.6f}, "
                   f"Peak Memory: {max(metrics.memory_usage):.1f}MB")
        
        return result
        
    except Exception as e:
        logger.error(f"LSTM training failed: {e}\n{traceback.format_exc()}")
        # Record failure metrics
        TRAINING_DURATION.labels(model_type='lstm', ticker=ticker).observe(time.time() - training_start_time)
        raise

def build_and_train_xgboost_optimized(df, hparams, validation_split=0.2,
                                    progress_callback: Optional[Callable] = None) -> Dict:
    """
    OPTIMIZED: XGBoost training with feature caching and monitoring
    """
    training_start_time = time.time()
    ticker = hparams.get('ticker', 'unknown')
    
    try:
        if progress_callback:
            progress_callback(10, "Preparing XGBoost data...")
        
        # Data quality check
        data_quality_check(df)
        
        # Initialize feature engine
        feature_engine = OptimizedFeatureEngine()
        
        # Create features using optimized engine
        df_with_features = feature_engine.create_features_optimized(df.copy())
        df_with_features['target'] = df_with_features['close'].shift(-1)
        df_with_features.dropna(inplace=True)
        
        if df_with_features.empty:
            raise ValueError("No data left after feature creation.")
        
        # Feature selection
        selected_features = select_features(df_with_features, max_features=20)
        logger.info(f"XGBoost using {len(selected_features)} features")
        
        if progress_callback:
            progress_callback(30, f"Selected {len(selected_features)} features...")
        
        # Prepare training data
        X = df_with_features[selected_features]
        y = df_with_features['target']
        
        # Train/validation split
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, shuffle=False
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        if progress_callback:
            progress_callback(50, "Training XGBoost model...")
        
        # Enhanced hyperparameters with optimization
        optimized_hparams = {
            **hparams,
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
            'predictor': 'gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor',
            'enable_categorical': True,
            'max_cat_to_onehot': 4,
            'verbosity': 0
        }
        
        # Model training with monitoring
        model = xgb.XGBRegressor(**optimized_hparams)
        
        # Training with early stopping if validation data available
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        start_training_time = time.time()
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10 if eval_set else None,
            verbose=False
        )
        
        training_time = time.time() - start_training_time
        
        if progress_callback:
            progress_callback(80, "Analyzing feature importance...")
        
        # Feature importance analysis
        feature_importance = {}
        try:
            importance_scores = model.get_booster().get_score(importance_type='weight')
            # Map feature indices back to names
            for i, feature in enumerate(selected_features):
                feature_key = f'f{i}'
                if feature_key in importance_scores:
                    feature_importance[feature] = importance_scores[feature_key]
                else:
                    feature_importance[feature] = 0.0
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            feature_importance = {f: 0.0 for f in selected_features}
        
        # Model evaluation
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val) if X_val is not None else train_score
        
        total_training_time = time.time() - training_start_time
        
        # Prometheus metrics
        TRAINING_DURATION.labels(model_type='xgboost', ticker=ticker).observe(total_training_time)
        TRAINING_LOSS.labels(model_type='xgboost', ticker=ticker).set(-val_score)  # Negative because higher score is better
        
        if progress_callback:
            progress_callback(95, "XGBoost training completed...")
        
        result = {
            'model': model,
            'train_score': train_score,
            'val_score': val_score,
            'feature_importance': feature_importance,
            'selected_features': selected_features,
            'training_time': total_training_time,
            'hyperparameters': optimized_hparams,
            'n_estimators': model.n_estimators,
            'best_iteration': getattr(model, 'best_iteration', model.n_estimators)
        }
        
        logger.info(f"XGBoost training completed: {total_training_time:.2f}s, "
                   f"Train Score: {train_score:.4f}, Val Score: {val_score:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}\n{traceback.format_exc()}")
        # Record failure metrics
        TRAINING_DURATION.labels(model_type='xgboost', ticker=ticker).observe(time.time() - training_start_time)
        raise

class TrainingManager:
    """
    ORCHESTRATION: Manages training workflows with monitoring and optimization
    """
    
    def __init__(self):
        self.active_trainings = {}
        self.training_history = []
    
    def train_model_optimized(self, df, model_type: str, selected_features: list,
                            hparams: dict, validation_split: float = 0.2,
                            progress_callback: Optional[Callable] = None) -> Dict:
        """
        Main training orchestrator with automatic optimization
        """
        training_id = f"{model_type}_{int(time.time())}"
        self.active_trainings[training_id] = {
            'start_time': time.time(),
            'model_type': model_type,
            'status': 'starting'
        }
        
        try:
            if progress_callback:
                progress_callback(5, f"Initializing {model_type} training...")
            
            # Add training ID to hyperparameters for tracking
            hparams['training_id'] = training_id
            
            if model_type == 'lstm':
                result = build_and_train_lstm_optimized(
                    df, selected_features, hparams, validation_split, progress_callback
                )
            elif model_type == 'xgboost':
                result = build_and_train_xgboost_optimized(
                    df, hparams, validation_split, progress_callback
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Update training record
            training_record = {
                'training_id': training_id,
                'model_type': model_type,
                'start_time': self.active_trainings[training_id]['start_time'],
                'end_time': time.time(),
                'success': True,
                'result': result
            }
            
            self.training_history.append(training_record)
            del self.active_trainings[training_id]
            
            if progress_callback:
                progress_callback(100, "Training completed successfully!")
            
            return result
            
        except Exception as e:
            # Record failed training
            training_record = {
                'training_id': training_id,
                'model_type': model_type,
                'start_time': self.active_trainings[training_id]['start_time'],
                'end_time': time.time(),
                'success': False,
                'error': str(e)
            }
            
            self.training_history.append(training_record)
            del self.active_trainings[training_id]
            
            raise
    
    def get_training_stats(self) -> Dict:
        """Get comprehensive training statistics"""
        successful_trainings = [t for t in self.training_history if t['success']]
        failed_trainings = [t for t in self.training_history if not t['success']]
        
        return {
            'total_trainings': len(self.training_history),
            'successful_trainings': len(successful_trainings),
            'failed_trainings': len(failed_trainings),
            'success_rate': len(successful_trainings) / len(self.training_history) if self.training_history else 0,
            'active_trainings': len(self.active_trainings),
            'avg_training_time': np.mean([
                t['end_time'] - t['start_time'] for t in successful_trainings
            ]) if successful_trainings else 0
        }

# Global training manager instance
training_manager = TrainingManager()

# Backward compatibility functions
def build_and_train_lstm(df, selected_features, hparams, validation_split=0.2, 
                        update_callback=None):
    """Legacy wrapper with progress callback conversion"""
    progress_callback = None
    if update_callback:
        def progress_callback(progress, message):
            update_callback({'progress': progress, 'status': message})
    
    return build_and_train_lstm_optimized(df, selected_features, hparams, 
                                        validation_split, progress_callback)

def build_and_train_xgboost(df, hparams, validation_split=0.2):
    """Legacy wrapper for backward compatibility"""
    return build_and_train_xgboost_optimized(df, hparams, validation_split)

# Performance monitoring functions
def get_training_performance_stats() -> Dict:
    """Get detailed training performance statistics"""
    return {
        'training_manager': training_manager.get_training_stats(),
        'gpu_available': torch.cuda.is_available(),
        'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        'gpu_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        'system_memory_usage': psutil.virtual_memory()._asdict()
    }

if __name__ == "__main__":
    # Performance test
    print("Testing optimized training system...")
    
    # Generate test data
    import pandas as pd
    dates = pd.date_range('2020-01-01', periods=5000, freq='1H')
    test_df = pd.DataFrame({
        'datetime': dates,
        'open': np.random.randn(5000).cumsum() + 100,
        'high': np.random.randn(5000).cumsum() + 102,
        'low': np.random.randn(5000).cumsum() + 98,
        'close': np.random.randn(5000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 5000)
    })
    
    # Test LSTM training
    test_hparams = {
        'hidden_dim': 64,
        'n_layers': 2,
        'dropout_prob': 0.3,
        'learning_rate': 1e-3,
        'epochs': 5,  # Short test
        'batch_size': 16,
        'teacher_forcing_ratio': 0.5
    }
    
    feature_engine = OptimizedFeatureEngine()
    test_features = feature_engine.get_feature_columns()[:10]  # Use subset for testing
    
    def test_progress(progress, message):
        print(f"Progress: {progress}% - {message}")
    
    try:
        print("Starting LSTM training test...")
        result = training_manager.train_model_optimized(
            test_df, 'lstm', test_features, test_hparams, 
            validation_split=0.2, progress_callback=test_progress
        )
        print(f"LSTM test completed - Training time: {result['training_time']:.2f}s")
        
        # Performance stats
        stats = get_training_performance_stats()
        print(f"Performance stats: {stats['training_manager']}")
        
    except Exception as e:
        print(f"Training test failed: {e}")
        import traceback
        traceback.print_exc()
                                