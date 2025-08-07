"""
Strategy Module - Contains trading strategy implementations
Separated from backtesting orchestration for better maintainability
"""

import backtrader as bt
import torch
import numpy as np
import traceback
import logging
from project.config import Config
from project.features import prepare_inference_from_df, _create_features
from project.signals import calculate_breakout_signal

logger = logging.getLogger(__name__)

class EnhancedModelBasedStrategy(bt.Strategy):
    """
    Enhanced Backtesting-Strategie mit verbessertem Logging und Performance-Tracking
    Separated into dedicated strategy module for better organization
    """
    params = (
        ('lstm_model', None),
        ('scaler', None),
        ('xgboost_model', None),
        ('full_test_df', None),
        ('selected_features', None),
        ('fold_id', 0)  # Track which fold this is
    )
    
    def __init__(self):
        self.order = None
        self.trade_log = []
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.prediction_log = []  # Track predictions
        self.signal_log = []      # Track signals
        
        # Performance tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info(f"Initialized strategy for fold {self.p.fold_id}")
    
    def log(self, txt, dt=None):
        """Enhanced logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.debug(f'Fold {self.p.fold_id} - {dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Enhanced order notification"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size:.2f}, Cost: {order.executed.value:.2f}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size:.2f}, Cost: {order.executed.value:.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.status}')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Enhanced trade notification with detailed logging"""
        if trade.isclosed:
            self.trade_count += 1
            
            trade_info = {
                'entry_price': trade.price,
                'exit_price': trade.price + trade.pnl / trade.size,
                'pnl': trade.pnl,
                'pnl_comm': trade.pnlcomm,
                'size': trade.size,
                'trade_id': self.trade_count,
                'fold_id': self.p.fold_id,
                'entry_date': self.data.datetime.date(-trade.barlen+1).isoformat(),
                'exit_date': self.data.datetime.date(0).isoformat(),
                'duration_bars': trade.barlen
            }
            
            self.trade_log.append(trade_info)
            
            if trade.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            win_rate = self.winning_trades / self.trade_count if self.trade_count > 0 else 0
            
            self.log(f'TRADE CLOSED - PnL: {trade.pnlcomm:.2f}, '
                    f'Size: {trade.size:.2f}, Duration: {trade.barlen} bars, '
                    f'Win Rate: {win_rate:.2%}')
    
    def next(self):
        """Enhanced main trading logic with comprehensive logging"""
        if self.order: 
            return
            
        current_idx = len(self.data)
        if current_idx < Config.DATA_WINDOW_SIZE: 
            return
        
        df_for_prediction = self.p.full_test_df.iloc[:current_idx]
        
        try:
            # Prepare data for prediction
            sequences = prepare_inference_from_df(
                df_for_prediction, 
                self.p.scaler, 
                Config.DATA_WINDOW_SIZE, 
                self.p.selected_features
            )
            
            if sequences.size == 0: 
                return
                
            # LSTM Prediction
            device = self.p.lstm_model.device
            src_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                lstm_preds = self.p.lstm_model.predict(
                    src_tensor, 
                    Config.DATA_PREDICTION_LENGTH
                )
            
            unscaled_lstm_preds = self.p.scaler.inverse_transform(
                lstm_preds.squeeze(0).cpu().numpy()
            )
            
            # XGBoost Prediction
            features = _create_features(df_for_prediction)
            latest_features = features[self.p.selected_features].iloc[-1].values.reshape(1, -1)
            xgboost_pred = self.p.xgboost_model.predict(latest_features)[0]
            
            # Calculate signal
            current_close = self.data.close[0]
            signal, confidence, metadata = calculate_breakout_signal(
                unscaled_lstm_preds, 
                xgboost_pred, 
                current_close,
                df_for_prediction
            )
            
            # Log predictions and signals
            prediction_info = {
                'date': self.data.datetime.date(0).isoformat(),
                'current_price': current_close,
                'lstm_pred_mean': float(np.mean(unscaled_lstm_preds)),
                'xgboost_pred': float(xgboost_pred),
                'signal': signal,
                'confidence': confidence if confidence is not None else 0,
                'atr': self.atr[0]
            }
            
            self.prediction_log.append(prediction_info)
            self.signal_log.append({
                'date': self.data.datetime.date(0).isoformat(),
                'signal': signal,
                'position_size': self.position.size if self.position else 0
            })
            
            # Enhanced Trading logic with better risk management
            if not self.position and self.atr[0] > 0:
                risk_per_trade = getattr(Config, 'RISK_PER_TRADE', 0.01)
                atr_multiplier = getattr(Config, 'ATR_MULTIPLIER', 2.0)
                
                if signal == "STRONG_BULLISH_BREAKOUT":
                    stop_price = current_close - atr_multiplier * self.atr[0]
                    risk_amount = current_close - stop_price
                    
                    if risk_amount > 0:  # Additional safety check
                        size = (self.broker.getvalue() * risk_per_trade) / risk_amount
                        size = min(size, self.broker.getvalue() * 0.95 / current_close)  # Max 95% of capital
                        
                        if size > 0:
                            self.buy(size=size)
                            self.sell(exectype=bt.Order.Stop, price=stop_price, size=size)
                            self.log(f'LONG SIGNAL - Entry: {current_close:.2f}, Stop: {stop_price:.2f}, Size: {size:.2f}')
                        
                elif signal == "STRONG_BEARISH_BREAKOUT":
                    stop_price = current_close + atr_multiplier * self.atr[0]
                    risk_amount = stop_price - current_close
                    
                    if risk_amount > 0:  # Additional safety check
                        size = (self.broker.getvalue() * risk_per_trade) / risk_amount
                        size = min(size, self.broker.getvalue() * 0.95 / current_close)  # Max 95% of capital
                        
                        if size > 0:
                            self.sell(size=size)
                            self.buy(exectype=bt.Order.Stop, price=stop_price, size=size)
                            self.log(f'SHORT SIGNAL - Entry: {current_close:.2f}, Stop: {stop_price:.2f}, Size: {size:.2f}')
                            
            elif self.position and signal == "NO_SIGNAL":
                self.close()
                self.log(f'CLOSING POSITION - Exit signal received')
                
        except Exception as e:
            logger.error(f"Fehler im Backtest bei Index {current_idx}: {traceback.format_exc()}")

# Legacy compatibility - kept for backward compatibility
class ModelBasedStrategy(EnhancedModelBasedStrategy):
    """
    Legacy strategy class for backward compatibility
    Simply inherits from EnhancedModelBasedStrategy
    """
    pass
