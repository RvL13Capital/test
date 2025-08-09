# project/signal_performance_tracker.py
"""
Track and analyze signal performance for continuous improvement
Feeds back into the optimization loop
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import logging

from .storage import get_gcs_storage
from .config import Config

logger = logging.getLogger(__name__)

class UserAction(Enum):
    """User actions on signals"""
    TAKEN = "TAKEN"
    IGNORED = "IGNORED"
    MODIFIED = "MODIFIED"
    WATCHING = "WATCHING"
    DISMISSED = "DISMISSED"

class SignalOutcome(Enum):
    """Signal outcome categories"""
    SUCCESS_T1 = "SUCCESS_T1"      # Hit target 1
    SUCCESS_T2 = "SUCCESS_T2"      # Hit target 2
    SUCCESS_T3 = "SUCCESS_T3"      # Hit target 3
    STOPPED_OUT = "STOPPED_OUT"    # Hit stop loss
    EXPIRED = "EXPIRED"            # Signal expired
    PARTIAL = "PARTIAL"            # Partial success
    PENDING = "PENDING"            # Still active

@dataclass
class SignalPerformanceRecord:
    """Complete record of signal performance"""
    
    # Signal identification
    signal_id: str
    ticker: str
    signal_timestamp: datetime
    signal_type: str
    signal_strength: str
    signal_confidence: float
    
    # User interaction
    user_id: str
    user_action: UserAction
    action_timestamp: Optional[datetime] = None
    user_notes: Optional[str] = None
    user_rating: Optional[int] = None  # 1-5 stars
    
    # Actual execution (if taken)
    actual_entry: Optional[float] = None
    actual_entry_time: Optional[datetime] = None
    actual_exit: Optional[float] = None
    actual_exit_time: Optional[datetime] = None
    position_size: Optional[float] = None
    
    # Outcome
    outcome: SignalOutcome = SignalOutcome.PENDING
    outcome_timestamp: Optional[datetime] = None
    actual_return: float = 0.0
    actual_return_percent: float = 0.0
    holding_period_days: int = 0
    
    # Performance vs prediction
    hit_target_1: bool = False
    hit_target_2: bool = False
    hit_target_3: bool = False
    hit_stop_loss: bool = False
    max_favorable_excursion: float = 0.0  # Best unrealized gain
    max_adverse_excursion: float = 0.0    # Worst unrealized loss
    
    # Original signal data
    original_targets: Dict = field(default_factory=dict)
    original_prediction: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Convert enums
        data['user_action'] = self.user_action.value if self.user_action else None
        data['outcome'] = self.outcome.value
        # Convert datetimes
        for key in ['signal_timestamp', 'action_timestamp', 'actual_entry_time', 
                    'actual_exit_time', 'outcome_timestamp']:
            if data.get(key):
                data[key] = data[key].isoformat() if isinstance(data[key], datetime) else data[key]
        return data

class SignalPerformanceTracker:
    """
    Track and analyze signal performance for optimization
    """
    
    def __init__(self):
        self.performance_records: List[SignalPerformanceRecord] = []
        self.active_signals: Dict[str, SignalPerformanceRecord] = {}
        self.performance_metrics: Dict = {}
        self.feedback_buffer: List[Dict] = []
        
        # Storage
        self.gcs = get_gcs_storage()
        
        # Thresholds
        self.min_feedback_for_learning = 50
        self.performance_window_days = 30
        
        logger.info("Signal Performance Tracker initialized")
    
    def track_signal_issued(self, signal: Any, user_id: str) -> SignalPerformanceRecord:
        """Track when a signal is issued"""
        
        record = SignalPerformanceRecord(
            signal_id=signal.signal_id,
            ticker=signal.ticker,
            signal_timestamp=signal.timestamp,
            signal_type=signal.signal_type.value,
            signal_strength=signal.strength.value,
            signal_confidence=signal.confidence,
            user_id=user_id,
            user_action=UserAction.WATCHING,
            original_targets={
                'entry': signal.targets.entry,
                'stop_loss': signal.targets.stop_loss,
                'target_1': signal.targets.target_1,
                'target_2': signal.targets.target_2,
                'target_3': signal.targets.target_3
            },
            original_prediction={
                'expected_return': signal.expected_return,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'expected_days': signal.expected_breakout_days
            }
        )
        
        self.active_signals[signal.signal_id] = record
        
        logger.info(f"Tracking signal {signal.signal_id} for user {user_id}")
        
        return record
    
    def record_user_action(self, signal_id: str, user_id: str, 
                          action: UserAction, details: Optional[Dict] = None):
        """Record user action on a signal"""
        
        if signal_id not in self.active_signals:
            logger.warning(f"Signal {signal_id} not found in active signals")
            return
        
        record = self.active_signals[signal_id]
        
        record.user_action = action
        record.action_timestamp = datetime.now()
        
        if details:
            record.actual_entry = details.get('entry_price')
            record.actual_entry_time = details.get('entry_time')
            record.position_size = details.get('position_size')
            record.user_notes = details.get('notes')
        
        logger.info(f"Recorded {action.value} for signal {signal_id}")
    
    def record_signal_outcome(self, signal_id: str, outcome_data: Dict) -> Optional[SignalPerformanceRecord]:
        """Record the actual outcome of a signal"""
        
        if signal_id not in self.active_signals:
            logger.warning(f"Signal {signal_id} not found")
            return None
        
        record = self.active_signals[signal_id]
        
        # Update outcome data
        record.outcome_timestamp = datetime.now()
        
        if 'exit_price' in outcome_data:
            record.actual_exit = outcome_data['exit_price']
            record.actual_exit_time = outcome_data.get('exit_time', datetime.now())
            
            # Calculate returns
            if record.actual_entry:
                record.actual_return = record.actual_exit - record.actual_entry
                record.actual_return_percent = (record.actual_return / record.actual_entry) * 100
                
                # Calculate holding period
                if record.actual_entry_time:
                    delta = record.actual_exit_time - record.actual_entry_time
                    record.holding_period_days = delta.days
        
        # Determine outcome
        if 'hit_target' in outcome_data:
            target_num = outcome_data['hit_target']
            if target_num == 1:
                record.hit_target_1 = True
                record.outcome = SignalOutcome.SUCCESS_T1
            elif target_num == 2:
                record.hit_target_2 = True
                record.outcome = SignalOutcome.SUCCESS_T2
            elif target_num == 3:
                record.hit_target_3 = True
                record.outcome = SignalOutcome.SUCCESS_T3
        
        elif outcome_data.get('stopped_out', False):
            record.hit_stop_loss = True
            record.outcome = SignalOutcome.STOPPED_OUT
        
        elif outcome_data.get('expired', False):
            record.outcome = SignalOutcome.EXPIRED
        
        # Record excursions if provided
        record.max_favorable_excursion = outcome_data.get('max_favorable_excursion', 0)
        record.max_adverse_excursion = outcome_data.get('max_adverse_excursion', 0)
        
        # Move to completed records
        self.performance_records.append(record)
        del self.active_signals[signal_id]
        
        # Update metrics
        self._update_performance_metrics(record)
        
        # Store to persistent storage
        self._store_performance_record(record)
        
        logger.info(f"Recorded outcome {record.outcome.value} for signal {signal_id}")
        
        return record
    
    def record_user_feedback(self, signal_id: str, user_id: str, feedback: Dict):
        """Record user feedback on signal quality"""
        
        feedback_entry = {
            'signal_id': signal_id,
            'user_id': user_id,
            'timestamp': datetime.now(),
            'rating': feedback.get('rating'),  # 1-5 stars
            'usefulness': feedback.get('usefulness'),  # Was it useful?
            'accuracy': feedback.get('accuracy'),  # Was it accurate?
            'notes': feedback.get('notes'),
            'would_follow_again': feedback.get('would_follow_again')
        }
        
        self.feedback_buffer.append(feedback_entry)
        
        # Update signal record if exists
        if signal_id in self.active_signals:
            record = self.active_signals[signal_id]
            record.user_rating = feedback.get('rating')
            record.user_notes = feedback.get('notes')
        
        logger.info(f"Recorded feedback for signal {signal_id}")
    
    def _update_performance_metrics(self, record: SignalPerformanceRecord):
        """Update aggregate performance metrics"""
        
        # Get recent records for metrics
        recent_records = self._get_recent_records(self.performance_window_days)
        
        if not recent_records:
            return
        
        # Calculate metrics
        self.performance_metrics = {
            'total_signals': len(recent_records),
            'signals_taken': len([r for r in recent_records if r.user_action == UserAction.TAKEN]),
            'win_rate': self._calculate_win_rate(recent_records),
            'average_return': self._calculate_average_return(recent_records),
            'hit_rate_t1': self._calculate_target_hit_rate(recent_records, 1),
            'hit_rate_t2': self._calculate_target_hit_rate(recent_records, 2),
            'hit_rate_t3': self._calculate_target_hit_rate(recent_records, 3),
            'stop_loss_rate': self._calculate_stop_loss_rate(recent_records),
            'average_holding_days': self._calculate_average_holding_period(recent_records),
            'sharpe_ratio': self._calculate_sharpe_ratio(recent_records),
            'profit_factor': self._calculate_profit_factor(recent_records),
            'accuracy_by_type': self._calculate_accuracy_by_type(recent_records),
            'accuracy_by_strength': self._calculate_accuracy_by_strength(recent_records),
            'user_satisfaction': self._calculate_user_satisfaction(recent_records)
        }
    
    def _calculate_win_rate(self, records: List[SignalPerformanceRecord]) -> float:
        """Calculate win rate"""
        
        taken_records = [r for r in records if r.user_action == UserAction.TAKEN]
        if not taken_records:
            return 0
        
        wins = [r for r in taken_records if r.actual_return_percent > 0]
        return len(wins) / len(taken_records)
    
    def _calculate_average_return(self, records: List[SignalPerformanceRecord]) -> float:
        """Calculate average return"""
        
        returns = [r.actual_return_percent for r in records 
                  if r.user_action == UserAction.TAKEN and r.actual_return_percent != 0]
        
        return np.mean(returns) if returns else 0
    
    def _calculate_target_hit_rate(self, records: List[SignalPerformanceRecord], 
                                  target_num: int) -> float:
        """Calculate hit rate for specific target"""
        
        relevant_records = [r for r in records if r.user_action == UserAction.TAKEN]
        if not relevant_records:
            return 0
        
        if target_num == 1:
            hits = [r for r in relevant_records if r.hit_target_1]
        elif target_num == 2:
            hits = [r for r in relevant_records if r.hit_target_2]
        else:
            hits = [r for r in relevant_records if r.hit_target_3]
        
        return len(hits) / len(relevant_records)
    
    def _calculate_stop_loss_rate(self, records: List[SignalPerformanceRecord]) -> float:
        """Calculate stop loss hit rate"""
        
        relevant_records = [r for r in records if r.user_action == UserAction.TAKEN]
        if not relevant_records:
            return 0
        
        stopped = [r for r in relevant_records if r.hit_stop_loss]
        return len(stopped) / len(relevant_records)
    
    def _calculate_average_holding_period(self, records: List[SignalPerformanceRecord]) -> float:
        """Calculate average holding period"""
        
        periods = [r.holding_period_days for r in records 
                  if r.user_action == UserAction.TAKEN and r.holding_period_days > 0]
        
        return np.mean(periods) if periods else 0
    
    def _calculate_sharpe_ratio(self, records: List[SignalPerformanceRecord]) -> float:
        """Calculate Sharpe ratio"""
        
        returns = [r.actual_return_percent for r in records 
                  if r.user_action == UserAction.TAKEN]
        
        if len(returns) < 2:
            return 0
        
        return_array = np.array(returns)
        if return_array.std() == 0:
            return 0
        
        # Annualized Sharpe (simplified)
        return (return_array.mean() / return_array.std()) * np.sqrt(252 / 20)  # Assuming 20 day holding
    
    def _calculate_profit_factor(self, records: List[SignalPerformanceRecord]) -> float:
        """Calculate profit factor"""
        
        wins = [r.actual_return_percent for r in records 
               if r.user_action == UserAction.TAKEN and r.actual_return_percent > 0]
        losses = [abs(r.actual_return_percent) for r in records 
                 if r.user_action == UserAction.TAKEN and r.actual_return_percent < 0]
        
        if not losses:
            return float('inf') if wins else 0
        
        return sum(wins) / sum(losses) if losses else 0
    
    def _calculate_accuracy_by_type(self, records: List[SignalPerformanceRecord]) -> Dict:
        """Calculate accuracy by signal type"""
        
        accuracy = {}
        
        # Group by signal type
        for signal_type in set(r.signal_type for r in records):
            type_records = [r for r in records if r.signal_type == signal_type]
            
            if type_records:
                wins = [r for r in type_records if r.actual_return_percent > 0]
                accuracy[signal_type] = len(wins) / len(type_records)
        
        return accuracy
    
    def _calculate_accuracy_by_strength(self, records: List[SignalPerformanceRecord]) -> Dict:
        """Calculate accuracy by signal strength"""
        
        accuracy = {}
        
        # Group by signal strength
        for strength in set(r.signal_strength for r in records):
            strength_records = [r for r in records if r.signal_strength == strength]
            
            if strength_records:
                wins = [r for r in strength_records if r.actual_return_percent > 0]
                accuracy[strength] = len(wins) / len(strength_records)
        
        return accuracy
    
    def _calculate_user_satisfaction(self, records: List[SignalPerformanceRecord]) -> float:
        """Calculate average user satisfaction"""
        
        ratings = [r.user_rating for r in records if r.user_rating is not None]
        return np.mean(ratings) if ratings else 0
    
    def _get_recent_records(self, days: int) -> List[SignalPerformanceRecord]:
        """Get records from recent days"""
        
        cutoff = datetime.now() - timedelta(days=days)
        return [r for r in self.performance_records if r.signal_timestamp >= cutoff]
    
    def _store_performance_record(self, record: SignalPerformanceRecord):
        """Store performance record to persistent storage"""
        
        if self.gcs:
            try:
                date_str = record.signal_timestamp.strftime('%Y%m%d')
                path = f"performance/{date_str}/{record.signal_id}.json"
                self.gcs.upload_json(record.to_dict(), path)
            except Exception as e:
                logger.error(f"Failed to store performance record: {e}")
    
    def generate_performance_report(self, period_days: int = 30) -> Dict:
        """Generate comprehensive performance report"""
        
        records = self._get_recent_records(period_days)
        
        if not records:
            return {'message': 'No performance data available'}
        
        report = {
            'period': f"{period_days} days",
            'period_start': (datetime.now() - timedelta(days=period_days)).isoformat(),
            'period_end': datetime.now().isoformat(),
            'summary': self.performance_metrics,
            'detailed_analysis': {
                'signals_by_outcome': self._analyze_by_outcome(records),
                'performance_by_ticker': self._analyze_by_ticker(records),
                'time_analysis': self._analyze_timing(records),
                'user_behavior': self._analyze_user_behavior(records),
                'prediction_accuracy': self._analyze_prediction_accuracy(records)
            },
            'improvement_suggestions': self._generate_improvement_suggestions(records),
            'top_performing_signals': self._get_top_signals(records, 10),
            'worst_performing_signals': self._get_worst_signals(records, 10)
            # project/signal_performance_tracker.py (Fortsetzung)

        }
        
        return report
    
    def _analyze_by_outcome(self, records: List[SignalPerformanceRecord]) -> Dict:
        """Analyze signals by outcome"""
        
        outcome_counts = {}
        for outcome in SignalOutcome:
            count = len([r for r in records if r.outcome == outcome])
            outcome_counts[outcome.value] = count
        
        return outcome_counts
    
    def _analyze_by_ticker(self, records: List[SignalPerformanceRecord]) -> Dict:
        """Analyze performance by ticker"""
        
        ticker_performance = {}
        
        for ticker in set(r.ticker for r in records):
            ticker_records = [r for r in records if r.ticker == ticker]
            
            ticker_performance[ticker] = {
                'total_signals': len(ticker_records),
                'win_rate': self._calculate_win_rate(ticker_records),
                'average_return': self._calculate_average_return(ticker_records),
                'best_return': max([r.actual_return_percent for r in ticker_records], default=0),
                'worst_return': min([r.actual_return_percent for r in ticker_records], default=0)
            }
        
        return ticker_performance
    
    def _analyze_timing(self, records: List[SignalPerformanceRecord]) -> Dict:
        """Analyze timing accuracy"""
        
        timing_analysis = {
            'average_predicted_days': np.mean([r.original_prediction.get('expected_days', 0) 
                                              for r in records]),
            'average_actual_days': np.mean([r.holding_period_days for r in records 
                                           if r.holding_period_days > 0]),
            'timing_accuracy': 0,
            'early_exits': 0,
            'late_exits': 0
        }
        
        # Calculate timing accuracy
        timing_errors = []
        for r in records:
            if r.holding_period_days > 0 and r.original_prediction.get('expected_days'):
                error = abs(r.holding_period_days - r.original_prediction['expected_days'])
                timing_errors.append(error)
                
                if r.holding_period_days < r.original_prediction['expected_days'] - 2:
                    timing_analysis['early_exits'] += 1
                elif r.holding_period_days > r.original_prediction['expected_days'] + 2:
                    timing_analysis['late_exits'] += 1
        
        if timing_errors:
            timing_analysis['timing_accuracy'] = 1 - (np.mean(timing_errors) / 10)  # Normalized
        
        return timing_analysis
    
    def _analyze_user_behavior(self, records: List[SignalPerformanceRecord]) -> Dict:
        """Analyze user behavior patterns"""
        
        behavior = {
            'action_distribution': {},
            'average_time_to_action': 0,
            'modification_rate': 0,
            'follow_through_rate': 0
        }
        
        # Action distribution
        for action in UserAction:
            count = len([r for r in records if r.user_action == action])
            behavior['action_distribution'][action.value] = count
        
        # Time to action
        action_times = []
        for r in records:
            if r.action_timestamp and r.signal_timestamp:
                delta = r.action_timestamp - r.signal_timestamp
                action_times.append(delta.total_seconds() / 3600)  # Hours
        
        if action_times:
            behavior['average_time_to_action'] = np.mean(action_times)
        
        # Modification rate
        total_taken = len([r for r in records if r.user_action == UserAction.TAKEN])
        modified = len([r for r in records if r.user_action == UserAction.MODIFIED])
        
        if total_taken + modified > 0:
            behavior['modification_rate'] = modified / (total_taken + modified)
        
        # Follow-through rate
        watching = len([r for r in records if r.user_action == UserAction.WATCHING])
        if watching > 0:
            behavior['follow_through_rate'] = total_taken / watching
        
        return behavior
    
    def _analyze_prediction_accuracy(self, records: List[SignalPerformanceRecord]) -> Dict:
        """Analyze prediction accuracy"""
        
        accuracy = {
            'return_prediction_error': 0,
            'risk_reward_accuracy': 0,
            'target_accuracy': {},
            'confidence_calibration': 0
        }
        
        # Return prediction error
        return_errors = []
        for r in records:
            if r.actual_return_percent != 0 and r.original_prediction.get('expected_return'):
                expected = r.original_prediction['expected_return'] * 100
                error = abs(r.actual_return_percent - expected)
                return_errors.append(error)
        
        if return_errors:
            accuracy['return_prediction_error'] = np.mean(return_errors)
        
        # Risk/Reward accuracy
        rr_achieved = []
        for r in records:
            if r.actual_return_percent != 0 and r.original_prediction.get('risk_reward_ratio'):
                actual_rr = abs(r.actual_return_percent / r.original_targets.get('risk_percent', 5))
                expected_rr = r.original_prediction['risk_reward_ratio']
                rr_achieved.append(actual_rr / expected_rr if expected_rr > 0 else 0)
        
        if rr_achieved:
            accuracy['risk_reward_accuracy'] = np.mean(rr_achieved)
        
        # Target accuracy
        for i in range(1, 4):
            target_key = f'target_{i}'
            hits = len([r for r in records if getattr(r, f'hit_target_{i}', False)])
            total = len([r for r in records if r.original_targets.get(target_key)])
            
            if total > 0:
                accuracy['target_accuracy'][target_key] = hits / total
        
        # Confidence calibration
        confidence_buckets = {}
        for r in records:
            bucket = round(r.signal_confidence * 10) / 10  # Round to nearest 0.1
            if bucket not in confidence_buckets:
                confidence_buckets[bucket] = {'total': 0, 'wins': 0}
            
            confidence_buckets[bucket]['total'] += 1
            if r.actual_return_percent > 0:
                confidence_buckets[bucket]['wins'] += 1
        
        calibration_errors = []
        for confidence, outcomes in confidence_buckets.items():
            if outcomes['total'] > 0:
                actual_win_rate = outcomes['wins'] / outcomes['total']
                error = abs(actual_win_rate - confidence)
                calibration_errors.append(error)
        
        if calibration_errors:
            accuracy['confidence_calibration'] = 1 - np.mean(calibration_errors)
        
        return accuracy
    
    def _generate_improvement_suggestions(self, records: List[SignalPerformanceRecord]) -> List[str]:
        """Generate improvement suggestions based on performance"""
        
        suggestions = []
        metrics = self.performance_metrics
        
        # Win rate suggestions
        if metrics.get('win_rate', 0) < 0.5:
            suggestions.append("Consider increasing minimum confidence threshold - current win rate below 50%")
        
        # Stop loss suggestions
        if metrics.get('stop_loss_rate', 0) > 0.4:
            suggestions.append("High stop-loss rate detected - consider wider stops or better entry timing")
        
        # Target suggestions
        if metrics.get('hit_rate_t1', 0) < 0.6:
            suggestions.append("Target 1 hit rate low - consider more conservative first targets")
        
        # User behavior suggestions
        taken_rate = metrics.get('signals_taken', 0) / max(metrics.get('total_signals', 1), 1)
        if taken_rate < 0.2:
            suggestions.append("Low signal utilization - consider adjusting signal parameters to match risk tolerance")
        
        # Timing suggestions
        timing_analysis = self._analyze_timing(records)
        if timing_analysis['timing_accuracy'] < 0.7:
            suggestions.append("Timing predictions need improvement - consider retraining timing models")
        
        # Confidence calibration
        pred_accuracy = self._analyze_prediction_accuracy(records)
        if pred_accuracy['confidence_calibration'] < 0.8:
            suggestions.append("Confidence scores not well calibrated - adjust confidence calculation")
        
        # Type-specific suggestions
        accuracy_by_type = metrics.get('accuracy_by_type', {})
        for signal_type, accuracy in accuracy_by_type.items():
            if accuracy < 0.4:
                suggestions.append(f"Poor performance for {signal_type} signals - consider removing or improving")
        
        return suggestions
    
    def _get_top_signals(self, records: List[SignalPerformanceRecord], n: int) -> List[Dict]:
        """Get top performing signals"""
        
        # Sort by return
        sorted_records = sorted(records, key=lambda r: r.actual_return_percent, reverse=True)
        
        top_signals = []
        for record in sorted_records[:n]:
            top_signals.append({
                'signal_id': record.signal_id,
                'ticker': record.ticker,
                'return': f"{record.actual_return_percent:.2f}%",
                'holding_days': record.holding_period_days,
                'signal_type': record.signal_type,
                'confidence': record.signal_confidence
            })
        
        return top_signals
    
    def _get_worst_signals(self, records: List[SignalPerformanceRecord], n: int) -> List[Dict]:
        """Get worst performing signals"""
        
        # Sort by return (ascending)
        sorted_records = sorted(records, key=lambda r: r.actual_return_percent)
        
        worst_signals = []
        for record in sorted_records[:n]:
            worst_signals.append({
                'signal_id': record.signal_id,
                'ticker': record.ticker,
                'return': f"{record.actual_return_percent:.2f}%",
                'holding_days': record.holding_period_days,
                'signal_type': record.signal_type,
                'confidence': record.signal_confidence,
                'outcome': record.outcome.value
            })
        
        return worst_signals
    
    def should_trigger_learning(self) -> bool:
        """Check if enough feedback accumulated to trigger learning"""
        
        return len(self.feedback_buffer) >= self.min_feedback_for_learning
    
    def get_recent_feedback(self) -> List[Dict]:
        """Get recent feedback for learning"""
        
        feedback = self.feedback_buffer.copy()
        self.feedback_buffer.clear()  # Clear after retrieval
        return feedback
    
    def export_performance_data(self, format: str = 'csv') -> Any:
        """Export performance data for analysis"""
        
        if not self.performance_records:
            return None
        
        df = pd.DataFrame([r.to_dict() for r in self.performance_records])
        
        if format == 'csv':
            return df.to_csv(index=False)
        elif format == 'json':
            return df.to_json(orient='records')
        elif format == 'dataframe':
            return df
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_optimization_feedback(self) -> Dict:
        """Get feedback data formatted for optimizer"""
        
        recent_records = self._get_recent_records(30)
        
        if not recent_records:
            return {}
        
        return {
            'performance_metrics': self.performance_metrics,
            'user_satisfaction': self._calculate_user_satisfaction(recent_records),
            'signal_accuracy': {
                'by_type': self._calculate_accuracy_by_type(recent_records),
                'by_strength': self._calculate_accuracy_by_strength(recent_records)
            },
            'prediction_errors': self._analyze_prediction_accuracy(recent_records),
            'improvement_areas': self._generate_improvement_suggestions(recent_records),
            'sample_size': len(recent_records),
            'period_days': 30
        }
