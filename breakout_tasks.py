# project/breakout_tasks.py
"""
Asynchronous tasks for the self-optimizing breakout system
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import yfinance as yf

from .extensions import celery
from .tasks import SecureTask
from .consolidation_network import NetworkConsolidationAnalyzer, extract_consolidation_features
from .breakout_strategy import BreakoutScreener, BreakoutPredictor
from .auto_optimizer import get_auto_optimizer
from .storage import get_gcs_storage
from .monitoring import CELERY_TASKS_TOTAL, CELERY_TASK_DURATION

logger = logging.getLogger(__name__)

@celery.task(bind=True, base=SecureTask, time_limit=300)
def analyze_breakout_async(self, ticker: str, lookback_days: int = 90, 
                          user_id: str = None) -> Dict:
    """
    Asynchronously analyze breakout potential for a single stock
    """
    start_time = datetime.now()
    
    try:
        self.update_state(state='PROGRESS', meta={
            'status': f'Fetching data for {ticker}...',
            'progress': 10
        })
        
        # Fetch stock data
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{lookback_days}d")
        
        if len(df) < 30:
            return {
                'status': 'FAILED',
                'error': 'Insufficient historical data',
                'ticker': ticker
            }
        
        # Get market cap
        info = stock.info
        market_cap = info.get('marketCap', 0)
        
        self.update_state(state='PROGRESS', meta={
            'status': 'Analyzing consolidation patterns...',
            'progress': 40
        })
        
        # Analyze consolidation
        analyzer = NetworkConsolidationAnalyzer()
        consolidations = analyzer.analyze_consolidation(df, market_cap)
        
        # Extract features
        df_features = extract_consolidation_features(df, market_cap)
        
        self.update_state(state='PROGRESS', meta={
            'status': 'Calculating breakout probability...',
            'progress': 70
        })
        
        # Get current phase
        current_phase = analyzer.get_current_phase(df)
        
        # Prepare result
        result = {
            'status': 'SUCCESS',
            'ticker': ticker,
            'market_cap': market_cap,
            'sector': info.get('sector', 'Unknown'),
            'current_phase': current_phase,
            'consolidations': [
                {
                    'duration_days': c.duration_days,
                    'breakout_probability': c.breakout_probability,
                    'expected_move': c.expected_move,
                    'phase_transition_score': c.phase_transition_score
                }
                for c in consolidations
            ],
            'latest_features': df_features.tail(1).to_dict('records')[0] if len(df_features) > 0 else {},
            'analysis_timestamp': datetime.now().isoformat(),
            'execution_time': (datetime.now() - start_time).total_seconds()
        }
        
        # Store result
        if get_gcs_storage():
            storage_path = f"analysis_results/{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            get_gcs_storage().upload_json(result, storage_path, user_id=user_id)
        
        # Audit log
        self._audit_log('breakout_analysis_completed', user_id or 'system', {
            'ticker': ticker,
            'breakout_probability': result['consolidations'][-1]['breakout_probability'] if result['consolidations'] else 0
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Breakout analysis failed for {ticker}: {e}")
        
        self._audit_log('breakout_analysis_failed', user_id or 'system', {
            'ticker': ticker,
            'error': str(e)
        }, success=False)
        
        return {
            'status': 'FAILED',
            'error': str(e),
            'ticker': ticker
        }

@celery.task(bind=True, base=SecureTask, time_limit=600)
def screen_market_async(self, min_market_cap: float = 10e6, 
                       max_market_cap: float = 2e9,
                       min_consolidation_days: int = 20,
                       limit: int = 20,
                       sector_filter: Optional[List[str]] = None,
                       user_id: str = None) -> Dict:
    """
    Asynchronously screen market for breakout candidates
    """
    start_time = datetime.now()
    
    try:
        self.update_state(state='PROGRESS', meta={
            'status': 'Loading stock universe...',
            'progress': 5
        })
        
        # Get stock universe (in production, query from database)
        stock_universe = _get_small_cap_universe(min_market_cap, max_market_cap, sector_filter)
        
        self.update_state(state='PROGRESS', meta={
            'status': f'Screening {len(stock_universe)} stocks...',
            'progress': 20
        })
        
        screener = BreakoutScreener(min_market_cap, max_market_cap)
        analyzer = NetworkConsolidationAnalyzer(min_consolidation_days=min_consolidation_days)
        
        candidates = []
        stocks_processed = 0
        
        for ticker in stock_universe[:limit * 3]:  # Screen 3x to get enough candidates
            try:
                # Update progress
                progress = 20 + (60 * stocks_processed / len(stock_universe))
                self.update_state(state='PROGRESS', meta={
                    'status': f'Analyzing {ticker}...',
                    'progress': min(progress, 80)
                })
                
                # Get stock data
                stock = yf.Ticker(ticker)
                df = stock.history(period="120d")
                
                if len(df) < min_consolidation_days:
                    continue
                
                # Get info
                info = stock.info
                market_cap = info.get('marketCap', 0)
                
                # Check market cap
                if not (min_market_cap <= market_cap <= max_market_cap):
                    continue
                
                # Check sector filter
                if sector_filter and info.get('sector') not in sector_filter:
                    continue
                
                # Analyze current phase
                current_phase = analyzer.get_current_phase(df, lookback_days=60)
                
                if current_phase['status'] == 'consolidation':
                    # Analyze full consolidation
                    consolidations = analyzer.analyze_consolidation(df, market_cap)
                    
                    if consolidations:
                        latest = consolidations[-1]
                        
                        candidate = {
                            'ticker': ticker,
                            'company_name': info.get('longName', ticker),
                            'market_cap': market_cap,
                            'sector': info.get('sector', 'Unknown'),
                            'industry': info.get('industry', 'Unknown'),
                            'avg_volume': df['Volume'].mean(),
                            'consolidation_days': latest.duration_days,
                            'phase_transition_score': latest.phase_transition_score,
                            'accumulation_score': latest.accumulation_score,
                            'breakout_probability': latest.breakout_probability,
                            'expected_move': latest.expected_move,
                            'volume_pattern': latest.volume_pattern,
                            'current_price': df['Close'].iloc[-1],
                            'score': latest.breakout_probability * latest.expected_move
                        }
                        
                        candidates.append(candidate)
                
                stocks_processed += 1
                
            except Exception as e:
                logger.warning(f"Error screening {ticker}: {e}")
                continue
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        candidates = candidates[:limit]
        
        self.update_state(state='PROGRESS', meta={
            'status': 'Finalizing results...',
            'progress': 90
        })
        
        # Prepare result
        result = {
            'status': 'SUCCESS',
            'candidates': candidates,
            'total_screened': stocks_processed,
            'total_found': len(candidates),
            'screening_criteria': {
                'min_market_cap': min_market_cap,
                'max_market_cap': max_market_cap,
                'min_consolidation_days': min_consolidation_days,
                'sector_filter': sector_filter
            },
            'timestamp': datetime.now().isoformat(),
            'execution_time': (datetime.now() - start_time).total_seconds()
        }
        
        # Store result
        if get_gcs_storage():
            storage_path = f"screening_results/screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            get_gcs_storage().upload_json(result, storage_path, user_id=user_id)
        
        # Update market snapshot
        _update_market_snapshot(candidates)
        
        # Audit log
        self._audit_log('market_screening_completed', user_id or 'system', {
            'candidates_found': len(candidates),
            'stocks_screened': stocks_processed
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Market screening failed: {e}")
        
        self._audit_log('market_screening_failed', user_id or 'system', {
            'error': str(e)
        }, success=False)
        
        return {
            'status': 'FAILED',
            'error': str(e)
        }

@celery.task(bind=True, base=SecureTask, time_limit=900)
def optimize_system_async(self, force: bool = False, user_id: str = None) -> Dict:
    """
    Run system optimization cycle asynchronously
    """
    start_time = datetime.now()
    
    try:
        self.update_state(state='PROGRESS', meta={
            'status': 'Starting optimization cycle...',
            'progress': 5
        })
        
        optimizer = get_auto_optimizer()
        
        # Check if optimization is needed
        if not force and (datetime.now() - optimizer.last_optimization) < timedelta(hours=12):
            return {
                'status': 'SKIPPED',
                'message': 'Optimization not needed yet',
                'last_optimization': optimizer.last_optimization.isoformat()
            }
        
        self.update_state(state='PROGRESS', meta={
            'status': 'Collecting performance data...',
            'progress': 20
        })
        
        # Trigger optimization
        optimizer._run_optimization_cycle()
        
        # Get results
        if optimizer.optimization_history:
            latest_result = optimizer.optimization_history[-1]
            
            return {
                'status': 'SUCCESS',
                'optimization_result': {
                    'models_evaluated': latest_result.models_evaluated,
                    'improvement_pct': latest_result.improvement_pct,
                    'new_model_id': latest_result.best_model_id,
                    'validation_metrics': latest_result.validation_metrics
                },
                'timestamp': datetime.now().isoformat(),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
        else:
            return {
                'status': 'FAILED',
                'error': 'Optimization completed but no results available'
            }
        
    except Exception as e:
        logger.error(f"System optimization failed: {e}")
        
        return {
            'status': 'FAILED',
            'error': str(e)
        }

@celery.task(bind=True, time_limit=120)
def monitor_breakouts_async(self) -> Dict:
    """
    Monitor current positions and alert on breakouts
    """
    try:
        # Get current watchlist
        watchlist = _get_breakout_watchlist()
        
        alerts = []
        
        for ticker in watchlist:
            try:
                # Get latest data
                stock = yf.Ticker(ticker)
                df = stock.history(period="5d")
                
                if len(df) < 2:
                    continue
                
                # Check for breakout
                latest_close = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2]
                latest_volume = df['Volume'].iloc[-1]
                avg_volume = df['Volume'].mean()
                
                # Simple breakout detection
                price_change = (latest_close - prev_close) / prev_close
                volume_surge = latest_volume / avg_volume
                
                if price_change > 0.05 and volume_surge > 2.0:  # 5% move on 2x volume
                    alert = {
                        'ticker': ticker,
                        'price_change': price_change,
                        'volume_surge': volume_surge,
                        'current_price': latest_close,
                        'alert_type': 'BREAKOUT',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    alerts.append(alert)
                    
                    # Send real-time alert
                    _send_breakout_alert(alert)
                    
            except Exception as e:
                logger.error(f"Error monitoring {ticker}: {e}")
                continue
        
        return {
            'status': 'SUCCESS',
            'alerts_generated': len(alerts),
            'alerts': alerts,
            'watchlist_size': len(watchlist)
        }
        
    except Exception as e:
        logger.error(f"Breakout monitoring failed: {e}")
        return {
            'status': 'FAILED',
            'error': str(e)
        }

# Helper functions

def _get_small_cap_universe(min_cap: float, max_cap: float, 
                          sector_filter: Optional[List[str]] = None) -> List[str]:
    """Get universe of small cap stocks"""
    # In production, query from database
    # For now, return curated list
    
    universe = [
        # Nano caps
        'BNGO', 'GEVO', 'SENS', 'PROG', 'ATOS', 'XELA', 'BBIG', 'CEI',
        'MULN', 'NILE', 'INDO', 'IMPP', 'COSM', 'RELI', 'BKTI', 'APRN',
        # Micro caps
        'ALLK', 'PRPO', 'CLOV', 'WISH', 'MILE', 'GOEV', 'RIDE', 'WKHS',
        'ARVL', 'LCID', 'RIVN', 'FFIE', 'NKLA', 'HYLN', 'PTRA', 'LEV',
        # Small caps
        'FUBO', 'VERU', 'ATER', 'BBAI', 'GREE', 'SPRT', 'XPEV', 'LI',
        'NIO', 'FSR', 'PSFE', 'OPEN', 'SOFI', 'UPST', 'AFRM', 'HOOD'
    ]
    
    return universe

def _update_market_snapshot(candidates: List[Dict]):
    """Update market snapshot with latest candidates"""
    try:
        gcs = get_gcs_storage()
        if gcs:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'candidates': candidates,
                'market_conditions': _get_market_conditions()
            }
            
            gcs.upload_json(
                snapshot,
                f"market_data/snapshot_{datetime.now().strftime('%Y%m%d_%H')}.json"
            )
            
    except Exception as e:
        logger.error(f"Failed to update market snapshot: {e}")

def _get_market_conditions() -> Dict:
    """Get current market conditions"""
    try:
        # Get market indices
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="30d")
        
        iwm = yf.Ticker("IWM")
        iwm_hist = iwm.history(period="30d")
        
        return {
            'spy_trend': 'up' if spy_hist['Close'].iloc[-1] > spy_hist['Close'].iloc[0] else 'down',
            'iwm_trend': 'up' if iwm_hist['Close'].iloc[-1] > iwm_hist['Close'].iloc[0] else 'down',
            'market_volatility': spy_hist['Close'].pct_change().std() * np.sqrt(252),
            'small_cap_strength': (iwm_hist['Close'].iloc[-1] / iwm_hist['Close'].iloc[0]) / 
                                 (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0])
        }
        
    except Exception as e:
        logger.error(f"Failed to get market conditions: {e}")
        return {}

def _get_breakout_watchlist() -> List[str]:
    """Get current breakout watchlist"""
    # In production, query from database based on screening results
    # For now, return top candidates from recent screening
    
    try:
        gcs = get_gcs_storage()
        if gcs:
            # Get latest screening results
            results = gcs.list_models(prefix="screening_results/")
            if results:
                latest = max(results, key=lambda x: x['updated'])
                data = gcs.download_json(latest['name'])
                
                return [c['ticker'] for c in data.get('candidates', [])[:20]]
                
    except Exception as e:
        logger.error(f"Failed to get watchlist: {e}")
    
    return []

def _send_breakout_alert(alert: Dict):
    """Send real-time breakout alert"""
    try:
        # In production, send via WebSocket, email, webhook, etc.
        logger.info(f"BREAKOUT ALERT: {alert['ticker']} - "
                   f"{alert['price_change']:.2%} on {alert['volume_surge']:.1f}x volume")
        
        # Store alert
        gcs = get_gcs_storage()
        if gcs:
            gcs.upload_json(
                alert,
                f"alerts/breakout_{alert['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")

# Scheduled tasks
@celery.task
def scheduled_market_scan():
    """Scheduled task to scan market"""
    screen_market_async.delay(
        min_market_cap=10e6,
        max_market_cap=2e9,
        limit=50,
        user_id='scheduler'
    )

@celery.task
def scheduled_optimization():
    """Scheduled task to run optimization"""
    optimize_system_async.delay(force=False, user_id='scheduler')

@celery.task
def scheduled_monitoring():
    """Scheduled task to monitor breakouts"""
    monitor_breakouts_async.delay()