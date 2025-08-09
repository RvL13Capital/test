# ======================== ANPASSUNG 1: secure_tasks.py ========================
# Ersetzen Sie die bestehenden Task-Implementierungen durch diese:

# project/tasks_integrated.py
"""
Updated Celery tasks that use the integrated system
Replace the content of secure_tasks.py with this
"""

from celery import Task
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

from .extensions import celery
from .integrated_system import IntegratedMLTradingSystem, IntegratedSystemConfig

logger = logging.getLogger(__name__)

class IntegratedTask(Task):
    """Base task class for integrated system"""
    
    _system = None
    
    @property
    def system(self):
        """Lazy load integrated system"""
        if IntegratedTask._system is None:
            config = IntegratedSystemConfig()
            IntegratedTask._system = IntegratedMLTradingSystem(config)
        return IntegratedTask._system
    
    def _audit_log(self, event: str, user_id: str, details: Dict, success: bool = True):
        """Audit logging"""
        logger.info(f"AUDIT: {event} by {user_id}: {details} - Success: {success}")

@celery.task(bind=True, base=IntegratedTask, name='tasks.train_model')
def train_model_integrated(self, ticker: str, model_type: str, user_id: str,
                          custom_params: Optional[Dict] = None):
    """
    Simplified training task using integrated system
    """
    try:
        self.update_state(state='PROGRESS', meta={
            'status': 'Initializing integrated pipeline...',
            'progress': 5
        })
        
        # Run through integrated system
        result = asyncio.run(
            self.system.run_complete_training_pipeline(ticker, model_type)
        )
        
        self._audit_log('training_completed', user_id, {
            'ticker': ticker,
            'model_type': model_type,
            'model_id': result.get('model_id')
        })
        
        return result
        
    except Exception as e:
        self._audit_log('training_failed', user_id, {
            'ticker': ticker,
            'error': str(e)
        }, success=False)
        raise

@celery.task(bind=True, base=IntegratedTask, name='tasks.update_eod_data')
def update_eod_data(self, tickers: list, user_id: str):
    """
    Update EOD data through integrated pipeline
    """
    try:
        result = asyncio.run(
            self.system.eod_pipeline.run_daily_update(tickers)
        )
        
        self._audit_log('eod_update_completed', user_id, {
            'tickers': tickers,
            'records': result.get('total_records', 0)
        })
        
        return result
        
    except Exception as e:
        self._audit_log('eod_update_failed', user_id, {
            'error': str(e)
        }, success=False)
        raise
