# project/eod_data_pipeline.py
"""
EOD (End-of-Day) Data Pipeline - Complete Implementation
Focused exclusively on daily OHLCV data with long history support
"""

import pandas as pd
import numpy as np
import logging
import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, time, date
from dataclasses import dataclass, asdict, field
from enum import Enum
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2.pool import ThreadedConnectionPool
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.pool import QueuePool
import redis
import pickle
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import schedule
import time as time_module
import holidays
import yfinance as yf
from retrying import retry

# Import from project modules where available
try:
    from .fixed_market_data import get_market_data_manager
    from .config import Config
except ImportError:
    # Fallback for standalone testing
    Config = type('Config', (), {
        'DB_HOST': 'localhost',
        'DB_PORT': 5432,
        'DB_NAME': 'trading_eod_db',
        'DB_USER': 'postgres',
        'DB_PASSWORD': 'password',
        'CELERY_RESULT_BACKEND': 'redis://localhost:6379/0'
    })

logger = logging.getLogger(__name__)

# ======================== Configuration ========================

@dataclass
class EODConfig:
    """Complete EOD Pipeline Configuration"""
    # Database settings
    db_host: str = field(default_factory=lambda: os.getenv('DB_HOST', 'localhost'))
    db_port: int = field(default_factory=lambda: int(os.getenv('DB_PORT', '5432')))
    db_name: str = field(default_factory=lambda: os.getenv('DB_NAME', 'trading_eod_db'))
    db_user: str = field(default_factory=lambda: os.getenv('DB_USER', 'postgres'))
    db_password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD', 'password'))
    
    # Data settings
    max_history_years: int = 20  # Maximum years of history to fetch
    default_lookback_days: int = 252 * 5  # 5 years default (252 trading days/year)
    batch_size: int = 100  # Records per batch insert
    chunk_size: int = 500  # Records per chunk for processing
    
    # Cache settings
    use_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    redis_url: str = field(default_factory=lambda: os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
    
    # API settings
    rate_limit_delay: float = 0.1  # Delay between API calls in seconds
    max_retries: int = 3
    timeout: int = 30
    
    # Processing settings
    parallel_workers: int = 4
    use_multiprocessing: bool = False  # Use threading by default
    
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

# ======================== Market Calendar ========================

class MarketCalendar:
    """Handles market holidays and trading days for EOD data"""
    
    def __init__(self, country: str = 'US'):
        self.country = country
        self.holidays = holidays.US(years=range(2000, 2030))
        self.market_open = time(9, 30)  # 9:30 AM EST
        self.market_close = time(16, 0)  # 4:00 PM EST
        self._cache = {}  # Cache for is_trading_day checks
    
    def is_trading_day(self, check_date: Union[datetime, date]) -> bool:
        """Check if given date is a trading day"""
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        
        # Check cache first
        if check_date in self._cache:
            return self._cache[check_date]
        
        # Weekend check
        if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            self._cache[check_date] = False
            return False
        
        # Holiday check
        if check_date in self.holidays:
            self._cache[check_date] = False
            return False
        
        self._cache[check_date] = True
        return True
    
    def get_last_trading_day(self, from_date: datetime = None) -> datetime:
        """Get the last trading day before or on the given date"""
        if from_date is None:
            from_date = datetime.now()
        
        current = from_date.date() if isinstance(from_date, datetime) else from_date
        
        while not self.is_trading_day(current):
            current -= timedelta(days=1)
        
        return datetime.combine(current, time(16, 0))  # Market close time
    
    def get_next_trading_day(self, from_date: datetime = None) -> datetime:
        """Get the next trading day after the given date"""
        if from_date is None:
            from_date = datetime.now()
        
        current = from_date.date() if isinstance(from_date, datetime) else from_date
        current += timedelta(days=1)
        
        while not self.is_trading_day(current):
            current += timedelta(days=1)
        
        return datetime.combine(current, time(9, 30))  # Market open time
    
    def get_trading_days_between(self, start_date: datetime, end_date: datetime) -> List[date]:
        """Get all trading days between two dates"""
        trading_days = []
        current = start_date.date() if isinstance(start_date, datetime) else start_date
        end = end_date.date() if isinstance(end_date, datetime) else end_date
        
        while current <= end:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)
        
        return trading_days
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        
        if not self.is_trading_day(now):
            return False
        
        current_time = now.time()
        return self.market_open <= current_time <= self.market_close

# ======================== EOD Database Manager ========================

class EODDatabase:
    """
    Complete database manager for EOD data using TimescaleDB
    """
    
    def __init__(self, config: EODConfig):
        self.config = config
        self.engine = None
        self.pool = None
        self.metadata = MetaData()
        self.market_calendar = MarketCalendar()
        self._init_connection()
        self._init_tables()
    
    def _init_connection(self):
        """Initialize database connection with proper error handling"""
        try:
            # Create SQLAlchemy engine
            self.engine = create_engine(
                self.config.connection_string,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False
            )
            
            # Create psycopg2 connection pool
            self.pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"Connected to EOD database at {self.config.db_host}:{self.config.db_port}")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def _init_tables(self):
        """Create all necessary EOD tables with TimescaleDB optimization"""
        with self.engine.connect() as conn:
            try:
                # Enable TimescaleDB extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
                conn.commit()
                
                # Main EOD data table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS eod_data (
                        date DATE NOT NULL,
                        ticker VARCHAR(10) NOT NULL,
                        open DOUBLE PRECISION,
                        high DOUBLE PRECISION,
                        low DOUBLE PRECISION,
                        close DOUBLE PRECISION,
                        adjusted_close DOUBLE PRECISION,
                        volume BIGINT,
                        dividends DOUBLE PRECISION DEFAULT 0,
                        stock_splits DOUBLE PRECISION DEFAULT 1,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (ticker, date)
                    )
                """))
                
                # Convert to hypertable for time-series optimization
                try:
                    conn.execute(text("""
                        SELECT create_hypertable('eod_data', 'date',
                            chunk_time_interval => INTERVAL '1 year',
                            if_not_exists => TRUE)
                    """))
                except Exception as e:
                    logger.debug(f"Hypertable already exists or not supported: {e}")
                
                # Create indexes for performance
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_eod_ticker_date 
                    ON eod_data (ticker, date DESC)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_eod_date 
                    ON eod_data (date DESC)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_eod_ticker 
                    ON eod_data (ticker)
                """))
                
                # Features table for computed indicators
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS eod_features (
                        date DATE NOT NULL,
                        ticker VARCHAR(10) NOT NULL,
                        features JSONB NOT NULL,
                        feature_version INTEGER DEFAULT 1,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (ticker, date, feature_version)
                    )
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_eod_features_ticker_date
                    ON eod_features (ticker, date DESC)
                """))
                
                # Data quality tracking table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS eod_data_quality (
                        ticker VARCHAR(10) NOT NULL,
                        last_update TIMESTAMPTZ NOT NULL,
                        first_date DATE,
                        last_date DATE,
                        total_days INTEGER,
                        missing_days INTEGER,
                        data_gaps JSONB,
                        quality_score FLOAT,
                        PRIMARY KEY (ticker)
                    )
                """))
                
                # ETL job tracking
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS eod_etl_jobs (
                        job_id SERIAL PRIMARY KEY,
                        job_type VARCHAR(50) NOT NULL,
                        started_at TIMESTAMPTZ NOT NULL,
                        completed_at TIMESTAMPTZ,
                        status VARCHAR(20) NOT NULL,
                        tickers_processed INTEGER DEFAULT 0,
                        records_inserted INTEGER DEFAULT 0,
                        records_updated INTEGER DEFAULT 0,
                        errors JSONB,
                        metadata JSONB
                    )
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_etl_jobs_started 
                    ON eod_etl_jobs (started_at DESC)
                """))
                
                conn.commit()
                logger.info("EOD database tables initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize tables: {e}")
                raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool with context manager"""
        conn = self.pool.getconn()
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.pool.putconn(conn)
    
    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def insert_eod_data(self, data: List[Dict]) -> Tuple[int, int]:
        """
        Bulk insert/update EOD data
        Returns: (inserted_count, updated_count)
        """
        if not data:
            return 0, 0
        
        inserted = 0
        updated = 0
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            try:
                # First, check which records exist
                check_query = """
                    SELECT ticker, date FROM eod_data 
                    WHERE (ticker, date) IN %s
                """
                
                # Create list of (ticker, date) tuples
                check_params = [(d['ticker'], d['date']) for d in data]
                cur.execute(check_query, (tuple(check_params),))
                existing = set(cur.fetchall())
                
                # Separate into insert and update batches
                to_insert = []
                to_update = []
                
                for record in data:
                    key = (record['ticker'], record['date'])
                    if key in existing:
                        to_update.append(record)
                    else:
                        to_insert.append(record)
                
                # Bulk insert new records
                if to_insert:
                    insert_query = """
                        INSERT INTO eod_data 
                        (date, ticker, open, high, low, close, adjusted_close, 
                         volume, dividends, stock_splits)
                        VALUES (%(date)s, %(ticker)s, %(open)s, %(high)s, %(low)s, 
                                %(close)s, %(adjusted_close)s, %(volume)s, 
                                %(dividends)s, %(stock_splits)s)
                    """
                    execute_batch(cur, insert_query, to_insert, page_size=self.config.batch_size)
                    inserted = len(to_insert)
                
                # Bulk update existing records
                if to_update:
                    update_query = """
                        UPDATE eod_data SET
                            open = %(open)s,
                            high = %(high)s,
                            low = %(low)s,
                            close = %(close)s,
                            adjusted_close = %(adjusted_close)s,
                            volume = %(volume)s,
                            dividends = %(dividends)s,
                            stock_splits = %(stock_splits)s,
                            updated_at = NOW()
                        WHERE ticker = %(ticker)s AND date = %(date)s
                    """
                    execute_batch(cur, update_query, to_update, page_size=self.config.batch_size)
                    updated = len(to_update)
                
                conn.commit()
                logger.info(f"EOD data operation: {inserted} inserted, {updated} updated")
                
                return inserted, updated
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to insert/update EOD data: {e}")
                raise
            finally:
                cur.close()
    
    def get_eod_data(self, ticker: str, start_date: datetime = None, 
                    end_date: datetime = None) -> pd.DataFrame:
        """Retrieve EOD data for a ticker within date range"""
        # Set defaults if not provided
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=self.config.default_lookback_days)
        
        query = """
            SELECT date, open, high, low, close, adjusted_close, volume, 
                   dividends, stock_splits
            FROM eod_data
            WHERE ticker = %s AND date >= %s AND date <= %s
            ORDER BY date ASC
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(ticker, start_date.date(), end_date.date()),
                parse_dates=['date']
            )
            
            if not df.empty:
                df.set_index('date', inplace=True)
                # Ensure numeric columns are float
                numeric_cols = ['open', 'high', 'low', 'close', 'adjusted_close', 
                               'volume', 'dividends', 'stock_splits']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
    
    def get_latest_date(self, ticker: str) -> Optional[datetime]:
        """Get the latest date we have data for a ticker"""
        query = "SELECT MAX(date) FROM eod_data WHERE ticker = %s"
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (ticker,))
            result = cur.fetchone()
            
            if result and result[0]:
                return datetime.combine(result[0], time(0, 0))
            return None
    
    def get_tickers_list(self) -> List[str]:
        """Get list of all tickers in database"""
        query = "SELECT DISTINCT ticker FROM eod_data ORDER BY ticker"
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]
    
    def get_date_range(self, ticker: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the date range for a ticker's data"""
        query = """
            SELECT MIN(date), MAX(date) 
            FROM eod_data 
            WHERE ticker = %s
        """
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (ticker,))
            result = cur.fetchone()
            
            if result and result[0]:
                start = datetime.combine(result[0], time(0, 0))
                end = datetime.combine(result[1], time(0, 0))
                return start, end
            return None, None
    
    def update_data_quality(self, ticker: str):
        """Update data quality metrics for a ticker"""
        try:
            # Get date range
            start_date, end_date = self.get_date_range(ticker)
            
            if not start_date:
                return
            
            # Get all dates we have
            query = """
                SELECT date FROM eod_data 
                WHERE ticker = %s 
                ORDER BY date
            """
            
            with self.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(query, (ticker,))
                existing_dates = [row[0] for row in cur.fetchall()]
                
                # Calculate expected trading days
                calendar = MarketCalendar()
                expected_dates = calendar.get_trading_days_between(start_date, end_date)
                
                # Find gaps
                missing_dates = set(expected_dates) - set(existing_dates)
                
                # Calculate quality score
                total_expected = len(expected_dates)
                quality_score = (total_expected - len(missing_dates)) / total_expected if total_expected > 0 else 0
                
                # Find data gaps (consecutive missing days)
                gaps = []
                if missing_dates:
                    sorted_missing = sorted(missing_dates)
                    gap_start = sorted_missing[0]
                    gap_end = sorted_missing[0]
                    
                    for i in range(1, len(sorted_missing)):
                        if (sorted_missing[i] - sorted_missing[i-1]).days == 1:
                            gap_end = sorted_missing[i]
                        else:
                            gaps.append({'start': str(gap_start), 'end': str(gap_end)})
                            gap_start = sorted_missing[i]
                            gap_end = sorted_missing[i]
                    
                    gaps.append({'start': str(gap_start), 'end': str(gap_end)})
                
                # Update quality table
                update_query = """
                    INSERT INTO eod_data_quality 
                    (ticker, last_update, first_date, last_date, total_days, 
                     missing_days, data_gaps, quality_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker) DO UPDATE SET
                        last_update = EXCLUDED.last_update,
                        first_date = EXCLUDED.first_date,
                        last_date = EXCLUDED.last_date,
                        total_days = EXCLUDED.total_days,
                        missing_days = EXCLUDED.missing_days,
                        data_gaps = EXCLUDED.data_gaps,
                        quality_score = EXCLUDED.quality_score
                """
                
                cur.execute(update_query, (
                    ticker, datetime.now(), start_date.date(), end_date.date(),
                    len(existing_dates), len(missing_dates), 
                    json.dumps(gaps), quality_score
                ))
                
                conn.commit()
                logger.info(f"Updated data quality for {ticker}: score={quality_score:.2%}")
                
        except Exception as e:
            logger.error(f"Failed to update data quality for {ticker}: {e}")

# ======================== EOD Data Fetcher ========================

class EODDataFetcher:
    """
    Fetches EOD data from multiple sources with fallback logic
    """
    
    def __init__(self, config: EODConfig):
        self.config = config
        self.market_data_manager = None
        self.initialized = False
        self._rate_limiter = asyncio.Semaphore(10)  # Limit concurrent requests
    
    async def initialize(self):
        """Initialize data sources"""
        if not self.initialized:
            try:
                from .fixed_market_data import get_market_data_manager
                self.market_data_manager = await get_market_data_manager()
                self.initialized = True
                logger.info("Market data manager initialized")
            except Exception as e:
                logger.warning(f"Market data manager not available: {e}")
                logger.info("Will use yfinance as fallback")
                self.initialized = True
    
    async def fetch_eod_history(self, ticker: str, years: int = 5) -> pd.DataFrame:
        """
        Fetch EOD history for a ticker with fallback to yfinance
        """
        if not self.initialized:
            await self.initialize()
        
        # Try primary data source first
        if self.market_data_manager:
            df = await self._fetch_from_market_manager(ticker, years)
            if not df.empty:
                return df
        
        # Fallback to yfinance
        return await self._fetch_from_yfinance(ticker, years)
    
    async def _fetch_from_market_manager(self, ticker: str, years: int) -> pd.DataFrame:
        """Fetch from market data manager"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            async with self._rate_limiter:
                df = await self.market_data_manager.get_historical_data(
                    ticker, start_date, end_date
                )
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Process the data
            df = self._process_raw_data(df, ticker)
            logger.info(f"Fetched {len(df)} days from market manager for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Market manager fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    
    async def _fetch_from_yfinance(self, ticker: str, years: int) -> pd.DataFrame:
        """Fallback to yfinance for data fetching"""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None, 
                self._yfinance_fetch_sync,
                ticker, years
            )
            return df
            
        except Exception as e:
            logger.error(f"yfinance fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def _yfinance_fetch_sync(self, ticker: str, years: int) -> pd.DataFrame:
        """Synchronous yfinance fetch"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=False,
                actions=True
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # Rename columns to match our schema
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adjusted_close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            })
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Clean data
            df = df.dropna(subset=['close'])
            
            logger.info(f"Fetched {len(df)} days from yfinance for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"yfinance sync fetch failed: {e}")
            return pd.DataFrame()
    
    def _process_raw_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Process raw data into standard format"""
        # Ensure we have date as index
        if 'datetime' in df.columns:
            df['date'] = pd.to_datetime(df['datetime']).dt.date
            df = df.set_index('date')
        elif 'date' in df.columns and df.index.name != 'date':
            df = df.set_index('date')
        
        # Ensure column names are lowercase
        df.columns = df.columns.str.lower()
        
        # Add ticker if not present
        if 'ticker' not in df.columns:
            df['ticker'] = ticker
        
        # Add missing columns with defaults
        if 'adjusted_close' not in df.columns:
            df['adjusted_close'] = df.get('close', 0)
        if 'dividends' not in df.columns:
            df['dividends'] = 0
        if 'stock_splits' not in df.columns:
            df['stock_splits'] = 1
        
        # Remove any duplicate dates
        df = df[~df.index.duplicated(keep='last')]
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    async def fetch_multiple_tickers(self, tickers: List[str], 
                                   years: int = 5) -> Dict[str, pd.DataFrame]:
        """Fetch EOD data for multiple tickers in parallel"""
        tasks = []
        for ticker in tickers:
            tasks.append(self.fetch_eod_history(ticker, years))
            # Add small delay to avoid rate limiting
            await asyncio.sleep(self.config.rate_limit_delay)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                data_dict[ticker] = result
            elif isinstance(result, Exception):
                logger.error(f"Error fetching {ticker}: {result}")
        
        logger.info(f"Successfully fetched data for {len(data_dict)}/{len(tickers)} tickers")
        return data_dict

# ======================== Feature Calculator ========================

class EODFeatureCalculator:
    """
    Calculates technical indicators and features from EOD data
    """
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive feature set for EOD data"""
        if df.empty or len(df) < 2:
            return df
        
        features = df.copy()
        
        # Price-based features
        features = EODFeatureCalculator._calculate_returns(features)
        features = EODFeatureCalculator._calculate_moving_averages(features)
        features = EODFeatureCalculator._calculate_price_channels(features)
        
        # Volume features
        features = EODFeatureCalculator._calculate_volume_features(features)
        
        # Volatility features
        features = EODFeatureCalculator._calculate_volatility(features)
        
        # Technical indicators
        features = EODFeatureCalculator._calculate_rsi(features)
        features = EODFeatureCalculator._calculate_macd(features)
        features = EODFeatureCalculator._calculate_bollinger_bands(features)
        features = EODFeatureCalculator._calculate_atr(features)
        
        # Pattern features
        features = EODFeatureCalculator._calculate_patterns(features)
        
        return features
    
    @staticmethod
    def _calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various return metrics"""
        # Simple returns
        df['returns_1d'] = df['close'].pct_change()
        df['returns_5d'] = df['close'].pct_change(periods=5)
        df['returns_10d'] = df['close'].pct_change(periods=10)
        df['returns_20d'] = df['close'].pct_change(periods=20)
        df['returns_60d'] = df['close'].pct_change(periods=60)
        
        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Overnight returns
        df['overnight_returns'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Intraday returns
        df['intraday_returns'] = (df['close'] - df['open']) / df['open']
        
        return df
    
    @staticmethod
    def _calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            if len(df) >= period:
                # Simple moving average
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                
                # Exponential moving average
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                
                # Price to MA ratio
                df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        
        return df
    
    @staticmethod
    def _calculate_price_channels(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price channels and positions"""
        periods = [20, 52 * 5, 252]  # 20 days, 52 weeks, 252 days (1 year)
        names = ['20d', '52w', '252d']
        
        for period, name in zip(periods, names):
            if len(df) >= period:
                df[f'high_{name}'] = df['high'].rolling(window=period).max()
                df[f'low_{name}'] = df['low'].rolling(window=period).min()
                
                # Price position in channel
                df[f'price_position_{name}'] = (
                    (df['close'] - df[f'low_{name}']) / 
                    (df[f'high_{name}'] - df[f'low_{name}'])
                ).fillna(0.5)
                
                # Distance from high/low
                df[f'pct_from_high_{name}'] = (
                    (df['close'] - df[f'high_{name}']) / df[f'high_{name}']
                )
                df[f'pct_from_low_{name}'] = (
                    (df['close'] - df[f'low_{name}']) / df[f'low_{name}']
                )
        
        return df
    
    @staticmethod
    def _calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        if 'volume' not in df.columns:
            return df
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # On-Balance Volume (OBV)
        df['price_change'] = df['close'].diff()
        df['obv'] = (df['volume'] * np.sign(df['price_change'])).cumsum()
        df.drop('price_change', axis=1, inplace=True)
        
        # Volume-Price Trend (VPT)
        df['vpt'] = (df['returns_1d'] * df['volume']).cumsum()
        
        # Money Flow
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = df['typical_price'] * df['volume']
        
        return df
    
    @staticmethod
    def _calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility metrics"""
        if 'returns_1d' not in df.columns:
            df['returns_1d'] = df['close'].pct_change()
        
        # Historical volatility (annualized)
        for period in [10, 20, 60]:
            if len(df) >= period:
                df[f'volatility_{period}d'] = (
                    df['returns_1d'].rolling(window=period).std() * np.sqrt(252)
                )
        
        # Parkinson volatility (using high-low)
        if len(df) >= 20:
            df['parkinson_vol'] = np.sqrt(
                252 / (4 * np.log(2)) * 
                (np.log(df['high'] / df['low']) ** 2).rolling(window=20).mean()
            )
        
        # Garman-Klass volatility
        if len(df) >= 20:
            df['gk_vol'] = np.sqrt(
                252 * (
                    0.5 * (np.log(df['high'] / df['low']) ** 2) -
                    (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2)
                ).rolling(window=20).mean()
            )
        
        return df
    
    @staticmethod
    def _calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        if len(df) < period + 1:
            return df
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI-based features
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        return df
    
    @staticmethod
    def _calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        if len(df) < 26:
            return df
        
        # MACD line
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        
        # Signal line
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # MACD histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD cross signals
        df['macd_cross_up'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)
        
        df['macd_cross_down'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        ).astype(int)
        
        return df
    
    @staticmethod
    def _calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        if len(df) < period:
            return df
        
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
        
        # Bollinger Band features
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Squeeze indicator
        df['bb_squeeze'] = df['bb_width'] / df['bb_middle']
        
        return df
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range"""
        if len(df) < period + 1:
            return df
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        
        # ATR percentage
        df['atr_pct'] = df['atr'] / df['close']
        
        return df
    
    @staticmethod
    def _calculate_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price pattern features"""
        if len(df) < 3:
            return df
        
        # Gap detection
        df['gap_up'] = ((df['open'] > df['close'].shift(1)) * 
                       (df['open'] - df['close'].shift(1)) / df['close'].shift(1))
        
        df['gap_down'] = ((df['open'] < df['close'].shift(1)) * 
                         (df['close'].shift(1) - df['open']) / df['close'].shift(1))
        
        # Doji detection (small body)
        body = abs(df['close'] - df['open'])
        range_hl = df['high'] - df['low']
        df['doji'] = (body / range_hl < 0.1).astype(int)
        
        # Hammer pattern
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        df['hammer'] = ((lower_shadow > 2 * body) & 
                       (df['close'] > df['open'])).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) &  # Current is bullish
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous was bearish
            (df['open'] < df['close'].shift(1)) &  # Current opens below previous close
            (df['close'] > df['open'].shift(1))  # Current closes above previous open
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['close'] < df['open']) &  # Current is bearish
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous was bullish
            (df['open'] > df['close'].shift(1)) &  # Current opens above previous close
            (df['close'] < df['open'].shift(1))  # Current closes below previous open
        ).astype(int)
        
        return df

# ======================== Cache Manager ========================

class CacheManager:
    """Manages Redis caching for EOD data"""
    
    def __init__(self, config: EODConfig):
        self.config = config
        self.client = None
        self.enabled = False
        self._init_cache()
    
    def _init_cache(self):
        """Initialize Redis connection"""
        if not self.config.use_cache:
            return
        
        try:
            self.client = redis.from_url(
                self.config.redis_url,
                decode_responses=False
            )
            self.client.ping()
            self.enabled = True
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.enabled = False
    
    def _make_key(self, ticker: str, key_type: str) -> str:
        """Generate cache key"""
        return f"eod:{ticker}:{key_type}"
    
    def get_dataframe(self, ticker: str, key_type: str = "data") -> Optional[pd.DataFrame]:
        """Get DataFrame from cache"""
        if not self.enabled:
            return None
        
        try:
            key = self._make_key(ticker, key_type)
            data = self.client.get(key)
            
            if data:
                return pickle.loads(data)
            
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        
        return None
    
    def set_dataframe(self, ticker: str, df: pd.DataFrame, 
                     key_type: str = "data", ttl: int = None):
        """Store DataFrame in cache"""
        if not self.enabled or df is None:
            return
        
        try:
            key = self._make_key(ticker, key_type)
            data = pickle.dumps(df)
            
            if ttl is None:
                ttl = self.config.cache_ttl
            
            self.client.setex(key, ttl, data)
            
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
    
    def invalidate(self, ticker: str):
        """Invalidate all cache entries for a ticker"""
        if not self.enabled:
            return
        
        try:
            pattern = f"eod:{ticker}:*"
            for key in self.client.scan_iter(match=pattern):
                self.client.delete(key)
            
        except Exception as e:
            logger.debug(f"Cache invalidation failed: {e}")

# ======================== Main EOD Pipeline ========================

class EODPipeline:
    """
    Main EOD data pipeline orchestrator
    """
    
    def __init__(self, config: Optional[EODConfig] = None):
        self.config = config or EODConfig()
        self.db = EODDatabase(self.config)
        self.fetcher = EODDataFetcher(self.config)
        self.calculator = EODFeatureCalculator()
        self.cache = CacheManager(self.config)
        self.calendar = MarketCalendar()
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        self.running = False
        self.job_id = None
    
    async def initialize(self):
        """Initialize all pipeline components"""
        await self.fetcher.initialize()
        logger.info("EOD Pipeline fully initialized")
    
    def _start_job(self, job_type: str, metadata: Dict = None) -> int:
        """Start ETL job tracking"""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO eod_etl_jobs (job_type, started_at, status, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING job_id
            """, (job_type, datetime.now(), 'RUNNING', json.dumps(metadata)))
            
            job_id = cur.fetchone()[0]
            conn.commit()
            return job_id
    
    def _complete_job(self, job_id: int, status: str, stats: Dict):
        """Complete ETL job tracking"""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE eod_etl_jobs SET
                    completed_at = %s,
                    status = %s,
                    tickers_processed = %s,
                    records_inserted = %s,
                    records_updated = %s,
                    errors = %s
                WHERE job_id = %s
            """, (
                datetime.now(),
                status,
                stats.get('tickers_processed', 0),
                stats.get('records_inserted', 0),
                stats.get('records_updated', 0),
                json.dumps(stats.get('errors', [])),
                job_id
            ))
            conn.commit()
    
    async def update_ticker(self, ticker: str, years: int = None, 
                          force: bool = False) -> Dict:
        """
        Update EOD data for a single ticker
        """
        if years is None:
            # Check what we have and fetch only missing data
            latest_date = self.db.get_latest_date(ticker)
            if latest_date and not force:
                # Calculate how many days to fetch
                days_missing = (datetime.now() - latest_date).days
                if days_missing < 2:  # Data is current
                    logger.info(f"{ticker} data is current")
                    return {
                        'ticker': ticker,
                        'success': True,
                        'records_inserted': 0,
                        'records_updated': 0,
                        'message': 'Data is current'
                    }
                years = max(1, days_missing / 365)
            else:
                years = 5  # Default to 5 years
        
        result = {
            'ticker': ticker,
            'success': False,
            'records_inserted': 0,
            'records_updated': 0,
            'features_calculated': 0,
            'errors': []
        }
        
        try:
            # Check cache first
            if not force:
                cached_df = self.cache.get_dataframe(ticker, "raw")
                if cached_df is not None:
                    logger.info(f"Using cached data for {ticker}")
                    df = cached_df
                else:
                    # Fetch fresh data
                    df = await self.fetcher.fetch_eod_history(ticker, years)
                    if not df.empty:
                        self.cache.set_dataframe(ticker, df, "raw")
            else:
                # Force fetch
                df = await self.fetcher.fetch_eod_history(ticker, years)
                self.cache.invalidate(ticker)
            
            if df.empty:
                result['errors'].append("No data fetched")
                return result
            
            # Calculate features
            df_features = self.calculator.calculate_all_features(df)
            result['features_calculated'] = len([c for c in df_features.columns 
                                                if c not in df.columns])
            
            # Prepare records for database
            records = []
            for date_idx, row in df_features.iterrows():
                record = {
                    'date': date_idx.date() if hasattr(date_idx, 'date') else date_idx,
                    'ticker': ticker,
                    'open': row.get('open'),
                    'high': row.get('high'),
                    'low': row.get('low'),
                    'close': row.get('close'),
                    'adjusted_close': row.get('adjusted_close', row.get('close')),
                    'volume': int(row.get('volume', 0)),
                    'dividends': row.get('dividends', 0),
                    'stock_splits': row.get('stock_splits', 1)
                }
                records.append(record)
            
            # Insert to database
            inserted, updated = self.db.insert_eod_data(records)
            result['records_inserted'] = inserted
            result['records_updated'] = updated
            
            # Store features
            self._store_features(ticker, df_features)
            
            # Update data quality
            self.db.update_data_quality(ticker)
            
            # Cache the processed data
            self.cache.set_dataframe(ticker, df_features, "processed")
            
            result['success'] = True
            result['date_range'] = (df.index[0].strftime('%Y-%m-%d'), 
                                   df.index[-1].strftime('%Y-%m-%d'))
            
            logger.info(f"Updated {ticker}: {inserted} inserted, {updated} updated, "
                       f"{result['features_calculated']} features calculated")
            
        except Exception as e:
            error_msg = f"Failed to update {ticker}: {str(e)}"
            logger.error(error_msg)
            result['errors'].append(error_msg)
        
        return result
    
    def _store_features(self, ticker: str, df: pd.DataFrame):
        """Store computed features in database"""
        try:
            # Get feature columns (exclude OHLCV and basic columns)
            exclude_cols = ['open', 'high', 'low', 'close', 'adjusted_close', 
                          'volume', 'ticker', 'dividends', 'stock_splits']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                
                for date_idx, row in df.iterrows():
                    features = {}
                    for col in feature_cols:
                        value = row[col]
                        # Handle NaN and infinite values
                        if pd.notna(value) and not np.isinf(value):
                            features[col] = float(value) if isinstance(value, (int, float, np.number)) else value
                    
                    if features:  # Only store if we have features
                        query = """
                            INSERT INTO eod_features (date, ticker, features, feature_version)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (ticker, date, feature_version) DO UPDATE
                            SET features = EXCLUDED.features,
                                created_at = NOW()
                        """
                        
                        cur.execute(query, (
                            date_idx.date() if hasattr(date_idx, 'date') else date_idx,
                            ticker,
                            json.dumps(features),
                            1  # Feature version
                        ))
                
                conn.commit()
                logger.debug(f"Stored {len(feature_cols)} features for {ticker}")
                
        except Exception as e:
            logger.error(f"Failed to store features for {ticker}: {e}")
    
    async def update_universe(self, tickers: List[str], years: int = None,
                            force: bool = False) -> Dict:
        """Update EOD data for multiple tickers"""
        job_id = self._start_job('UNIVERSE_UPDATE', {
            'tickers': tickers,
            'years': years,
            'force': force
        })
        
        logger.info(f"Updating {len(tickers)} tickers")
        
        results = []
        stats = {
            'tickers_processed': 0,
            'records_inserted': 0,
            'records_updated': 0,
            'errors': []
        }
        
        # Process tickers with rate limiting
        for ticker in tickers:
            try:
                result = await self.update_ticker(ticker, years, force)
                results.append(result)
                
                if result['success']:
                    stats['tickers_processed'] += 1
                    stats['records_inserted'] += result['records_inserted']
                    stats['records_updated'] += result['records_updated']
                else:
                    stats['errors'].extend(result.get('errors', []))
                
                # Rate limiting
                await asyncio.sleep(self.config.rate_limit_delay)
                
            except Exception as e:
                error_msg = f"Failed to process {ticker}: {str(e)}"
                logger.error(error_msg)
                stats['errors'].append(error_msg)
        
        # Complete job
        status = 'SUCCESS' if stats['tickers_processed'] == len(tickers) else 'PARTIAL'
        self._complete_job(job_id, status, stats)
        
        return {
            'job_id': job_id,
            'tickers_requested': len(tickers),
            'tickers_processed': stats['tickers_processed'],
            'records_inserted': stats['records_inserted'],
            'records_updated': stats['records_updated'],
            'errors': stats['errors'][:10],  # Limit errors in response
            'details': results
        }
    
    async def backfill_history(self, ticker: str, years: int = 20) -> Dict:
        """
        Backfill long history for a ticker (up to 20 years)
        """
        logger.info(f"Backfilling {years} years of history for {ticker}")
        
        job_id = self._start_job('BACKFILL', {
            'ticker': ticker,
            'years': years
        })
        
        result = await self.update_ticker(ticker, years, force=True)
        
        # Complete job
        status = 'SUCCESS' if result['success'] else 'FAILED'
        self._complete_job(job_id, status, {
            'tickers_processed': 1 if result['success'] else 0,
            'records_inserted': result.get('records_inserted', 0),
            'records_updated': result.get('records_updated', 0),
            'errors': result.get('errors', [])
        })
        
        return result
    
    def get_data_for_training(self, ticker: str, 
                            start_date: datetime = None,
                            end_date: datetime = None,
                            include_features: bool = True) -> pd.DataFrame:
        """Get EOD data with features for model training"""
        # Check cache first
        cache_key = f"training_{include_features}"
        cached_df = self.cache.get_dataframe(ticker, cache_key)
        
        if cached_df is not None and not start_date and not end_date:
            logger.debug(f"Using cached training data for {ticker}")
            return cached_df
        
        # Get OHLCV data
        df = self.db.get_eod_data(ticker, start_date, end_date)
        
        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return df
        
        if not include_features:
            return df
        
        # Get stored features
        try:
            with self.db.get_connection() as conn:
                query = """
                    SELECT date, features
                    FROM eod_features
                    WHERE ticker = %s AND feature_version = 1
                """
                
                params = [ticker]
                
                if start_date:
                    query += " AND date >= %s"
                    params.append(start_date.date())
                
                if end_date:
                    query += " AND date <= %s"
                    params.append(end_date.date())
                
                query += " ORDER BY date ASC"
                
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute(query, tuple(params))
                
                # Merge features with OHLCV
                for row in cur.fetchall():
                    date = pd.Timestamp(row['date'])
                    if date in df.index:
                        features = row['features']
                        for key, value in features.items():
                            df.loc[date, key] = value
                
                # Cache if using default parameters
                if not start_date and not end_date:
                    self.cache.set_dataframe(ticker, df, cache_key)
                
        except Exception as e:
            logger.error(f"Failed to load features for {ticker}: {e}")
        
        return df
    
    def get_latest_data(self, ticker: str, days: int = 60) -> pd.DataFrame:
        """Get latest N days of EOD data with features"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))  # Buffer for weekends
        return self.get_data_for_training(ticker, start_date, end_date)
    
    async def run_daily_update(self, tickers: List[str] = None) -> Dict:
        """Run daily update for configured tickers"""
        # Check if market is open
        if not self.calendar.is_trading_day(datetime.now()):
            logger.info("Not a trading day, skipping daily update")
            return {'status': 'skipped', 'reason': 'not_trading_day'}
        
        # Use default tickers if none provided
        if not tickers:
            tickers = self.db.get_tickers_list()
        
        if not tickers:
            logger.warning("No tickers to update")
            return {'status': 'skipped', 'reason': 'no_tickers'}
        
        logger.info(f"Running daily update for {len(tickers)} tickers")
        
        # Update with 1 year of data (enough to catch up)
        result = await self.update_universe(tickers, years=1)
        
        return result
    
    def schedule_daily_update(self, tickers: List[str] = None, 
                            update_time: str = "17:00"):
        """Schedule daily EOD update"""
        schedule.every().day.at(update_time).do(
            lambda: asyncio.run(self.run_daily_update(tickers))
        )
        logger.info(f"Scheduled daily update at {update_time}")
    
    async def start(self):
        """Start the pipeline with scheduling"""
        await self.initialize()
        self.running = True
        
        logger.info("EOD Pipeline started")
        
        while self.running:
            schedule.run_pending()
            await asyncio.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop the pipeline"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("EOD Pipeline stopped")
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        with self.db.get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get recent jobs
            cur.execute("""
                SELECT job_type, status, started_at, completed_at,
                       tickers_processed, records_inserted, records_updated
                FROM eod_etl_jobs
                ORDER BY started_at DESC
                LIMIT 10
            """)
            recent_jobs = cur.fetchall()
            
            # Get data statistics
            cur.execute("""
                SELECT COUNT(DISTINCT ticker) as ticker_count,
                       COUNT(*) as total_records,
                       MIN(date) as earliest_date,
                       MAX(date) as latest_date
                FROM eod_data
            """)
            data_stats = cur.fetchone()
            
            # Get quality statistics
            cur.execute("""
                SELECT AVG(quality_score) as avg_quality,
                       COUNT(*) as tickers_tracked
                FROM eod_data_quality
            """)
            quality_stats = cur.fetchone()
            
            return {
                'pipeline_running': self.running,
                'recent_jobs': recent_jobs,
                'data_statistics': data_stats,
                'quality_statistics': quality_stats,
                'cache_enabled': self.cache.enabled,
                'last_check': datetime.now().isoformat()
            }

# ======================== Convenience Functions ========================

_pipeline_instance = None

def get_eod_pipeline() -> EODPipeline:
    """Get or create EOD pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = EODPipeline()
    return _pipeline_instance

async def update_eod_data(tickers: List[str], years: int = 5) -> Dict:
    """Convenience function to update EOD data"""
    pipeline = get_eod_pipeline()
    await pipeline.initialize()
    return await pipeline.update_universe(tickers, years)

async def get_training_data(ticker: str, include_features: bool = True) -> pd.DataFrame:
    """Convenience function to get training data"""
    pipeline = get_eod_pipeline()
    return pipeline.get_data_for_training(ticker, include_features=include_features)

# ======================== Main ========================

async def main():
    """Test and demonstrate the EOD pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create pipeline
    config = EODConfig()
    pipeline = EODPipeline(config)
    await pipeline.initialize()
    
    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print("\n=== Testing EOD Pipeline ===\n")
    
    # 1. Update with recent data
    print("1. Updating recent data...")
    results = await pipeline.update_universe(test_tickers[:3], years=1)
    print(f"   Results: {results['tickers_processed']}/{len(test_tickers[:3])} successful")
    print(f"   Records: {results['records_inserted']} inserted, {results['records_updated']} updated")
    
    # 2. Get training data
    print("\n2. Getting training data for AAPL...")
    df = pipeline.get_data_for_training('AAPL')
    if not df.empty:
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Features: {len([c for c in df.columns if c not in ['open','high','low','close','volume']])} calculated")
    
    # 3. Get latest data
    print("\n3. Getting latest 30 days for AAPL...")
    latest = pipeline.get_latest_data('AAPL', days=30)
    if not latest.empty:
        print(f"   Records: {len(latest)}")
        print(f"   Latest close: ${latest['close'].iloc[-1]:.2f}")
    
    # 4. Check pipeline status
    print("\n4. Pipeline status...")
    status = pipeline.get_pipeline_status()
    print(f"   Total tickers: {status['data_statistics']['ticker_count']}")
    print(f"   Total records: {status['data_statistics']['total_records']}")
    print(f"   Data quality: {status['quality_statistics']['avg_quality']:.2%}")
    
    # 5. Test backfill (optional - takes longer)
    # print("\n5. Testing backfill for AAPL (10 years)...")
    # backfill = await pipeline.backfill_history('AAPL', years=10)
    # print(f"   Success: {backfill['success']}")
    # print(f"   Records: {backfill.get('records_inserted', 0)}")
    
    print("\n=== EOD Pipeline Test Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
