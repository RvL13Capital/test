# project/eod_data_pipeline.py
"""
EOD (End-of-Day) Data Pipeline with TimescaleDB and Feature Store
Optimized for daily candles and volume data
"""

import pandas as pd
import numpy as np
import logging
import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, time
from dataclasses import dataclass, asdict
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2.pool import ThreadedConnectionPool
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Integer, DateTime, Boolean, Index, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.pool import QueuePool
import redis
import pickle
from contextlib import contextmanager
import schedule
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback
from retrying import retry
import pyarrow as pa
import pyarrow.parquet as pq
import holidays

# Import existing modules
from .fixed_market_data import MarketDataManager, get_market_data_manager
from .optimized_features import OptimizedFeatureEngine, feature_engine
from .config import Config

logger = logging.getLogger(__name__)

# ======================== Database Configuration ========================

@dataclass
class EODDatabaseConfig:
    """Database configuration for EOD data"""
    host: str = Config.DB_HOST if hasattr(Config, 'DB_HOST') else "localhost"
    port: int = Config.DB_PORT if hasattr(Config, 'DB_PORT') else 5432
    database: str = Config.DB_NAME if hasattr(Config, 'DB_NAME') else "trading_eod_db"
    user: str = Config.DB_USER if hasattr(Config, 'DB_USER') else "postgres"
    password: str = Config.DB_PASSWORD if hasattr(Config, 'DB_PASSWORD') else "password"
    pool_size: int = 10  # Reduced for EOD data (less concurrent operations)
    max_overflow: int = 20
    
    @property
    def connection_string(self) -> str:
        """Generate connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

# ======================== Market Calendar ========================

class MarketCalendar:
    """Handles market holidays and trading days"""
    
    def __init__(self, country: str = 'US'):
        self.holidays = holidays.US(years=range(2020, 2030))
        self.market_open = time(9, 30)  # 9:30 AM EST
        self.market_close = time(16, 0)  # 4:00 PM EST
    
    def is_trading_day(self, date: datetime) -> bool:
        """Check if given date is a trading day"""
        # Check if weekend
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if holiday
        if date.date() in self.holidays:
            return False
        
        return True
    
    def get_last_trading_day(self, date: datetime = None) -> datetime:
        """Get the last trading day before or on the given date"""
        if date is None:
            date = datetime.now()
        
        while not self.is_trading_day(date):
            date -= timedelta(days=1)
        
        return date
    
    def get_next_trading_day(self, date: datetime = None) -> datetime:
        """Get the next trading day after the given date"""
        if date is None:
            date = datetime.now()
        
        date += timedelta(days=1)
        while not self.is_trading_day(date):
            date += timedelta(days=1)
        
        return date
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        if not self.is_trading_day(now):
            return False
        
        current_time = now.time()
        return self.market_open <= current_time <= self.market_close

# ======================== EOD Database Manager ========================

class EODDatabaseManager:
    """
    Manages TimescaleDB for EOD data
    Optimized for daily candles with efficient storage and retrieval
    """
    
    def __init__(self, config: EODDatabaseConfig):
        self.config = config
        self.engine = None
        self.connection_pool = None
        self.metadata = MetaData()
        self.market_calendar = MarketCalendar()
        self._init_database()
    
    def _init_database(self):
        """Initialize database connection and tables"""
        try:
            # Create SQLAlchemy engine
            self.engine = create_engine(
                self.config.connection_string,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # Create connection pool
            self.connection_pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=self.config.pool_size,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            
            # Create tables
            self._create_eod_tables()
            
            logger.info(f"EOD Database initialized at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize EOD Database: {e}")
            raise
    
    def _create_eod_tables(self):
        """Create database tables optimized for EOD data"""
        with self.engine.connect() as conn:
            # Enable TimescaleDB
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
            conn.commit()
            
            # EOD market data table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS eod_market_data (
                    date DATE NOT NULL,
                    ticker VARCHAR(10) NOT NULL,
                    open DOUBLE PRECISION NOT NULL,
                    high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL,
                    close DOUBLE PRECISION NOT NULL,
                    adjusted_close DOUBLE PRECISION,
                    volume BIGINT NOT NULL,
                    market_cap DOUBLE PRECISION,
                    shares_outstanding DOUBLE PRECISION,
                    dividend DOUBLE PRECISION DEFAULT 0,
                    split_factor DOUBLE PRECISION DEFAULT 1,
                    sector VARCHAR(100),
                    industry VARCHAR(100),
                    exchange VARCHAR(20),
                    data_source VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (date, ticker)
                )
            """))
            
            # Convert to hypertable with monthly chunks (optimal for EOD data)
            try:
                conn.execute(text("""
                    SELECT create_hypertable('eod_market_data', 'date', 
                                            chunk_time_interval => INTERVAL '1 month',
                                            if_not_exists => TRUE)
                """))
            except Exception as e:
                logger.debug(f"Hypertable already exists: {e}")
            
            # Indexes for efficient queries
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_eod_ticker_date 
                ON eod_market_data (ticker, date DESC)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_eod_date 
                ON eod_market_data (date DESC)
            """))
            
            # EOD Feature store
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS eod_features (
                    feature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    feature_set VARCHAR(100) NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    ticker VARCHAR(10) NOT NULL,
                    date DATE NOT NULL,
                    features JSONB NOT NULL,
                    feature_list TEXT[],
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(feature_set, version, ticker, date)
                )
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_eod_features_lookup 
                ON eod_features (ticker, date DESC)
                WHERE is_active = TRUE
            """))
            
            # Technical indicators table (denormalized for performance)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS eod_technical_indicators (
                    date DATE NOT NULL,
                    ticker VARCHAR(10) NOT NULL,
                    sma_5 DOUBLE PRECISION,
                    sma_20 DOUBLE PRECISION,
                    sma_50 DOUBLE PRECISION,
                    sma_200 DOUBLE PRECISION,
                    ema_12 DOUBLE PRECISION,
                    ema_26 DOUBLE PRECISION,
                    rsi_14 DOUBLE PRECISION,
                    macd DOUBLE PRECISION,
                    macd_signal DOUBLE PRECISION,
                    bb_upper DOUBLE PRECISION,
                    bb_middle DOUBLE PRECISION,
                    bb_lower DOUBLE PRECISION,
                    atr_14 DOUBLE PRECISION,
                    adx_14 DOUBLE PRECISION,
                    volume_sma_20 DOUBLE PRECISION,
                    relative_volume DOUBLE PRECISION,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (date, ticker)
                )
            """))
            
            # ETL tracking
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS eod_etl_jobs (
                    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    job_date DATE NOT NULL,
                    job_type VARCHAR(50) NOT NULL,
                    tickers TEXT[],
                    start_time TIMESTAMPTZ NOT NULL,
                    end_time TIMESTAMPTZ,
                    status VARCHAR(20) NOT NULL,
                    records_processed INTEGER DEFAULT 0,
                    records_failed INTEGER DEFAULT 0,
                    error_messages JSONB,
                    execution_time_seconds FLOAT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_eod_etl_jobs_date 
                ON eod_etl_jobs (job_date DESC)
            """))
            
            # Data quality tracking
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS eod_data_quality (
                    date DATE NOT NULL,
                    ticker VARCHAR(10) NOT NULL,
                    has_gaps BOOLEAN DEFAULT FALSE,
                    gap_dates DATE[],
                    completeness_score FLOAT,
                    volume_anomaly BOOLEAN DEFAULT FALSE,
                    price_anomaly BOOLEAN DEFAULT FALSE,
                    last_check TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (date, ticker)
                )
            """))
            
            # Weekly/Monthly aggregates for faster backtesting
            conn.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS eod_weekly_data AS
                SELECT 
                    date_trunc('week', date) AS week,
                    ticker,
                    first(open, date) as open,
                    max(high) as high,
                    min(low) as low,
                    last(close, date) as close,
                    sum(volume) as volume,
                    avg(market_cap) as avg_market_cap,
                    count(*) as trading_days
                FROM eod_market_data
                GROUP BY week, ticker
                WITH NO DATA
            """))
            
            conn.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS eod_monthly_data AS
                SELECT 
                    date_trunc('month', date) AS month,
                    ticker,
                    first(open, date) as open,
                    max(high) as high,
                    min(low) as low,
                    last(close, date) as close,
                    sum(volume) as volume,
                    avg(market_cap) as avg_market_cap,
                    count(*) as trading_days
                FROM eod_market_data
                GROUP BY month, ticker
                WITH NO DATA
            """))
            
            # Retention policy (keep daily data for 10 years)
            try:
                conn.execute(text("""
                    SELECT add_retention_policy('eod_market_data', 
                        drop_after => INTERVAL '10 years',
                        if_not_exists => TRUE)
                """))
            except Exception as e:
                logger.debug(f"Retention policy already exists: {e}")
            
            conn.commit()
            logger.info("EOD database tables created successfully")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)
    
    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def insert_eod_data(self, data: List[Dict]) -> int:
        """Insert EOD market data with upsert logic"""
        if not data:
            return 0
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            try:
                insert_query = """
                    INSERT INTO eod_market_data (
                        date, ticker, open, high, low, close, adjusted_close,
                        volume, market_cap, shares_outstanding, dividend, split_factor,
                        sector, industry, exchange, data_source, updated_at
                    ) VALUES (
                        %(date)s, %(ticker)s, %(open)s, %(high)s, %(low)s, %(close)s,
                        %(adjusted_close)s, %(volume)s, %(market_cap)s, %(shares_outstanding)s,
                        %(dividend)s, %(split_factor)s, %(sector)s, %(industry)s,
                        %(exchange)s, %(data_source)s, NOW()
                    )
                    ON CONFLICT (date, ticker) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        adjusted_close = EXCLUDED.adjusted_close,
                        volume = EXCLUDED.volume,
                        market_cap = EXCLUDED.market_cap,
                        updated_at = NOW()
                """
                
                execute_batch(cur, insert_query, data, page_size=500)
                conn.commit()
                
                rows_inserted = len(data)
                logger.info(f"Inserted/Updated {rows_inserted} EOD records")
                return rows_inserted
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to insert EOD data: {e}")
                raise
            finally:
                cur.close()
    
    def get_eod_data(self, ticker: str, start_date: datetime, 
                    end_date: datetime) -> pd.DataFrame:
        """Get EOD data for a ticker"""
        query = """
            SELECT date, open, high, low, close, adjusted_close, volume,
                   market_cap, shares_outstanding
            FROM eod_market_data
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
            df.set_index('date', inplace=True)
            return df
    
    def get_latest_eod_data(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Get latest EOD data (default 1 trading year = 252 days)"""
        query = """
            SELECT date, open, high, low, close, adjusted_close, volume
            FROM eod_market_data
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT %s
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(ticker, days),
                parse_dates=['date']
            )
            df = df.sort_values('date')
            df.set_index('date', inplace=True)
            return df
    
    def check_data_gaps(self, ticker: str, start_date: datetime, 
                       end_date: datetime) -> Dict:
        """Check for missing trading days in data"""
        # Get existing data
        existing_data = self.get_eod_data(ticker, start_date, end_date)
        existing_dates = set(existing_data.index.date)
        
        # Generate expected trading days
        expected_dates = []
        current_date = start_date
        while current_date <= end_date:
            if self.market_calendar.is_trading_day(current_date):
                expected_dates.append(current_date.date())
            current_date += timedelta(days=1)
        
        expected_dates = set(expected_dates)
        
        # Find gaps
        missing_dates = expected_dates - existing_dates
        
        return {
            'ticker': ticker,
            'expected_days': len(expected_dates),
            'actual_days': len(existing_dates),
            'missing_days': len(missing_dates),
            'missing_dates': sorted(list(missing_dates)),
            'completeness': len(existing_dates) / len(expected_dates) if expected_dates else 0
        }

# ======================== EOD Feature Store ========================

class EODFeatureStore:
    """
    Feature store optimized for EOD data
    Manages daily features with efficient storage and retrieval
    """
    
    def __init__(self, db_manager: EODDatabaseManager, cache_client: Optional[redis.Redis] = None):
        self.db = db_manager
        self.cache = cache_client
        self.feature_engine = OptimizedFeatureEngine(use_cache=True)
        
    def compute_eod_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Compute features specifically for EOD data"""
        # Ensure we have enough data
        if len(df) < 200:  # Need at least 200 days for SMA_200
            logger.warning(f"Insufficient data for {ticker}: {len(df)} days")
            
        # Use optimized feature engine for base features
        features_df = self.feature_engine.create_features_optimized(df)
        
        # Add EOD-specific features
        # Returns
        features_df['returns_1d'] = features_df['close'].pct_change()
        features_df['returns_5d'] = features_df['close'].pct_change(periods=5)
        features_df['returns_20d'] = features_df['close'].pct_change(periods=20)
        features_df['returns_60d'] = features_df['close'].pct_change(periods=60)
        
        # Log returns
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        
        # Moving averages (additional)
        features_df['sma_50'] = features_df['close'].rolling(window=50).mean()
        features_df['sma_200'] = features_df['close'].rolling(window=200).mean()
        
        # Volume features
        features_df['volume_sma_20'] = features_df['volume'].rolling(window=20).mean()
        features_df['relative_volume'] = features_df['volume'] / features_df['volume_sma_20']
        
        # Price position indicators
        features_df['price_to_sma20'] = features_df['close'] / features_df['sma_20']
        features_df['price_to_sma50'] = features_df['close'] / features_df['sma_50']
        features_df['price_to_sma200'] = features_df['close'] / features_df['sma_200']
        
        # Volatility measures
        features_df['volatility_20d'] = features_df['returns_1d'].rolling(window=20).std()
        features_df['volatility_60d'] = features_df['returns_1d'].rolling(window=60).std()
        
        # High/Low indicators
        features_df['high_20d'] = features_df['high'].rolling(window=20).max()
        features_df['low_20d'] = features_df['low'].rolling(window=20).min()
        features_df['price_position_20d'] = (features_df['close'] - features_df['low_20d']) / (features_df['high_20d'] - features_df['low_20d'])
        
        features_df['high_52w'] = features_df['high'].rolling(window=252).max()
        features_df['low_52w'] = features_df['low'].rolling(window=252).min()
        features_df['price_position_52w'] = (features_df['close'] - features_df['low_52w']) / (features_df['high_52w'] - features_df['low_52w'])
        
        # MACD
        exp1 = features_df['close'].ewm(span=12, adjust=False).mean()
        exp2 = features_df['close'].ewm(span=26, adjust=False).mean()
        features_df['macd'] = exp1 - exp2
        features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # ADX (Average Directional Index)
        features_df['adx_14'] = self._calculate_adx(df, period=14)
        
        # Money Flow Index
        features_df['mfi_14'] = self._calculate_mfi(df, period=14)
        
        # Fill NaN values
        features_df.fillna(method='ffill', inplace=True)
        features_df.fillna(0, inplace=True)
        
        return features_df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = abs(100 * (minus_dm.rolling(period).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow.append(money_flow[i])
                negative_flow.append(0)
            elif typical_price[i] < typical_price[i-1]:
                positive_flow.append(0)
                negative_flow.append(money_flow[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        
        positive_flow = pd.Series(positive_flow, index=df.index[1:])
        negative_flow = pd.Series(negative_flow, index=df.index[1:])
        
        positive_flow_sum = positive_flow.rolling(period).sum()
        negative_flow_sum = negative_flow.rolling(period).sum()
        
        money_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_ratio))
        
        # Add NaN for the first value to match the DataFrame length
        mfi = pd.concat([pd.Series([np.nan], index=df.index[:1]), mfi])
        
        return mfi
    
    def save_eod_features(self, ticker: str, features: pd.DataFrame, 
                         feature_set: str = "eod_default") -> bool:
        """Save EOD features to database"""
        try:
            feature_records = []
            for date, row in features.iterrows():
                feature_dict = row.to_dict()
                
                # Remove infinite values
                feature_dict = {k: v if not np.isinf(v) else 0 for k, v in feature_dict.items()}
                
                feature_records.append({
                    'feature_set': feature_set,
                    'version': 1,
                    'ticker': ticker,
                    'date': date,
                    'features': json.dumps(feature_dict, default=str),
                    'feature_list': list(feature_dict.keys())
                })
            
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                
                insert_query = """
                    INSERT INTO eod_features (
                        feature_set, version, ticker, date, features, feature_list
                    ) VALUES (
                        %(feature_set)s, %(version)s, %(ticker)s, %(date)s,
                        %(features)s, %(feature_list)s
                    )
                    ON CONFLICT (feature_set, version, ticker, date) 
                    DO UPDATE SET
                        features = EXCLUDED.features,
                        feature_list = EXCLUDED.feature_list
                """
                
                execute_batch(cur, insert_query, feature_records, page_size=500)
                conn.commit()
                
                logger.info(f"Saved {len(feature_records)} feature records for {ticker}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save EOD features: {e}")
            return False
    
    def load_eod_features(self, ticker: str, start_date: datetime, 
                         end_date: datetime, feature_set: str = "eod_default") -> pd.DataFrame:
        """Load EOD features from database"""
        query = """
            SELECT date, features
            FROM eod_features
            WHERE ticker = %s 
                AND feature_set = %s
                AND date >= %s 
                AND date <= %s
                AND is_active = TRUE
            ORDER BY date ASC
        """
        
        with self.db.get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, (ticker, feature_set, start_date.date(), end_date.date()))
            rows = cur.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            data = []
            for row in rows:
                feature_dict = json.loads(row['features'])
                feature_dict['date'] = row['date']
                data.append(feature_dict)
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
    
    def save_technical_indicators(self, ticker: str, indicators: pd.DataFrame) -> bool:
        """Save pre-computed technical indicators for fast access"""
        try:
            records = []
            for date, row in indicators.iterrows():
                record = {
                    'date': date,
                    'ticker': ticker,
                    'sma_5': row.get('sma_5'),
                    'sma_20': row.get('sma_20'),
                    'sma_50': row.get('sma_50'),
                    'sma_200': row.get('sma_200'),
                    'ema_12': row.get('ema_12'),
                    'ema_26': row.get('ema_26'),
                    'rsi_14': row.get('rsi_14'),
                    'macd': row.get('macd'),
                    'macd_signal': row.get('macd_signal'),
                    'bb_upper': row.get('bb_upper'),
                    'bb_middle': row.get('sma_20'),  # BB middle is SMA20
                    'bb_lower': row.get('bb_lower'),
                    'atr_14': row.get('atr_14'),
                    'adx_14': row.get('adx_14'),
                    'volume_sma_20': row.get('volume_sma_20'),
                    'relative_volume': row.get('relative_volume')
                }
                records.append(record)
            
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                
                insert_query = """
                    INSERT INTO eod_technical_indicators (
                        date, ticker, sma_5, sma_20, sma_50, sma_200,
                        ema_12, ema_26, rsi_14, macd, macd_signal,
                        bb_upper, bb_middle, bb_lower, atr_14, adx_14,
                        volume_sma_20, relative_volume
                    ) VALUES (
                        %(date)s, %(ticker)s, %(sma_5)s, %(sma_20)s, %(sma_50)s, %(sma_200)s,
                        %(ema_12)s, %(ema_26)s, %(rsi_14)s, %(macd)s, %(macd_signal)s,
                        %(bb_upper)s, %(bb_middle)s, %(bb_lower)s, %(atr_14)s, %(adx_14)s,
                        %(volume_sma_20)s, %(relative_volume)s
                    )
                    ON CONFLICT (date, ticker) DO UPDATE SET
                        sma_5 = EXCLUDED.sma_5,
                        sma_20 = EXCLUDED.sma_20,
                        sma_50 = EXCLUDED.sma_50,
                        sma_200 = EXCLUDED.sma_200,
                        rsi_14 = EXCLUDED.rsi_14,
                        macd = EXCLUDED.macd
                """
                
                execute_batch(cur, insert_query, records, page_size=500)
                conn.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to save technical indicators: {e}")
            return False

# ======================== EOD ETL Pipeline ========================

class EODETLPipeline:
    """
    ETL Pipeline optimized for End-of-Day data processing
    Runs once daily after market close
    """
    
    def __init__(self, config: EODDatabaseConfig):
        self.db_manager = EODDatabaseManager(config)
        self.feature_store = EODFeatureStore(self.db_manager, self._init_redis())
        self.market_data_manager = None
        self.market_calendar = MarketCalendar()
        self.executor = ThreadPoolExecutor(max_workers=5)  # Less parallelism needed for EOD
        self.running = False
        
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis for caching"""
        try:
            redis_client = redis.from_url(
                Config.CELERY_RESULT_BACKEND if hasattr(Config, 'CELERY_RESULT_BACKEND') 
                else "redis://localhost:6379/0",
                decode_responses=False
            )
            redis_client.ping()
            logger.info("Redis cache initialized for EOD pipeline")
            return redis_client
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            return None
    
    async def initialize(self):
        """Initialize the pipeline"""
        self.market_data_manager = await get_market_data_manager()
        logger.info("EOD ETL Pipeline initialized")
    
    def _track_job(self, job_type: str, tickers: List[str], job_date: datetime) -> str:
        """Track ETL job execution"""
        with self.db_manager.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO eod_etl_jobs (job_date, job_type, tickers, start_time, status)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING job_id
            """, (job_date.date(), job_type, tickers, datetime.now(), "RUNNING"))
            
            job_id = cur.fetchone()[0]
            conn.commit()
            
        return job_id
    
    def _complete_job(self, job_id: str, status: str, records_processed: int = 0, 
                     records_failed: int = 0, errors: List[str] = None):
        """Complete job tracking"""
        end_time = datetime.now()
        
        with self.db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            # Get start time
            cur.execute("SELECT start_time FROM eod_etl_jobs WHERE job_id = %s", (job_id,))
            start_time = cur.fetchone()[0]
            execution_time = (end_time - start_time).total_seconds()
            
            cur.execute("""
                UPDATE eod_etl_jobs 
                SET end_time = %s, status = %s, records_processed = %s, 
                    records_failed = %s, error_messages = %s, execution_time_seconds = %s
                WHERE job_id = %s
            """, (end_time, status, records_processed, records_failed, 
                  json.dumps(errors) if errors else None, execution_time, job_id))
            conn.commit()
    
    async def extract_eod_data(self, ticker: str, days_back: int = 252) -> pd.DataFrame:
        """Extract EOD data for a ticker (default 1 year)"""
        try:
            end_date = self.market_calendar.get_last_trading_day()
            start_date = end_date - timedelta(days=days_back * 1.5)  # Extra buffer for holidays
            
            # Get historical data
            historical_data = await self.market_data_manager.get_historical_data(
                ticker, start_date, end_date
            )
            
            if historical_data is None or historical_data.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Get stock info for metadata
            stock_info = await self.market_data_manager.get_stock_info(ticker)
            
            # Add metadata
            if stock_info:
                historical_data['market_cap'] = stock_info.market_cap
                historical_data['shares_outstanding'] = stock_info.shares_outstanding
                historical_data['sector'] = stock_info.sector
                historical_data['industry'] = stock_info.industry
                historical_data['exchange'] = stock_info.exchange
            
            historical_data['ticker'] = ticker
            historical_data['data_source'] = "polygon"
            
            # Ensure we have date column
            if 'datetime' in historical_data.columns:
                historical_data['date'] = pd.to_datetime(historical_data['datetime']).dt.date
            else:
                historical_data['date'] = historical_data.index.date
            
            # Calculate adjusted close (simplified - would need actual dividend/split data)
            historical_data['adjusted_close'] = historical_data['close']
            historical_data['dividend'] = 0
            historical_data['split_factor'] = 1
            
            logger.info(f"Extracted {len(historical_data)} EOD records for {ticker}")
            return historical_data
            
        except Exception as e:
            logger.error(f"Failed to extract EOD data for {ticker}: {e}")
            raise
    
    def validate_eod_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Validate and clean EOD data"""
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date'])
        
        # Validate OHLC relationships
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['volume'] < 0)
        )
        
        if invalid_mask.any():
            logger.warning(f"Found {invalid_mask.sum()} invalid records for {ticker}")
            df = df[~invalid_mask]
        
        # Handle missing values
        df['volume'] = df['volume'].fillna(0)
        price_cols = ['open', 'high', 'low', 'close', 'adjusted_close']
        df[price_cols] = df[price_cols].ffill()
        
        # Remove extreme outliers (> 50% daily change)
        df['daily_change'] = df['close'].pct_change()
        extreme_moves = df['daily_change'].abs() > 0.5
        if extreme_moves.any():
            logger.warning(f"Found {extreme_moves.sum()} extreme moves for {ticker}")
            # Keep them but flag for review
        
        df = df.drop(columns=['daily_change'])
        
        final_count = len(df)
        if final_count < initial_count:
            logger.info(f"Cleaned {initial_count - final_count} invalid records for {ticker}")
        
        return df
    
    async def process_ticker_eod(self, ticker: str, lookback_days: int = 252) -> Dict:
        """Process complete EOD pipeline for a single ticker"""
        job_date = self.market_calendar.get_last_trading_day()
        job_id = self._track_job("EOD_FULL", [ticker], job_date)
        
        result = {
            'ticker': ticker,
            'success': False,
            'records_processed': 0,
            'features_computed': 0,
            'errors': []
        }
        
        try:
            # Extract
            logger.info(f"Extracting EOD data for {ticker}")
            raw_data = await self.extract_eod_data(ticker, lookback_days)
            
            # Validate
            clean_data = self.validate_eod_data(raw_data, ticker)
            
            # Load to database
            records_to_insert = []
            for _, row in clean_data.iterrows():
                records_to_insert.append(row.to_dict())
            
            records_inserted = self.db_manager.insert_eod_data(records_to_insert)
            result['records_processed'] = records_inserted
            
            # Compute features
            logger.info(f"Computing features for {ticker}")
            features = self.feature_store.compute_eod_features(clean_data, ticker)
            
            # Save features
            self.feature_store.save_eod_features(ticker, features)
            result['features_computed'] = len(features.columns)
            
            # Save technical indicators separately for fast access
            self.feature_store.save_technical_indicators(ticker, features)
            
            # Check data quality
            gap_check = self.db_manager.check_data_gaps(
                ticker, 
                job_date - timedelta(days=lookback_days),
                job_date
            )
            
            if gap_check['missing_days'] > 0:
                logger.warning(f"Data gaps found for {ticker}: {gap_check['missing_days']} days missing")
                result['data_gaps'] = gap_check['missing_dates']
            
            result['success'] = True
            self._complete_job(job_id, "SUCCESS", records_inserted)
            
            logger.info(f"EOD processing completed for {ticker}")
            
        except Exception as e:
            error_msg = str(e)
            result['errors'].append(error_msg)
            self._complete_job(job_id, "FAILED", 0, 0, [error_msg])
            logger.error(f"EOD processing failed for {ticker}: {e}")
        
        return result
    
    async def run_daily_update(self, tickers: List[str]) -> Dict:
        """Run daily EOD update for multiple tickers"""
        if not self.market_calendar.is_trading_day(datetime.now()):
            logger.info("Not a trading day, skipping daily update")
            return {'status': 'skipped', 'reason': 'not_trading_day'}
        
        logger.info(f"Starting daily EOD update for {len(tickers)} tickers")
        
        # Process tickers in parallel
        tasks = []
        for ticker in tickers:
            task = self.process_ticker_eod(ticker, lookback_days=30)  # Only update last 30 days
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Summarize results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failed = len(results) - successful
        total_records = sum(r.get('records_processed', 0) for r in results if isinstance(r, dict))
        
        summary = {
            'date': self.market_calendar.get_last_trading_day(),
            'tickers_processed': len(tickers),
            'successful': successful,
            'failed': failed,
            'total_records': total_records,
            'results': results
        }
        
        logger.info(f"Daily update completed: {successful}/{len(tickers)} successful")
        
        # Refresh materialized views
        self._refresh_aggregates()
        
        return summary
    
    def _refresh_aggregates(self):
        """Refresh materialized views for performance"""
        try:
            with self.db_manager.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY eod_weekly_data")
                cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY eod_monthly_data")
                conn.commit()
                logger.info("Refreshed materialized views")
        except Exception as e:
            logger.error(f"Failed to refresh materialized views: {e}")
    
    def schedule_daily_job(self, tickers: List[str], run_time: str = "17:00"):
        """Schedule daily EOD update job (default 5 PM EST after market close)"""
        schedule.every().day.at(run_time).do(
            lambda: asyncio.run(self.run_daily_update(tickers))
        )
        logger.info(f"Scheduled daily EOD update at {run_time} for {len(tickers)} tickers")
    
    def get_pipeline_health(self) -> Dict:
        """Get pipeline health metrics"""
        with self.db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Recent job status
            cur.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE status = 'SUCCESS') as successful_jobs,
                    COUNT(*) FILTER (WHERE status = 'FAILED') as failed_jobs,
                    AVG(execution_time_seconds) as avg_execution_time,
                    MAX(job_date) as last_run_date
                FROM eod_etl_jobs
                WHERE job_date >= CURRENT_DATE - INTERVAL '7 days'
            """)
            job_metrics = cur.fetchone()
            
            # Data coverage
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT ticker) as tickers_count,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    COUNT(*) as total_records
                FROM eod_market_data
            """)
            data_coverage = cur.fetchone()
            
            # Data quality
            cur.execute("""
                SELECT 
                    AVG(completeness_score) as avg_completeness,
                    COUNT(*) FILTER (WHERE has_gaps = TRUE) as tickers_with_gaps
                FROM eod_data_quality
                WHERE date >= CURRENT_DATE - INTERVAL '7 days'
            """)
            quality_metrics = cur.fetchone()
            
            return {
                'job_metrics': job_metrics,
                'data_coverage': data_coverage,
                'quality_metrics': quality_metrics,
                'pipeline_status': 'RUNNING' if self.running else 'STOPPED',
                'last_check': datetime.now().isoformat()
            }
    
    async def backfill_historical_data(self, ticker: str, years: int = 5) -> Dict:
        """Backfill historical EOD data"""
        logger.info(f"Starting historical backfill for {ticker} ({years} years)")
        
        result = await self.process_ticker_eod(ticker, lookback_days=years * 252)
        
        if result['success']:
            logger.info(f"Backfill completed for {ticker}: {result['records_processed']} records")
        else:
            logger.error(f"Backfill failed for {ticker}: {result['errors']}")
        
        return result
    
    async def start(self):
        """Start the EOD pipeline"""
        await self.initialize()
        self.running = True
        
        logger.info("EOD ETL Pipeline started")
        
        # Run scheduled jobs
        while self.running:
            schedule.run_pending()
            await asyncio.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop the pipeline"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("EOD ETL Pipeline stopped")

# ======================== Main Execution ========================

async def main():
    """Main function for EOD pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = EODDatabaseConfig()
    
    # Initialize pipeline
    pipeline = EODETLPipeline(config)
    
    try:
        await pipeline.initialize()
        
        # Define tickers to track
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'JPM', 'JNJ', 'V', 'PG', 'UNH',
            'NVDA', 'META', 'BRK.B', 'XOM', 'LLY'
        ]
        
        # Run initial update
        logger.info("Running initial EOD update...")
        results = await pipeline.run_daily_update(tickers[:5])  # Start with 5 tickers
        
        print(f"Initial update results: {json.dumps(results, indent=2, default=str)}")
        
        # Schedule daily updates
        pipeline.schedule_daily_job(tickers, "17:00")
        
        # Get pipeline health
        health = pipeline.get_pipeline_health()
        print(f"Pipeline health: {json.dumps(health, indent=2, default=str)}")
        
        # Optional: Backfill historical data for one ticker
        # backfill_result = await pipeline.backfill_historical_data('AAPL', years=2)
        # print(f"Backfill result: {backfill_result}")
        
        # Start continuous operation
        await pipeline.start()
        
    except KeyboardInterrupt:
        logger.info("Shutting down EOD pipeline...")
        pipeline.stop()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        traceback.print_exc()
    finally:
        pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())
