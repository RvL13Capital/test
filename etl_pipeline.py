# project/data_pipeline.py
"""
Robust ETL Pipeline with TimescaleDB and Feature Store
Provides fault-tolerant data extraction, transformation, and loading
with versioned feature management
"""

import pandas as pd
import numpy as np
import logging
import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
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

# Import existing modules
from .fixed_market_data import MarketDataManager, get_market_data_manager
from .optimized_features import OptimizedFeatureEngine, feature_engine
from .config import Config

logger = logging.getLogger(__name__)

# ======================== Database Configuration ========================

class DatabaseType(Enum):
    """Supported database types"""
    TIMESCALEDB = "timescaledb"
    INFLUXDB = "influxdb"
    POSTGRESQL = "postgresql"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: DatabaseType = DatabaseType.TIMESCALEDB
    host: str = Config.DB_HOST if hasattr(Config, 'DB_HOST') else "localhost"
    port: int = Config.DB_PORT if hasattr(Config, 'DB_PORT') else 5432
    database: str = Config.DB_NAME if hasattr(Config, 'DB_NAME') else "trading_db"
    user: str = Config.DB_USER if hasattr(Config, 'DB_USER') else "postgres"
    password: str = Config.DB_PASSWORD if hasattr(Config, 'DB_PASSWORD') else "password"
    pool_size: int = 20
    max_overflow: int = 40
    
    @property
    def connection_string(self) -> str:
        """Generate connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

# ======================== TimescaleDB Manager ========================

class TimescaleDBManager:
    """
    Manages TimescaleDB connections and operations
    Provides high-performance time-series data storage
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.connection_pool = None
        self.metadata = MetaData()
        self._init_database()
    
    def _init_database(self):
        """Initialize database connection and tables"""
        try:
            # Create SQLAlchemy engine with connection pooling
            self.engine = create_engine(
                self.config.connection_string,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # Create connection pool for raw psycopg2 operations
            self.connection_pool = ThreadedConnectionPool(
                minconn=5,
                maxconn=self.config.pool_size,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            
            # Create tables
            self._create_tables()
            
            logger.info(f"TimescaleDB initialized successfully at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables with TimescaleDB hypertables"""
        with self.engine.connect() as conn:
            # Enable TimescaleDB extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
            conn.commit()
            
            # Market data table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS market_data (
                    time TIMESTAMPTZ NOT NULL,
                    ticker VARCHAR(10) NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION NOT NULL,
                    volume BIGINT,
                    market_cap DOUBLE PRECISION,
                    shares_outstanding DOUBLE PRECISION,
                    sector VARCHAR(100),
                    industry VARCHAR(100),
                    exchange VARCHAR(20),
                    data_source VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            # Convert to hypertable if not already
            try:
                conn.execute(text("""
                    SELECT create_hypertable('market_data', 'time', 
                                            chunk_time_interval => INTERVAL '1 day',
                                            if_not_exists => TRUE)
                """))
            except Exception as e:
                logger.debug(f"Hypertable already exists or creation skipped: {e}")
            
            # Create indexes for better query performance
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_market_data_ticker_time 
                ON market_data (ticker, time DESC)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_market_data_time 
                ON market_data (time DESC)
            """))
            
            # Feature store table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS feature_store (
                    feature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    feature_set_name VARCHAR(100) NOT NULL,
                    feature_version INTEGER NOT NULL DEFAULT 1,
                    ticker VARCHAR(10) NOT NULL,
                    time TIMESTAMPTZ NOT NULL,
                    features JSONB NOT NULL,
                    feature_columns TEXT[],
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(feature_set_name, feature_version, ticker, time)
                )
            """))
            
            # Create index for feature lookups
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_feature_store_lookup 
                ON feature_store (feature_set_name, ticker, time DESC)
                WHERE is_active = TRUE
            """))
            
            # ETL job tracking table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS etl_jobs (
                    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    job_type VARCHAR(50) NOT NULL,
                    ticker VARCHAR(10),
                    start_time TIMESTAMPTZ NOT NULL,
                    end_time TIMESTAMPTZ,
                    status VARCHAR(20) NOT NULL,
                    records_processed INTEGER DEFAULT 0,
                    error_message TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            # Data quality metrics table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    ticker VARCHAR(10) NOT NULL,
                    metric_date DATE NOT NULL,
                    completeness_score FLOAT,
                    accuracy_score FLOAT,
                    timeliness_score FLOAT,
                    consistency_score FLOAT,
                    missing_data_points INTEGER,
                    anomaly_count INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(ticker, metric_date)
                )
            """))
            
            # Create continuous aggregates for real-time analytics
            conn.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_hourly
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 hour', time) AS bucket,
                    ticker,
                    first(open, time) as open,
                    max(high) as high,
                    min(low) as low,
                    last(close, time) as close,
                    sum(volume) as volume,
                    avg(market_cap) as avg_market_cap
                FROM market_data
                GROUP BY bucket, ticker
                WITH NO DATA
            """))
            
            # Create refresh policy for continuous aggregate
            try:
                conn.execute(text("""
                    SELECT add_continuous_aggregate_policy('market_data_hourly',
                        start_offset => INTERVAL '3 days',
                        end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '1 hour',
                        if_not_exists => TRUE)
                """))
            except Exception as e:
                logger.debug(f"Continuous aggregate policy already exists: {e}")
            
            # Data retention policy (keep detailed data for 2 years)
            try:
                conn.execute(text("""
                    SELECT add_retention_policy('market_data', 
                        drop_after => INTERVAL '2 years',
                        if_not_exists => TRUE)
                """))
            except Exception as e:
                logger.debug(f"Retention policy already exists: {e}")
            
            conn.commit()
            logger.info("Database tables created successfully")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)
    
    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def insert_market_data(self, data: List[Dict]) -> int:
        """Insert market data with retry logic"""
        if not data:
            return 0
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            try:
                # Prepare insert query
                insert_query = """
                    INSERT INTO market_data (
                        time, ticker, open, high, low, close, volume,
                        market_cap, shares_outstanding, sector, industry, 
                        exchange, data_source
                    ) VALUES (
                        %(time)s, %(ticker)s, %(open)s, %(high)s, %(low)s, 
                        %(close)s, %(volume)s, %(market_cap)s, %(shares_outstanding)s,
                        %(sector)s, %(industry)s, %(exchange)s, %(data_source)s
                    )
                    ON CONFLICT (time, ticker) DO UPDATE SET
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        market_cap = EXCLUDED.market_cap
                """
                
                # Batch insert for better performance
                execute_batch(cur, insert_query, data, page_size=1000)
                conn.commit()
                
                rows_inserted = len(data)
                logger.info(f"Inserted {rows_inserted} market data records")
                return rows_inserted
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to insert market data: {e}")
                raise
            finally:
                cur.close()
    
    def get_market_data(self, ticker: str, start_time: datetime, 
                       end_time: datetime) -> pd.DataFrame:
        """Retrieve market data for a ticker within time range"""
        query = """
            SELECT time, ticker, open, high, low, close, volume, 
                   market_cap, shares_outstanding
            FROM market_data
            WHERE ticker = %s AND time >= %s AND time <= %s
            ORDER BY time ASC
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(ticker, start_time, end_time),
                parse_dates=['time']
            )
            return df
    
    def get_latest_market_data(self, ticker: str, limit: int = 1000) -> pd.DataFrame:
        """Get latest market data for a ticker"""
        query = """
            SELECT time, ticker, open, high, low, close, volume
            FROM market_data
            WHERE ticker = %s
            ORDER BY time DESC
            LIMIT %s
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(ticker, limit),
                parse_dates=['time']
            )
            # Reverse to get chronological order
            return df.iloc[::-1].reset_index(drop=True)

# ======================== Feature Store Manager ========================

class FeatureStore:
    """
    Centralized feature store for consistent feature management
    Ensures feature consistency between training and inference
    """
    
    def __init__(self, db_manager: TimescaleDBManager, cache_client: Optional[redis.Redis] = None):
        self.db = db_manager
        self.cache = cache_client
        self.feature_engine = OptimizedFeatureEngine(use_cache=True)
        self.feature_versions = {}
        
    def _generate_feature_hash(self, features: pd.DataFrame) -> str:
        """Generate hash for feature versioning"""
        feature_str = pd.util.hash_pandas_object(features).sum()
        return hashlib.md5(str(feature_str).encode()).hexdigest()
    
    def save_features(self, ticker: str, features: pd.DataFrame, 
                     feature_set_name: str = "default", 
                     metadata: Optional[Dict] = None) -> str:
        """Save features to feature store with versioning"""
        try:
            # Get or create feature version
            version = self.feature_versions.get(feature_set_name, 1)
            
            # Prepare feature data
            feature_records = []
            for idx, row in features.iterrows():
                # Extract time from index or datetime column
                if isinstance(idx, pd.Timestamp):
                    time = idx
                elif 'datetime' in features.columns:
                    time = row['datetime']
                else:
                    time = datetime.now()
                
                # Convert row to dict excluding time
                feature_dict = row.to_dict()
                if 'datetime' in feature_dict:
                    del feature_dict['datetime']
                
                feature_records.append({
                    'feature_set_name': feature_set_name,
                    'feature_version': version,
                    'ticker': ticker,
                    'time': time,
                    'features': json.dumps(feature_dict),
                    'feature_columns': list(feature_dict.keys()),
                    'metadata': json.dumps(metadata) if metadata else None
                })
            
            # Insert into database
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                
                insert_query = """
                    INSERT INTO feature_store (
                        feature_set_name, feature_version, ticker, time,
                        features, feature_columns, metadata
                    ) VALUES (
                        %(feature_set_name)s, %(feature_version)s, %(ticker)s,
                        %(time)s, %(features)s, %(feature_columns)s, %(metadata)s
                    )
                    ON CONFLICT (feature_set_name, feature_version, ticker, time) 
                    DO UPDATE SET
                        features = EXCLUDED.features,
                        feature_columns = EXCLUDED.feature_columns,
                        metadata = EXCLUDED.metadata
                """
                
                execute_batch(cur, insert_query, feature_records, page_size=1000)
                conn.commit()
                
                logger.info(f"Saved {len(feature_records)} feature records for {ticker}")
                
                # Cache latest features if cache is available
                if self.cache:
                    cache_key = f"features:{ticker}:{feature_set_name}:latest"
                    self.cache.setex(
                        cache_key, 
                        3600,  # 1 hour TTL
                        pickle.dumps(features.tail(100))  # Cache last 100 rows
                    )
                
                return f"{feature_set_name}_v{version}"
                
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
            raise
    
    def load_features(self, ticker: str, start_time: datetime, end_time: datetime,
                     feature_set_name: str = "default",
                     version: Optional[int] = None) -> pd.DataFrame:
        """Load features from feature store"""
        
        # Try cache first for recent data
        if self.cache and (end_time - start_time).days <= 7:
            cache_key = f"features:{ticker}:{feature_set_name}:{start_time}:{end_time}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Loaded features from cache for {ticker}")
                return pickle.loads(cached_data)
        
        # Query from database
        query = """
            SELECT time, features, feature_columns
            FROM feature_store
            WHERE ticker = %s 
                AND feature_set_name = %s
                AND time >= %s 
                AND time <= %s
                AND is_active = TRUE
        """
        
        params = [ticker, feature_set_name, start_time, end_time]
        
        if version is not None:
            query += " AND feature_version = %s"
            params.append(version)
        else:
            query += " AND feature_version = (SELECT MAX(feature_version) FROM feature_store WHERE ticker = %s AND feature_set_name = %s)"
            params.extend([ticker, feature_set_name])
        
        query += " ORDER BY time ASC"
        
        with self.db.get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, params)
            rows = cur.fetchall()
            
            if not rows:
                logger.warning(f"No features found for {ticker} in {feature_set_name}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for row in rows:
                feature_dict = json.loads(row['features'])
                feature_dict['time'] = row['time']
                data.append(feature_dict)
            
            df = pd.DataFrame(data)
            df.set_index('time', inplace=True)
            
            # Cache the result
            if self.cache:
                cache_key = f"features:{ticker}:{feature_set_name}:{start_time}:{end_time}"
                self.cache.setex(cache_key, 3600, pickle.dumps(df))
            
            return df
    
    def get_feature_metadata(self, feature_set_name: str = "default") -> Dict:
        """Get metadata about a feature set"""
        query = """
            SELECT 
                feature_version,
                COUNT(DISTINCT ticker) as n_tickers,
                COUNT(DISTINCT time) as n_timestamps,
                MIN(time) as earliest_time,
                MAX(time) as latest_time,
                array_agg(DISTINCT unnest(feature_columns)) as all_features
            FROM feature_store
            WHERE feature_set_name = %s AND is_active = TRUE
            GROUP BY feature_version
            ORDER BY feature_version DESC
        """
        
        with self.db.get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, (feature_set_name,))
            results = cur.fetchall()
            
            return {
                'versions': results,
                'latest_version': results[0]['feature_version'] if results else 0,
                'total_versions': len(results)
            }

# ======================== ETL Pipeline ========================

class ETLPipeline:
    """
    Main ETL Pipeline orchestrator
    Handles data extraction, transformation, and loading
    """
    
    def __init__(self, config: DatabaseConfig):
        self.db_manager = TimescaleDBManager(config)
        self.feature_store = FeatureStore(self.db_manager, self._init_redis())
        self.market_data_manager = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.jobs = {}
        
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection for caching"""
        try:
            redis_client = redis.from_url(
                Config.CELERY_RESULT_BACKEND if hasattr(Config, 'CELERY_RESULT_BACKEND') 
                else "redis://localhost:6379/0",
                decode_responses=False
            )
            redis_client.ping()
            logger.info("Redis cache initialized for ETL pipeline")
            return redis_client
        except Exception as e:
            logger.warning(f"Redis not available, continuing without cache: {e}")
            return None
    
    async def initialize(self):
        """Initialize the ETL pipeline"""
        self.market_data_manager = await get_market_data_manager()
        logger.info("ETL Pipeline initialized")
    
    def _track_job(self, job_type: str, ticker: Optional[str] = None) -> str:
        """Track ETL job execution"""
        job_id = str(datetime.now().timestamp())
        
        with self.db_manager.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO etl_jobs (job_type, ticker, start_time, status)
                VALUES (%s, %s, %s, %s)
                RETURNING job_id
            """, (job_type, ticker, datetime.now(), "RUNNING"))
            
            job_id = cur.fetchone()[0]
            conn.commit()
            
        self.jobs[job_id] = {
            'type': job_type,
            'ticker': ticker,
            'start_time': datetime.now()
        }
        
        return job_id
    
    def _complete_job(self, job_id: str, status: str, records: int = 0, 
                     error: Optional[str] = None):
        """Mark job as complete"""
        with self.db_manager.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE etl_jobs 
                SET end_time = %s, status = %s, records_processed = %s, error_message = %s
                WHERE job_id = %s
            """, (datetime.now(), status, records, error, job_id))
            conn.commit()
        
        if job_id in self.jobs:
            del self.jobs[job_id]
    
    async def extract_market_data(self, ticker: str, days_back: int = 30) -> pd.DataFrame:
        """Extract market data from providers"""
        job_id = self._track_job("EXTRACT", ticker)
        
        try:
            # Get stock info
            stock_info = await self.market_data_manager.get_stock_info(ticker)
            if not stock_info:
                raise ValueError(f"Could not get stock info for {ticker}")
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            historical_data = await self.market_data_manager.get_historical_data(
                ticker, start_date, end_date
            )
            
            if historical_data is None or historical_data.empty:
                raise ValueError(f"No historical data found for {ticker}")
            
            # Combine with stock info
            historical_data['ticker'] = ticker
            historical_data['market_cap'] = stock_info.market_cap
            historical_data['shares_outstanding'] = stock_info.shares_outstanding
            historical_data['sector'] = stock_info.sector
            historical_data['industry'] = stock_info.industry
            historical_data['exchange'] = stock_info.exchange
            historical_data['data_source'] = "polygon"
            
            self._complete_job(job_id, "SUCCESS", len(historical_data))
            logger.info(f"Extracted {len(historical_data)} records for {ticker}")
            
            return historical_data
            
        except Exception as e:
            self._complete_job(job_id, "FAILED", 0, str(e))
            logger.error(f"Failed to extract data for {ticker}: {e}")
            raise
    
    def transform_and_compute_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Transform data and compute features"""
        job_id = self._track_job("TRANSFORM", ticker)
        
        try:
            # Data quality checks
            df = self._validate_and_clean_data(df)
            
            # Compute features using the optimized feature engine
            features_df = self.feature_store.feature_engine.create_features_optimized(df)
            
            # Additional custom features
            features_df['ticker'] = ticker
            
            # Calculate additional metrics
            features_df['price_momentum'] = features_df['close'].pct_change(periods=20)
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume'].rolling(20).mean()
            
            self._complete_job(job_id, "SUCCESS", len(features_df))
            logger.info(f"Computed {len(features_df.columns)} features for {ticker}")
            
            return features_df
            
        except Exception as e:
            self._complete_job(job_id, "FAILED", 0, str(e))
            logger.error(f"Failed to transform data for {ticker}: {e}")
            raise
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean market data"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df['volume'] = df['volume'].fillna(0)
        
        # Forward fill price data
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill()
        
        # Remove rows with all NaN prices
        df = df.dropna(subset=price_cols, how='all')
        
        # Validate price relationships
        df = df[(df['high'] >= df['low']) & 
               (df['high'] >= df['open']) & 
               (df['high'] >= df['close']) &
               (df['low'] <= df['open']) & 
               (df['low'] <= df['close'])]
        
        # Remove outliers (prices outside 5 standard deviations)
        for col in price_cols:
            mean = df[col].mean()
            std = df[col].std()
            df = df[np.abs(df[col] - mean) <= (5 * std)]
        
        return df
    
    async def load_to_database(self, df: pd.DataFrame, ticker: str) -> int:
        """Load data to database"""
        job_id = self._track_job("LOAD", ticker)
        
        try:
            # Prepare market data records
            market_records = []
            for idx, row in df.iterrows():
                if isinstance(idx, pd.Timestamp):
                    time = idx
                elif 'datetime' in df.columns:
                    time = row['datetime'] 
                else:
                    time = datetime.now()
                
                market_records.append({
                    'time': time,
                    'ticker': ticker,
                    'open': row.get('open'),
                    'high': row.get('high'),
                    'low': row.get('low'),
                    'close': row.get('close'),
                    'volume': row.get('volume'),
                    'market_cap': row.get('market_cap'),
                    'shares_outstanding': row.get('shares_outstanding'),
                    'sector': row.get('sector'),
                    'industry': row.get('industry'),
                    'exchange': row.get('exchange'),
                    'data_source': row.get('data_source', 'unknown')
                })
            
            # Insert market data
            records_inserted = self.db_manager.insert_market_data(market_records)
            
            # Save features to feature store
            self.feature_store.save_features(ticker, df, "etl_pipeline")
            
            self._complete_job(job_id, "SUCCESS", records_inserted)
            logger.info(f"Loaded {records_inserted} records for {ticker}")
            
            return records_inserted
            
        except Exception as e:
            self._complete_job(job_id, "FAILED", 0, str(e))
            logger.error(f"Failed to load data for {ticker}: {e}")
            raise
    
    async def run_etl_for_ticker(self, ticker: str, days_back: int = 30) -> Dict:
        """Run complete ETL pipeline for a ticker"""
        logger.info(f"Starting ETL pipeline for {ticker}")
        
        results = {
            'ticker': ticker,
            'success': False,
            'records_processed': 0,
            'features_computed': 0,
            'errors': []
        }
        
        try:
            # Extract
            raw_data = await self.extract_market_data(ticker, days_back)
            results['records_extracted'] = len(raw_data)
            
            # Transform
            features_df = self.transform_and_compute_features(raw_data, ticker)
            results['features_computed'] = len(features_df.columns)
            
            # Load
            records_loaded = await self.load_to_database(features_df, ticker)
            results['records_processed'] = records_loaded
            
            results['success'] = True
            logger.info(f"ETL pipeline completed for {ticker}: {results}")
            
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"ETL pipeline failed for {ticker}: {e}")
            
        return results
    
    async def run_batch_etl(self, tickers: List[str], days_back: int = 30) -> List[Dict]:
        """Run ETL for multiple tickers in parallel"""
        logger.info(f"Starting batch ETL for {len(tickers)} tickers")
        
        tasks = []
        for ticker in tickers:
            task = self.run_etl_for_ticker(ticker, days_back)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failed = len(results) - successful
        
        logger.info(f"Batch ETL completed: {successful} successful, {failed} failed")
        
        return results
    
    def schedule_etl_jobs(self):
        """Schedule periodic ETL jobs"""
        # Daily data refresh at 6 AM
        schedule.every().day.at("06:00").do(
            lambda: asyncio.run(self.run_batch_etl(['AAPL', 'MSFT', 'GOOGL'], 7))
        )
        
        # Hourly updates during market hours
        for hour in range(9, 17):  # 9 AM to 4 PM
            schedule.every().day.at(f"{hour:02d}:30").do(
                lambda: asyncio.run(self.run_batch_etl(['SPY'], 1))
            )
        
        logger.info("ETL jobs scheduled")
    
    def calculate_data_quality_metrics(self, ticker: str, date: datetime) -> Dict:
        """Calculate data quality metrics"""
        metrics = {
            'ticker': ticker,
            'date': date,
            'completeness_score': 0.0,
            'accuracy_score': 0.0,
            'timeliness_score': 0.0,
            'consistency_score': 0.0,
            'missing_data_points': 0,
            'anomaly_count': 0
        }
        
        try:
            # Get data for the date
            start_time = date.replace(hour=0, minute=0, second=0)
            end_time = date.replace(hour=23, minute=59, second=59)
            
            df = self.db_manager.get_market_data(ticker, start_time, end_time)
            
            if df.empty:
                logger.warning(f"No data found for {ticker} on {date}")
                return metrics
            
            # Completeness: Check for missing values
            total_fields = len(df.columns) * len(df)
            missing_values = df.isnull().sum().sum()
            metrics['completeness_score'] = 1 - (missing_values / total_fields)
            metrics['missing_data_points'] = missing_values
            
            # Accuracy: Check price relationships
            valid_prices = ((df['high'] >= df['low']) & 
                          (df['high'] >= df['open']) & 
                          (df['high'] >= df['close'])).sum()
            metrics['accuracy_score'] = valid_prices / len(df)
            
            # Timeliness: Check data recency
            latest_update = df['time'].max()
            delay_hours = (datetime.now(latest_update.tzinfo) - latest_update).total_seconds() / 3600
            metrics['timeliness_score'] = max(0, 1 - (delay_hours / 24))
            
            # Consistency: Check for outliers
            price_cols = ['open', 'high', 'low', 'close']
            anomalies = 0
            for col in price_cols:
                mean = df[col].mean()
                std = df[col].std()
                outliers = ((df[col] - mean).abs() > (3 * std)).sum()
                anomalies += outliers
            
            metrics['anomaly_count'] = anomalies
            metrics['consistency_score'] = 1 - (anomalies / (len(df) * len(price_cols)))
            
            # Save metrics to database
            with self.db_manager.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO data_quality_metrics (
                        ticker, metric_date, completeness_score, accuracy_score,
                        timeliness_score, consistency_score, missing_data_points,
                        anomaly_count, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, metric_date) DO UPDATE SET
                        completeness_score = EXCLUDED.completeness_score,
                        accuracy_score = EXCLUDED.accuracy_score,
                        timeliness_score = EXCLUDED.timeliness_score,
                        consistency_score = EXCLUDED.consistency_score,
                        missing_data_points = EXCLUDED.missing_data_points,
                        anomaly_count = EXCLUDED.anomaly_count
                """, (
                    ticker, date.date(),
                    metrics['completeness_score'],
                    metrics['accuracy_score'],
                    metrics['timeliness_score'],
                    metrics['consistency_score'],
                    metrics['missing_data_points'],
                    metrics['anomaly_count'],
                    json.dumps({'calculated_at': datetime.now().isoformat()})
                ))
                conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to calculate data quality metrics: {e}")
        
        return metrics
    
    def get_pipeline_stats(self) -> Dict:
        """Get ETL pipeline statistics"""
        with self.db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get job statistics
            cur.execute("""
                SELECT 
                    job_type,
                    status,
                    COUNT(*) as count,
                    AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_duration_seconds,
                    SUM(records_processed) as total_records
                FROM etl_jobs
                WHERE start_time >= NOW() - INTERVAL '24 hours'
                GROUP BY job_type, status
            """)
            job_stats = cur.fetchall()
            
            # Get data volume statistics
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT ticker) as unique_tickers,
                    COUNT(*) as total_records,
                    MIN(time) as earliest_record,
                    MAX(time) as latest_record
                FROM market_data
            """)
            data_stats = cur.fetchone()
            
            # Get feature store statistics
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT feature_set_name) as feature_sets,
                    COUNT(DISTINCT ticker) as tickers_with_features,
                    COUNT(*) as total_feature_records
                FROM feature_store
                WHERE is_active = TRUE
            """)
            feature_stats = cur.fetchone()
            
            return {
                'job_statistics': job_stats,
                'data_statistics': data_stats,
                'feature_statistics': feature_stats,
                'active_jobs': len(self.jobs),
                'pipeline_status': 'RUNNING' if self.running else 'STOPPED'
            }
    
    async def cleanup_old_data(self, days_to_keep: int = 730):
        """Clean up old data beyond retention period"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            # Clean market data
            cur.execute("""
                DELETE FROM market_data 
                WHERE time < %s
                RETURNING COUNT(*)
            """, (cutoff_date,))
            
            market_deleted = cur.fetchone()[0] if cur.rowcount > 0 else 0
            
            # Clean old feature versions
            cur.execute("""
                UPDATE feature_store 
                SET is_active = FALSE 
                WHERE time < %s AND is_active = TRUE
                RETURNING COUNT(*)
            """, (cutoff_date,))
            
            features_archived = cur.fetchone()[0] if cur.rowcount > 0 else 0
            
            # Clean old ETL jobs
            cur.execute("""
                DELETE FROM etl_jobs 
                WHERE start_time < %s
                RETURNING COUNT(*)
            """, (cutoff_date,))
            
            jobs_deleted = cur.fetchone()[0] if cur.rowcount > 0 else 0
            
            conn.commit()
            
            logger.info(f"Cleanup completed: {market_deleted} market records, "
                       f"{features_archived} features archived, {jobs_deleted} jobs deleted")
            
            return {
                'market_records_deleted': market_deleted,
                'features_archived': features_archived,
                'jobs_deleted': jobs_deleted
            }
    
    def export_to_parquet(self, ticker: str, output_path: str, 
                         start_date: datetime, end_date: datetime):
        """Export data to Parquet format for efficient storage"""
        # Get market data
        market_df = self.db_manager.get_market_data(ticker, start_date, end_date)
        
        # Get features
        features_df = self.feature_store.load_features(
            ticker, start_date, end_date, "etl_pipeline"
        )
        
        # Merge data
        if not features_df.empty and not market_df.empty:
            combined_df = pd.merge(
                market_df, 
                features_df, 
                left_on='time', 
                right_index=True, 
                how='outer'
            )
        else:
            combined_df = market_df
        
        # Write to Parquet
        table = pa.Table.from_pandas(combined_df)
        pq.write_table(table, output_path, compression='snappy')
        
        logger.info(f"Exported {len(combined_df)} records to {output_path}")
        
        return output_path
    
    async def start(self):
        """Start the ETL pipeline"""
        await self.initialize()
        self.running = True
        self.schedule_etl_jobs()
        
        logger.info("ETL Pipeline started")
        
        # Run scheduled jobs
        while self.running:
            schedule.run_pending()
            await asyncio.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop the ETL pipeline"""
        self.running = False
        self.executor.shutdown(wait=True)
        
        if self.market_data_manager:
            asyncio.run(self.market_data_manager.close())
        
        logger.info("ETL Pipeline stopped")

# ======================== Main Execution ========================

async def main():
    """Main function to run the ETL pipeline"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create database configuration
    db_config = DatabaseConfig()
    
    # Initialize ETL pipeline
    pipeline = ETLPipeline(db_config)
    
    try:
        # Run initial batch ETL for key tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        results = await pipeline.run_batch_etl(tickers, days_back=30)
        
        # Print results
        for result in results:
            if isinstance(result, dict):
                print(f"Ticker: {result['ticker']}, Success: {result['success']}, "
                      f"Records: {result.get('records_processed', 0)}")
        
        # Calculate data quality metrics
        for ticker in tickers:
            metrics = pipeline.calculate_data_quality_metrics(ticker, datetime.now())
            print(f"Data quality for {ticker}: Completeness={metrics['completeness_score']:.2f}, "
                  f"Accuracy={metrics['accuracy_score']:.2f}")
        
        # Get pipeline statistics
        stats = pipeline.get_pipeline_stats()
        print(f"Pipeline stats: {json.dumps(stats, indent=2, default=str)}")
        
        # Start continuous pipeline
        await pipeline.start()
        
    except KeyboardInterrupt:
        logger.info("Shutting down ETL pipeline...")
        pipeline.stop()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        traceback.print_exc()
    finally:
        pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())
