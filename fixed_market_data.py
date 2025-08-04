# project/market_data.py
"""
Proper market data handling with multiple providers and accurate market cap
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class StockInfo:
    """Complete stock information"""
    ticker: str
    market_cap: float
    shares_outstanding: float
    sector: str
    industry: str
    exchange: str
    last_price: float
    avg_volume_30d: float
    last_updated: datetime

class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    @abstractmethod
    async def get_stock_info(self, ticker: str) -> Optional[StockInfo]:
        pass
    
    @abstractmethod
    async def get_historical_data(self, ticker: str, start_date: datetime, 
                                end_date: datetime) -> Optional[pd.DataFrame]:
        pass
    
    @abstractmethod
    async def get_market_index(self, index: str, period: int = 30) -> Optional[pd.DataFrame]:
        pass

class PolygonDataProvider(MarketDataProvider):
    """Polygon.io data provider - reliable for production"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = None
    
    async def _get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_stock_info(self, ticker: str) -> Optional[StockInfo]:
        """Get accurate stock information from Polygon"""
        try:
            session = await self._get_session()
            
            # Get ticker details
            details_url = f"{self.base_url}/v3/reference/tickers/{ticker}"
            params = {"apiKey": self.api_key}
            
            async with session.get(details_url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to get details for {ticker}: {resp.status}")
                    return None
                
                data = await resp.json()
                details = data.get('results', {})
            
            # Get latest quote for current price
            quote_url = f"{self.base_url}/v2/last/nbbo/{ticker}"
            async with session.get(quote_url, params=params) as resp:
                if resp.status == 200:
                    quote_data = await resp.json()
                    last_price = quote_data.get('results', {}).get('p', 0)
                else:
                    last_price = 0
            
            # Calculate accurate market cap
            shares_outstanding = details.get('share_class_shares_outstanding', 0)
            market_cap = shares_outstanding * last_price
            
            # Get 30-day average volume
            volume_url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')}/{datetime.now().strftime('%Y-%m-%d')}"
            async with session.get(volume_url, params=params) as resp:
                if resp.status == 200:
                    volume_data = await resp.json()
                    volumes = [bar['v'] for bar in volume_data.get('results', [])]
                    avg_volume = np.mean(volumes) if volumes else 0
                else:
                    avg_volume = 0
            
            return StockInfo(
                ticker=ticker,
                market_cap=market_cap,
                shares_outstanding=shares_outstanding,
                sector=details.get('sic_description', 'Unknown'),
                industry=details.get('sic_code', 'Unknown'),
                exchange=details.get('primary_exchange', 'Unknown'),
                last_price=last_price,
                avg_volume_30d=avg_volume,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting stock info for {ticker}: {e}")
            return None
    
    async def get_historical_data(self, ticker: str, start_date: datetime, 
                                end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data"""
        try:
            session = await self._get_session()
            
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apiKey": self.api_key, "adjusted": "true"}
            
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to get historical data for {ticker}: {resp.status}")
                    return None
                
                data = await resp.json()
                results = data.get('results', [])
                
                if not results:
                    return None
                
                df = pd.DataFrame(results)
                df['datetime'] = pd.to_datetime(df['t'], unit='ms')
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                })
                df = df.set_index('datetime')
                
                return df[['open', 'high', 'low', 'close', 'volume']]
                
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {e}")
            return None
    
    async def get_market_index(self, index: str = 'SPY', period: int = 30) -> Optional[pd.DataFrame]:
        """Get market index data for relative strength calculation"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period)
        return await self.get_historical_data(index, start_date, end_date)
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

class AlphaVantageDataProvider(MarketDataProvider):
    """Alpha Vantage as backup provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = None
    
    async def _get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_stock_info(self, ticker: str) -> Optional[StockInfo]:
        """Get stock overview from Alpha Vantage"""
        try:
            session = await self._get_session()
            
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": self.api_key
            }
            
            async with session.get(self.base_url, params=params) as resp:
                if resp.status != 200:
                    return None
                
                data = await resp.json()
                
                # Get quote for current price
                quote_params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": ticker,
                    "apikey": self.api_key
                }
                
                async with session.get(self.base_url, params=quote_params) as quote_resp:
                    quote_data = await quote_resp.json()
                    last_price = float(quote_data.get("Global Quote", {}).get("05. price", 0))
                
                market_cap = float(data.get("MarketCapitalization", 0))
                shares_outstanding = float(data.get("SharesOutstanding", 0))
                
                # If shares outstanding is not available, calculate from market cap
                if shares_outstanding == 0 and market_cap > 0 and last_price > 0:
                    shares_outstanding = market_cap / last_price
                
                return StockInfo(
                    ticker=ticker,
                    market_cap=market_cap,
                    shares_outstanding=shares_outstanding,
                    sector=data.get("Sector", "Unknown"),
                    industry=data.get("Industry", "Unknown"),
                    exchange=data.get("Exchange", "Unknown"),
                    last_price=last_price,
                    avg_volume_30d=float(data.get("50DayMovingAverage", 0)),
                    last_updated=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Alpha Vantage error for {ticker}: {e}")
            return None
    
    async def get_historical_data(self, ticker: str, start_date: datetime, 
                                end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical data from Alpha Vantage"""
        try:
            session = await self._get_session()
            
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "apikey": self.api_key,
                "outputsize": "full"
            }
            
            async with session.get(self.base_url, params=params) as resp:
                if resp.status != 200:
                    return None
                
                data = await resp.json()
                time_series = data.get("Time Series (Daily)", {})
                
                if not time_series:
                    return None
                
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.rename(columns={
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '6. volume': 'volume'
                })
                
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                # Filter date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                return df[['open', 'high', 'low', 'close', 'volume']].sort_index()
                
        except Exception as e:
            logger.error(f"Alpha Vantage historical data error: {e}")
            return None
    
    async def get_market_index(self, index: str = 'SPY', period: int = 30) -> Optional[pd.DataFrame]:
        """Get market index data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period)
        return await self.get_historical_data(index, start_date, end_date)
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

class MarketDataManager:
    """Manager for multiple data providers with fallback"""
    
    def __init__(self, primary_provider: MarketDataProvider, 
                 backup_providers: List[MarketDataProvider] = None):
        self.primary_provider = primary_provider
        self.backup_providers = backup_providers or []
        self._cache = {}  # Simple in-memory cache
        self._cache_ttl = 300  # 5 minutes
    
    def _get_cache_key(self, method: str, *args) -> str:
        """Generate cache key"""
        return f"{method}:{':'.join(str(arg) for arg in args)}"
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached data is still valid"""
        return (datetime.now() - timestamp).total_seconds() < self._cache_ttl
    
    async def get_stock_info(self, ticker: str) -> Optional[StockInfo]:
        """Get stock info with fallback to backup providers"""
        cache_key = self._get_cache_key("stock_info", ticker)
        
        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if self._is_cache_valid(timestamp):
                return cached_data
        
        # Try primary provider
        info = await self.primary_provider.get_stock_info(ticker)
        
        # Fallback to backup providers
        if not info:
            for provider in self.backup_providers:
                info = await provider.get_stock_info(ticker)
                if info:
                    break
        
        # Cache result
        if info:
            self._cache[cache_key] = (info, datetime.now())
        
        return info
    
    async def get_historical_data(self, ticker: str, start_date: datetime,
                                end_date: datetime) -> Optional[pd.DataFrame]:
        """Get historical data with fallback"""
        # Try primary provider
        data = await self.primary_provider.get_historical_data(ticker, start_date, end_date)
        
        # Fallback to backup providers
        if data is None or data.empty:
            for provider in self.backup_providers:
                data = await provider.get_historical_data(ticker, start_date, end_date)
                if data is not None and not data.empty:
                    break
        
        return data
    
    async def get_market_index(self, index: str = 'SPY', period: int = 30) -> Optional[pd.DataFrame]:
        """Get market index data"""
        cache_key = self._get_cache_key("market_index", index, period)
        
        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if self._is_cache_valid(timestamp):
                return cached_data
        
        # Try providers
        data = await self.primary_provider.get_market_index(index, period)
        
        if data is None:
            for provider in self.backup_providers:
                data = await provider.get_market_index(index, period)
                if data is not None:
                    break
        
        # Cache result
        if data is not None:
            self._cache[cache_key] = (data, datetime.now())
        
        return data
    
    async def calculate_relative_strength(self, ticker: str, period: int = 20) -> Optional[float]:
        """Calculate true relative strength vs market index"""
        try:
            # Get stock data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period + 5)  # Extra days for calculation
            
            stock_data = await self.get_historical_data(ticker, start_date, end_date)
            if stock_data is None or len(stock_data) < period:
                return None
            
            # Get market index
            market_data = await self.get_market_index('SPY', period + 5)
            if market_data is None or len(market_data) < period:
                return None
            
            # Align dates
            common_dates = stock_data.index.intersection(market_data.index)
            if len(common_dates) < period:
                return None
            
            stock_data = stock_data.loc[common_dates].tail(period)
            market_data = market_data.loc[common_dates].tail(period)
            
            # Calculate returns
            stock_return = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[0]) - 1
            market_return = (market_data['close'].iloc[-1] / market_data['close'].iloc[0]) - 1
            
            # Relative strength
            relative_strength = stock_return - market_return
            
            return relative_strength
            
        except Exception as e:
            logger.error(f"Error calculating relative strength for {ticker}: {e}")
            return None
    
    async def close(self):
        """Close all providers"""
        await self.primary_provider.close()
        for provider in self.backup_providers:
            await provider.close()

# Factory function to create market data manager
def create_market_data_manager(config: Dict) -> MarketDataManager:
    """Create market data manager with configured providers"""
    
    # Primary provider (Polygon)
    polygon_key = config.get('POLYGON_API_KEY')
    if polygon_key:
        primary = PolygonDataProvider(polygon_key)
    else:
        raise ValueError("No primary data provider configured")
    
    # Backup providers
    backup_providers = []
    
    alpha_vantage_key = config.get('ALPHA_VANTAGE_API_KEY')
    if alpha_vantage_key:
        backup_providers.append(AlphaVantageDataProvider(alpha_vantage_key))
    
    return MarketDataManager(primary, backup_providers)

# Global instance
_market_data_manager = None

async def get_market_data_manager() -> MarketDataManager:
    """Get or create market data manager"""
    global _market_data_manager
    
    if _market_data_manager is None:
        from .config import Config
        config = {
            'POLYGON_API_KEY': Config.POLYGON_API_KEY,
            'ALPHA_VANTAGE_API_KEY': Config.ALPHA_VANTAGE_API_KEY
        }
        _market_data_manager = create_market_data_manager(config)
    
    return _market_data_manager