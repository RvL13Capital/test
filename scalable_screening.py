# project/screening_optimized.py
"""
Optimized and scalable market screening system
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .market_data import get_market_data_manager
from .stock_universe import get_universe_manager, StockListing
from .consolidation_network import NetworkConsolidationAnalyzer
from .auto_optimizer_working import WorkingAutoOptimizer
from .config import ExtendedConfig

logger = logging.getLogger(__name__)

@dataclass
class ScreeningCandidate:
    """Screening result for a stock"""
    ticker: str
    market_cap: float
    sector: str
    consolidation_score: float
    breakout_probability: float
    expected_magnitude: float
    phase_transition_score: float
    volume_surge: float
    relative_strength: float
    score: float
    last_updated: datetime

class ParallelScreener:
    """Highly optimized parallel screening system"""
    
    def __init__(self):
        self.process_pool = None
        self.semaphore = None
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)
    
    async def initialize(self):
        """Initialize the screening system"""
        # Create process pool for CPU-intensive work
        self.process_pool = ProcessPoolExecutor(
            max_workers=min(mp.cpu_count(), ExtendedConfig.MAX_CONCURRENT_ANALYSES)
        )
        
        # Semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(ExtendedConfig.SCREENING_BATCH_SIZE)
        
        logger.info("Parallel screening system initialized")
    
    async def screen_market(self, min_market_cap: float = 10e6,
                          max_market_cap: float = 2e9,
                          sectors: Optional[List[str]] = None,
                          limit: int = 50) -> List[ScreeningCandidate]:
        """Screen market with optimized parallel processing"""
        
        start_time = datetime.now()
        
        # Get universe with server-side filtering
        universe_stocks = await self._get_filtered_universe(
            min_market_cap, max_market_cap, sectors
        )
        
        if not universe_stocks:
            logger.warning("No stocks found in universe")
            return []
        
        logger.info(f"Screening {len(universe_stocks)} stocks...")
        
        # Filter cached results
        candidates = []
        stocks_to_screen = []
        
        for stock in universe_stocks:
            cached = self._get_cached_result(stock.ticker)
            if cached:
                candidates.append(cached)
            else:
                stocks_to_screen.append(stock)
        
        logger.info(f"Using {len(candidates)} cached results, screening {len(stocks_to_screen)} new stocks")
        
        # Batch process remaining stocks
        if stocks_to_screen:
            # Sort by market cap for better distribution
            stocks_to_screen.sort(key=lambda x: x.market_cap or 0, reverse=True)
            
            # Process in optimized batches
            batch_results = await self._process_batches(stocks_to_screen, limit * 2)
            candidates.extend(batch_results)
        
        # Sort by score and limit
        candidates.sort(key=lambda x: x.score, reverse=True)
        top_candidates = candidates[:limit]
        
        screening_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Screening completed in {screening_time:.1f}s, found {len(top_candidates)} candidates")
        
        return top_candidates
    
    async def _get_filtered_universe(self, min_cap: float, max_cap: float,
                                   sectors: Optional[List[str]] = None) -> List[StockListing]:
        """Get universe with server-side filtering where possible"""
        
        universe_manager = get_universe_manager()
        
        # Try to use provider-specific filtering
        if ExtendedConfig.PRIMARY_DATA_PROVIDER == 'polygon' and ExtendedConfig.POLYGON_API_KEY:
            # Polygon supports market cap filtering in the API
            return await self._get_polygon_filtered_universe(min_cap, max_cap, sectors)
        else:
            # Fallback to local filtering
            all_stocks = await universe_manager.get_stocks_by_market_cap(min_cap, max_cap)
            
            if sectors:
                all_stocks = [s for s in all_stocks if s.sector in sectors]
            
            return all_stocks
    
    async def _get_polygon_filtered_universe(self, min_cap: float, max_cap: float,
                                           sectors: Optional[List[str]] = None) -> List[StockListing]:
        """Get filtered universe directly from Polygon API"""
        stocks = []
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.polygon.io/v3/reference/tickers"
                params = {
                    "apiKey": ExtendedConfig.POLYGON_API_KEY,
                    "market": "stocks",
                    "active": "true",
                    "gte.market_cap": str(min_cap),
                    "lte.market_cap": str(max_cap),
                    "limit": 1000
                }
                
                # Add sector filter if specified
                if sectors and len(sectors) == 1:
                    # Polygon supports single sector filter
                    params["sic_description"] = sectors[0]
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        for item in data.get('results', []):
                            stock = StockListing(
                                ticker=item['ticker'],
                                name=item.get('name', ''),
                                exchange=item.get('primary_exchange', ''),
                                market_cap=item.get('market_cap'),
                                sector=item.get('sic_description'),
                                last_updated=datetime.now()
                            )
                            
                            # Additional client-side sector filter if needed
                            if not sectors or stock.sector in sectors:
                                stocks.append(stock)
                
                logger.info(f"Got {len(stocks)} filtered stocks from Polygon")
                
        except Exception as e:
            logger.error(f"Polygon filtered query failed: {e}")
        
        return stocks
    
    async def _process_batches(self, stocks: List[StockListing], max_total: int) -> List[ScreeningCandidate]:
        """Process stocks in optimized batches"""
        results = []
        
        # Dynamic batch sizing based on system load
        batch_size = self._calculate_optimal_batch_size(len(stocks))
        
        for i in range(0, min(len(stocks), max_total), batch_size):
            batch = stocks[i:i + batch_size]
            
            # Process batch with rate limiting
            batch_tasks = []
            for stock in batch:
                task = self._process_single_stock_limited(stock)
                batch_tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Filter successful results
            for result in batch_results:
                if isinstance(result, ScreeningCandidate):
                    results.append(result)
                    self._cache_result(result)
            
            # Early exit if we have enough good candidates
            if len(results) >= max_total:
                break
            
            # Small delay between batches to avoid overwhelming APIs
            await asyncio.sleep(0.1)
        
        return results
    
    async def _process_single_stock_limited(self, stock: StockListing) -> Optional[ScreeningCandidate]:
        """Process single stock with rate limiting"""
        async with self.semaphore:
            return await self._process_single_stock(stock)
    
    async def _process_single_stock(self, stock: StockListing) -> Optional[ScreeningCandidate]:
        """Process a single stock for screening"""
        try:
            market_manager = await get_market_data_manager()
            
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            df = await market_manager.get_historical_data(
                stock.ticker, start_date, end_date
            )
            
            if df is None or len(df) < 60:
                return None
            
            # Quick pre-filter: check basic volatility
            returns = df['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)
            
            # Skip if too volatile or too stable
            if volatility < 0.15 or volatility > 1.0:
                return None
            
            # Get consolidation analysis
            analyzer = NetworkConsolidationAnalyzer()
            current_phase = analyzer.get_current_phase(df)
            
            if current_phase['status'] != 'consolidation':
                return None
            
            # Calculate additional metrics
            volume_surge = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            relative_strength = await market_manager.calculate_relative_strength(stock.ticker, 20)
            
            # Quick scoring without full model prediction (for initial filtering)
            consolidation_score = current_phase.get('phase_transition_score', 0)
            
            # Heuristic scoring
            score = (
                consolidation_score * 0.4 +
                min(volume_surge / 3, 1.0) * 0.3 +
                (relative_strength + 1) / 2 * 0.3
            )
            
            candidate = ScreeningCandidate(
                ticker=stock.ticker,
                market_cap=stock.market_cap or 0,
                sector=stock.sector or 'Unknown',
                consolidation_score=consolidation_score,
                breakout_probability=0,  # Will be filled by detailed analysis
                expected_magnitude=0,
                phase_transition_score=current_phase.get('phase_transition_score', 0),
                volume_surge=volume_surge,
                relative_strength=relative_strength or 0,
                score=score,
                last_updated=datetime.now()
            )
            
            return candidate
            
        except Exception as e:
            logger.debug(f"Error screening {stock.ticker}: {e}")
            return None
    
    def _calculate_optimal_batch_size(self, total_stocks: int) -> int:
        """Calculate optimal batch size based on system resources"""
        # Base it on CPU count and configured limits
        cpu_count = mp.cpu_count()
        max_concurrent = ExtendedConfig.MAX_CONCURRENT_ANALYSES
        
        # Dynamic sizing
        if total_stocks < 100:
            return min(10, max_concurrent)
        elif total_stocks < 500:
            return min(20, max_concurrent)
        else:
            return min(cpu_count * 2, max_concurrent)
    
    def _get_cached_result(self, ticker: str) -> Optional[ScreeningCandidate]:
        """Get cached screening result"""
        if ticker in self.cache:
            cached_data, timestamp = self.cache[ticker]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_data
            else:
                del self.cache[ticker]
        return None
    
    def _cache_result(self, result: ScreeningCandidate):
        """Cache screening result"""
        self.cache[result.ticker] = (result, datetime.now())
    
    async def get_detailed_predictions(self, candidates: List[ScreeningCandidate],
                                     optimizer: WorkingAutoOptimizer) -> List[ScreeningCandidate]:
        """Get detailed model predictions for top candidates"""
        if not optimizer.current_model:
            logger.warning("No model available for detailed predictions")
            return candidates
        
        # Process top candidates with full model
        detailed_candidates = []
        
        for candidate in candidates[:20]:  # Limit to top 20 for performance
            try:
                # Get full prediction
                market_manager = await get_market_data_manager()
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
                
                df = await market_manager.get_historical_data(
                    candidate.ticker, start_date, end_date
                )
                
                if df is not None and len(df) >= 60:
                    # Extract features and make prediction
                    from .consolidation_network import extract_consolidation_features
                    df_features = extract_consolidation_features(df, candidate.market_cap)
                    
                    prediction = await self._make_model_prediction(
                        optimizer.current_model,
                        optimizer.current_scaler,
                        df_features
                    )
                    
                    if prediction:
                        # Update candidate with model predictions
                        candidate.breakout_probability = prediction['breakout_probability']
                        candidate.expected_magnitude = prediction['expected_magnitude']
                        
                        # Recalculate score with model predictions
                        candidate.score = (
                            candidate.breakout_probability * 0.5 +
                            candidate.expected_magnitude * 0.3 +
                            candidate.consolidation_score * 0.2
                        )
                
                detailed_candidates.append(candidate)
                
            except Exception as e:
                logger.debug(f"Error getting detailed prediction for {candidate.ticker}: {e}")
                detailed_candidates.append(candidate)
        
        # Re-sort by updated scores
        detailed_candidates.sort(key=lambda x: x.score, reverse=True)
        
        return detailed_candidates
    
    async def _make_model_prediction(self, model, scaler, df_features: pd.DataFrame) -> Optional[Dict]:
        """Make prediction using the model"""
        # Similar to the prediction logic in working_api.py
        # but extracted for reuse
        try:
            import torch
            
            device = next(model.parameters()).device
            window_size = 60
            
            if len(df_features) < window_size:
                return None
            
            feature_cols = [col for col in df_features.columns 
                          if col not in ['datetime', 'ticker', 'market_cap']]
            
            sequence = df_features[feature_cols].tail(window_size).values
            sequence_flat = sequence.reshape(-1, len(feature_cols))
            sequence_normalized = scaler.transform(sequence_flat).reshape(1, window_size, -1)
            
            input_tensor = torch.tensor(sequence_normalized, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
            
            return {
                'breakout_probability': outputs['breakout_probability'].cpu().item(),
                'expected_magnitude': outputs['expected_magnitude'].cpu().item()
            }
            
        except Exception as e:
            logger.debug(f"Model prediction error: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)

# Global instance
_parallel_screener = None

async def get_parallel_screener() -> ParallelScreener:
    """Get or create parallel screener"""
    global _parallel_screener
    if _parallel_screener is None:
        _parallel_screener = ParallelScreener()
        await _parallel_screener.initialize()
    return _parallel_screener