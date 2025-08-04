# project/consolidation_network.py
"""
Advanced Consolidation Detection with Network Analysis
Treats consolidation phases as complex networks undergoing phase transitions
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ConsolidationMetrics:
    """Metrics for consolidation phase analysis"""
    duration_days: int
    price_range: float
    volume_pattern: str
    network_density: float
    clustering_coefficient: float
    phase_transition_score: float
    mispricing_indicator: float
    accumulation_score: float
    breakout_probability: float
    expected_move: float

class NetworkConsolidationAnalyzer:
    """
    Analyzes consolidation phases as complex networks to detect phase transitions
    and predict explosive breakouts for nano/small-cap stocks
    """
    
    def __init__(self, 
                 min_consolidation_days: int = 20,
                 max_consolidation_days: int = 120,
                 price_tolerance: float = 0.15,  # 15% range for consolidation
                 min_volume_ratio: float = 0.7):
        
        self.min_consolidation_days = min_consolidation_days
        self.max_consolidation_days = max_consolidation_days
        self.price_tolerance = price_tolerance
        self.min_volume_ratio = min_volume_ratio
        
        # Network analysis parameters
        self.correlation_window = 10
        self.network_threshold = 0.65
        
        # Phase transition parameters
        self.critical_density = 0.68  # Percolation threshold
        self.critical_clustering = 0.45
        
    @jit(nopython=True)
    def _detect_consolidation_boundaries(self, prices: np.ndarray, volumes: np.ndarray) -> List[Tuple[int, int]]:
        """
        Fast detection of consolidation boundaries using price and volume patterns
        """
        n = len(prices)
        consolidations = []
        
        i = 0
        while i < n - self.min_consolidation_days:
            # Check if we're entering a consolidation
            window_end = min(i + self.max_consolidation_days, n)
            
            for j in range(i + self.min_consolidation_days, window_end):
                window_prices = prices[i:j]
                window_volumes = volumes[i:j]
                
                # Calculate price range
                price_range = (np.max(window_prices) - np.min(window_prices)) / np.mean(window_prices)
                
                # Check volume consistency
                avg_volume = np.mean(window_volumes)
                volume_consistency = np.min(window_volumes) / avg_volume if avg_volume > 0 else 0
                
                # Consolidation criteria
                if price_range <= self.price_tolerance and volume_consistency >= self.min_volume_ratio:
                    # Extend consolidation as far as possible
                    end_idx = j
                    for k in range(j, window_end):
                        extended_prices = prices[i:k+1]
                        extended_range = (np.max(extended_prices) - np.min(extended_prices)) / np.mean(extended_prices)
                        
                        if extended_range <= self.price_tolerance:
                            end_idx = k
                        else:
                            break
                    
                    consolidations.append((i, end_idx))
                    i = end_idx  # Skip to end of consolidation
                    break
            else:
                i += 1
        
        return consolidations
    
    def _build_correlation_network(self, df_window: pd.DataFrame) -> nx.Graph:
        """
        Build correlation network from price, volume, and technical indicators
        """
        # Features for correlation analysis
        features = ['close', 'volume', 'high_low_ratio', 'close_open_ratio', 
                   'volume_price_correlation', 'price_efficiency']
        
        # Calculate additional features
        df_window['high_low_ratio'] = df_window['high'] / df_window['low']
        df_window['close_open_ratio'] = df_window['close'] / df_window['open']
        
        # Rolling correlation between volume and price
        df_window['volume_price_correlation'] = (
            df_window['volume'].rolling(5).corr(df_window['close'])
        )
        
        # Price efficiency (how directional vs random the movement is)
        returns = df_window['close'].pct_change()
        df_window['price_efficiency'] = returns.rolling(5).sum() / returns.rolling(5).std()
        
        # Create correlation matrix using rolling windows
        correlation_matrices = []
        
        for i in range(self.correlation_window, len(df_window)):
            window = df_window.iloc[i-self.correlation_window:i]
            
            # Normalize features
            scaler = StandardScaler()
            normalized = scaler.fit_transform(window[features].fillna(0))
            
            # Calculate correlation
            corr = np.corrcoef(normalized.T)
            correlation_matrices.append(corr)
        
        # Average correlation matrix
        avg_correlation = np.mean(correlation_matrices, axis=0)
        
        # Build network
        G = nx.Graph()
        n_features = len(features)
        
        # Add nodes
        for i, feature in enumerate(features):
            G.add_node(i, name=feature)
        
        # Add edges based on correlation threshold
        for i in range(n_features):
            for j in range(i+1, n_features):
                if abs(avg_correlation[i, j]) > self.network_threshold:
                    G.add_edge(i, j, weight=abs(avg_correlation[i, j]))
        
        return G
    
    def _calculate_phase_transition_indicators(self, G: nx.Graph, df_window: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate network metrics that indicate phase transitions
        """
        # Network density (connectedness)
        density = nx.density(G) if len(G) > 1 else 0
        
        # Average clustering coefficient
        clustering = nx.average_clustering(G) if len(G) > 0 else 0
        
        # Degree centrality variance (inequality in connections)
        if len(G) > 1:
            centralities = list(nx.degree_centrality(G).values())
            centrality_variance = np.var(centralities)
        else:
            centrality_variance = 0
        
        # Price-volume divergence (potential mispricing)
        price_trend = stats.linregress(range(len(df_window)), df_window['close'])[0]
        volume_trend = stats.linregress(range(len(df_window)), df_window['volume'])[0]
        
        # Normalize trends
        price_trend_norm = price_trend / df_window['close'].mean()
        volume_trend_norm = volume_trend / df_window['volume'].mean()
        
        divergence = abs(price_trend_norm - volume_trend_norm)
        
        # Accumulation detection (volume increases while price stays flat)
        accumulation_score = 0
        if abs(price_trend_norm) < 0.01:  # Flat price
            if volume_trend_norm > 0.05:  # Increasing volume
                accumulation_score = volume_trend_norm / (abs(price_trend_norm) + 0.001)
        
        # Phase transition score (proximity to critical values)
        density_score = 1 / (1 + abs(density - self.critical_density))
        clustering_score = 1 / (1 + abs(clustering - self.critical_clustering))
        
        phase_transition_score = (density_score + clustering_score) / 2
        
        return {
            'density': density,
            'clustering': clustering,
            'centrality_variance': centrality_variance,
            'divergence': divergence,
            'accumulation_score': accumulation_score,
            'phase_transition_score': phase_transition_score
        }
    
    def _analyze_microstructure(self, df_window: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze market microstructure for hidden accumulation patterns
        """
        # Intraday volatility patterns
        df_window['intraday_range'] = (df_window['high'] - df_window['low']) / df_window['close']
        
        # Volume at price levels (simplified)
        df_window['price_level'] = pd.qcut(df_window['close'], q=10, labels=False, duplicates='drop')
        volume_profile = df_window.groupby('price_level')['volume'].sum()
        
        # Volume concentration (is volume concentrated at certain price levels?)
        volume_concentration = volume_profile.std() / volume_profile.mean() if len(volume_profile) > 0 else 0
        
        # Smart money indicators
        # 1. Large volume with small price movement (absorption)
        df_window['absorption'] = df_window['volume'] / (df_window['intraday_range'] + 0.001)
        absorption_trend = stats.linregress(range(len(df_window)), df_window['absorption'])[0]
        
        # 2. Close near high (buying pressure)
        df_window['close_to_high'] = (df_window['close'] - df_window['low']) / (df_window['high'] - df_window['low'] + 0.001)
        buying_pressure = df_window['close_to_high'].mean()
        
        # 3. Volume-weighted price trend
        vwap = (df_window['close'] * df_window['volume']).sum() / df_window['volume'].sum()
        price_vs_vwap = df_window['close'].iloc[-1] / vwap - 1
        
        return {
            'volume_concentration': volume_concentration,
            'absorption_trend': absorption_trend,
            'buying_pressure': buying_pressure,
            'price_vs_vwap': price_vs_vwap
        }
    
    def _predict_breakout_magnitude(self, metrics: Dict[str, float], 
                                  consolidation_days: int,
                                  market_cap: float) -> float:
        """
        Predict expected breakout magnitude based on consolidation metrics
        """
        # Base prediction from phase transition score
        base_magnitude = 0.15 + (metrics['phase_transition_score'] * 0.25)
        
        # Adjust for accumulation patterns
        accumulation_bonus = min(metrics['accumulation_score'] * 0.1, 0.2)
        
        # Adjust for consolidation duration (longer = more explosive)
        duration_factor = min(consolidation_days / 60, 1.5)
        
        # Adjust for market cap (smaller = more volatile)
        market_cap_factor = 1.0
        if market_cap < 50e6:  # Nano cap
            market_cap_factor = 1.5
        elif market_cap < 300e6:  # Micro cap  
            market_cap_factor = 1.3
        elif market_cap < 2e9:  # Small cap
            market_cap_factor = 1.1
        
        # Microstructure adjustments
        micro_factor = 1.0
        if metrics.get('buying_pressure', 0) > 0.7:
            micro_factor *= 1.1
        if metrics.get('absorption_trend', 0) > 0:
            micro_factor *= 1.1
        
        # Calculate final magnitude
        expected_magnitude = base_magnitude * duration_factor * market_cap_factor * micro_factor
        expected_magnitude += accumulation_bonus
        
        # Cap at realistic values
        return min(expected_magnitude, 1.0)  # Max 100% move
    
    def analyze_consolidation(self, df: pd.DataFrame, market_cap: float) -> List[ConsolidationMetrics]:
        """
        Main method to analyze consolidation phases and predict breakouts
        """
        if len(df) < self.min_consolidation_days:
            return []
        
        # Detect consolidation periods
        prices = df['close'].values
        volumes = df['volume'].values
        
        consolidations = self._detect_consolidation_boundaries(prices, volumes)
        
        results = []
        
        for start_idx, end_idx in consolidations:
            df_consolidation = df.iloc[start_idx:end_idx+1].copy()
            
            # Build correlation network
            G = self._build_correlation_network(df_consolidation)
            
            # Calculate phase transition indicators
            phase_metrics = self._calculate_phase_transition_indicators(G, df_consolidation)
            
            # Analyze microstructure
            micro_metrics = self._analyze_microstructure(df_consolidation)
            
            # Combine all metrics
            all_metrics = {**phase_metrics, **micro_metrics}
            
            # Calculate breakout probability
            breakout_probability = self._calculate_breakout_probability(all_metrics)
            
            # Predict breakout magnitude
            expected_move = self._predict_breakout_magnitude(
                all_metrics, 
                end_idx - start_idx,
                market_cap
            )
            
            # Determine volume pattern
            volume_pattern = self._classify_volume_pattern(df_consolidation)
            
            # Create consolidation metrics
            metrics = ConsolidationMetrics(
                duration_days=end_idx - start_idx,
                price_range=(df_consolidation['high'].max() - df_consolidation['low'].min()) / df_consolidation['close'].mean(),
                volume_pattern=volume_pattern,
                network_density=phase_metrics['density'],
                clustering_coefficient=phase_metrics['clustering'],
                phase_transition_score=phase_metrics['phase_transition_score'],
                mispricing_indicator=phase_metrics['divergence'],
                accumulation_score=phase_metrics['accumulation_score'],
                breakout_probability=breakout_probability,
                expected_move=expected_move
            )
            
            results.append(metrics)
            
            logger.info(f"Consolidation detected: {metrics.duration_days} days, "
                       f"breakout probability: {metrics.breakout_probability:.2%}, "
                       f"expected move: {metrics.expected_move:.2%}")
        
        return results
    
    def _calculate_breakout_probability(self, metrics: Dict[str, float]) -> float:
        """
        Calculate probability of explosive breakout based on all metrics
        """
        # Weight different factors
        weights = {
            'phase_transition_score': 0.3,
            'accumulation_score': 0.25,
            'buying_pressure': 0.2,
            'absorption_trend': 0.15,
            'volume_concentration': 0.1
        }
        
        # Calculate weighted score
        score = 0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0)
            
            # Normalize values
            if metric == 'accumulation_score':
                normalized = min(value / 10, 1.0)
            elif metric == 'absorption_trend':
                normalized = 1 / (1 + np.exp(-value))  # Sigmoid
            elif metric == 'volume_concentration':
                normalized = min(value / 2, 1.0)
            else:
                normalized = value
            
            score += normalized * weight
        
        # Convert to probability
        probability = 1 / (1 + np.exp(-4 * (score - 0.5)))  # Sigmoid with steeper curve
        
        return probability
    
    def _classify_volume_pattern(self, df: pd.DataFrame) -> str:
        """
        Classify volume pattern during consolidation
        """
        # Calculate volume trend
        volume_trend = stats.linregress(range(len(df)), df['volume'])[0]
        
        # Calculate volume volatility
        volume_volatility = df['volume'].std() / df['volume'].mean()
        
        # Classify pattern
        if volume_trend > 0 and volume_volatility < 0.5:
            return "steady_accumulation"
        elif volume_trend > 0 and volume_volatility > 0.5:
            return "volatile_accumulation"
        elif abs(volume_trend) < 0.01:
            return "flat"
        else:
            return "declining"
    
    def get_current_phase(self, df: pd.DataFrame, lookback_days: int = 60) -> Dict[str, any]:
        """
        Get current market phase for real-time monitoring
        """
        if len(df) < lookback_days:
            return {'status': 'insufficient_data'}
        
        recent_df = df.tail(lookback_days).copy()
        
        # Quick consolidation check
        price_range = (recent_df['close'].max() - recent_df['close'].min()) / recent_df['close'].mean()
        
        if price_range <= self.price_tolerance:
            # We're in consolidation, analyze it
            G = self._build_correlation_network(recent_df)
            phase_metrics = self._calculate_phase_transition_indicators(G, recent_df)
            
            return {
                'status': 'consolidation',
                'duration': len(recent_df),
                'phase_transition_score': phase_metrics['phase_transition_score'],
                'accumulation_score': phase_metrics['accumulation_score'],
                'network_density': phase_metrics['density'],
                'alert': phase_metrics['phase_transition_score'] > 0.7
            }
        else:
            return {
                'status': 'trending',
                'price_range': price_range
            }


# Specialized features for consolidation breakout prediction
def extract_consolidation_features(df: pd.DataFrame, market_cap: float) -> pd.DataFrame:
    """
    Extract specialized features for consolidation breakout prediction
    """
    analyzer = NetworkConsolidationAnalyzer()
    
    # Get consolidation metrics
    consolidations = analyzer.analyze_consolidation(df, market_cap)
    
    # Create feature columns
    feature_df = df.copy()
    
    # Initialize consolidation features
    feature_df['in_consolidation'] = 0
    feature_df['consolidation_days'] = 0
    feature_df['phase_transition_score'] = 0
    feature_df['accumulation_score'] = 0
    feature_df['breakout_probability'] = 0
    feature_df['expected_move'] = 0
    feature_df['network_density'] = 0
    
    # Mark consolidation periods and add features
    for i, (start_idx, end_idx) in enumerate(analyzer._detect_consolidation_boundaries(
        df['close'].values, df['volume'].values)):
        
        if i < len(consolidations):
            metrics = consolidations[i]
            
            # Mark consolidation period
            feature_df.loc[feature_df.index[start_idx:end_idx+1], 'in_consolidation'] = 1
            
            # Add rolling features
            for j in range(start_idx, end_idx + 1):
                days_in = j - start_idx + 1
                feature_df.loc[feature_df.index[j], 'consolidation_days'] = days_in
                
                # Progressive scores (increase as we approach breakout)
                progress = days_in / metrics.duration_days
                
                feature_df.loc[feature_df.index[j], 'phase_transition_score'] = (
                    metrics.phase_transition_score * progress
                )
                feature_df.loc[feature_df.index[j], 'accumulation_score'] = metrics.accumulation_score
                feature_df.loc[feature_df.index[j], 'breakout_probability'] = (
                    metrics.breakout_probability * (0.5 + 0.5 * progress)
                )
                feature_df.loc[feature_df.index[j], 'expected_move'] = metrics.expected_move
                feature_df.loc[feature_df.index[j], 'network_density'] = metrics.network_density
    
    # Add momentum features specific to breakouts
    feature_df['volume_surge'] = feature_df['volume'] / feature_df['volume'].rolling(20).mean()
    feature_df['price_squeeze'] = feature_df['high'] - feature_df['low']
    feature_df['price_squeeze_ratio'] = (
        feature_df['price_squeeze'] / feature_df['close'].rolling(20).std()
    )
    
    # Relative strength vs market
    feature_df['relative_strength'] = (
        feature_df['close'].pct_change(20) / 0.02  # Assume 2% market return
    )
    
    return feature_df