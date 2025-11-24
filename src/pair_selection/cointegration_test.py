"""
Cointegration testing module for identifying trading pairs.

This module implements:
- Correlation-based pair filtering
- Augmented Dickey-Fuller (ADF) test for cointegration
- Spread calculation with hedge ratio estimation
- Z-score calculation for trading signals
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from config import COINTEGRATION_CONFIG, DATA_CONFIG, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CointegrationTester:
    """Tests pairs for cointegration and calculates trading signals."""
    
    def __init__(self, config=None):
        """
        Initialize cointegration tester.
        
        Args:
            config: CointegrationConfig instance. If None, uses default config.
        """
        self.config = config or COINTEGRATION_CONFIG
    
    def calculate_correlation_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix of log returns.
        
        Args:
            prices: DataFrame with prices (columns = tickers, index = dates)
            
        Returns:
            Correlation matrix
        """
        logger.info("Calculating correlation matrix...")
        
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # Calculate correlation
        corr_matrix = log_returns.corr()
        
        return corr_matrix
    
    def filter_correlated_pairs(
        self, 
        corr_matrix: pd.DataFrame,
        min_correlation: float = None
    ) -> List[Tuple[str, str, float]]:
        """
        Filter pairs by correlation threshold.
        
        Args:
            corr_matrix: Correlation matrix
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of (ticker1, ticker2, correlation) tuples
        """
        min_corr = min_correlation or self.config.min_correlation
        logger.info(f"Filtering pairs with correlation >= {min_corr}")
        
        pairs = []
        tickers = corr_matrix.columns.tolist()
        
        for i, ticker1 in enumerate(tickers):
            for ticker2 in tickers[i+1:]:
                corr = corr_matrix.loc[ticker1, ticker2]
                if corr >= min_corr:
                    pairs.append((ticker1, ticker2, corr))
        
        logger.info(f"Found {len(pairs)} pairs with correlation >= {min_corr}")
        return pairs
    
    def estimate_hedge_ratio(
        self, 
        price1: pd.Series, 
        price2: pd.Series,
        method: str = "ols"
    ) -> Tuple[float, pd.Series]:
        """
        Estimate hedge ratio (beta) for pair.
        
        Args:
            price1: Price series for first stock
            price2: Price series for second stock
            method: 'ols' for ordinary least squares
            
        Returns:
            Tuple of (hedge_ratio, residuals)
        """
        if self.config.use_log_prices:
            y = np.log(price1)
            x = np.log(price2)
        else:
            y = price1
            x = price2
        
        # Remove NaN values
        valid_idx = ~(y.isna() | x.isna())
        y = y[valid_idx]
        x = x[valid_idx]
        
        if len(y) < 100:  # Need sufficient data
            return None, None
        
        # OLS regression: y = alpha + beta * x + epsilon
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate residuals (spread)
        residuals = y - (intercept + slope * x)
        
        return slope, residuals
    
    def test_cointegration(
        self, 
        price1: pd.Series, 
        price2: pd.Series
    ) -> Dict:
        """
        Test if two price series are cointegrated using ADF test.
        
        Args:
            price1: Price series for first stock
            price2: Price series for second stock
            
        Returns:
            Dictionary with test results
        """
        # Estimate hedge ratio and calculate spread
        hedge_ratio, residuals = self.estimate_hedge_ratio(price1, price2)
        
        if residuals is None:
            return {
                'is_cointegrated': False,
                'adf_statistic': None,
                'p_value': None,
                'hedge_ratio': None,
                'reason': 'Insufficient data'
            }
        
        # Perform ADF test on residuals
        try:
            adf_result = adfuller(
                residuals.dropna(),
                maxlag=self.config.adf_maxlag,
                regression=self.config.adf_regression,
                autolag='AIC'
            )
            
            adf_statistic = adf_result[0]
            p_value = adf_result[1]
            critical_values = adf_result[4]
            
            # Determine if cointegrated
            is_cointegrated = (
                p_value < self.config.p_value_threshold and
                adf_statistic < self.config.min_adf_statistic
            )
            
            return {
                'is_cointegrated': is_cointegrated,
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'hedge_ratio': hedge_ratio,
                'critical_values': critical_values,
                'residuals': residuals
            }
            
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            return {
                'is_cointegrated': False,
                'adf_statistic': None,
                'p_value': None,
                'hedge_ratio': None,
                'reason': str(e)
            }
    
    def calculate_zscore(
        self, 
        spread: pd.Series,
        lookback: int = None
    ) -> pd.Series:
        """
        Calculate z-score of spread.
        
        Args:
            spread: Spread time series
            lookback: Rolling window for mean/std calculation
            
        Returns:
            Z-score series
        """
        lookback = lookback or self.config.z_score_lookback
        
        rolling_mean = spread.rolling(window=lookback).mean()
        rolling_std = spread.rolling(window=lookback).std()
        
        zscore = (spread - rolling_mean) / rolling_std
        
        return zscore
    
    def find_cointegrated_pairs(
        self, 
        prices: pd.DataFrame,
        date_range: Optional[Tuple[str, str]] = None
    ) -> pd.DataFrame:
        """
        Find all cointegrated pairs from price data.
        
        Args:
            prices: DataFrame with prices (columns = tickers, index = dates)
            date_range: Optional tuple of (start_date, end_date) to filter data
            
        Returns:
            DataFrame with cointegrated pairs and statistics
        """
        logger.info("Starting cointegration testing...")
        
        # Filter date range if provided
        if date_range:
            start_date, end_date = date_range
            prices = prices.loc[start_date:end_date]
            logger.info(f"Filtered to date range: {start_date} to {end_date}")
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(prices)
        
        # Filter by correlation
        correlated_pairs = self.filter_correlated_pairs(corr_matrix)
        
        if not correlated_pairs:
            logger.warning("No correlated pairs found!")
            return pd.DataFrame()
        
        # Test each pair for cointegration
        logger.info(f"Testing {len(correlated_pairs)} pairs for cointegration...")
        
        results = []
        
        for ticker1, ticker2, correlation in correlated_pairs:
            price1 = prices[ticker1]
            price2 = prices[ticker2]
            
            # Test cointegration
            test_result = self.test_cointegration(price1, price2)
            
            if test_result['is_cointegrated']:
                # Calculate additional statistics
                hedge_ratio = test_result['hedge_ratio']
                residuals = test_result['residuals']
                
                spread_mean = residuals.mean()
                spread_std = residuals.std()
                half_life = self._calculate_half_life(residuals)
                
                results.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'correlation': correlation,
                    'adf_statistic': test_result['adf_statistic'],
                    'p_value': test_result['p_value'],
                    'hedge_ratio': hedge_ratio,
                    'spread_mean': spread_mean,
                    'spread_std': spread_std,
                    'half_life': half_life,
                })
        
        if not results:
            logger.warning("No cointegrated pairs found!")
            return pd.DataFrame()
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by ADF statistic (more negative = stronger)
        results_df = results_df.sort_values('adf_statistic').reset_index(drop=True)
        
        # Select top N pairs
        top_n = min(self.config.top_n_pairs, len(results_df))
        results_df = results_df.head(top_n)
        
        logger.info(f"Found {len(results_df)} cointegrated pairs")
        
        return results_df
    
    def _calculate_half_life(self, residuals: pd.Series) -> float:
        """
        Calculate half-life of mean reversion.
        
        Half-life = -log(2) / theta, where theta comes from:
        delta_y = alpha + theta * y_{t-1} + epsilon
        
        Args:
            residuals: Spread time series
            
        Returns:
            Half-life in days
        """
        try:
            delta_y = residuals.diff().dropna()
            y_lag = residuals.shift(1).dropna()
            
            # Align indices
            valid_idx = delta_y.index.intersection(y_lag.index)
            delta_y = delta_y.loc[valid_idx]
            y_lag = y_lag.loc[valid_idx]
            
            # Remove NaN
            valid = ~(delta_y.isna() | y_lag.isna())
            delta_y = delta_y[valid]
            y_lag = y_lag[valid]
            
            if len(delta_y) < 10:
                return np.nan
            
            # OLS: delta_y = alpha + theta * y_lag
            slope, _, _, _, _ = stats.linregress(y_lag, delta_y)
            
            if slope >= 0:
                return np.inf  # Not mean-reverting
            
            half_life = -np.log(2) / slope
            
            return max(0, half_life)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"Half-life calculation failed: {e}")
            return np.nan


def main():
    """Main function for running cointegration testing as a script."""
    # Load price data
    price_path = PROCESSED_DATA_DIR / DATA_CONFIG.stock_prices_file
    if not price_path.exists():
        logger.error(f"Price data not found at {price_path}")
        logger.error("Please run data collection first: python -m data.data_collection")
        return
    
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    
    # Initialize tester
    tester = CointegrationTester()
    
    # Find cointegrated pairs (in-sample period)
    pairs_df = tester.find_cointegrated_pairs(
        prices,
        date_range=(COINTEGRATION_CONFIG.train_start, COINTEGRATION_CONFIG.train_end)
    )
    
    # Save results
    output_path = PROCESSED_DATA_DIR.parent.parent / "results" / "cointegrated_pairs.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(output_path, index=False)
    
    print(f"\nCointegration Testing Summary:")
    print(f"  Cointegrated pairs found: {len(pairs_df)}")
    print(f"  Results saved to: {output_path}")
    
    if len(pairs_df) > 0:
        print(f"\nTop 5 pairs:")
        print(pairs_df.head().to_string())


if __name__ == "__main__":
    main()



