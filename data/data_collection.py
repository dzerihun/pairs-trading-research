"""
Data collection module for downloading and preprocessing stock price data.

This module handles:
- Downloading S&P 500 ticker list
- Filtering by sector (Financials)
- Downloading historical price data
- Data cleaning and validation
- Saving processed data
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import DATA_CONFIG, DATA_DIR, PROCESSED_DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Handles collection and preprocessing of stock price data."""
    
    def __init__(self, config=None):
        """
        Initialize data collector.
        
        Args:
            config: DataConfig instance. If None, uses default config.
        """
        self.config = config or DATA_CONFIG
        self.processed_data_path = PROCESSED_DATA_DIR / self.config.stock_prices_file
        self.metadata_path = PROCESSED_DATA_DIR / self.config.metadata_file
        
    def get_sp500_tickers(self) -> List[str]:
        """
        Fetch S&P 500 ticker list from Wikipedia.
        
        Returns:
            List of ticker symbols
        """
        logger.info("Fetching S&P 500 ticker list from Wikipedia...")
        
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        try:
            # Add User-Agent header to avoid 403 Forbidden error
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'constituents'})
            
            if table is None:
                raise ValueError("Could not find S&P 500 table on Wikipedia")
            
            tickers = []
            rows = table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    ticker = cells[0].text.strip()
                    sector = cells[2].text.strip() if len(cells) > 2 else ""
                    
                    # Filter by sector
                    if self.config.sector.lower() in sector.lower():
                        tickers.append(ticker.replace('.', '-'))  # Yahoo Finance uses hyphens
            
            logger.info(f"Found {len(tickers)} stocks in {self.config.sector} sector")
            return tickers
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            raise
    
    def download_stock_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str,
        retry_count: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Download historical price data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            retry_count: Number of retry attempts
            
        Returns:
            DataFrame with OHLCV data, or None if download fails
        """
        for attempt in range(retry_count):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    return None
                
                # Reset index to make Date a column
                df.reset_index(inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Select only Close price and Volume
                df = df[['Date', 'Close', 'Volume']].copy()
                df.columns = ['Date', 'Price', 'Volume']
                df['Ticker'] = ticker
                
                # Check data quality
                if len(df) < self.config.min_trading_days:
                    logger.warning(
                        f"{ticker}: Only {len(df)} days of data "
                        f"(minimum: {self.config.min_trading_days})"
                    )
                    return None
                
                # Check for missing data
                missing_pct = df['Price'].isna().sum() / len(df)
                if missing_pct > self.config.max_missing_days_pct:
                    logger.warning(
                        f"{ticker}: {missing_pct:.1%} missing data "
                        f"(max: {self.config.max_missing_days_pct:.1%})"
                    )
                    return None
                
                # Check minimum price
                if df['Price'].min() < self.config.min_price:
                    logger.warning(
                        f"{ticker}: Minimum price ${df['Price'].min():.2f} "
                        f"below threshold ${self.config.min_price}"
                    )
                    return None
                
                return df
                
            except Exception as e:
                if attempt < retry_count - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}. Retrying...")
                    time.sleep(1)  # Brief pause before retry
                else:
                    logger.error(f"Failed to download {ticker} after {retry_count} attempts: {e}")
                    return None
        
        return None
    
    def download_all_data(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Download historical data for all tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Tuple of (combined DataFrame, metadata dictionary)
        """
        logger.info(f"Downloading data for {len(tickers)} tickers...")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        all_data = []
        metadata = {
            'download_date': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'sector': self.config.sector,
            'total_tickers': len(tickers),
            'successful_downloads': 0,
            'failed_downloads': 0,
            'failed_tickers': []
        }
        
        for ticker in tqdm(tickers, desc="Downloading stock data"):
            df = self.download_stock_data(ticker, start_date, end_date)
            
            if df is not None:
                all_data.append(df)
                metadata['successful_downloads'] += 1
            else:
                metadata['failed_downloads'] += 1
                metadata['failed_tickers'].append(ticker)
            
            # Rate limiting to avoid overwhelming the API
            time.sleep(0.1)
        
        if not all_data:
            raise ValueError("No data was successfully downloaded!")
        
        # Combine all DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Pivot to wide format (one column per ticker)
        price_df = combined_df.pivot_table(
            index='Date',
            columns='Ticker',
            values='Price'
        )
        
        # Sort by date
        price_df.sort_index(inplace=True)
        
        # Fill missing values (forward fill, then backward fill)
        price_df.ffill(inplace=True)
        price_df.bfill(inplace=True)
        
        logger.info(f"Successfully downloaded {metadata['successful_downloads']} tickers")
        logger.info(f"Failed downloads: {metadata['failed_downloads']}")
        logger.info(f"Final data shape: {price_df.shape}")
        
        return price_df, metadata
    
    def is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if not self.processed_data_path.exists():
            return False
        
        if not self.config.use_cache:
            return False
        
        # Check if cache is older than cache_days
        cache_age = datetime.now() - datetime.fromtimestamp(
            self.processed_data_path.stat().st_mtime
        )
        
        return cache_age.days < self.config.cache_days
    
    def collect_data(self, force_refresh: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Main method to collect all data.
        
        Args:
            force_refresh: If True, ignore cache and re-download
            
        Returns:
            Tuple of (price DataFrame, metadata dictionary)
        """
        # Check cache
        if not force_refresh and self.is_cache_valid():
            logger.info("Loading data from cache...")
            price_df = pd.read_csv(
                self.processed_data_path,
                index_col=0,
                parse_dates=True
            )
            
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            logger.info(f"Loaded {price_df.shape[1]} tickers from cache")
            return price_df, metadata
        
        # Download fresh data
        logger.info("Cache invalid or force_refresh=True. Downloading fresh data...")
        
        # Get tickers
        tickers = self.get_sp500_tickers()
        
        # Download data
        price_df, metadata = self.download_all_data(
            tickers,
            self.config.start_date,
            self.config.end_date
        )
        
        # Save to disk
        logger.info(f"Saving data to {self.processed_data_path}")
        price_df.to_csv(self.processed_data_path)
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Data collection complete!")
        return price_df, metadata


def main():
    """Main function for running data collection as a script."""
    collector = DataCollector()
    price_df, metadata = collector.collect_data()
    
    print(f"\nData Collection Summary:")
    print(f"  Tickers: {price_df.shape[1]}")
    print(f"  Date range: {price_df.index.min()} to {price_df.index.max()}")
    print(f"  Trading days: {len(price_df)}")
    print(f"  Successful downloads: {metadata.get('successful_downloads', 0)}")
    print(f"  Failed downloads: {metadata.get('failed_downloads', 0)}")


if __name__ == "__main__":
    main()

