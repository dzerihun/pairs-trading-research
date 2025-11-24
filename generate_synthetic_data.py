"""
Generate synthetic stock price data for testing the pairs trading pipeline.

This script creates realistic synthetic data when external APIs are unavailable.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

def generate_correlated_prices(
    ticker1: str,
    ticker2: str,
    start_date: str,
    end_date: str,
    correlation: float = 0.85,
    initial_price1: float = 100.0,
    initial_price2: float = 95.0,
    volatility1: float = 0.02,
    volatility2: float = 0.02,
    drift1: float = 0.0001,
    drift2: float = 0.0001
) -> pd.DataFrame:
    """Generate two correlated price series using geometric Brownian motion."""

    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n_days = len(dates)

    # Generate correlated random returns
    cov_matrix = [[volatility1**2, correlation * volatility1 * volatility2],
                  [correlation * volatility1 * volatility2, volatility2**2]]

    returns = np.random.multivariate_normal([drift1, drift2], cov_matrix, n_days)

    # Generate price paths using geometric Brownian motion
    price1 = initial_price1 * np.exp(np.cumsum(returns[:, 0]))
    price2 = initial_price2 * np.exp(np.cumsum(returns[:, 1]))

    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        ticker1: price1,
        ticker2: price2
    })
    df.set_index('Date', inplace=True)

    return df

def generate_financial_sector_data(
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    n_cointegrated_pairs: int = 5,
    n_random_stocks: int = 20
) -> tuple:
    """
    Generate synthetic financial sector data with some cointegrated pairs.

    Returns:
        prices (pd.DataFrame): Price data for all stocks
        metadata (dict): Metadata about the stocks
    """

    print(f"Generating synthetic data from {start_date} to {end_date}...")
    print(f"Creating {n_cointegrated_pairs} cointegrated pairs + {n_random_stocks} independent stocks")

    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # Initialize prices DataFrame
    prices = pd.DataFrame(index=dates)

    # Bank pairs (highly correlated)
    bank_pairs = [
        ('JPM', 'BAC', 0.88),
        ('WFC', 'USB', 0.85),
        ('PNC', 'TFC', 0.82),
    ]

    # Financial services pairs
    finserv_pairs = [
        ('GS', 'MS', 0.90),
        ('BLK', 'TROW', 0.83),
    ]

    # Generate cointegrated pairs
    all_pairs = bank_pairs + finserv_pairs

    for ticker1, ticker2, corr in all_pairs[:n_cointegrated_pairs]:
        # Generate correlated prices with mean reversion
        pair_df = generate_correlated_prices(
            ticker1, ticker2,
            start_date, end_date,
            correlation=corr,
            initial_price1=np.random.uniform(80, 150),
            initial_price2=np.random.uniform(70, 140),
            volatility1=np.random.uniform(0.015, 0.025),
            volatility2=np.random.uniform(0.015, 0.025),
            drift1=np.random.uniform(-0.0001, 0.0003),
            drift2=np.random.uniform(-0.0001, 0.0003)
        )
        prices = prices.join(pair_df, how='outer')

    # Generate additional independent stocks
    independent_tickers = [
        'SCHW', 'AXP', 'COF', 'STT', 'DFS', 'KEY', 'CFG', 'FITB',
        'HBAN', 'RF', 'CMA', 'AIG', 'PRU', 'MET', 'ALL', 'TRV',
        'PGR', 'CB', 'AFL', 'HIG'
    ]

    for i, ticker in enumerate(independent_tickers[:n_random_stocks]):
        # Generate independent random walk
        returns = np.random.normal(0.0001, 0.02, len(dates))
        price = 100 * np.exp(np.cumsum(returns))
        prices[ticker] = price

    # Create metadata
    metadata = {
        'sector': 'Financials',
        'start_date': start_date,
        'end_date': end_date,
        'n_tickers': len(prices.columns),
        'tickers': prices.columns.tolist(),
        'note': 'Synthetic data generated for demonstration purposes',
        'cointegrated_pairs': [(t1, t2) for t1, t2, _ in all_pairs[:n_cointegrated_pairs]]
    }

    print(f"✓ Generated {len(prices.columns)} stocks with {len(prices)} trading days")

    return prices, metadata

def main():
    """Generate and save synthetic data."""

    # Create data directory
    data_dir = Path('data/processed')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    prices, metadata = generate_financial_sector_data(
        start_date="2018-01-01",
        end_date="2024-12-31",
        n_cointegrated_pairs=5,
        n_random_stocks=20
    )

    # Save to CSV
    prices_file = data_dir / 'stock_prices.csv'
    prices.to_csv(prices_file)
    print(f"✓ Saved prices to {prices_file}")

    # Save metadata
    metadata_file = data_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
    print(f"Trading days: {len(prices)}")
    print(f"Number of stocks: {len(prices.columns)}")
    print(f"\nPrice statistics:")
    print(prices.describe().T[['mean', 'std', 'min', 'max']].round(2))

    print("\n✓ Synthetic data generation complete!")
    print(f"You can now run: python run_pipeline.py")

if __name__ == "__main__":
    main()
