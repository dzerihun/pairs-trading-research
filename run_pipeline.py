"""
Master pipeline script for pairs trading research project.

This script orchestrates the complete research pipeline:
1. Data collection
2. Pair selection (cointegration testing)
3. Backtesting (in-sample and out-of-sample)
4. Performance analysis and visualization
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from config import (
    BACKTEST_CONFIG,
    COINTEGRATION_CONFIG,
    DATA_CONFIG,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from data.data_collection import DataCollector
from src.analysis.performance_analysis import PerformanceAnalyzer
from src.backtesting.backtest_engine import BacktestEngine
from src.pair_selection.cointegration_test import CointegrationTester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete research pipeline."""
    logger.info("=" * 80)
    logger.info("PAIRS TRADING RESEARCH PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Data Collection
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("=" * 80)
    
    collector = DataCollector()
    prices, metadata = collector.collect_data()
    
    logger.info(f"Data collection complete:")
    logger.info(f"  - Tickers: {prices.shape[1]}")
    logger.info(f"  - Date range: {prices.index.min()} to {prices.index.max()}")
    logger.info(f"  - Trading days: {len(prices)}")
    
    # Step 2: Pair Selection
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: PAIR SELECTION (COINTEGRATION TESTING)")
    logger.info("=" * 80)
    
    tester = CointegrationTester()
    pairs_df = tester.find_cointegrated_pairs(
        prices,
        date_range=(COINTEGRATION_CONFIG.train_start, COINTEGRATION_CONFIG.train_end)
    )
    
    if pairs_df.empty:
        logger.error("No cointegrated pairs found! Exiting.")
        return
    
    # Save pairs
    pairs_path = RESULTS_DIR / "cointegrated_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)
    logger.info(f"Found {len(pairs_df)} cointegrated pairs")
    logger.info(f"Results saved to: {pairs_path}")
    
    # Step 3: Backtesting
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: BACKTESTING")
    logger.info("=" * 80)
    
    engine = BacktestEngine()
    
    # In-sample backtest
    logger.info("\nRunning in-sample backtest...")
    is_trades, is_metrics, is_returns = engine.run_backtest(
        prices,
        pairs_df,
        (COINTEGRATION_CONFIG.train_start, COINTEGRATION_CONFIG.train_end),
        "is"
    )
    
    # Out-of-sample backtest
    logger.info("\nRunning out-of-sample backtest...")
    oos_trades, oos_metrics, oos_returns = engine.run_backtest(
        prices,
        pairs_df,
        (COINTEGRATION_CONFIG.test_start, COINTEGRATION_CONFIG.test_end),
        "oos"
    )
    
    # Save backtest results
    logger.info("\nSaving backtest results...")
    is_trades.to_csv(RESULTS_DIR / BACKTEST_CONFIG.is_trades_file, index=False)
    oos_trades.to_csv(RESULTS_DIR / BACKTEST_CONFIG.oos_trades_file, index=False)
    is_metrics.to_csv(RESULTS_DIR / BACKTEST_CONFIG.is_metrics_file, index=False)
    oos_metrics.to_csv(RESULTS_DIR / BACKTEST_CONFIG.oos_metrics_file, index=False)
    
    is_returns.to_csv(RESULTS_DIR / "is_daily_returns.csv")
    oos_returns.to_csv(RESULTS_DIR / "oos_daily_returns.csv")
    
    logger.info(f"In-sample trades: {len(is_trades)}")
    logger.info(f"Out-of-sample trades: {len(oos_trades)}")
    
    # Step 4: Performance Analysis
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: PERFORMANCE ANALYSIS & VISUALIZATION")
    logger.info("=" * 80)
    
    analyzer = PerformanceAnalyzer()
    
    # Generate all plots
    analyzer.generate_all_plots(is_returns, oos_returns, is_trades, oos_trades)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("=" * 80)
    
    print("\n" + "=" * 80)
    print("IN-SAMPLE PERFORMANCE METRICS")
    print("=" * 80)
    if not is_metrics.empty:
        print(is_metrics.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE PERFORMANCE METRICS")
    print("=" * 80)
    if not oos_metrics.empty:
        print(oos_metrics.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("RESULTS LOCATION")
    print("=" * 80)
    print(f"  Figures: {RESULTS_DIR / 'figures'}")
    print(f"  Tables: {RESULTS_DIR / 'tables'}")
    print(f"  Trade data: {RESULTS_DIR}")
    print("\nPipeline execution complete!")


if __name__ == "__main__":
    main()

