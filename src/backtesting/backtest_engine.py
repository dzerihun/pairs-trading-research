"""
Backtesting engine for pairs trading strategy.

This module implements:
- Walk-forward backtesting
- Portfolio-level simulation
- Trade recording
- Performance metric calculation
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BACKTEST_CONFIG,
    COINTEGRATION_CONFIG,
    DATA_CONFIG,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    TRADING_CONFIG,
)
from src.analysis.performance_analysis import PerformanceAnalyzer
from src.pair_selection.cointegration_test import CointegrationTester
from src.trading.strategy import PairsTradingStrategy, PortfolioManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """Main backtesting engine for pairs trading strategy."""
    
    def __init__(
        self,
        trading_config=None,
        cointegration_config=None,
        backtest_config=None
    ):
        """
        Initialize backtesting engine.
        
        Args:
            trading_config: TradingConfig instance
            cointegration_config: CointegrationConfig instance
            backtest_config: BacktestConfig instance
        """
        self.trading_config = trading_config or TRADING_CONFIG
        self.cointegration_config = cointegration_config or COINTEGRATION_CONFIG
        self.backtest_config = backtest_config or BACKTEST_CONFIG
        
        self.strategy = PairsTradingStrategy(
            self.trading_config,
            self.cointegration_config
        )
        self.cointegration_tester = CointegrationTester(self.cointegration_config)
        self.performance_analyzer = PerformanceAnalyzer()
    
    def prepare_pair_data(
        self,
        prices: pd.DataFrame,
        pair_info: Dict,
        date_range: Tuple[str, str]
    ) -> Tuple[pd.Series, pd.Series, pd.Series, float]:
        """
        Prepare data for a single pair.
        
        Args:
            prices: Full price DataFrame
            pair_info: Dictionary with pair information (ticker1, ticker2, hedge_ratio)
            date_range: Tuple of (start_date, end_date)
            
        Returns:
            Tuple of (price1, price2, zscore, hedge_ratio)
        """
        ticker1 = pair_info['ticker1']
        ticker2 = pair_info['ticker2']
        hedge_ratio = pair_info['hedge_ratio']
        
        # Filter date range
        start_date, end_date = date_range
        price1 = prices.loc[start_date:end_date, ticker1]
        price2 = prices.loc[start_date:end_date, ticker2]
        
        # Calculate spread
        spread = self.strategy.calculate_spread(price1, price2, hedge_ratio)
        
        # Calculate z-score
        zscore = self.cointegration_tester.calculate_zscore(spread)
        
        return price1, price2, zscore, hedge_ratio
    
    def backtest_pair(
        self,
        prices: pd.DataFrame,
        pair_info: Dict,
        date_range: Tuple[str, str],
        portfolio_manager: PortfolioManager
    ) -> pd.DataFrame:
        """
        Backtest a single pair.
        
        Args:
            prices: Full price DataFrame
            pair_info: Dictionary with pair information
            date_range: Tuple of (start_date, end_date)
            portfolio_manager: PortfolioManager instance
            
        Returns:
            DataFrame with trade records
        """
        ticker1 = pair_info['ticker1']
        ticker2 = pair_info['ticker2']
        pair_id = f"{ticker1}_{ticker2}"
        
        # Prepare data
        price1, price2, zscore, hedge_ratio = self.prepare_pair_data(
            prices, pair_info, date_range
        )
        
        # Calculate spread for signals
        spread = self.strategy.calculate_spread(price1, price2, hedge_ratio)
        
        trades = []
        current_position = 0.0
        entry_date = None
        entry_zscore = None
        
        for date in price1.index:
            if pd.isna(zscore.loc[date]):
                continue
            
            current_z = zscore.loc[date]
            p1 = price1.loc[date]
            p2 = price2.loc[date]
            
            # Generate signal
            signal = self.strategy.generate_signals(
                spread.loc[:date],
                zscore.loc[:date],
                current_position
            ).loc[date]
            
            # Check for stop loss / take profit on existing position
            if current_position != 0 and entry_date:
                position_type = 'long' if current_position > 0 else 'short'
                spread_entry = spread.loc[entry_date]
                spread_current = spread.loc[date]
                
                # Check stop loss / take profit on spread
                if self.strategy.check_stop_loss(
                    spread_entry, spread_current, position_type
                ) or self.strategy.check_take_profit(
                    spread_entry, spread_current, position_type
                ):
                    signal = -current_position  # Exit signal
            
            # Execute trades
            if signal != 0 and current_position == 0:
                # Open new position
                shares1, shares2 = self.strategy.calculate_position_size(
                    portfolio_manager.cash,
                    p1, p2, hedge_ratio, signal
                )
                
                if shares1 != 0 or shares2 != 0:
                    transaction_cost = self.strategy.calculate_transaction_cost(
                        shares1, shares2, p1, p2
                    )
                    
                    if portfolio_manager.open_position(
                        pair_id, date, shares1, shares2, p1, p2, transaction_cost
                    ):
                        current_position = signal
                        entry_date = date
                        entry_zscore = current_z
                        
                        trades.append({
                            'date': date,
                            'pair_id': pair_id,
                            'ticker1': ticker1,
                            'ticker2': ticker2,
                            'action': 'OPEN',
                            'signal': signal,
                            'shares1': shares1,
                            'shares2': shares2,
                            'price1': p1,
                            'price2': p2,
                            'zscore': current_z,
                            'spread': spread.loc[date],
                            'transaction_cost': transaction_cost,
                            'portfolio_value': portfolio_manager.total_value
                        })
            
            elif signal != 0 and current_position != 0:
                # Close existing position
                if pair_id in portfolio_manager.positions:
                    transaction_cost = self.strategy.calculate_transaction_cost(
                        portfolio_manager.positions[pair_id]['shares1'],
                        portfolio_manager.positions[pair_id]['shares2'],
                        p1, p2
                    )
                    
                    pnl = portfolio_manager.close_position(
                        pair_id, date, p1, p2, transaction_cost
                    )
                    
                    if pnl is not None:
                        trades.append({
                            'date': date,
                            'pair_id': pair_id,
                            'ticker1': ticker1,
                            'ticker2': ticker2,
                            'action': 'CLOSE',
                            'signal': 0,
                            'shares1': 0,
                            'shares2': 0,
                            'price1': p1,
                            'price2': p2,
                            'zscore': current_z,
                            'spread': spread.loc[date],
                            'pnl': pnl,
                            'transaction_cost': transaction_cost,
                            'portfolio_value': portfolio_manager.total_value,
                            'entry_date': entry_date,
                            'entry_zscore': entry_zscore,
                            'holding_period': (date - entry_date).days
                        })
                        
                        current_position = 0.0
                        entry_date = None
                        entry_zscore = None
            
            # Update portfolio value
            portfolio_manager.update_portfolio_value(date, prices)
        
        # Close any remaining positions at end of period
        if current_position != 0 and pair_id in portfolio_manager.positions:
            last_date = price1.index[-1]
            p1 = price1.loc[last_date]
            p2 = price2.loc[last_date]
            
            transaction_cost = self.strategy.calculate_transaction_cost(
                portfolio_manager.positions[pair_id]['shares1'],
                portfolio_manager.positions[pair_id]['shares2'],
                p1, p2
            )
            
            pnl = portfolio_manager.close_position(
                pair_id, last_date, p1, p2, transaction_cost
            )
            
            if pnl is not None:
                trades.append({
                    'date': last_date,
                    'pair_id': pair_id,
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'action': 'CLOSE',
                    'signal': 0,
                    'pnl': pnl,
                    'transaction_cost': transaction_cost,
                    'portfolio_value': portfolio_manager.total_value,
                    'entry_date': entry_date,
                    'holding_period': (last_date - entry_date).days if entry_date else None
                })
        
        if trades:
            return pd.DataFrame(trades)
        else:
            return pd.DataFrame()
    
    def run_backtest(
        self,
        prices: pd.DataFrame,
        pairs_df: pd.DataFrame,
        date_range: Tuple[str, str],
        period_name: str = "backtest"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Run full backtest on all pairs.
        
        Args:
            prices: Full price DataFrame
            pairs_df: DataFrame with cointegrated pairs
            date_range: Tuple of (start_date, end_date)
            period_name: Name for this backtest period (e.g., "is" or "oos")
            
        Returns:
            Tuple of (trades_df, metrics_df, daily_returns)
        """
        logger.info(f"Running backtest for {period_name} period: {date_range[0]} to {date_range[1]}")
        
        portfolio_manager = PortfolioManager(self.trading_config)
        all_trades = []
        
        # Backtest each pair
        for idx, pair_row in pairs_df.iterrows():
            pair_info = {
                'ticker1': pair_row['ticker1'],
                'ticker2': pair_row['ticker2'],
                'hedge_ratio': pair_row['hedge_ratio']
            }
            
            logger.info(f"Backtesting pair {idx+1}/{len(pairs_df)}: {pair_info['ticker1']} - {pair_info['ticker2']}")
            
            trades_df = self.backtest_pair(
                prices, pair_info, date_range, portfolio_manager
            )
            
            if not trades_df.empty:
                all_trades.append(trades_df)
        
        # Combine all trades
        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
            combined_trades = combined_trades.sort_values('date').reset_index(drop=True)
        else:
            combined_trades = pd.DataFrame()
            logger.warning("No trades generated!")
        
        # Calculate daily returns
        daily_returns = self._calculate_daily_returns(
            combined_trades, portfolio_manager, date_range
        )
        
        # Calculate performance metrics
        metrics_df = self.performance_analyzer.calculate_metrics(
            daily_returns, combined_trades, period_name
        )
        
        return combined_trades, metrics_df, daily_returns
    
    def _calculate_daily_returns(
        self,
        trades_df: pd.DataFrame,
        portfolio_manager: PortfolioManager,
        date_range: Tuple[str, str]
    ) -> pd.Series:
        """
        Calculate daily portfolio returns.
        
        Args:
            trades_df: DataFrame with all trades
            portfolio_manager: PortfolioManager instance
            date_range: Tuple of (start_date, end_date)
            
        Returns:
            Series of daily returns
        """
        start_date, end_date = date_range
        
        # Initialize with initial capital
        initial_capital = self.trading_config.initial_capital
        
        # Get all unique dates
        if not trades_df.empty:
            all_dates = pd.date_range(
                start=start_date,
                end=end_date,
                freq='D'
            )
            all_dates = [d for d in all_dates if d.weekday() < 5]  # Business days only
            
            # Create portfolio value series
            portfolio_values = pd.Series(index=all_dates, dtype=float)
            portfolio_values.iloc[0] = initial_capital
            
            # Update based on trades
            current_value = initial_capital
            for _, trade in trades_df.iterrows():
                date = pd.to_datetime(trade['date'])
                if 'portfolio_value' in trade and pd.notna(trade['portfolio_value']):
                    current_value = trade['portfolio_value']
                
                if date in portfolio_values.index:
                    portfolio_values.loc[date] = current_value
            
            # Forward fill portfolio values
            portfolio_values.ffill(inplace=True)
            portfolio_values.fillna(initial_capital, inplace=True)
            
            # Calculate returns
            daily_returns = portfolio_values.pct_change().dropna()
        else:
            # No trades - return zero returns
            all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            all_dates = [d for d in all_dates if d.weekday() < 5]
            daily_returns = pd.Series(0.0, index=all_dates[1:])
        
        return daily_returns


def main():
    """Main function for running backtest as a script."""
    # Load data
    price_path = PROCESSED_DATA_DIR / DATA_CONFIG.stock_prices_file
    if not price_path.exists():
        logger.error(f"Price data not found at {price_path}")
        return
    
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    
    # Load cointegrated pairs
    pairs_path = RESULTS_DIR / "cointegrated_pairs.csv"
    if not pairs_path.exists():
        logger.error(f"Cointegrated pairs not found at {pairs_path}")
        logger.error("Please run cointegration testing first")
        return
    
    pairs_df = pd.read_csv(pairs_path)
    
    # Initialize engine
    engine = BacktestEngine()
    
    # Run in-sample backtest
    logger.info("Running in-sample backtest...")
    is_trades, is_metrics, is_returns = engine.run_backtest(
        prices,
        pairs_df,
        (COINTEGRATION_CONFIG.train_start, COINTEGRATION_CONFIG.train_end),
        "is"
    )
    
    # Run out-of-sample backtest
    logger.info("Running out-of-sample backtest...")
    oos_trades, oos_metrics, oos_returns = engine.run_backtest(
        prices,
        pairs_df,
        (COINTEGRATION_CONFIG.test_start, COINTEGRATION_CONFIG.test_end),
        "oos"
    )
    
    # Save results
    is_trades.to_csv(RESULTS_DIR / BACKTEST_CONFIG.is_trades_file, index=False)
    oos_trades.to_csv(RESULTS_DIR / BACKTEST_CONFIG.oos_trades_file, index=False)
    is_metrics.to_csv(RESULTS_DIR / BACKTEST_CONFIG.is_metrics_file, index=False)
    oos_metrics.to_csv(RESULTS_DIR / BACKTEST_CONFIG.oos_metrics_file, index=False)
    
    is_returns.to_csv(RESULTS_DIR / "is_daily_returns.csv")
    oos_returns.to_csv(RESULTS_DIR / "oos_daily_returns.csv")
    
    print("\nBacktest Complete!")
    print(f"In-sample trades: {len(is_trades)}")
    print(f"Out-of-sample trades: {len(oos_trades)}")
    print(f"\nIn-sample metrics:")
    print(is_metrics.to_string())
    print(f"\nOut-of-sample metrics:")
    print(oos_metrics.to_string())


if __name__ == "__main__":
    main()

