"""Fix metrics calculation by recalculating returns from trades."""

import sys
sys.path.insert(0, '.')

from src.analysis.performance_analysis import PerformanceAnalyzer
from config import *
import pandas as pd

def calc_returns_from_trades(trades_df, initial_capital, start_date, end_date):
    """Calculate daily returns from closed trades."""
    closed = trades_df[trades_df['action'] == 'CLOSE'].copy()
    if closed.empty:
        # Return zero returns
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        all_dates = [d for d in all_dates if d.weekday() < 5]
        return pd.Series(0.0, index=all_dates[1:])
    
    closed['date'] = pd.to_datetime(closed['date'], utc=True).dt.tz_localize(None)  # Remove timezone
    closed = closed.sort_values('date')
    closed['cumulative_pnl'] = closed['pnl'].cumsum()
    
    # Create date range (timezone-naive)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    all_dates = [d for d in all_dates if d.weekday() < 5]
    all_dates = pd.DatetimeIndex(all_dates)
    
    # Portfolio values
    portfolio_values = pd.Series(index=all_dates, dtype=float)
    portfolio_values.iloc[0] = initial_capital
    
    # Fill in values on trade dates
    for _, trade in closed.iterrows():
        trade_date = pd.to_datetime(trade['date']).normalize()
        # Find closest date
        if len(all_dates) > 0:
            try:
                idx = all_dates.get_loc(trade_date)
                portfolio_values.iloc[idx] = initial_capital + trade['cumulative_pnl']
            except KeyError:
                # Find nearest date
                trade_date_naive = trade_date.tz_localize(None) if trade_date.tz else trade_date
                nearest_idx = (all_dates - trade_date_naive).abs().argmin()
                portfolio_values.iloc[nearest_idx] = initial_capital + trade['cumulative_pnl']
    
    portfolio_values.ffill(inplace=True)
    portfolio_values.fillna(initial_capital, inplace=True)
    
    return portfolio_values.pct_change().dropna()

# Load existing data
print('Loading data...')
is_trades = pd.read_csv(RESULTS_DIR / 'is_trades.csv')
oos_trades = pd.read_csv(RESULTS_DIR / 'oos_trades.csv')

print('Recalculating returns...')
is_returns_new = calc_returns_from_trades(
    is_trades, 
    TRADING_CONFIG.initial_capital,
    COINTEGRATION_CONFIG.train_start,
    COINTEGRATION_CONFIG.train_end
)
oos_returns_new = calc_returns_from_trades(
    oos_trades,
    TRADING_CONFIG.initial_capital,
    COINTEGRATION_CONFIG.test_start,
    COINTEGRATION_CONFIG.test_end
)

print('Recalculating metrics...')
analyzer = PerformanceAnalyzer()
is_metrics = analyzer.calculate_metrics(is_returns_new, is_trades, 'is')
oos_metrics = analyzer.calculate_metrics(oos_returns_new, oos_trades, 'oos')

# Save
is_metrics.to_csv(RESULTS_DIR / 'is_metrics.csv', index=False)
oos_metrics.to_csv(RESULTS_DIR / 'oos_metrics.csv', index=False)
is_returns_new.to_csv(RESULTS_DIR / 'is_daily_returns.csv')
oos_returns_new.to_csv(RESULTS_DIR / 'oos_daily_returns.csv')

print('\n' + '='*80)
print('IN-SAMPLE METRICS')
print('='*80)
print(is_metrics.to_string(index=False))

print('\n' + '='*80)
print('OUT-OF-SAMPLE METRICS')
print('='*80)
print(oos_metrics.to_string(index=False))

