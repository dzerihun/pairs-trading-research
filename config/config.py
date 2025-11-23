"""
Configuration file for pairs trading research project.
All parameters are centralized here for easy modification and reproducibility.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data collection and preprocessing parameters."""
    # Date range
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    
    # Universe selection
    sector: str = "Financials"
    min_correlation: float = 0.70
    min_trading_days: int = 1000  # Minimum days of data required
    
    # Data quality filters
    min_price: float = 5.0  # Minimum stock price to avoid penny stocks
    max_missing_days_pct: float = 0.05  # Max 5% missing data
    
    # Data files
    stock_prices_file: str = "stock_prices.csv"
    metadata_file: str = "metadata.json"
    
    # Data source
    use_cache: bool = True  # Use cached data if available
    cache_days: int = 1  # Refresh cache after N days


@dataclass
class CointegrationConfig:
    """Cointegration testing parameters."""
    # ADF test parameters
    adf_maxlag: Optional[int] = None  # Auto-select based on AIC
    adf_regression: str = "c"  # 'c' for constant, 'ct' for constant+trend
    
    # Significance threshold
    p_value_threshold: float = 0.05
    
    # Pair selection
    min_adf_statistic: float = -3.0  # More negative = stronger
    min_correlation: float = 0.70
    top_n_pairs: int = 10
    
    # Lookback window for training (in-sample)
    train_start: str = "2018-01-01"
    train_end: str = "2022-12-31"
    
    # Test window (out-of-sample)
    test_start: str = "2023-01-01"
    test_end: str = "2024-12-31"
    
    # Spread calculation
    use_log_prices: bool = True
    hedge_ratio_method: str = "ols"  # 'ols' or 'kalman'
    
    # Z-score calculation
    z_score_lookback: int = 60  # Days for rolling mean/std


@dataclass
class TradingConfig:
    """Trading strategy parameters."""
    # Entry/exit thresholds (in standard deviations)
    entry_threshold: float = 2.0  # Enter when |z-score| > 2
    exit_threshold: float = 0.5   # Exit when |z-score| < 0.5
    
    # Position sizing
    initial_capital: float = 100000.0  # $100k
    position_size_pct: float = 0.10    # 10% of capital per pair
    max_concurrent_positions: int = 5
    
    # Risk management
    stop_loss_pct: float = 0.10  # 10% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    max_position_size: float = 20000.0  # Max $20k per position
    min_position_size: float = 5000.0   # Min $5k per position
    
    # Transaction costs
    commission_per_trade: float = 1.0  # $1 per trade
    slippage_bps: float = 5.0  # 5 basis points (0.05%)
    
    # Position management
    rebalance_frequency: str = "D"  # Daily
    allow_partial_exits: bool = True


@dataclass
class BacktestConfig:
    """Backtesting engine parameters."""
    # Rebalancing
    rebalance_frequency: str = "D"  # Daily
    
    # Performance metrics
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    trading_days_per_year: int = 252
    
    # Output files
    is_trades_file: str = "is_trades.csv"
    oos_trades_file: str = "oos_trades.csv"
    is_metrics_file: str = "is_metrics.csv"
    oos_metrics_file: str = "oos_metrics.csv"
    daily_returns_file: str = "daily_returns.csv"
    
    # Logging
    verbose: bool = True
    log_trades: bool = True


@dataclass
class AnalysisConfig:
    """Analysis and visualization parameters."""
    # Figure settings
    figure_dpi: int = 300
    figure_format: str = "png"
    figure_size: tuple = (12, 8)
    
    # Style
    style: str = "seaborn-v0_8"
    color_palette: str = "husl"
    font_size: int = 12
    
    # Output files
    equity_curve_file: str = "equity_curve.png"
    drawdown_file: str = "drawdown.png"
    returns_dist_file: str = "returns_distribution.png"
    trade_analysis_file: str = "trade_analysis.png"
    monthly_returns_file: str = "monthly_returns.png"
    pair_performance_file: str = "pair_performance.png"
    
    # Table settings
    table_format: str = "csv"
    decimal_places: int = 4


# Global configuration instances
DATA_CONFIG = DataConfig()
COINTEGRATION_CONFIG = CointegrationConfig()
TRADING_CONFIG = TradingConfig()
BACKTEST_CONFIG = BacktestConfig()
ANALYSIS_CONFIG = AnalysisConfig()

