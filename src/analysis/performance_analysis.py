"""
Performance analysis and visualization module.

This module implements:
- Performance metric calculation
- Equity curve generation
- Drawdown analysis
- Return distribution analysis
- Trade analysis
"""

import logging
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    ANALYSIS_CONFIG,
    BACKTEST_CONFIG,
    FIGURES_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    TRADING_CONFIG,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use(ANALYSIS_CONFIG.style)
sns.set_palette(ANALYSIS_CONFIG.color_palette)


class PerformanceAnalyzer:
    """Analyzes and visualizes backtest performance."""
    
    def __init__(self, config=None):
        """
        Initialize performance analyzer.
        
        Args:
            config: AnalysisConfig instance
        """
        self.config = config or ANALYSIS_CONFIG
    
    def calculate_metrics(
        self,
        daily_returns: pd.Series,
        trades_df: pd.DataFrame,
        period_name: str = "backtest"
    ) -> pd.DataFrame:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            daily_returns: Series of daily returns
            trades_df: DataFrame with trade records
            period_name: Name of the period (e.g., "is", "oos")
            
        Returns:
            DataFrame with performance metrics
        """
        if daily_returns.empty:
            return pd.DataFrame()
        
        # Basic return metrics
        total_return = (1 + daily_returns).prod() - 1
        annualized_return = (1 + total_return) ** (BACKTEST_CONFIG.trading_days_per_year / len(daily_returns)) - 1
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(BACKTEST_CONFIG.trading_days_per_year)
        sharpe_ratio = (annualized_return - BACKTEST_CONFIG.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade metrics
        if not trades_df.empty and 'pnl' in trades_df.columns:
            closed_trades = trades_df[trades_df['action'] == 'CLOSE'].copy()
            if not closed_trades.empty and 'pnl' in closed_trades.columns:
                winning_trades = closed_trades[closed_trades['pnl'] > 0]
                losing_trades = closed_trades[closed_trades['pnl'] <= 0]
                
                win_rate = len(winning_trades) / len(closed_trades) if len(closed_trades) > 0 else 0
                avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
                profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
                total_trades = len(closed_trades)
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0
                total_trades = 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            total_trades = 0
        
        # Additional metrics
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(daily_returns, annualized_return)
        
        metrics = {
            'period': period_name,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': total_trades,
            'trading_days': len(daily_returns)
        }
        
        return pd.DataFrame([metrics])
    
    def _calculate_sortino_ratio(
        self,
        daily_returns: pd.Series,
        annualized_return: float
    ) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) == 0:
            return 0
        
        downside_std = downside_returns.std() * np.sqrt(BACKTEST_CONFIG.trading_days_per_year)
        if downside_std == 0:
            return 0
        
        return (annualized_return - BACKTEST_CONFIG.risk_free_rate) / downside_std
    
    def plot_equity_curve(
        self,
        daily_returns: pd.Series,
        period_name: str = "backtest",
        save_path: Optional[str] = None
    ):
        """
        Plot equity curve.
        
        Args:
            daily_returns: Series of daily returns
            period_name: Name of the period
            save_path: Path to save figure
        """
        cumulative_returns = (1 + daily_returns).cumprod()
        portfolio_value = cumulative_returns * TRADING_CONFIG.initial_capital
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        ax.plot(portfolio_value.index, portfolio_value.values, linewidth=2)
        ax.axhline(y=TRADING_CONFIG.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.set_xlabel('Date', fontsize=self.config.font_size)
        ax.set_ylabel('Portfolio Value ($)', fontsize=self.config.font_size)
        ax.set_title(f'Equity Curve - {period_name.upper()}', fontsize=self.config.font_size + 2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            logger.info(f"Saved equity curve to {save_path}")
        else:
            save_path = FIGURES_DIR / self.config.equity_curve_file.replace('.png', f'_{period_name}.png')
            plt.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
            logger.info(f"Saved equity curve to {save_path}")
        
        plt.close()
    
    def plot_drawdown(
        self,
        daily_returns: pd.Series,
        period_name: str = "backtest",
        save_path: Optional[str] = None
    ):
        """
        Plot drawdown analysis.
        
        Args:
            daily_returns: Series of daily returns
            period_name: Name of the period
            save_path: Path to save figure
        """
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown.values, linewidth=1, color='darkred')
        ax.set_xlabel('Date', fontsize=self.config.font_size)
        ax.set_ylabel('Drawdown', fontsize=self.config.font_size)
        ax.set_title(f'Drawdown Analysis - {period_name.upper()}', fontsize=self.config.font_size + 2)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        else:
            save_path = FIGURES_DIR / self.config.drawdown_file.replace('.png', f'_{period_name}.png')
            plt.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        
        logger.info(f"Saved drawdown plot to {save_path}")
        plt.close()
    
    def plot_returns_distribution(
        self,
        daily_returns: pd.Series,
        period_name: str = "backtest",
        save_path: Optional[str] = None
    ):
        """
        Plot return distribution.
        
        Args:
            daily_returns: Series of daily returns
            period_name: Name of the period
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        ax1.hist(daily_returns, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(daily_returns.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {daily_returns.mean():.4f}')
        ax1.set_xlabel('Daily Return', fontsize=self.config.font_size)
        ax1.set_ylabel('Frequency', fontsize=self.config.font_size)
        ax1.set_title(f'Return Distribution - {period_name.upper()}', fontsize=self.config.font_size + 2)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(daily_returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=self.config.font_size + 2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        else:
            save_path = FIGURES_DIR / self.config.returns_dist_file.replace('.png', f'_{period_name}.png')
            plt.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        
        logger.info(f"Saved returns distribution to {save_path}")
        plt.close()
    
    def plot_trade_analysis(
        self,
        trades_df: pd.DataFrame,
        period_name: str = "backtest",
        save_path: Optional[str] = None
    ):
        """
        Plot trade analysis.
        
        Args:
            trades_df: DataFrame with trade records
            period_name: Name of the period
            save_path: Path to save figure
        """
        if trades_df.empty or 'pnl' not in trades_df.columns:
            logger.warning("No trade data available for analysis")
            return
        
        closed_trades = trades_df[trades_df['action'] == 'CLOSE'].copy()
        if closed_trades.empty:
            logger.warning("No closed trades to analyze")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # P&L over time
        closed_trades['cumulative_pnl'] = closed_trades['pnl'].cumsum()
        axes[0, 0].plot(closed_trades['date'], closed_trades['cumulative_pnl'], linewidth=2)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Date', fontsize=self.config.font_size)
        axes[0, 0].set_ylabel('Cumulative P&L ($)', fontsize=self.config.font_size)
        axes[0, 0].set_title('Cumulative P&L Over Time', fontsize=self.config.font_size + 2)
        axes[0, 0].grid(True, alpha=0.3)
        
        # P&L distribution
        axes[0, 1].hist(closed_trades['pnl'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('P&L per Trade ($)', fontsize=self.config.font_size)
        axes[0, 1].set_ylabel('Frequency', fontsize=self.config.font_size)
        axes[0, 1].set_title('P&L Distribution', fontsize=self.config.font_size + 2)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Holding period distribution
        if 'holding_period' in closed_trades.columns:
            axes[1, 0].hist(closed_trades['holding_period'].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Holding Period (Days)', fontsize=self.config.font_size)
            axes[1, 0].set_ylabel('Frequency', fontsize=self.config.font_size)
            axes[1, 0].set_title('Holding Period Distribution', fontsize=self.config.font_size + 2)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Monthly returns
        if 'date' in closed_trades.columns:
            closed_trades['date'] = pd.to_datetime(closed_trades['date'], utc=True).dt.tz_localize(None)
            monthly_pnl = closed_trades.groupby(closed_trades['date'].dt.to_period('M'))['pnl'].sum()
            axes[1, 1].bar(range(len(monthly_pnl)), monthly_pnl.values, alpha=0.7)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Month', fontsize=self.config.font_size)
            axes[1, 1].set_ylabel('Monthly P&L ($)', fontsize=self.config.font_size)
            axes[1, 1].set_title('Monthly P&L', fontsize=self.config.font_size + 2)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xticks(range(0, len(monthly_pnl), max(1, len(monthly_pnl)//10)))
            axes[1, 1].set_xticklabels([str(monthly_pnl.index[i]) for i in range(0, len(monthly_pnl), max(1, len(monthly_pnl)//10))], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        else:
            save_path = FIGURES_DIR / self.config.trade_analysis_file.replace('.png', f'_{period_name}.png')
            plt.savefig(save_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        
        logger.info(f"Saved trade analysis to {save_path}")
        plt.close()
    
    def generate_all_plots(
        self,
        is_returns: pd.Series,
        oos_returns: pd.Series,
        is_trades: pd.DataFrame,
        oos_trades: pd.DataFrame
    ):
        """
        Generate all visualization plots.
        
        Args:
            is_returns: In-sample daily returns
            oos_returns: Out-of-sample daily returns
            is_trades: In-sample trades DataFrame
            oos_trades: Out-of-sample trades DataFrame
        """
        logger.info("Generating all visualizations...")
        
        # In-sample plots
        if not is_returns.empty:
            self.plot_equity_curve(is_returns, "is")
            self.plot_drawdown(is_returns, "is")
            self.plot_returns_distribution(is_returns, "is")
        
        if not is_trades.empty:
            self.plot_trade_analysis(is_trades, "is")
        
        # Out-of-sample plots
        if not oos_returns.empty:
            self.plot_equity_curve(oos_returns, "oos")
            self.plot_drawdown(oos_returns, "oos")
            self.plot_returns_distribution(oos_returns, "oos")
        
        if not oos_trades.empty:
            self.plot_trade_analysis(oos_trades, "oos")
        
        logger.info("All visualizations generated!")



