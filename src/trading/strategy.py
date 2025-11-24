"""
Trading strategy implementation for pairs trading.

This module implements:
- Signal generation based on z-scores
- Position management
- Entry/exit logic
- Risk management
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import COINTEGRATION_CONFIG, TRADING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairsTradingStrategy:
    """Implements cointegration-based pairs trading strategy."""
    
    def __init__(self, trading_config=None, cointegration_config=None):
        """
        Initialize trading strategy.
        
        Args:
            trading_config: TradingConfig instance
            cointegration_config: CointegrationConfig instance
        """
        self.trading_config = trading_config or TRADING_CONFIG
        self.cointegration_config = cointegration_config or COINTEGRATION_CONFIG
    
    def calculate_spread(
        self,
        price1: pd.Series,
        price2: pd.Series,
        hedge_ratio: float
    ) -> pd.Series:
        """
        Calculate spread between two price series.
        
        Args:
            price1: Price series for first stock
            price2: Price series for second stock
            hedge_ratio: Hedge ratio (beta) from cointegration test
            
        Returns:
            Spread time series
        """
        if self.cointegration_config.use_log_prices:
            spread = np.log(price1) - hedge_ratio * np.log(price2)
        else:
            spread = price1 - hedge_ratio * price2
        
        return spread
    
    def generate_signals(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        current_position: float = 0.0
    ) -> pd.Series:
        """
        Generate trading signals based on z-score thresholds.
        
        Args:
            spread: Spread time series
            zscore: Z-score time series
            current_position: Current position size (-1 to 1)
            
        Returns:
            Signal series: 1 for long spread, -1 for short spread, 0 for no position
        """
        signals = pd.Series(0.0, index=spread.index)
        
        entry_threshold = self.trading_config.entry_threshold
        exit_threshold = self.trading_config.exit_threshold
        
        for i in range(len(zscore)):
            z = zscore.iloc[i]
            
            # Entry signals
            if current_position == 0:
                if z < -entry_threshold:
                    # Spread is low, expect it to rise -> Long spread
                    signals.iloc[i] = 1.0
                elif z > entry_threshold:
                    # Spread is high, expect it to fall -> Short spread
                    signals.iloc[i] = -1.0
            
            # Exit signals
            elif current_position > 0:  # Long position
                if z >= -exit_threshold:
                    signals.iloc[i] = -1.0  # Exit long
            elif current_position < 0:  # Short position
                if z <= exit_threshold:
                    signals.iloc[i] = 1.0  # Exit short
        
        return signals
    
    def calculate_position_size(
        self,
        capital: float,
        price1: float,
        price2: float,
        hedge_ratio: float,
        signal: float
    ) -> Tuple[float, float]:
        """
        Calculate position sizes for both stocks.
        
        Args:
            capital: Available capital for this position
            price1: Current price of stock 1
            price2: Current price of stock 2
            hedge_ratio: Hedge ratio
            signal: Trading signal (1 for long spread, -1 for short spread)
            
        Returns:
            Tuple of (shares1, shares2)
        """
        # Position size based on config
        position_value = min(
            capital * self.trading_config.position_size_pct,
            self.trading_config.max_position_size
        )
        
        position_value = max(
            position_value,
            self.trading_config.min_position_size
        )
        
        # For dollar-neutral position:
        # If long spread: long stock1, short stock2
        # If short spread: short stock1, long stock2
        
        if signal > 0:  # Long spread
            # Long stock1, short stock2
            shares1 = position_value / price1
            shares2 = -hedge_ratio * shares1  # Short position
        elif signal < 0:  # Short spread
            # Short stock1, long stock2
            shares1 = -position_value / price1
            shares2 = -hedge_ratio * shares1  # Long position
        else:
            return 0.0, 0.0
        
        return shares1, shares2
    
    def check_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        position_type: str
    ) -> bool:
        """
        Check if stop loss is triggered.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            position_type: 'long' or 'short'
            
        Returns:
            True if stop loss triggered
        """
        if position_type == 'long':
            loss_pct = (current_price - entry_price) / entry_price
        else:  # short
            loss_pct = (entry_price - current_price) / entry_price
        
        return loss_pct <= -self.trading_config.stop_loss_pct
    
    def check_take_profit(
        self,
        entry_price: float,
        current_price: float,
        position_type: str
    ) -> bool:
        """
        Check if take profit is triggered.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            position_type: 'long' or 'short'
            
        Returns:
            True if take profit triggered
        """
        if position_type == 'long':
            profit_pct = (current_price - entry_price) / entry_price
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price
        
        return profit_pct >= self.trading_config.take_profit_pct
    
    def calculate_transaction_cost(
        self,
        shares1: float,
        shares2: float,
        price1: float,
        price2: float
    ) -> float:
        """
        Calculate total transaction cost.
        
        Args:
            shares1: Number of shares for stock 1
            shares2: Number of shares for stock 2
            price1: Price of stock 1
            price2: Price of stock 2
            
        Returns:
            Total transaction cost
        """
        # Commission
        num_trades = (abs(shares1) > 0) + (abs(shares2) > 0)
        commission = num_trades * self.trading_config.commission_per_trade
        
        # Slippage
        slippage_bps = self.trading_config.slippage_bps / 10000
        value1 = abs(shares1) * price1
        value2 = abs(shares2) * price2
        slippage = (value1 + value2) * slippage_bps
        
        return commission + slippage


class PortfolioManager:
    """Manages portfolio-level positions and constraints."""
    
    def __init__(self, trading_config=None):
        """
        Initialize portfolio manager.
        
        Args:
            trading_config: TradingConfig instance
        """
        self.trading_config = trading_config or TRADING_CONFIG
        self.positions: Dict[str, Dict] = {}  # {pair_id: position_info}
        self.cash = self.trading_config.initial_capital
        self.total_value = self.trading_config.initial_capital
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        active_positions = sum(
            1 for pos in self.positions.values() 
            if pos['shares1'] != 0 or pos['shares2'] != 0
        )
        return active_positions < self.trading_config.max_concurrent_positions
    
    def open_position(
        self,
        pair_id: str,
        date: pd.Timestamp,
        shares1: float,
        shares2: float,
        price1: float,
        price2: float,
        transaction_cost: float
    ) -> bool:
        """
        Open a new position.
        
        Args:
            pair_id: Unique identifier for the pair
            date: Trade date
            shares1: Shares of stock 1
            shares2: Shares of stock 2
            price1: Price of stock 1
            price2: Price of stock 2
            transaction_cost: Transaction cost
            
        Returns:
            True if position opened successfully
        """
        if not self.can_open_position():
            return False
        
        # Calculate required capital (for margin on short positions)
        value1 = abs(shares1) * price1
        value2 = abs(shares2) * price2
        required_capital = max(value1, value2)  # Margin requirement
        
        if self.cash < required_capital + transaction_cost:
            return False
        
        # Record position
        self.positions[pair_id] = {
            'date_opened': date,
            'shares1': shares1,
            'shares2': shares2,
            'entry_price1': price1,
            'entry_price2': price2,
            'transaction_cost': transaction_cost,
            'position_type': 'long' if shares1 > 0 else 'short'
        }
        
        # Update cash
        self.cash -= transaction_cost
        
        return True
    
    def close_position(
        self,
        pair_id: str,
        date: pd.Timestamp,
        price1: float,
        price2: float,
        transaction_cost: float
    ) -> Optional[float]:
        """
        Close an existing position.
        
        Args:
            pair_id: Unique identifier for the pair
            date: Trade date
            price1: Current price of stock 1
            price2: Current price of stock 2
            transaction_cost: Transaction cost
            
        Returns:
            P&L if position was closed, None otherwise
        """
        if pair_id not in self.positions:
            return None
        
        position = self.positions[pair_id]
        
        # Calculate P&L
        pnl1 = (price1 - position['entry_price1']) * position['shares1']
        pnl2 = (price2 - position['entry_price2']) * position['shares2']
        total_pnl = pnl1 + pnl2 - position['transaction_cost'] - transaction_cost
        
        # Update cash
        self.cash += total_pnl
        
        # Remove position
        del self.positions[pair_id]
        
        return total_pnl
    
    def update_portfolio_value(
        self,
        date: pd.Timestamp,
        prices: pd.DataFrame
    ) -> float:
        """
        Update total portfolio value (cash + positions).
        
        Args:
            date: Current date
            prices: DataFrame with current prices
            
        Returns:
            Total portfolio value
        """
        position_value = 0.0
        
        for pair_id, position in self.positions.items():
            ticker1, ticker2 = pair_id.split('_')
            
            if ticker1 in prices.columns and ticker2 in prices.columns:
                price1 = prices.loc[date, ticker1]
                price2 = prices.loc[date, ticker2]
                
                # Current value of position
                value1 = position['shares1'] * price1
                value2 = position['shares2'] * price2
                position_value += value1 + value2
        
        self.total_value = self.cash + position_value
        
        return self.total_value
    
    def get_active_positions(self) -> List[str]:
        """Get list of active position pair IDs."""
        return [
            pair_id for pair_id, pos in self.positions.items()
            if pos['shares1'] != 0 or pos['shares2'] != 0
        ]



