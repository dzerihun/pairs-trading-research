# Pairs Trading Research - Results Summary

**Generated:** November 24, 2024  
**Strategy:** Cointegration-based Pairs Trading  
**Universe:** S&P 500 Financial Sector Stocks  
**Period:** 2018-2024

---

## Executive Summary

This research implements a cointegration-based pairs trading strategy on S&P 500 Financial Sector stocks. The strategy identifies statistically cointegrated pairs and trades on mean-reverting spread movements.

### Key Findings

- **10 cointegrated pairs** identified from 73 financial sector stocks
- **235 closed trades** in-sample (2018-2022)
- **84 closed trades** out-of-sample (2023-2024)
- **70.2% win rate** out-of-sample
- **Positive Sharpe ratio** in both periods

---

## Performance Metrics

### In-Sample Performance (2018-2022)

| Metric | Value |
|--------|-------|
| **Total Return** | 69.5% |
| **Annualized Return** | 10.7% |
| **Volatility** | 16.4% |
| **Sharpe Ratio** | 0.53 |
| **Sortino Ratio** | 0.15 |
| **Calmar Ratio** | 0.27 |
| **Maximum Drawdown** | -40.1% |
| **Win Rate** | 69.8% |
| **Profit Factor** | 1.56 |
| **Average Win** | $1,174 |
| **Average Loss** | -$1,733 |
| **Total Trades** | 235 |
| **Trading Days** | 1,304 |

### Out-of-Sample Performance (2023-2024)

| Metric | Value |
|--------|-------|
| **Total Return** | 7.1% |
| **Annualized Return** | 3.4% |
| **Volatility** | 13.3% |
| **Sharpe Ratio** | 0.10 |
| **Sortino Ratio** | 0.03 |
| **Calmar Ratio** | 0.25 |
| **Maximum Drawdown** | -13.3% |
| **Win Rate** | 70.2% |
| **Profit Factor** | 1.18 |
| **Average Win** | $786 |
| **Average Loss** | -$1,572 |
| **Total Trades** | 84 |
| **Trading Days** | 521 |

---

## Identified Trading Pairs

The following 10 pairs were identified as cointegrated (sorted by ADF statistic strength):

1. **NTRS - TFC** (Northern Trust - Truist Financial)
   - Correlation: 0.787
   - ADF Statistic: -4.57
   - P-value: 0.0001
   - Half-life: 25.3 days

2. **BAC - PNC** (Bank of America - PNC Financial)
   - Correlation: 0.883
   - ADF Statistic: -4.52
   - P-value: 0.0002
   - Half-life: 21.4 days

3. **SYF - TFC** (Synchrony Financial - Truist Financial)
   - Correlation: 0.770
   - ADF Statistic: -4.52
   - P-value: 0.0002
   - Half-life: 28.1 days

4. **COF - NTRS** (Capital One - Northern Trust)
   - Correlation: 0.749
   - ADF Statistic: -4.40
   - P-value: 0.0003
   - Half-life: 22.8 days

5. **MA - V** (Mastercard - Visa)
   - Correlation: 0.918
   - ADF Statistic: -4.32
   - P-value: 0.0004
   - Half-life: 15.4 days

6. **AIG - L** (AIG - Loews)
   - Correlation: 0.774
   - ADF Statistic: -4.31
   - P-value: 0.0004
   - Half-life: 31.0 days

7. **NTRS - SYF** (Northern Trust - Synchrony Financial)
   - Correlation: 0.708
   - ADF Statistic: -4.23
   - P-value: 0.0006
   - Half-life: 30.9 days

8. **KEY - NTRS** (KeyCorp - Northern Trust)
   - Correlation: 0.791
   - ADF Statistic: -4.21
   - P-value: 0.0006
   - Half-life: 24.6 days

9. **GL - TRV** (Globe Life - Travelers)
   - Correlation: 0.703
   - ADF Statistic: -4.02
   - P-value: 0.0013
   - Half-life: 29.0 days

10. **GL - WRB** (Globe Life - WR Berkley)
    - Correlation: 0.703
    - ADF Statistic: -4.02
    - P-value: 0.0013
    - Half-life: 29.0 days

---

## Strategy Parameters

- **Entry Threshold:** |z-score| > 2.0
- **Exit Threshold:** |z-score| < 0.5
- **Position Size:** 10% of capital per pair
- **Max Concurrent Positions:** 5
- **Stop Loss:** 10%
- **Take Profit:** 15%
- **Transaction Costs:** $1 commission + 5 bps slippage
- **Initial Capital:** $100,000

---

## Key Observations

### Strengths

1. **High Win Rate:** 70% win rate indicates the strategy successfully identifies mean-reverting opportunities
2. **Positive Sharpe Ratio:** Both in-sample and out-of-sample periods show positive risk-adjusted returns
3. **Consistent Performance:** Win rate remained stable between in-sample (69.8%) and out-of-sample (70.2%)
4. **Strong Pair Selection:** All pairs have statistically significant cointegration (p < 0.05)

### Weaknesses

1. **Performance Degradation:** Out-of-sample returns (3.4% annualized) significantly lower than in-sample (10.7%)
2. **Large Drawdowns:** Maximum drawdown of -40.1% in-sample is substantial
3. **Lower Profit Factor OOS:** Profit factor decreased from 1.56 to 1.18 out-of-sample
4. **Volatility:** 13-16% annualized volatility is moderate but not negligible

### Potential Improvements

1. **Dynamic Hedge Ratios:** Implement Kalman filter for adaptive hedge ratios
2. **Regime Detection:** Add market regime filters (bull/bear markets)
3. **Position Sizing:** Implement Kelly criterion or volatility-based sizing
4. **Pair Selection:** Test additional cointegration methods (Johansen test)
5. **Risk Management:** Implement portfolio-level risk limits

---

## Statistical Significance

All identified pairs passed the Augmented Dickey-Fuller (ADF) test with:
- **P-value < 0.05** (statistically significant)
- **ADF statistic < -3.0** (strong mean reversion)
- **Half-life < 31 days** (reasonable mean reversion speed)

---

## Files Generated

### Data Files
- `cointegrated_pairs.csv` - All identified pairs with statistics
- `is_trades.csv` - In-sample trade records (470 trades)
- `oos_trades.csv` - Out-of-sample trade records (168 trades)
- `is_daily_returns.csv` - Daily portfolio returns (in-sample)
- `oos_daily_returns.csv` - Daily portfolio returns (out-of-sample)

### Metrics Files
- `is_metrics.csv` - In-sample performance metrics
- `oos_metrics.csv` - Out-of-sample performance metrics

### Visualizations
- `equity_curve_is.png` - In-sample equity curve
- `equity_curve_oos.png` - Out-of-sample equity curve
- `drawdown_is.png` - In-sample drawdown analysis
- `drawdown_oos.png` - Out-of-sample drawdown analysis
- `returns_distribution_is.png` - In-sample return distribution
- `returns_distribution_oos.png` - Out-of-sample return distribution
- `trade_analysis_is.png` - In-sample trade analysis
- `trade_analysis_oos.png` - Out-of-sample trade analysis

---

## Next Steps

1. **Deep Dive Analysis:** Examine individual pair performance
2. **Sensitivity Analysis:** Test different entry/exit thresholds
3. **Market Regime Analysis:** Analyze performance across different market conditions
4. **Literature Review:** Compare results with academic papers (Gatev et al., 2006)
5. **Paper Writing:** Document methodology, results, and findings

---

## Conclusion

The cointegration-based pairs trading strategy demonstrates:
- **Statistical validity** through rigorous cointegration testing
- **Positive risk-adjusted returns** in both periods
- **High win rate** indicating successful mean reversion identification
- **Performance degradation** out-of-sample suggesting need for improvements

The strategy shows promise but requires further refinement, particularly in risk management and adaptive parameter selection.

---

*This summary was automatically generated from backtest results.*

