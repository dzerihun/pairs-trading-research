# Research Paper Writing Guide

## Structure

### 1. Abstract (150-250 words)
- Brief summary of strategy, methodology, and key findings
- Mention in-sample vs out-of-sample results
- Key performance metrics

### 2. Introduction
- Motivation for pairs trading
- Research question/hypothesis
- Contribution of this work
- Overview of paper structure

### 3. Literature Review
Key papers to cite:
- **Gatev et al. (2006)** - "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
  - Found that simple distance-based pairs trading can be profitable
  - Discussed mean reversion in stock prices
  
- **Engle & Granger (1987)** - Cointegration methodology
  - Introduced cointegration concept
  - Statistical framework for testing cointegration
  
- **Vidyamurthy (2004)** - "Pairs Trading: Quantitative Methods and Analysis"
  - Comprehensive guide to pairs trading strategies
  - Discussed various statistical methods

- **Alexander & Dimitriu (2005)** - "Indexing and Statistical Arbitrage"
  - Applied cointegration to pairs trading
  - Performance analysis

### 4. Methodology

#### 4.1 Data Description
- Source: S&P 500 Financial Sector stocks
- Period: 2018-2024
- Split: In-sample (2018-2022) vs Out-of-sample (2023-2024)
- Data cleaning procedures

#### 4.2 Pair Selection
- Correlation filtering (threshold: 0.70)
- Cointegration testing using ADF test
- Selection criteria (p-value < 0.05, ADF statistic < -3.0)
- Hedge ratio estimation (OLS regression)

#### 4.3 Trading Strategy
- Signal generation based on z-scores
- Entry thresholds: |z-score| > 2.0
- Exit thresholds: |z-score| < 0.5
- Position sizing: 10% of capital per pair
- Risk management: Stop loss (10%), take profit (15%)

#### 4.4 Backtesting Framework
- Walk-forward analysis
- Transaction costs: $1 commission + 5 bps slippage
- Portfolio constraints: Max 5 concurrent positions
- Performance metrics calculation

### 5. Results

#### 5.1 Pair Selection Results
- Number of cointegrated pairs identified
- Top pairs with statistics
- Correlation and cointegration strength

#### 5.2 In-Sample Performance
- Total return
- Annualized return
- Sharpe ratio
- Maximum drawdown
- Win rate and profit factor
- Number of trades

#### 5.3 Out-of-Sample Performance
- Same metrics as in-sample
- Comparison with in-sample results
- Discussion of performance degradation (if any)

#### 5.4 Visualizations
- Equity curves
- Drawdown analysis
- Return distributions
- Trade analysis

### 6. Discussion

#### 6.1 Interpretation of Results
- Why the strategy performed as it did
- Market conditions during test period
- Pair-specific performance

#### 6.2 Limitations
- Data limitations
- Assumptions made
- Transaction cost assumptions
- Survivorship bias (if applicable)
- Look-ahead bias prevention

#### 6.3 Potential Improvements
- Alternative cointegration tests (Johansen)
- Dynamic hedge ratios (Kalman filter)
- Regime detection
- Machine learning for pair selection
- Different sectors or asset classes

### 7. Conclusion
- Summary of findings
- Key takeaways
- Future research directions
- Practical implications

### 8. References
- Proper academic citations
- Use consistent citation style (APA, MLA, Chicago, etc.)
- Include all papers mentioned in literature review

## Writing Tips

1. **Be Honest**: Discuss limitations and failures openly
2. **Be Specific**: Use exact numbers and statistics
3. **Be Clear**: Explain methodology in detail
4. **Be Critical**: Analyze why results occurred
5. **Be Professional**: Use academic writing style
6. **Be Concise**: Avoid unnecessary verbosity

## Key Metrics to Report

### Return Metrics
- Total return (in-sample and out-of-sample)
- Annualized return
- Cumulative returns over time

### Risk Metrics
- Volatility (annualized)
- Maximum drawdown
- Downside deviation

### Risk-Adjusted Metrics
- Sharpe ratio
- Sortino ratio
- Calmar ratio

### Trade Metrics
- Win rate
- Profit factor
- Average win/loss
- Total number of trades
- Average holding period

### Statistical Tests
- ADF test statistics
- P-values
- Half-life of mean reversion

## Common Sections to Include

### Tables
1. Summary statistics of selected pairs
2. Performance metrics comparison (IS vs OOS)
3. Pair-level performance breakdown
4. Sensitivity analysis results

### Figures
1. Equity curves (IS and OOS)
2. Drawdown charts
3. Return distributions
4. Trade analysis (P&L over time, holding periods)
5. Pair spread visualizations (for top pairs)

## Example Abstract Template

"This paper implements and backtests a cointegration-based pairs trading strategy on S&P 500 Financial Sector stocks from 2018 to 2024. Using the Augmented Dickey-Fuller (ADF) test, we identify cointegrated pairs and construct a dollar-neutral trading strategy based on z-score thresholds. The strategy achieves [X]% annualized return with a Sharpe ratio of [Y] in-sample, and [Z]% annualized return with a Sharpe ratio of [W] out-of-sample. We discuss the performance characteristics, limitations, and potential improvements to the strategy."

## Length Guidelines

- **Abstract**: 150-250 words
- **Introduction**: 2-3 pages
- **Literature Review**: 2-3 pages
- **Methodology**: 3-4 pages
- **Results**: 3-4 pages
- **Discussion**: 2-3 pages
- **Conclusion**: 1 page
- **Total**: 12-20 pages (excluding references and appendices)

## Final Checklist

- [ ] All figures are high quality (300 DPI)
- [ ] All tables are properly formatted
- [ ] All citations are complete and consistent
- [ ] Grammar and spelling checked
- [ ] Methodology is reproducible
- [ ] Results are honest and limitations discussed
- [ ] Code is available and documented
- [ ] Results match code output



