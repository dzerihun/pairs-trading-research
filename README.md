# Pairs Trading Research Project

A research-grade quantitative finance project implementing a cointegration-based pairs trading strategy on S&P 500 Financial Sector stocks (2018-2024).

## Project Overview

This project demonstrates:
- **Statistical Rigor**: Proper hypothesis testing (ADF test for cointegration), in-sample vs out-of-sample validation
- **Research Methodology**: Literature review, clear hypothesis testing, honest discussion of limitations
- **Technical Implementation**: Clean, modular code with comprehensive documentation
- **Practical Considerations**: Realistic transaction costs, position limits, risk management
- **Professional Presentation**: High-quality visualizations and comprehensive analysis

## Strategy Description

**Pairs Trading** is a market-neutral statistical arbitrage strategy that:
1. Identifies pairs of stocks with a stable long-term relationship (cointegration)
2. Monitors the spread between their prices
3. Enters positions when the spread deviates significantly from its mean
4. Exits when the spread reverts to its mean

**Methodology**:
- **Cointegration Testing**: Augmented Dickey-Fuller (ADF) test
- **Universe**: S&P 500 Financial Sector stocks
- **Period**: 2018-2024 (in-sample: 2018-2022, out-of-sample: 2023-2024)
- **Signal Generation**: Z-score based entry/exit thresholds

## Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pairs-trading-research
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python run_pipeline.py
```

This will execute:
1. Data collection
2. Pair selection (cointegration testing)
3. Backtesting (in-sample and out-of-sample)
4. Performance analysis and visualization

### Individual Components

#### Data Collection
```bash
python -m data.data_collection
```

#### Pair Selection
```bash
python -m src.pair_selection.cointegration_test
```

#### Backtesting
```bash
python -m src.backtesting.backtest_engine
```

#### Analysis
```bash
python -m src.analysis.performance_analysis
```

## Project Structure

```
pairs-trading-research/
├── config/
│   └── config.py              # All parameters
├── data/
│   ├── data_collection.py     # Downloads stock data
│   └── processed/             # Processed data
├── src/
│   ├── pair_selection/
│   │   └── cointegration_test.py
│   ├── trading/
│   │   └── strategy.py
│   ├── backtesting/
│   │   └── backtest_engine.py
│   └── analysis/
│       └── performance_analysis.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── results/
│   ├── figures/              # Visualizations
│   ├── tables/               # Performance tables
│   └── *.csv                 # Trade data and metrics
├── paper/
│   └── RESEARCH_PAPER_GUIDE.md
├── run_pipeline.py           # Master execution script
└── requirements.txt
```

## Results

After running the pipeline, results will be available in:
- `results/figures/` - All visualizations
- `results/tables/` - Performance metrics tables
- `results/*.csv` - Trade data and pair statistics

## Key Concepts

### Cointegration
Two time series are cointegrated if they maintain a stable long-run relationship, even though each series individually may be non-stationary.

### Z-Score
Measures how many standard deviations the spread is from its mean:
- Z > 2: Spread unusually high → Short the spread
- Z < -2: Spread unusually low → Long the spread
- Z ≈ 0: Spread at mean → Exit position

### ADF Test
Augmented Dickey-Fuller test for stationarity. Tests if a time series has a "unit root" (random walk). P-value < 0.05 indicates statistical significance.

## Performance Metrics

- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

## Research Paper

See `paper/RESEARCH_PAPER_GUIDE.md` for guidance on writing the research paper.

## Configuration

All parameters are centralized in `config/config.py`:
- Data collection settings
- Cointegration test parameters
- Trading strategy parameters
- Backtesting configuration
- Analysis and visualization settings

## License

MIT License - see LICENSE file for details.

## Author

Dagmawi Zerihun

## Acknowledgments

This project is designed as a portfolio piece for graduate school applications in quantitative finance.
