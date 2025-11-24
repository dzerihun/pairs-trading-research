# Getting Started Guide

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the complete pipeline:**
```bash
python run_pipeline.py
```

This will execute all steps automatically and generate results in the `results/` directory.

## Step-by-Step Execution

### Step 1: Data Collection
```bash
python -m data.data_collection
```

Downloads S&P 500 Financial Sector stock data (2018-2024).

**Expected time**: 5-10 minutes (depends on internet speed)

### Step 2: Cointegration Testing
```bash
python -m src.pair_selection.cointegration_test
```

Identifies cointegrated pairs using ADF test.

**Expected time**: 2-5 minutes

### Step 3: Backtesting
```bash
python -m src.backtesting.backtest_engine
```

Runs in-sample and out-of-sample backtests.

**Expected time**: 5-15 minutes

### Step 4: Analysis
```bash
python -m src.analysis.performance_analysis
```

Generates performance metrics and visualizations.

**Expected time**: 1-2 minutes

## Configuration

All parameters are in `config/config.py`. Key settings:
- Date ranges
- Entry/exit thresholds
- Position sizing
- Transaction costs

## Troubleshooting

### Data Download Issues

If data collection fails:
- Check internet connection
- Verify yfinance is up to date: `pip install --upgrade yfinance`
- Some stocks may be delisted - this is normal, the code handles it

### Memory Issues

If you run out of memory:
- Reduce the date range in `config/config.py`
- Test fewer pairs
- Process data in chunks

### Import Errors

If you get import errors:
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.9+)

### Pandas fillna() Deprecation Warning

If you see warnings about `fillna(method='ffill')`:
- This is a known issue with newer pandas versions
- The code will still work, but you may see deprecation warnings
- To fix: Update pandas or modify the fillna calls in the code

## Next Steps

1. Review the results in `results/`
2. Explore the code in `src/`
3. Modify parameters in `config/config.py`
4. Run exploratory analysis in `notebooks/exploratory_analysis.ipynb`
5. Start writing your research paper using `paper/RESEARCH_PAPER_GUIDE.md`

## Expected Output

After running the pipeline, you should see:
- `results/cointegrated_pairs.csv` - Identified trading pairs
- `results/is_trades.csv` - In-sample trades
- `results/oos_trades.csv` - Out-of-sample trades
- `results/is_metrics.csv` - In-sample performance metrics
- `results/oos_metrics.csv` - Out-of-sample performance metrics
- `results/figures/` - All visualizations (equity curves, drawdowns, etc.)



