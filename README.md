# Longevity Biotech — Earnings Volatility Predictor

A machine learning model that predicts implied volatility (IV) crush around earnings events for longevity and biohacking biotech companies, then backtests a short-vol straddle strategy on the filtered signals.

---

## Backtest Results (2023–2026)

| Metric | Value |
|---|---|
| Win Rate | **90%** |
| Trades Taken | **10 / 60 events** |
| Sharpe Ratio | **4.50** |
| Avg Return / Trade | **+0.361** |
| Total Simulated PnL | **+3.606** |

---

## What is IV Crush?

Before earnings, options become expensive as traders hedge uncertainty — implied volatility (IV) spikes. After results are released, that fear collapses. This is called **IV crush**.

A short-vol straddle profits when IV crushes hard: you buy the straddle before earnings when IV is elevated, then sell after the event when IV has collapsed. The challenge is knowing *which* earnings events will crush vs. expand.

This model answers that question for a focused universe of longevity and AI-driven biotech names.

---

## Why Longevity Biotech?

This sector sits at the intersection of the most well-capitalized private bets in the world:
- **BEAM Therapeutics** — base gene editing, Bezos-backed
- **Recursion Pharmaceuticals (RXRX)** — AI drug discovery, Nvidia-backed ($50M strategic investment)
- **CRISPR Therapeutics (CRSP)** — gene editing, Gates Foundation and Bayer-backed
- **Illumina (ILMN)** — genomics infrastructure, BlackRock and Vanguard anchor positions
- **Alnylam Pharmaceuticals (ALNY)** — RNA therapeutics, Blackstone-backed

Binary FDA events and trial readouts create extreme IV spikes in this sector — ideal conditions for a vol prediction model.

---

## Project Structure

```
earnings-vol-predictor/
├── data/
│   ├── fetch_earnings.py      # pull earnings dates via yfinance
│   ├── fetch_prices.py        # pull OHLCV price history (2020–present)
│   ├── fetch_options.py       # compute realized vol proxy for IV
│   ├── raw/                   # cached CSVs (earnings_dates, prices, iv_data)
│   └── processed/             # features.csv
├── features/
│   └── engineer.py            # feature engineering (IV crush target, realized vol, prior move)
├── model/
│   ├── train.py               # LinearRegression baseline + XGBoost stretch
│   ├── evaluate.py            # feature importance + scatter plot
│   └── output/                # model.pkl + metrics.txt
├── backtest/
│   ├── straddle_sim.py        # simulate short-vol straddle strategy
│   └── output/                # results.txt
├── dashboard/
│   └── app.py                 # Streamlit interactive dashboard
└── requirements.txt
```

---

## Features Used

| Feature | Description |
|---|---|
| `realized_vol_20d` | Annualized realized vol over 20 trading days before earnings |
| `iv_before` | Realized vol proxy for pre-earnings implied vol |
| `prior_move_pct` | % price change day after previous earnings event |
| `market_cap` | Size bucket: small / mid / large |

**Target variable:** `iv_crush_pct = (iv_before - iv_after) / iv_before`

---

## Model

- **Baseline:** Linear Regression — R² 0.25, MAE 0.33
- **Stretch:** XGBoost — underperformed at this dataset size (60 rows)
- **Key finding:** `prior_move_pct` and `realized_vol_20d` are the strongest predictors of crush magnitude

---

## How to Run

```bash
# 1. clone and set up environment
git clone https://github.com/YOUR_USERNAME/earnings-vol-predictor
cd earnings-vol-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. run data pipeline (in order)
python3 data/fetch_prices.py
python3 data/fetch_earnings.py
python3 data/fetch_options.py

# 3. feature engineering
python3 features/engineer.py

# 4. train model
python3 model/train.py

# 5. backtest
python3 backtest/straddle_sim.py

# 6. launch dashboard
streamlit run dashboard/app.py
```

---

## Data Limitation

True historical implied volatility snapshots require paid data sources (e.g. Polygon.io options history). This project approximates IV using **realized volatility windows** computed from free yfinance price data:

- `iv_before` = realized vol over the 20 trading days before earnings
- `iv_after` = realized vol over the 10 trading days after earnings

This is a standard academic proxy and is clearly labeled throughout the codebase. The signal direction and relative rankings between tickers are valid; absolute IV levels would differ with real options chain data.

---

## Tech Stack

Python 3.11 · yfinance · pandas · numpy · scikit-learn · XGBoost · Streamlit · matplotlib

---

*Not financial advice. Resume portfolio project.*
