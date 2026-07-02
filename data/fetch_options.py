"""
Build IV data around each earnings event.
Output: data/raw/iv_data.csv  [ticker, earnings_date, iv_before, iv_after]

NOTE: yfinance only provides current option chain IV, not historical snapshots.
Approximation used (stated honestly in README):
  - iv_before: realized volatility over the 20 trading days BEFORE earnings
  - iv_after:  realized volatility over the 10 trading days AFTER earnings
Both computed from price history (fetch_prices.py must run first).
This is a standard proxy used in academic literature when historical IV is unavailable.
"""
import numpy as np
import pandas as pd
from pathlib import Path

PRICES_PATH = Path(__file__).parent / "raw" / "prices.csv"
EARNINGS_PATH = Path(__file__).parent / "raw" / "earnings_dates.csv"
OUT_PATH = Path(__file__).parent / "raw" / "iv_data.csv"

# trading days windows
WINDOW_BEFORE = 20
WINDOW_AFTER = 10
ANNUALIZE = np.sqrt(252)


def realized_vol(prices: pd.Series) -> float:
    """Annualized realized vol from a series of daily close prices."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < 2:
        return np.nan
    return float(log_returns.std() * ANNUALIZE)


def fetch_iv_data() -> pd.DataFrame:
    prices = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    earnings = pd.read_csv(EARNINGS_PATH, parse_dates=["earnings_date"])

    rows = []
    for _, row in earnings.iterrows():
        ticker = row["ticker"]
        edate = row["earnings_date"]

        tk_prices = prices[prices["ticker"] == ticker].sort_values("date")
        tk_prices = tk_prices.set_index("date")["close"]

        # get index position of earnings date (or nearest trading day)
        idx = tk_prices.index.searchsorted(pd.Timestamp(edate))

        # before window
        before_start = idx - WINDOW_BEFORE
        before_end = idx
        if before_start < 0 or before_end > len(tk_prices):
            print(f"  Skipping {ticker} {edate}: insufficient price history")
            continue
        iv_before = realized_vol(tk_prices.iloc[before_start:before_end])

        # after window
        after_start = idx
        after_end = idx + WINDOW_AFTER
        if after_end > len(tk_prices):
            print(f"  Skipping {ticker} {edate}: insufficient post-earnings data")
            continue
        iv_after = realized_vol(tk_prices.iloc[after_start:after_end])

        rows.append({
            "ticker": ticker,
            "earnings_date": edate,
            "iv_before": round(iv_before, 4),
            "iv_after": round(iv_after, 4),
        })
        print(f"  {ticker} {edate}: iv_before={iv_before:.3f}  iv_after={iv_after:.3f}")

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_PATH}")
    return df


if __name__ == "__main__":
    df = fetch_iv_data()
    print(df)
