"""
Feature engineering: combine iv_data + prices into a model-ready feature table.
Output: data/processed/features.csv

Features built:
  - iv_crush_pct     : target variable — (iv_before - iv_after) / iv_before
  - realized_vol_20d : 20-day trailing realized vol before earnings
  - vol_gap          : realized_vol_20d - iv_before (how much IV exceeds recent realized vol)
  - prior_move_pct   : % price change the day after the PREVIOUS earnings event
  - sector           : sector string from yfinance
  - market_cap       : market cap bucket (large/mid/small)
"""
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

IV_PATH     = Path(__file__).parent.parent / "data" / "raw" / "iv_data.csv"
PRICES_PATH = Path(__file__).parent.parent / "data" / "raw" / "prices.csv"
OUT_PATH    = Path(__file__).parent.parent / "data" / "processed" / "features.csv"

ANNUALIZE = np.sqrt(252)


def realized_vol(prices: pd.Series, window: int = 20) -> float:
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < 2:
        return np.nan
    return float(log_returns.iloc[-window:].std() * ANNUALIZE)


def get_ticker_meta(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "Unknown")
        cap = info.get("marketCap", 0)
        if cap >= 10_000_000_000:
            cap_bucket = "large"
        elif cap >= 2_000_000_000:
            cap_bucket = "mid"
        else:
            cap_bucket = "small"
        return {"sector": sector, "market_cap": cap_bucket}
    except Exception:
        return {"sector": "Unknown", "market_cap": "Unknown"}


def build_features() -> pd.DataFrame:
    iv_df     = pd.read_csv(IV_PATH, parse_dates=["earnings_date"])
    prices_df = pd.read_csv(PRICES_PATH, parse_dates=["date"])

    # cache ticker meta (one call per ticker)
    tickers = iv_df["ticker"].unique()
    meta = {t: get_ticker_meta(t) for t in tickers}
    print(f"Fetched metadata for: {list(tickers)}")

    rows = []
    for _, row in iv_df.iterrows():
        ticker  = row["ticker"]
        edate   = row["earnings_date"]
        iv_pre  = row["iv_before"]
        iv_post = row["iv_after"]

        tk_prices = (
            prices_df[prices_df["ticker"] == ticker]
            .sort_values("date")
            .set_index("date")["close"]
        )

        idx = tk_prices.index.searchsorted(pd.Timestamp(edate))

        # 20-day trailing realized vol before earnings
        if idx < 20:
            print(f"  Skipping {ticker} {edate}: not enough history for rv20")
            continue
        rv20 = realized_vol(tk_prices.iloc[idx - 20: idx])

        # prior earnings move: % change day-after vs day-of for PREVIOUS event
        same_ticker_rows = iv_df[iv_df["ticker"] == ticker].sort_values("earnings_date")
        prev_rows = same_ticker_rows[same_ticker_rows["earnings_date"] < edate]
        if prev_rows.empty:
            prior_move = np.nan
        else:
            prev_date = prev_rows.iloc[-1]["earnings_date"]
            prev_idx  = tk_prices.index.searchsorted(pd.Timestamp(prev_date))
            if prev_idx + 1 < len(tk_prices):
                p0 = tk_prices.iloc[prev_idx]
                p1 = tk_prices.iloc[prev_idx + 1]
                prior_move = (p1 - p0) / p0
            else:
                prior_move = np.nan

        # target variable
        if iv_pre == 0:
            continue
        iv_crush_pct = (iv_pre - iv_post) / iv_pre

        rows.append({
            "ticker":           ticker,
            "earnings_date":    edate,
            "iv_crush_pct":     round(iv_crush_pct, 4),   # target
            "realized_vol_20d": round(rv20, 4),
            "iv_before":        round(iv_pre, 4),
            "vol_gap":          round(rv20 - iv_pre, 4),  # +ve = IV elevated vs realized
            "prior_move_pct":   round(prior_move, 4) if not np.isnan(prior_move) else np.nan,
            "sector":           meta[ticker]["sector"],
            "market_cap":       meta[ticker]["market_cap"],
        })

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_PATH}")
    return df


if __name__ == "__main__":
    df = build_features()
    print(df.to_string())
