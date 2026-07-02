"""
Fetch daily OHLCV price history for all tickers.
Output: data/raw/prices.csv  [ticker, date, open, high, low, close, volume]

Covers 3 years — enough for realized vol calc + backtest.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path

TICKERS = ["BEAM", "RXRX", "CRSP", "ILMN", "ALNY"]
START_DATE = "2020-01-01"
OUT_PATH = Path(__file__).parent / "raw" / "prices.csv"


def fetch_prices(tickers: list[str], start: str = START_DATE) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        print(f"Fetching prices: {ticker}")
        try:
            df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            if df.empty:
                print(f"  No data for {ticker}")
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df["ticker"] = ticker
            df.index.name = "date"
            df = df.reset_index()
            frames.append(df)
        except Exception as e:
            print(f"  Error {ticker}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(combined)} rows → {OUT_PATH}")
    return combined


if __name__ == "__main__":
    df = fetch_prices(TICKERS)
    print(df.head(10))
