"""
Fetch last 4 reported earnings dates for each ticker.
Output: data/raw/earnings_dates.csv  [ticker, earnings_date]

Tickers: longevity biotech backed by Bezos (BEAM), Nvidia (RXRX),
Gates/Bayer (CRSP), BlackRock/Vanguard (ILMN), Blackstone (ALNY)
"""
import pandas as pd
import yfinance as yf
from pathlib import Path

TICKERS = ["BEAM", "RXRX", "CRSP", "ILMN", "ALNY"]
OUT_PATH = Path(__file__).parent / "raw" / "earnings_dates.csv"


def fetch_earnings(tickers: list[str], quarters: int = 12) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        print(f"Fetching earnings dates: {ticker}")
        try:
            dates = yf.Ticker(ticker).get_earnings_dates(limit=quarters * 2)
            if dates is None or dates.empty:
                print(f"  No data for {ticker}")
                continue
            # keep only past reported dates (not future estimates)
            past = dates[dates.index <= pd.Timestamp.today(tz="UTC")]
            past = past.head(quarters)
            for dt in past.index:
                rows.append({"ticker": ticker, "earnings_date": dt.date()})
        except Exception as e:
            print(f"  Error {ticker}: {e}")

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_PATH}")
    return df


if __name__ == "__main__":
    df = fetch_earnings(TICKERS)
    print(df)
