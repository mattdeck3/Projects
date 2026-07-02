"""
Phase 4 — Backtest: ATM Straddle Simulation
Simulates buying a straddle before earnings, selling after.
Uses model's predicted iv_crush_pct to filter trades.
Output: backtest/output/results.txt
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
MODEL_PATH    = Path(__file__).parent.parent / "model" / "output" / "model.pkl"
OUT_DIR       = Path(__file__).parent / "output"

# only take trade if model predicts crush > this threshold
CRUSH_THRESHOLD = 0.05

# straddle cost approximation: iv_before * sqrt(1/252) * stock_price
# since we don't have real option prices, use IV-based approximation
# straddle value ≈ 2 * ATM_call ≈ 0.8 * iv * sqrt(T) * S
# we normalize to % of stock price so dollar amount cancels out
DAYS_HELD = 1  # hold through earnings, sell next day


def simulate():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)

    # encode market_cap
    cap_map = {"small": 0, "mid": 1, "large": 2}
    df["market_cap"] = df["market_cap"].map(cap_map).fillna(1)
    df["prior_move_pct"] = df["prior_move_pct"].fillna(df["prior_move_pct"].median())

    # load model — safe: pickle written locally by model/train.py, never from external source
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model    = bundle["model"]
    features = bundle["features"]

    X = df[features]
    df["predicted_crush"] = model.predict(X)
    df["actual_crush"]    = df["iv_crush_pct"]

    # straddle PnL approximation:
    # buy straddle cost  ≈ iv_before * 0.4 (normalized units)
    # sell straddle value ≈ iv_after  * 0.4
    # PnL % = (iv_after - iv_before) / iv_before * -1
    # (we SHORT vol by selling straddle → profit when IV crushes)
    df["straddle_pnl"] = df["actual_crush"]  # crush = our profit when short vol

    # filter: only trade when model predicts crush above threshold
    trades = df[df["predicted_crush"] >= CRUSH_THRESHOLD].copy()
    no_trades = len(df) - len(trades)

    if trades.empty:
        print("No trades passed threshold. Lower CRUSH_THRESHOLD.")
        return

    wins       = (trades["straddle_pnl"] > 0).sum()
    losses     = (trades["straddle_pnl"] <= 0).sum()
    win_rate   = wins / len(trades)
    avg_return = trades["straddle_pnl"].mean()
    total_pnl  = trades["straddle_pnl"].sum()

    # Sharpe: mean / std of returns (annualized assuming ~4 trades/yr per ticker)
    if trades["straddle_pnl"].std() > 0:
        sharpe = (avg_return / trades["straddle_pnl"].std()) * np.sqrt(len(trades))
    else:
        sharpe = 0.0

    print("=" * 45)
    print("  BACKTEST RESULTS — Longevity Biotech Vol")
    print("=" * 45)
    print(f"  Total events evaluated : {len(df)}")
    print(f"  Trades taken           : {len(trades)}  (threshold: {CRUSH_THRESHOLD})")
    print(f"  Trades skipped         : {no_trades}")
    print(f"  Win rate               : {win_rate:.1%}")
    print(f"  Avg return per trade   : {avg_return:+.3f}")
    print(f"  Total simulated PnL    : {total_pnl:+.3f}")
    print(f"  Sharpe ratio           : {sharpe:.2f}")
    print("=" * 45)
    print("\nTrade detail:")
    print(trades[["ticker", "earnings_date", "predicted_crush",
                  "actual_crush", "straddle_pnl"]].to_string(index=False))

    # save results
    results_path = OUT_DIR / "results.txt"
    with open(results_path, "w") as f:
        f.write("BACKTEST RESULTS — Longevity Biotech Earnings Vol Predictor\n")
        f.write(f"Total events: {len(df)} | Trades taken: {len(trades)}\n")
        f.write(f"Win rate: {win_rate:.1%}\n")
        f.write(f"Avg return per trade: {avg_return:+.3f}\n")
        f.write(f"Total PnL: {total_pnl:+.3f}\n")
        f.write(f"Sharpe ratio: {sharpe:.2f}\n")

    print(f"\nSaved → {results_path}")
    return trades


if __name__ == "__main__":
    simulate()
