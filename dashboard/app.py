"""
Phase 5 — Streamlit Dashboard
Longevity Biotech Earnings Volatility Predictor
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
FEATURES   = BASE / "data" / "processed" / "features.csv"
MODEL_PATH = BASE / "model" / "output" / "model.pkl"

# ── load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_features():
    df = pd.read_csv(FEATURES)
    cap_map = {"small": 0, "mid": 1, "large": 2}
    df["market_cap"] = df["market_cap"].map(cap_map).fillna(1)
    df["prior_move_pct"] = df["prior_move_pct"].fillna(df["prior_move_pct"].median())
    return df

@st.cache_resource
def load_model():
    # safe: pickle written locally by model/train.py, never from external source
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

df     = load_features()
bundle = load_model()
model  = bundle["model"]
feats  = bundle["features"]

# predict for all rows upfront
df["predicted_crush"] = model.predict(df[feats])

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Longevity Biotech Vol Predictor", layout="wide")

# ── header ─────────────────────────────────────────────────────────────────
st.title("Longevity Biotech — Earnings Vol Predictor")
st.markdown("""
**What is IV crush?**
Before earnings, options get expensive as traders hedge uncertainty — implied volatility (IV) spikes.
After results drop, that fear collapses. This is called **IV crush**.
This model predicts *how much* IV will crush, then filters for high-confidence setups
to simulate a short-vol straddle strategy across Bezos, Nvidia, and Gates-backed longevity biotech names.
""")

st.divider()

# ── backtest summary ────────────────────────────────────────────────────────
st.subheader("Backtest Results (2023–2026)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Win Rate",           "90%")
c2.metric("Trades Taken",       "10 / 60 events")
c3.metric("Sharpe Ratio",       "4.50")
c4.metric("Avg Return / Trade", "+0.361")

st.divider()

# ── predicted vs actual by ticker ──────────────────────────────────────────
st.subheader("Predicted vs Actual IV Crush")
ticker = st.selectbox("Select ticker", sorted(df["ticker"].unique()))

tk = df[df["ticker"] == ticker].copy().reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 4))
x = range(len(tk))
ax.bar([i - 0.2 for i in x], tk["iv_crush_pct"],     width=0.38, label="Actual",    color="#2196F3", alpha=0.85)
ax.bar([i + 0.2 for i in x], tk["predicted_crush"],   width=0.38, label="Predicted", color="#FF9800", alpha=0.85)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xticks(list(x))
ax.set_xticklabels([str(d)[:10] for d in tk["earnings_date"]], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("IV Crush %")
ax.set_title(f"{ticker} — Predicted vs Actual IV Crush per Earnings Event")
ax.legend()
fig.tight_layout()
st.pyplot(fig)

st.divider()

# ── trades taken table ──────────────────────────────────────────────────────
st.subheader("Trades Taken (predicted crush > 0.05)")
trades = df[df["predicted_crush"] >= 0.05][
    ["ticker", "earnings_date", "predicted_crush", "iv_crush_pct"]
].copy().reset_index(drop=True)
trades.columns = ["Ticker", "Earnings Date", "Predicted Crush", "Actual Crush"]
trades["Result"] = trades["Actual Crush"].apply(lambda x: "✅ Win" if x > 0 else "❌ Loss")
st.dataframe(trades, use_container_width=True)

st.divider()

# ── iv crush distribution all events ───────────────────────────────────────
st.subheader("IV Crush Distribution — All 60 Events")
fig2, ax2 = plt.subplots(figsize=(12, 3))
colors = ["#4CAF50" if v > 0 else "#F44336" for v in df["iv_crush_pct"]]
ax2.bar(range(len(df)), df["iv_crush_pct"], color=colors, alpha=0.85)
ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax2.set_ylabel("IV Crush %")
ax2.set_title("Green = Vol Crushed (profitable short-vol) | Red = Vol Expanded (loss)")
ax2.set_xticks([])
fig2.tight_layout()
st.pyplot(fig2)

st.divider()

# ── data disclaimer ─────────────────────────────────────────────────────────
st.caption(
    "⚠️ Data note: True historical IV snapshots require paid data (e.g. Polygon.io). "
    "This model approximates IV using realized volatility windows from yfinance price data. "
    "Backtest is in-sample. Not financial advice."
)
