"""
Phase 3 — Model Training
Baseline: LinearRegression. Stretch: XGBoost.
Input:  data/processed/features.csv
Output: model/output/metrics.txt + model/output/model.pkl
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
OUT_DIR       = Path(__file__).parent / "output"

FEATURE_COLS = ["realized_vol_20d", "iv_before", "prior_move_pct", "market_cap"]
TARGET_COL   = "iv_crush_pct"


def load_and_prep(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # encode market_cap: small=0, mid=1, large=2
    cap_map = {"small": 0, "mid": 1, "large": 2}
    df["market_cap"] = df["market_cap"].map(cap_map).fillna(1)

    # fill missing prior_move_pct with median
    df["prior_move_pct"] = df["prior_move_pct"].fillna(df["prior_move_pct"].median())

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def train():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_and_prep(FEATURES_PATH)
    print(f"Dataset: {len(X)} rows, {len(FEATURE_COLS)} features")
    print(f"Target mean: {y.mean():.3f}  std: {y.std():.3f}\n")

    # with only 20 rows, use 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Baseline: Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    r2_lr  = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    print("=== Linear Regression ===")
    print(f"R²:  {r2_lr:.4f}")
    print(f"MAE: {mae_lr:.4f}")
    print("\nCoefficients:")
    for col, coef in zip(FEATURE_COLS, lr.coef_):
        print(f"  {col:20s}: {coef:+.4f}")
    print(f"  {'intercept':20s}: {lr.intercept_:+.4f}")

    # --- Stretch: XGBoost ---
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                           random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        r2_xgb  = r2_score(y_test, y_pred_xgb)
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        print(f"\n=== XGBoost ===")
        print(f"R²:  {r2_xgb:.4f}")
        print(f"MAE: {mae_xgb:.4f}")
        best_model = xgb if r2_xgb > r2_lr else lr
        best_name  = "XGBoost" if r2_xgb > r2_lr else "LinearRegression"
    except ImportError:
        best_model = lr
        best_name  = "LinearRegression"

    # save best model
    model_path = OUT_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "features": FEATURE_COLS}, f)

    # save metrics
    metrics_path = OUT_DIR / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Linear Regression — R²: {r2_lr:.4f}  MAE: {mae_lr:.4f}\n")
        if 'r2_xgb' in dir():
            f.write(f"XGBoost           — R²: {r2_xgb:.4f}  MAE: {mae_xgb:.4f}\n")

    print(f"\nBest model: {best_name}")
    print(f"Saved → {model_path}")
    print(f"Saved → {metrics_path}")
    return best_model, FEATURE_COLS


if __name__ == "__main__":
    train()
