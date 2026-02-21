"""
backtest.py
-----------
Backtest the ML model against buy-and-hold on a stock's test split.
Uses the same model fallback chain as predict.py.
"""

import os
import sys
import logging

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.feature_engineering import create_features, FEATURE_COLS, SECTOR_MAP
from ml_models.dataset_loader import download_data
from stocks_list import INDIAN_STOCKS

log = logging.getLogger(__name__)

SYMBOL_MODEL_DIR   = os.path.join("ml_models", "models")
GENERAL_MODEL_PATH = os.path.join("ml_models", "stock_model_general.pkl")
LEGACY_MODEL_PATH  = os.path.join("ml_models", "stock_model.pkl")


def _get_model(symbol: str) -> tuple:
    sym_path = os.path.join(
        SYMBOL_MODEL_DIR,
        f"stock_model_{symbol.replace('.', '_')}.pkl"
    )
    if os.path.exists(sym_path):
        return joblib.load(sym_path), "per-symbol"
    if os.path.exists(GENERAL_MODEL_PATH):
        return joblib.load(GENERAL_MODEL_PATH), "generalized"
    if os.path.exists(LEGACY_MODEL_PATH):
        return joblib.load(LEGACY_MODEL_PATH), "legacy"
    raise FileNotFoundError(
        "No trained model found. Run: python -m ml_models.train_model"
    )


def backtest(symbol: str = "TCS.NS", initial_capital: float = 100_000.0,
             transaction_cost: float = 0.001) -> dict:
    """
    Backtest on the last 20% of historical data (time-safe).
    Returns dict with 'summary' and 'data' (full test DataFrame).
    """
    model, model_type = _get_model(symbol)

    file_path = os.path.join("data_storage", symbol.replace(".", "_") + ".csv")
    if not os.path.exists(file_path):
        try:
            download_data(symbol, period="2y")
        except Exception as e:
            raise FileNotFoundError(f"Could not download data for '{symbol}': {e}")

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    else:
        df.columns = [str(c).strip().split(" ")[0] for c in df.columns]
    df.dropna(how="all", inplace=True)

    sector_id = 0
    for s in INDIAN_STOCKS:
        if s["symbol"] == symbol:
            sector_id = SECTOR_MAP.get(s["sector"], 0)
            break

    df = create_features(df, sector_id=sector_id)

    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} rows after feature engineering for '{symbol}'. Need ≥ 100."
        )

    split = int(len(df) * 0.8)
    test  = df.iloc[split:].copy()

    test["Signal"]       = model.predict(test[FEATURE_COLS])
    test["Stock_Return"] = test["Close"].pct_change().fillna(0)

    position_change      = test["Signal"].diff().abs().fillna(0)
    test["Strat_Return"] = (
        test["Signal"].shift(1).fillna(0) * test["Stock_Return"]
        - position_change * transaction_cost
    )

    test["Portfolio"] = initial_capital * (1 + test["Strat_Return"]).cumprod()
    test["Buy_Hold"]  = initial_capital * (1 + test["Stock_Return"]).cumprod()

    strat_return = (test["Portfolio"].iloc[-1] / initial_capital - 1) * 100
    bh_return    = (test["Buy_Hold"].iloc[-1]  / initial_capital - 1) * 100

    pred_accuracy = None
    if "Target" in test.columns:
        pred_accuracy = round(
            float((model.predict(test[FEATURE_COLS]) == test["Target"].values).mean() * 100), 2
        )

    summary = {
        "symbol":              symbol,
        "model_used":          model_type,
        "backtest_days":       len(test),
        "initial_capital":     initial_capital,
        "final_value":         round(float(test["Portfolio"].iloc[-1]), 2),
        "strategy_return_pct": round(strat_return, 2),
        "buy_hold_return_pct": round(bh_return, 2),
        "alpha_pct":           round(strat_return - bh_return, 2),
        "sharpe_ratio":        round(_sharpe(test["Strat_Return"]), 2),
        "max_drawdown_pct":    round(_max_drawdown(test["Portfolio"]) * 100, 2),
        "win_rate_pct":        round(_win_rate(test["Strat_Return"]) * 100, 2),
        "prediction_accuracy": pred_accuracy,
    }

    return {"summary": summary, "data": test}


def _sharpe(returns, risk_free=0.06, trading_days=252):
    daily_rf = risk_free / trading_days
    excess   = returns - daily_rf
    std      = excess.std()
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(trading_days))


def _max_drawdown(portfolio):
    rolling_max = portfolio.cummax()
    drawdown    = (portfolio - rolling_max) / rolling_max
    return float(drawdown.min())


def _win_rate(returns):
    """Fraction of trading days with positive return."""
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())