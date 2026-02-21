"""
predict.py
----------
Smart prediction with model fallback chain:
  1. Per-symbol model  (ml_models/models/stock_model_TCS_NS.pkl)  ← most accurate
  2. Generalized model (ml_models/stock_model_general.pkl)        ← trained on all stocks
  3. Legacy model      (ml_models/stock_model.pkl)                ← backward compat
"""

import os
import sys
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.feature_engineering import create_features, FEATURE_COLS, SECTOR_MAP
from ml_models.dataset_loader import download_data
from stocks_list import INDIAN_STOCKS

SYMBOL_MODEL_DIR   = os.path.join("ml_models", "models")
GENERAL_MODEL_PATH = os.path.join("ml_models", "stock_model_general.pkl")
LEGACY_MODEL_PATH  = os.path.join("ml_models", "stock_model.pkl")


def _get_model(symbol: str) -> tuple:
    """
    Returns (model, model_type_string).
    Tries per-symbol → general → legacy, raises if none found.
    """
    # 1. Per-symbol model
    sym_path = os.path.join(
        SYMBOL_MODEL_DIR,
        f"stock_model_{symbol.replace('.', '_')}.pkl"
    )
    if os.path.exists(sym_path):
        return joblib.load(sym_path), "per-symbol"

    # 2. Generalized model
    if os.path.exists(GENERAL_MODEL_PATH):
        return joblib.load(GENERAL_MODEL_PATH), "generalized"

    # 3. Legacy fallback
    if os.path.exists(LEGACY_MODEL_PATH):
        return joblib.load(LEGACY_MODEL_PATH), "legacy"

    raise FileNotFoundError(
        "No trained model found. Run one of:\n"
        "  python -m ml_models.train_model              (generalized — recommended)\n"
        "  python -m ml_models.train_model --symbol TCS.NS  (per-symbol)\n"
        "  POST /ml/train/{symbol} via the API"
    )


def _load_df(symbol: str) -> pd.DataFrame:
    """Load and feature-engineer the CSV for a symbol."""
    file_path = os.path.join("data_storage", symbol.replace(".", "_") + ".csv")
    if not os.path.exists(file_path):
        try:
            download_data(symbol, period="2y")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not download data for '{symbol}': {e}"
            )

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    else:
        df.columns = [str(c).strip().split(" ")[0] for c in df.columns]
    df.dropna(how="all", inplace=True)

    # Get sector
    sector_id = 0
    for s in INDIAN_STOCKS:
        if s["symbol"] == symbol:
            sector_id = SECTOR_MAP.get(s["sector"], 0)
            break

    df = create_features(df, sector_id=sector_id)
    if len(df) == 0:
        raise ValueError(
            f"Not enough data for '{symbol}' after feature engineering."
        )
    return df


def predict(symbol: str = "TCS.NS") -> dict:
    """
    Predict next-day direction for a symbol.
    Returns a rich dict with prediction, confidence, conviction,
    model type used, and top contributing features.
    """
    model, model_type = _get_model(symbol)
    df = _load_df(symbol)

    latest      = df[FEATURE_COLS].iloc[-1:]
    prediction  = model.predict(latest)[0]
    probability = model.predict_proba(latest)[0]

    direction  = "UP"   if prediction == 1 else "DOWN"
    confidence = round(float(max(probability)) * 100, 1)

    if confidence >= 70:
        conviction = "HIGH"
    elif confidence >= 58:
        conviction = "MEDIUM"
    else:
        conviction = "LOW"

    # Top 5 features driving this prediction (from RF component)
    top_features = []
    try:
        rf = model.named_estimators_["rf"]
        importances = sorted(
            zip(FEATURE_COLS, rf.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        for feat, imp in importances[:5]:
            val = round(float(latest[feat].iloc[0]), 4)
            top_features.append({
                "feature":    feat,
                "importance": round(imp, 4),
                "value":      val,
            })
    except Exception:
        pass

    return {
        "symbol":       symbol,
        "prediction":   direction,
        "confidence":   f"{confidence}%",
        "conviction":   conviction,
        "prob_up":      f"{round(float(probability[1]) * 100, 1)}%",
        "prob_down":    f"{round(float(probability[0]) * 100, 1)}%",
        "model_used":   model_type,
        "top_features": top_features,
    }


if __name__ == "__main__":
    import json
    sym = sys.argv[1] if len(sys.argv) > 1 else "TCS.NS"
    print(json.dumps(predict(sym), indent=2))