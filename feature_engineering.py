"""
feature_engineering.py
-----------------------
Creates a rich set of 28 technical + price features from OHLCV data.

New vs original:
  - Added OBV (On-Balance Volume)
  - Added ATR (Average True Range) — volatility measure
  - Added Williams %R
  - Added Stochastic %K
  - Added price gap feature (open vs prev close)
  - Added Sector encoding (passed in as optional int)
  - Normalized all price-based features to be scale-invariant
    so the model generalizes across high-price (MRF ~90k) and
    low-price (SAIL ~100) stocks
"""

import pandas as pd
import numpy as np

FEATURE_COLS = [
    # Moving averages (price-normalized — scale invariant)
    "Price_vs_MA10",
    "Price_vs_MA20",
    "Price_vs_MA50",
    "MA10_vs_MA20",
    "MA20_vs_MA50",

    # Exponential MAs
    "Price_vs_EMA10",
    "Price_vs_EMA20",

    # Volatility
    "Volatility_10",
    "Volatility_20",
    "ATR_ratio",           # ATR / Close — normalized volatility

    # Momentum / returns
    "Return_1d",
    "Return_3d",
    "Return_5d",
    "Return_10d",
    "Momentum_10",

    # Oscillators
    "RSI",
    "RSI_slope",
    "Williams_R",
    "Stoch_K",

    # MACD
    "MACD_hist",
    "MACD_slope",

    # Bollinger Bands
    "BB_width",
    "BB_position",

    # Volume
    "Volume_ratio",
    "OBV_slope",           # 5-day slope of OBV (normalized)

    # Candle structure
    "High_Low_ratio",
    "Body_ratio",          # (Close-Open) / (High-Low) — candle body size

    # Gap
    "Gap",                 # (Open - prev Close) / prev Close
]


def create_features(df: pd.DataFrame, sector_id: int = 0) -> pd.DataFrame:
    """
    Build all features from an OHLCV DataFrame.

    Parameters
    ----------
    df        : DataFrame with columns Close, High, Low, Open, Volume
    sector_id : integer encoding of the stock's sector (0 = unknown)
                Used only when training multi-stock model.
    """
    df = df.copy()

    # Ensure required columns exist with fallbacks
    if "Open" not in df.columns:
        df["Open"] = df["Close"]
    if "High" not in df.columns:
        df["High"] = df["Close"]
    if "Low" not in df.columns:
        df["Low"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    # ── Returns ───────────────────────────────────────────────────────────────
    df["Return_1d"]  = df["Close"].pct_change()
    df["Return_3d"]  = df["Close"].pct_change(3)
    df["Return_5d"]  = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)

    # ── Moving Averages (normalized — scale invariant) ────────────────────────
    ma10  = df["Close"].rolling(10).mean()
    ma20  = df["Close"].rolling(20).mean()
    ma50  = df["Close"].rolling(50).mean()
    ema10 = df["Close"].ewm(span=10, adjust=False).mean()
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()

    df["Price_vs_MA10"]  = (df["Close"] - ma10)  / ma10
    df["Price_vs_MA20"]  = (df["Close"] - ma20)  / ma20
    df["Price_vs_MA50"]  = (df["Close"] - ma50)  / ma50
    df["MA10_vs_MA20"]   = (ma10 - ma20) / ma20
    df["MA20_vs_MA50"]   = (ma20 - ma50) / ma50
    df["Price_vs_EMA10"] = (df["Close"] - ema10) / ema10
    df["Price_vs_EMA20"] = (df["Close"] - ema20) / ema20

    # ── Volatility ────────────────────────────────────────────────────────────
    df["Volatility_10"] = df["Return_1d"].rolling(10).std()
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()

    # ── ATR (Average True Range) — normalized ─────────────────────────────────
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_ratio"] = true_range.rolling(14).mean() / df["Close"]

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["Momentum_10"] = df["Close"] / df["Close"].shift(10) - 1

    # ── RSI (14) + slope ──────────────────────────────────────────────────────
    delta     = df["Close"].diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_slope"] = df["RSI"].diff(3)

    # ── Williams %R (14) ─────────────────────────────────────────────────────
    highest_high = df["High"].rolling(14).max()
    lowest_low   = df["Low"].rolling(14).min()
    df["Williams_R"] = -100 * (highest_high - df["Close"]) / (
        (highest_high - lowest_low).replace(0, np.nan)
    )

    # ── Stochastic %K (14) ───────────────────────────────────────────────────
    df["Stoch_K"] = 100 * (df["Close"] - lowest_low) / (
        (highest_high - lowest_low).replace(0, np.nan)
    )

    # ── MACD histogram + slope ────────────────────────────────────────────────
    ema12           = df["Close"].ewm(span=12, adjust=False).mean()
    ema26           = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line       = ema12 - ema26
    signal_line     = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = (macd_line - signal_line) / df["Close"]   # normalized
    df["MACD_slope"] = df["MACD_hist"].diff(3)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_sma  = df["Close"].rolling(20).mean()
    bb_std  = df["Close"].rolling(20).std()
    bb_up   = bb_sma + 2 * bb_std
    bb_lo   = bb_sma - 2 * bb_std
    denom   = (bb_up - bb_lo).replace(0, np.nan)
    df["BB_width"]    = (bb_up - bb_lo) / bb_sma
    df["BB_position"] = (df["Close"] - bb_lo) / denom

    # ── Volume features ───────────────────────────────────────────────────────
    vol_ma = df["Volume"].rolling(20).mean().replace(0, np.nan)
    df["Volume_ratio"] = df["Volume"] / vol_ma

    # OBV
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    obv_ma = obv.rolling(5).mean().replace(0, np.nan)
    df["OBV_slope"] = (obv - obv_ma) / (df["Close"] * df["Volume"].mean() + 1e-9)

    # ── Candle structure ──────────────────────────────────────────────────────
    df["High_Low_ratio"] = (df["High"] - df["Low"]) / df["Close"]
    candle_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["Body_ratio"] = (df["Close"] - df["Open"]) / candle_range

    # ── Gap ───────────────────────────────────────────────────────────────────
    df["Gap"] = (df["Open"] - df["Close"].shift()) / df["Close"].shift()

    # ── Target: 1 if next close > today's close ───────────────────────────────
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


# ── Sector encoding map ───────────────────────────────────────────────────────
SECTOR_MAP = {
    "IT": 1, "Banking": 2, "Finance": 3, "Pharma": 4, "Auto": 5,
    "FMCG": 6, "Energy": 7, "Metals": 8, "Power": 9, "Infrastructure": 10,
    "Healthcare": 11, "Cement": 12, "Consumer": 13, "Telecom": 14,
    "Insurance": 15, "Real Estate": 16, "Auto Ancillary": 17,
    "Defence": 18, "Engineering": 19, "Chemicals": 20,
    "Electricals": 21, "Logistics": 22, "Mining": 23,
    "Retail": 24, "Tech": 25, "Fintech": 26, "Tyres": 27,
    "Beverages": 28, "Apparel": 29, "Aviation": 30,
    "Renewable Energy": 31, "Electronics": 32,
    "Pipes": 33, "E-Commerce": 34, "Conglomerate": 35,
    "Agro Chemicals": 36, "Abrasives": 37, "Jewellery": 38,
}