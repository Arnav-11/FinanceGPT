"""
train_model.py
--------------
Two training modes:

1. GENERALIZED (default) — trains on ALL stocks combined.
   Produces  ml_models/stock_model_general.pkl
   This model learns universal market patterns across sectors.

2. SYMBOL-SPECIFIC — trains on one stock only.
   Produces  ml_models/models/stock_model_TCS_NS.pkl
   Best accuracy for that specific stock.

API routes use the generalized model by default; per-symbol model
is used automatically if it exists for the requested ticker.

Usage:
    # Train generalized model on all stocks
    python -m ml_models.train_model

    # Train per-symbol model for a specific stock
    python -m ml_models.train_model --symbol TCS.NS

    # Train per-symbol models for EVERY stock (takes ~10 min)
    python -m ml_models.train_model --all-symbols
"""

import os
import sys
import glob
import logging
import argparse

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stocks_list import INDIAN_STOCKS
from ml_models.dataset_loader import download_data
from ml_models.feature_engineering import create_features, FEATURE_COLS, SECTOR_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Where per-symbol models are saved
SYMBOL_MODEL_DIR   = os.path.join("ml_models", "models")
GENERAL_MODEL_PATH = os.path.join("ml_models", "stock_model_general.pkl")
# Legacy path — kept for backward compatibility
LEGACY_MODEL_PATH  = os.path.join("ml_models", "stock_model.pkl")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_symbol_df(symbol: str, sector_id: int = 0) -> pd.DataFrame | None:
    """Load CSV, engineer features, return DataFrame or None on failure."""
    file_path = os.path.join("data_storage", symbol.replace(".", "_") + ".csv")
    if not os.path.exists(file_path):
        return None

    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        else:
            df.columns = [str(c).strip().split(" ")[0] for c in df.columns]
        df.dropna(how="all", inplace=True)

        if len(df) < 100:
            return None

        df = create_features(df, sector_id=sector_id)
        return df if len(df) >= 60 else None

    except Exception as e:
        log.warning(f"  Could not process {symbol}: {e}")
        return None


def _build_ensemble() -> VotingClassifier:
    """Build a fresh VotingClassifier ensemble."""
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=7,
        min_samples_leaf=8, random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8, random_state=42
    )
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, C=0.1, random_state=42))
    ])
    return VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft"
    )


def _cv_and_test(X: pd.DataFrame, y: pd.Series, label: str) -> tuple[VotingClassifier, float]:
    """
    5-fold time-series CV, then final train on 80% / test on 20%.
    Returns (fitted model trained on full data, test accuracy).
    """
    tscv      = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    log.info(f"  Time-series CV ({label}) — {len(X)} rows, {len(FEATURE_COLS)} features")

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        rf_quick = RandomForestClassifier(
            n_estimators=100, max_depth=6,
            min_samples_leaf=10, random_state=42, n_jobs=-1
        )
        rf_quick.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        score = accuracy_score(y.iloc[val_idx], rf_quick.predict(X.iloc[val_idx]))
        cv_scores.append(score)

    log.info(f"  CV mean: {np.mean(cv_scores):.4f}  (+/- {np.std(cv_scores):.4f})")

    split   = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    model = _build_ensemble()
    model.fit(X_tr, y_tr)
    preds    = model.predict(X_te)
    accuracy = accuracy_score(y_te, preds)

    log.info(f"  Test accuracy: {accuracy:.4f}")
    log.info("\n" + classification_report(y_te, preds, target_names=["DOWN", "UP"]))

    # Retrain on full data before returning
    model.fit(X, y)
    return model, accuracy


# ── Public API ────────────────────────────────────────────────────────────────

def train_general(download_missing: bool = False) -> float:
    """
    Train a generalized model on ALL available stock CSVs combined.
    Returns test accuracy.
    """
    stocks = [s for s in INDIAN_STOCKS if s["index"] != "INDEX"]
    log.info(f"=== Generalized Market Model ===")
    log.info(f"Loading data for up to {len(stocks)} stocks...")

    frames = []
    loaded = 0

    for stock in stocks:
        symbol    = stock["symbol"]
        sector_id = SECTOR_MAP.get(stock["sector"], 0)

        if download_missing:
            fp = os.path.join("data_storage", symbol.replace(".", "_") + ".csv")
            if not os.path.exists(fp):
                try:
                    download_data(symbol, period="5y")
                except Exception:
                    pass

        df = _load_symbol_df(symbol, sector_id=sector_id)
        if df is not None:
            frames.append(df)
            loaded += 1

    if loaded == 0:
        raise RuntimeError(
            "No stock data found. Run bulk_downloader first:\n"
            "  python -m ml_models.bulk_downloader"
        )

    log.info(f"Loaded {loaded}/{len(stocks)} stocks successfully")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    log.info(f"Combined dataset: {len(combined):,} rows")

    X = combined[FEATURE_COLS]
    y = combined["Target"]

    model, accuracy = _cv_and_test(X, y, label="ALL STOCKS")

    # Save as both general and legacy paths
    os.makedirs("ml_models", exist_ok=True)
    joblib.dump(model, GENERAL_MODEL_PATH)
    joblib.dump(model, LEGACY_MODEL_PATH)   # keeps backward compatibility
    log.info(f"Generalized model saved → {GENERAL_MODEL_PATH}")
    log.info(f"Legacy path updated    → {LEGACY_MODEL_PATH}")

    # Feature importance
    rf_fitted   = model.named_estimators_["rf"]
    importances = sorted(
        zip(FEATURE_COLS, rf_fitted.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    log.info("\nTop 10 features:")
    for feat, imp in importances[:10]:
        log.info(f"  {feat:<25}: {imp:.4f}")

    return accuracy


def train(symbol: str = "TCS.NS", download: bool = False) -> float:
    """
    Train a per-symbol model for one stock.
    Returns test accuracy.
    """
    log.info(f"=== Per-Symbol Model: {symbol} ===")

    if download:
        download_data(symbol, period="5y")

    # Look up sector
    sector_id = 0
    for s in INDIAN_STOCKS:
        if s["symbol"] == symbol:
            sector_id = SECTOR_MAP.get(s["sector"], 0)
            break

    df = _load_symbol_df(symbol, sector_id=sector_id)
    if df is None:
        raise FileNotFoundError(
            f"No usable data for '{symbol}'. "
            f"Download it first: POST /data/download/{symbol}"
        )

    if len(df) < 200:
        raise ValueError(
            f"Only {len(df)} rows after feature engineering for '{symbol}'. "
            "Need at least 200. Try a longer period."
        )

    X = df[FEATURE_COLS]
    y = df["Target"]

    model, accuracy = _cv_and_test(X, y, label=symbol)

    os.makedirs(SYMBOL_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(
        SYMBOL_MODEL_DIR,
        f"stock_model_{symbol.replace('.', '_')}.pkl"
    )
    joblib.dump(model, model_path)
    log.info(f"Per-symbol model saved → {model_path}")

    return accuracy


def train_all_symbols(download_missing: bool = False) -> dict:
    """
    Train per-symbol models for every stock in INDIAN_STOCKS.
    Returns dict of {symbol: accuracy}.
    """
    stocks  = [s for s in INDIAN_STOCKS if s["index"] != "INDEX"]
    results = {}

    log.info(f"=== Training per-symbol models for {len(stocks)} stocks ===")

    for i, stock in enumerate(stocks, 1):
        symbol = stock["symbol"]
        log.info(f"[{i}/{len(stocks)}] {symbol}")
        try:
            acc = train(symbol=symbol, download=download_missing)
            results[symbol] = round(acc, 4)
        except Exception as e:
            log.warning(f"  SKIP {symbol}: {e}")
            results[symbol] = None

    succeeded = sum(1 for v in results.values() if v is not None)
    avg_acc   = np.mean([v for v in results.values() if v is not None])
    log.info(f"\nDone. {succeeded}/{len(stocks)} models trained. Avg accuracy: {avg_acc:.4f}")

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FinanceGPT ML models")
    parser.add_argument("--symbol",      default=None,
                        help="Train per-symbol model for this ticker (e.g. TCS.NS)")
    parser.add_argument("--all-symbols", action="store_true",
                        help="Train per-symbol models for every stock")
    parser.add_argument("--download",    action="store_true",
                        help="Download missing data before training")
    args = parser.parse_args()

    if args.all_symbols:
        train_all_symbols(download_missing=args.download)
    elif args.symbol:
        train(symbol=args.symbol, download=args.download)
    else:
        # Default: train generalized model
        train_general(download_missing=args.download)