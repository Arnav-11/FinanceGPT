"""
lstm_model.py
-------------
LSTM-based deep learning model for stock direction prediction.

Architecture:
  Input  →  Bidirectional LSTM (128)  →  Dropout(0.3)
         →  LSTM (64)                 →  Dropout(0.3)
         →  Dense (32, relu)          →  Dropout(0.2)
         →  Dense (1, sigmoid)        →  UP / DOWN

Key design decisions:
  - Uses a 60-day sliding window (sequence_length=60)
  - Bidirectional LSTM on first layer
  - Per-feature MinMax scaling fitted on training data only
  - Early stopping + ReduceLROnPlateau to avoid overfitting
"""

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.feature_engineering import create_features, FEATURE_COLS, SECTOR_MAP
from ml_models.dataset_loader import download_data
from stocks_list import INDIAN_STOCKS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
LSTM_GENERAL_PATH = os.path.join("ml_models", "lstm_model_general.keras")
LSTM_SCALER_PATH  = os.path.join("ml_models", "lstm_scaler_general.pkl")
LSTM_SYMBOL_DIR   = os.path.join("ml_models", "models")

SEQUENCE_LENGTH  = 60
BATCH_SIZE       = 64
EPOCHS           = 50
VALIDATION_SPLIT = 0.1


# ── Data preparation ──────────────────────────────────────────────────────────

def _load_symbol_df(symbol, sector_id=0):
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
        if len(df) < SEQUENCE_LENGTH + 20:
            return None
        df = create_features(df, sector_id=sector_id)
        return df if len(df) >= SEQUENCE_LENGTH + 10 else None
    except Exception as e:
        log.warning(f"Could not process {symbol}: {e}")
        return None


def _make_sequences(X, y, seq_len=SEQUENCE_LENGTH):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _time_split(X_seq, y_seq, test_ratio=0.2):
    split = int(len(X_seq) * (1 - test_ratio))
    return X_seq[:split], X_seq[split:], y_seq[:split], y_seq[split:]


# ── Model architecture ────────────────────────────────────────────────────────

def _build_model(n_features, seq_len=SEQUENCE_LENGTH):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.LSTM(64, return_sequences=False,
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(32, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1
        ),
    ]


# ── Training ──────────────────────────────────────────────────────────────────

def train_general(download_missing=False):
    from sklearn.preprocessing import MinMaxScaler

    stocks = [s for s in INDIAN_STOCKS if s["index"] != "INDEX"]
    log.info("=== LSTM Generalized Training ===")
    log.info(f"Loading data for up to {len(stocks)} stocks...")

    all_X, all_y = [], []
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
            all_X.append(df[FEATURE_COLS].values)
            all_y.append(df["Target"].values)
            loaded += 1

    if loaded == 0:
        raise RuntimeError(
            "No stock data found. Run bulk download first:\n"
            "  python -m ml_models.bulk_downloader"
        )

    log.info(f"Loaded {loaded}/{len(stocks)} stocks")

    combined_X = np.vstack(all_X)
    combined_y = np.concatenate(all_y)
    log.info(f"Total rows before sequencing: {len(combined_X):,}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(combined_X)

    os.makedirs("ml_models", exist_ok=True)
    joblib.dump(scaler, LSTM_SCALER_PATH)
    log.info(f"Scaler saved -> {LSTM_SCALER_PATH}")

    # Build sequences per stock (no cross-stock contamination)
    seq_X_list, seq_y_list = [], []
    idx = 0
    for X_stock in all_X:
        n        = len(X_stock)
        X_scaled = scaler.transform(X_stock)
        y_stock  = combined_y[idx: idx + n]
        sx, sy   = _make_sequences(X_scaled, y_stock)
        seq_X_list.append(sx)
        seq_y_list.append(sy)
        idx += n

    X_seq = np.concatenate(seq_X_list)
    y_seq = np.concatenate(seq_y_list)
    log.info(f"Total sequences: {len(X_seq):,}  Shape: {X_seq.shape}")

    perm  = np.random.RandomState(42).permutation(len(X_seq))
    X_seq = X_seq[perm]
    y_seq = y_seq[perm]

    split   = int(len(X_seq) * 0.8)
    X_train = X_seq[:split]
    X_test  = X_seq[split:]
    y_train = y_seq[:split]
    y_test  = y_seq[split:]

    log.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    model = _build_model(n_features=len(FEATURE_COLS))
    model.summary()

    model.fit(
        X_train, y_train,
        epochs           = EPOCHS,
        batch_size       = BATCH_SIZE,
        validation_split = VALIDATION_SPLIT,
        callbacks        = _get_callbacks(),
        verbose          = 1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    log.info(f"Test Loss    : {loss:.4f}")
    log.info(f"Test Accuracy: {accuracy:.4f}")

    model.save(LSTM_GENERAL_PATH)
    log.info(f"LSTM model saved -> {LSTM_GENERAL_PATH}")

    return float(accuracy)


def train(symbol="TCS.NS", download=False):
    from sklearn.preprocessing import MinMaxScaler

    log.info(f"=== LSTM Per-Symbol Training: {symbol} ===")

    if download:
        download_data(symbol, period="5y")

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

    X_raw    = df[FEATURE_COLS].values
    y_raw    = df["Target"].values
    scaler   = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_raw)
    X_seq, y_seq = _make_sequences(X_scaled, y_raw)

    if len(X_seq) < 100:
        raise ValueError(f"Only {len(X_seq)} sequences for '{symbol}'. Need >= 100.")

    X_train, X_test, y_train, y_test = _time_split(X_seq, y_seq)
    log.info(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    model = _build_model(n_features=len(FEATURE_COLS))
    model.fit(
        X_train, y_train,
        epochs           = EPOCHS,
        batch_size       = min(BATCH_SIZE, len(X_train) // 4),
        validation_split = VALIDATION_SPLIT,
        callbacks        = _get_callbacks(),
        verbose          = 1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    log.info(f"Test Loss: {loss:.4f}  |  Test Accuracy: {accuracy:.4f}")

    os.makedirs(LSTM_SYMBOL_DIR, exist_ok=True)
    model_path  = os.path.join(LSTM_SYMBOL_DIR,
                               f"lstm_model_{symbol.replace('.', '_')}.keras")
    scaler_path = os.path.join(LSTM_SYMBOL_DIR,
                               f"lstm_scaler_{symbol.replace('.', '_')}.pkl")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    log.info(f"Model  saved -> {model_path}")
    log.info(f"Scaler saved -> {scaler_path}")

    return float(accuracy)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_lstm(symbol="TCS.NS"):
    # Find model + scaler
    sym_model_path  = os.path.join(LSTM_SYMBOL_DIR,
                                   f"lstm_model_{symbol.replace('.', '_')}.keras")
    sym_scaler_path = os.path.join(LSTM_SYMBOL_DIR,
                                   f"lstm_scaler_{symbol.replace('.', '_')}.pkl")

    if os.path.exists(sym_model_path) and os.path.exists(sym_scaler_path):
        model      = tf.keras.models.load_model(sym_model_path)
        scaler     = joblib.load(sym_scaler_path)
        model_type = "lstm-per-symbol"
    elif os.path.exists(LSTM_GENERAL_PATH) and os.path.exists(LSTM_SCALER_PATH):
        model      = tf.keras.models.load_model(LSTM_GENERAL_PATH)
        scaler     = joblib.load(LSTM_SCALER_PATH)
        model_type = "lstm-generalized"
    else:
        raise FileNotFoundError(
            "No LSTM model found. Train it first:\n"
            "  python -m ml_models.lstm_model"
        )

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

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Not enough data for '{symbol}'. "
            f"Need at least {SEQUENCE_LENGTH} rows."
        )

    X_raw    = df[FEATURE_COLS].values[-SEQUENCE_LENGTH:]
    X_scaled = scaler.transform(X_raw)
    X_input  = X_scaled.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLS))

    prob_up   = float(model.predict(X_input, verbose=0)[0][0])
    prob_down = 1.0 - prob_up
    direction = "UP" if prob_up >= 0.5 else "DOWN"
    confidence = round(max(prob_up, prob_down) * 100, 1)

    if confidence >= 70:
        conviction = "HIGH"
    elif confidence >= 58:
        conviction = "MEDIUM"
    else:
        conviction = "LOW"

    return {
        "symbol":     symbol,
        "prediction": direction,
        "confidence": f"{confidence}%",
        "conviction": conviction,
        "prob_up":    f"{round(prob_up * 100, 1)}%",
        "prob_down":  f"{round(prob_down * 100, 1)}%",
        "model_used": model_type,
    }


def predict_combined(symbol="TCS.NS"):
    from ml_models.predict import predict as predict_ml

    results = {"symbol": symbol}

    try:
        ml = predict_ml(symbol)
        results["ml"] = ml
    except Exception as e:
        results["ml"] = {"error": str(e)}
        ml = None

    try:
        lstm = predict_lstm(symbol)
        results["lstm"] = lstm
    except Exception as e:
        results["lstm"] = {"error": str(e)}
        lstm = None

    if ml and lstm and "error" not in ml and "error" not in lstm:
        ml_prob_up   = float(ml["prob_up"].replace("%", ""))   / 100
        lstm_prob_up = float(lstm["prob_up"].replace("%", "")) / 100

        combined_prob_up = (0.45 * ml_prob_up) + (0.55 * lstm_prob_up)
        combined_prob_dn = 1.0 - combined_prob_up

        direction  = "UP" if combined_prob_up >= 0.5 else "DOWN"
        confidence = round(max(combined_prob_up, combined_prob_dn) * 100, 1)

        if confidence >= 70:
            conviction = "HIGH"
        elif confidence >= 58:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"

        results["combined"] = {
            "prediction": direction,
            "confidence": f"{confidence}%",
            "conviction": conviction,
            "prob_up":    f"{round(combined_prob_up * 100, 1)}%",
            "prob_down":  f"{round(combined_prob_dn * 100, 1)}%",
            "model_used": "ml+lstm-ensemble",
        }

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / run LSTM model")
    parser.add_argument("--symbol",   default=None,
                        help="Train per-symbol LSTM (e.g. TCS.NS)")
    parser.add_argument("--predict",  default=None,
                        help="Run prediction for a symbol (e.g. TCS.NS)")
    parser.add_argument("--download", action="store_true",
                        help="Download missing data before training")
    args = parser.parse_args()

    if args.predict:
        import json
        print(json.dumps(predict_combined(args.predict), indent=2))
    elif args.symbol:
        train(symbol=args.symbol, download=args.download)
    else:
        train_general(download_missing=args.download)