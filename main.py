import math
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from data.stock_data import get_live_price, get_historical_data
from ml_models.dataset_loader import download_data
from services.analysis import analyze_stock
from services.watchlist import get_watchlist, add_to_watchlist, remove_from_watchlist
from ml_models.predict import predict
from ml_models.backtest import backtest
from stocks_list import INDIAN_STOCKS

log = logging.getLogger(__name__)

app = FastAPI(title="FinanceGPT API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/ui")
def ui():
    return FileResponse("frontend/index.html")


@app.get("/")
def root():
    return {"status": "ok", "message": "FinanceGPT API v2.1 is running"}


# ── Stocks list ───────────────────────────────────────────────────────────────

@app.get("/stocks/list")
def stocks_list(sector: str = None, index: str = None):
    """Return all stocks, optionally filtered by sector or index."""
    stocks = INDIAN_STOCKS
    if sector:
        stocks = [s for s in stocks if s["sector"].lower() == sector.lower()]
    if index:
        stocks = [s for s in stocks if s["index"].lower() == index.lower()]
    return stocks


# ── Data endpoints ────────────────────────────────────────────────────────────

@app.post("/data/download/{symbol}")
def api_download(symbol: str, period: str = "5y"):
    try:
        path = download_data(symbol, period)
        return {"message": f"Data saved to {path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/data/download-all")
def api_download_all(period: str = "5y", workers: int = 4):
    """Bulk download all 151 Indian stocks in parallel."""
    try:
        from ml_models.bulk_downloader import bulk_download
        results = bulk_download(period=period, workers=workers)
        return {
            "downloaded": len(results["success"]),
            "cached":     len(results["cached"]),
            "failed":     len(results["failed"]),
            "failed_symbols": [s for s, _ in results["failed"]],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/price/{symbol}")
def api_price(symbol: str):
    result = get_live_price(symbol)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/data/chart/{symbol}")
def api_chart(symbol: str, days: int = 180):
    """OHLCV + MA overlay data for charting (last N days)."""
    df = get_historical_data(symbol)
    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached data for '{symbol}'. Call POST /data/download/{symbol} first."
        )

    df = df.tail(days).copy()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    records = []
    for date, row in df.iterrows():
        ma20   = None if math.isnan(row["MA_20"]) else round(float(row["MA_20"]), 2)
        ma50   = None if math.isnan(row["MA_50"]) else round(float(row["MA_50"]), 2)
        volume = int(row["Volume"]) if "Volume" in row and not math.isnan(row["Volume"]) else None
        records.append({
            "date":   str(date)[:10],
            "open":   round(float(row["Open"]),  2) if "Open"  in row else None,
            "high":   round(float(row["High"]),  2) if "High"  in row else None,
            "low":    round(float(row["Low"]),   2) if "Low"   in row else None,
            "close":  round(float(row["Close"]), 2),
            "volume": volume,
            "ma20":   ma20,
            "ma50":   ma50,
        })
    return records


# ── Analysis ──────────────────────────────────────────────────────────────────

@app.get("/analysis/{symbol}")
def api_analysis(symbol: str):
    result = analyze_stock(symbol)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/compare")
def api_compare(symbol1: str, symbol2: str):
    """Compare two stocks side by side."""
    r1 = analyze_stock(symbol1)
    r2 = analyze_stock(symbol2)
    return {"stock1": r1, "stock2": r2}


# ── ML endpoints ──────────────────────────────────────────────────────────────

@app.post("/ml/train/general")
def api_train_general(download_fresh: bool = False):
    """Train the generalized model on ALL stocks combined."""
    try:
        from ml_models.train_model import train_general
        accuracy = train_general(download_missing=download_fresh)
        return {
            "message":  "Generalized market model trained successfully",
            "accuracy": round(accuracy, 4),
            "model":    "stock_model_general.pkl",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ml/train/{symbol}")
def api_train(symbol: str, download_fresh: bool = False):
    """Train a per-symbol model for one stock."""
    try:
        from ml_models.train_model import train
        accuracy = train(symbol=symbol, download=download_fresh)
        return {
            "message":  f"Per-symbol model trained for {symbol}",
            "accuracy": round(accuracy, 4),
            "model":    f"models/stock_model_{symbol.replace('.','_')}.pkl",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/predict/{symbol}")
def api_predict(symbol: str):
    """Predict next-day direction. Auto-selects best available model."""
    try:
        return predict(symbol)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/{symbol}")
def api_backtest(symbol: str, capital: float = 100000.0):
    try:
        result = backtest(symbol, initial_capital=capital)
        return result["summary"]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Watchlist ─────────────────────────────────────────────────────────────────

@app.get("/watchlist")
def api_get_watchlist():
    return get_watchlist()


@app.post("/watchlist/{symbol}")
def api_add_watchlist(symbol: str, name: str = ""):
    return add_to_watchlist(symbol, name)


@app.delete("/watchlist/{symbol}")
def api_remove_watchlist(symbol: str):
    return remove_from_watchlist(symbol)


# ── LSTM endpoints ────────────────────────────────────────────────────────────

@app.post("/ml/train-lstm/general")
def api_train_lstm_general(download_fresh: bool = False):
    """Train the generalized LSTM model on ALL stocks combined."""
    try:
        from ml_models.lstm_model import train_general as lstm_train_general
        accuracy = lstm_train_general(download_missing=download_fresh)
        return {
            "message":  "Generalized LSTM model trained successfully",
            "accuracy": round(accuracy, 4),
            "model":    "lstm_model_general.keras",
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ml/train-lstm/{symbol}")
def api_train_lstm_symbol(symbol: str, download_fresh: bool = False):
    """Train a per-symbol LSTM model for one stock."""
    try:
        from ml_models.lstm_model import train as lstm_train
        accuracy = lstm_train(symbol=symbol, download=download_fresh)
        return {
            "message":  f"Per-symbol LSTM model trained for {symbol}",
            "accuracy": round(accuracy, 4),
            "model":    f"models/lstm_model_{symbol.replace('.','_')}.keras",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/predict-lstm/{symbol}")
def api_predict_lstm(symbol: str):
    """Predict using LSTM model only."""
    try:
        from ml_models.lstm_model import predict_lstm
        return predict_lstm(symbol)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/predict-combined/{symbol}")
def api_predict_combined(symbol: str):
    """
    Combined ML + LSTM ensemble prediction.
    Returns predictions from both models + a weighted average.
    This is the most powerful prediction endpoint.
    """
    try:
        from ml_models.lstm_model import predict_combined
        return predict_combined(symbol)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))