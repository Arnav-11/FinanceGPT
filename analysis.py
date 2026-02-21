import os
import pandas as pd
from data.stock_data import get_historical_data
from indicators.technicals import calculate_rsi, calculate_macd, moving_average, bollinger_bands
from ml_models.dataset_loader import download_data


def _ensure_data(symbol):
    file_path = os.path.join("data_storage", symbol.replace(".", "_") + ".csv")
    if not os.path.exists(file_path):
        download_data(symbol, period="2y")


def generate_summary(rsi, macd, price, ma20, ma50, bb):
    signals = []
    score = 0
    max_score = 7  # RSI: ±2, MACD: ±2, MA50: ±1, MA20: ±1, BB: ±1

    # RSI signal
    if rsi < 30:
        signals.append("RSI is oversold (bullish)")
        score += 2
    elif rsi < 45:
        signals.append("RSI is leaning bullish")
        score += 1
    elif rsi > 70:
        signals.append("RSI is overbought (bearish)")
        score -= 2
    elif rsi > 55:
        signals.append("RSI is leaning bearish")
        score -= 1
    else:
        signals.append("RSI is neutral")

    # MACD signal
    if macd["macd"] > macd["signal"]:
        signals.append("MACD is bullish crossover")
        score += 2
    else:
        signals.append("MACD is bearish crossover")
        score -= 2

    # Price vs MAs
    if price > ma50:
        signals.append("Price is above MA50 (uptrend)")
        score += 1
    else:
        signals.append("Price is below MA50 (downtrend)")
        score -= 1

    if price > ma20:
        signals.append("Price is above MA20")
        score += 1
    else:
        signals.append("Price is below MA20")
        score -= 1

    # Bollinger Band position
    if price < bb["lower"]:
        signals.append("Price near lower Bollinger Band (oversold zone)")
        score += 1
    elif price > bb["upper"]:
        signals.append("Price near upper Bollinger Band (overbought zone)")
        score -= 1

    # Overall verdict
    if score >= 4:
        verdict = "STRONG BUY"
    elif score >= 2:
        verdict = "BUY"
    elif score <= -4:
        verdict = "STRONG SELL"
    elif score <= -2:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    return {
        "verdict": verdict,
        "score": score,
        "max_score": max_score,
        "signals": signals,
    }


def analyze_stock(symbol):
    try:
        _ensure_data(symbol)
    except Exception as e:
        return {"error": f"Could not download data for '{symbol}': {str(e)}"}

    df = get_historical_data(symbol)
    if df is None:
        return {"error": f"No data found for '{symbol}'. Download it first via /data/download/{symbol}"}

    try:
        rsi   = calculate_rsi(df)
        macd  = calculate_macd(df)
        ma20  = moving_average(df, 20)
        ma50  = moving_average(df, 50)
        bb    = bollinger_bands(df)
        price = round(float(df["Close"].iloc[-1]), 2)
        trend = "BULLISH" if price > ma50 else "BEARISH"

        # 52-week high/low (last 252 trading days)
        year_data   = df["Close"].tail(252)
        week52_high = round(float(year_data.max()), 2)
        week52_low  = round(float(year_data.min()), 2)

        pct_from_high = round((price - week52_high) / week52_high * 100, 2)
        pct_from_low  = round((price - week52_low)  / week52_low  * 100, 2)

        # pct_from_high is always <= 0; pct_from_low is always >= 0
        pct_high_str = f"{pct_from_high}%"
        pct_low_str  = f"+{pct_from_low}%" if pct_from_low >= 0 else f"{pct_from_low}%"

        summary = generate_summary(rsi, macd, price, ma20, ma50, bb)

        return {
            "symbol":  symbol,
            "price":   price,
            "trend":   trend,
            "week52": {
                "high":          week52_high,
                "low":           week52_low,
                "pct_from_high": pct_high_str,
                "pct_from_low":  pct_low_str,
            },
            "indicators": {
                "RSI":             rsi,
                "MACD":            macd,
                "MA_20":           ma20,
                "MA_50":           ma50,
                "Bollinger_Bands": bb,
            },
            "recommendation": summary["verdict"],
            "analyst_summary": {
                "verdict": summary["verdict"],
                "score":   f"{summary['score']} / {summary['max_score']}",
                "signals": summary["signals"],
            },
        }

    except Exception as e:
        return {"error": str(e)}