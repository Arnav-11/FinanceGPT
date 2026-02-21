import pandas as pd


def calculate_rsi(df, window=14):
    if len(df) < window + 1:
        raise ValueError(f"Need at least {window + 1} rows to compute RSI.")

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    return round(float(rsi.iloc[-1]), 2)


def calculate_macd(df, fast=12, slow=26, signal=9):
    if len(df) < slow + signal:
        raise ValueError(f"Need at least {slow + signal} rows to compute MACD.")

    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        "macd": round(float(macd_line.iloc[-1]), 2),
        "signal": round(float(signal_line.iloc[-1]), 2),
        "histogram": round(float(histogram.iloc[-1]), 2),
    }


def moving_average(df, window=20):
    if len(df) < window:
        raise ValueError(f"Need at least {window} rows to compute MA-{window}.")
    ma = df["Close"].rolling(window=window).mean()
    return round(float(ma.iloc[-1]), 2)


def bollinger_bands(df, window=20, num_std=2.0):
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    return {
        "upper": round(float((sma + num_std * std).iloc[-1]), 2),
        "middle": round(float(sma.iloc[-1]), 2),
        "lower": round(float((sma - num_std * std).iloc[-1]), 2),
    }