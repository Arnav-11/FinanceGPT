import os
import pandas as pd
import yfinance as yf

def _load_csv(symbol):
    file_name = symbol.replace(".", "_") + ".csv"
    file_path = os.path.join("data_storage", file_name)

    if not os.path.exists(file_path):
        return None, f"No cached data for '{symbol}'. Run dataset_loader.download_data('{symbol}') first."

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Normalise column names
    df.columns = [str(c).strip().split(" ")[0] for c in df.columns]
    df.dropna(how="all", inplace=True)

    return df, None


def get_live_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            return {"error": f"No live data for '{symbol}'"}
        price = round(float(data["Close"].iloc[-1]), 2)
        timestamp = str(data.index[-1])
        return {"symbol": symbol, "price": price, "as_of": timestamp}
    except Exception as e:
        return {"error": str(e)}


def get_historical_data(symbol):
    df, err = _load_csv(symbol)
    if err:
        return None
    return df