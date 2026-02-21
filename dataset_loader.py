import os
import yfinance as yf


def download_data(symbol="TCS.NS", period="5y"):
    """Download historical OHLCV data from Yahoo Finance and cache to CSV."""
    print(f"Downloading {symbol} ({period})...")

    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(
            f"No data returned for '{symbol}'. "
            "Check the ticker symbol (e.g. RELIANCE.NS for NSE stocks)."
        )

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, __import__("pandas").MultiIndex):
        df.columns = [col[0] for col in df.columns]
    else:
        df.columns = [str(c).strip().split(" ")[0] for c in df.columns]

    os.makedirs("data_storage", exist_ok=True)
    file_path = os.path.join("data_storage", symbol.replace(".", "_") + ".csv")
    df.to_csv(file_path)

    print(f"Saved {len(df)} rows to {file_path}")
    return file_path


if __name__ == "__main__":
    download_data("TCS.NS")