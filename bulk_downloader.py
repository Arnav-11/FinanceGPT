"""
bulk_downloader.py
------------------
Downloads historical OHLCV data for all Indian stocks in stocks_list.py.
Run this ONCE before training the generalized model.

Usage:
    python -m ml_models.bulk_downloader
    python -m ml_models.bulk_downloader --period 5y --workers 4
"""

import os
import sys
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stocks_list import INDIAN_STOCKS
from ml_models.dataset_loader import download_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _download_one(symbol: str, period: str, force: bool) -> tuple[str, bool, str]:
    """Download a single symbol. Returns (symbol, success, message)."""
    file_path = os.path.join(
        "data_storage", symbol.replace(".", "_") + ".csv"
    )
    if not force and os.path.exists(file_path):
        return symbol, True, "already cached"

    try:
        download_data(symbol, period=period)
        return symbol, True, "downloaded"
    except Exception as e:
        return symbol, False, str(e)


def bulk_download(period: str = "5y", workers: int = 4, force: bool = False):
    """
    Download data for all 151 real stocks (excludes INDEX symbols).
    Uses a thread pool to parallelize downloads — yfinance is I/O bound.
    """
    # Only download real stocks, not index symbols (^NSEI etc.)
    stocks = [s for s in INDIAN_STOCKS if s["index"] != "INDEX"]
    symbols = [s["symbol"] for s in stocks]

    log.info(f"Starting bulk download: {len(symbols)} stocks | period={period} | workers={workers}")
    os.makedirs("data_storage", exist_ok=True)

    results = {"success": [], "failed": [], "cached": []}
    total = len(symbols)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, sym, period, force): sym
            for sym in symbols
        }

        for i, future in enumerate(as_completed(futures), 1):
            symbol, success, msg = future.result()
            pct = i / total * 100

            if not success:
                results["failed"].append((symbol, msg))
                log.warning(f"[{i:3}/{total}] ({pct:5.1f}%) FAILED   {symbol:25s} — {msg}")
            elif msg == "already cached":
                results["cached"].append(symbol)
                log.info(f"[{i:3}/{total}] ({pct:5.1f}%) CACHED   {symbol}")
            else:
                results["success"].append(symbol)
                log.info(f"[{i:3}/{total}] ({pct:5.1f}%) OK       {symbol}")

            # Small delay to be polite to Yahoo Finance
            time.sleep(0.1)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info(f"  Downloaded  : {len(results['success'])}")
    log.info(f"  Already had : {len(results['cached'])}")
    log.info(f"  Failed      : {len(results['failed'])}")
    log.info("=" * 55)

    if results["failed"]:
        log.warning("Failed symbols:")
        for sym, err in results["failed"]:
            log.warning(f"  {sym:25s} — {err}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk download Indian stock data")
    parser.add_argument("--period",  default="5y",
                        choices=["1y", "2y", "5y", "10y"],
                        help="History period (default: 5y)")
    parser.add_argument("--workers", default=4, type=int,
                        help="Parallel download threads (default: 4)")
    parser.add_argument("--force",   action="store_true",
                        help="Re-download even if CSV already exists")
    args = parser.parse_args()

    bulk_download(period=args.period, workers=args.workers, force=args.force)