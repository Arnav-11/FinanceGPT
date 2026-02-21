import json
import os

# Use absolute path relative to this file to avoid cwd issues
_BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCHLIST_FILE = os.path.join(_BASE_DIR, "data_storage", "watchlist.json")


def _load():
    if not os.path.exists(WATCHLIST_FILE):
        return []
    with open(WATCHLIST_FILE, "r") as f:
        return json.load(f)


def _save(data):
    os.makedirs(os.path.dirname(WATCHLIST_FILE), exist_ok=True)
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_watchlist():
    return _load()


def add_to_watchlist(symbol: str, name: str = ""):
    watchlist = _load()
    if any(s["symbol"] == symbol for s in watchlist):
        return {"message": f"{symbol} is already in your watchlist"}
    watchlist.append({"symbol": symbol, "name": name})
    _save(watchlist)
    return {"message": f"{symbol} added to watchlist"}


def remove_from_watchlist(symbol: str):
    watchlist = _load()
    updated   = [s for s in watchlist if s["symbol"] != symbol]
    if len(updated) == len(watchlist):
        return {"message": f"{symbol} was not in watchlist"}
    _save(updated)
    return {"message": f"{symbol} removed from watchlist"}