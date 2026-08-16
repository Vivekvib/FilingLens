"""
ticker_lookup.py — resolves ANY ticker to its SEC CIK dynamically via
SEC's public company_tickers.json, for the on-demand "analyze any
company" feature. The 8 pre-baked companies still use the static
COMPANIES dict in config.py (unaffected) — this is only for tickers a
user types in that aren't on that list.

The lookup table (tens of thousands of tickers, a few MB) is fetched
once and cached in memory for the life of the process. This is the same
authoritative source scripts/verify_ciks.py already uses.
"""

import requests

from config import SEC_USER_AGENT

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {"User-Agent": SEC_USER_AGENT}

_cache: dict | None = None


def _load_lookup_table() -> dict:
    global _cache
    if _cache is None:
        resp = requests.get(TICKERS_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()  # {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
        _cache = {entry["ticker"].upper(): entry for entry in data.values()}
    return _cache


def resolve_ticker(ticker: str) -> dict | None:
    """Returns {"cik": "0000320193", "name": "Apple Inc."} for a
    recognized ticker, or None if SEC's own list doesn't have it (e.g. a
    typo, a delisted company, or a non-US filer)."""
    table = _load_lookup_table()
    entry = table.get(ticker.upper().strip())
    if entry is None:
        return None
    return {"cik": str(entry["cik_str"]).zfill(10), "name": entry["title"]}
