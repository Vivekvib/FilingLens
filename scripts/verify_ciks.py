"""
verify_ciks.py — one-off utility, not part of the main pipeline.

Fetches SEC's authoritative ticker -> CIK mapping (company_tickers.json,
maintained by SEC itself) and cross-checks it against config.COMPANIES,
printing any mismatches so you can fix config.py before the real
pipeline run. This is a more reliable check than manually searching
EDGAR's company-name search (which doesn't accept ticker symbols).

Run with: python scripts/verify_ciks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from config import COMPANIES, SEC_USER_AGENT

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {"User-Agent": SEC_USER_AGENT}


def main():
    resp = requests.get(TICKERS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()  # {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}

    # index by ticker for fast lookup
    by_ticker = {entry["ticker"]: entry for entry in data.values()}

    print(f"{'Ticker':<6} {'Config CIK':<14} {'SEC CIK':<14} {'Status'}")
    print("-" * 60)

    mismatches = []
    for ticker, info in COMPANIES.items():
        sec_entry = by_ticker.get(ticker)
        if sec_entry is None:
            print(f"{ticker:<6} {info['cik']:<14} {'NOT FOUND':<14} ticker not in SEC's list — check spelling")
            mismatches.append(ticker)
            continue

        sec_cik = str(sec_entry["cik_str"]).zfill(10)
        config_cik = info["cik"]
        status = "OK" if sec_cik == config_cik else "MISMATCH — update config.py"
        print(f"{ticker:<6} {config_cik:<14} {sec_cik:<14} {status}")
        if sec_cik != config_cik:
            mismatches.append(ticker)

    print()
    if mismatches:
        print(f"{len(mismatches)} ticker(s) need fixing in config.py: {mismatches}")
        print("Copy the 'SEC CIK' column values above into COMPANIES in config.py.")
    else:
        print("All CIKs verified correct. Safe to proceed to the pipeline run.")


if __name__ == "__main__":
    main()
