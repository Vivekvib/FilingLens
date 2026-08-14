"""
edgar_fetcher.py — pulls the latest 10-K filing for each company in
config.COMPANIES via the SEC EDGAR submissions API, and caches the raw
filing text/HTML to data/raw_filings/{ticker}.htm.

Why this shape: SEC EDGAR's submissions endpoint returns a company's full
filing history as JSON, we filter it down to the most recent 10-K, then
fetch that specific document. We never construct filing URLs by guessing —
always resolve them from the submissions JSON, since accession numbers and
document filenames aren't predictable.
"""

import json
import time
from pathlib import Path

import requests

from config import COMPANIES, RAW_FILINGS_DIR, SEC_SUBMISSIONS_URL, SEC_ARCHIVES_BASE, SEC_USER_AGENT

HEADERS = {"User-Agent": SEC_USER_AGENT}


def get_latest_10k_meta(cik: str) -> dict | None:
    """Return {accession_number, primary_document, filing_date} for the
    most recent 10-K on file for this CIK, or None if not found."""
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "accession_number": recent["accessionNumber"][i].replace("-", ""),
                "primary_document": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
            }
    return None


def fetch_filing_text(cik: str, meta: dict) -> str:
    """Download the primary 10-K document (HTML) as raw text."""
    cik_int = str(int(cik))  # archives URLs use the CIK without leading zeros
    url = f"{SEC_ARCHIVES_BASE}/{cik_int}/{meta['accession_number']}/{meta['primary_document']}"
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.text


def fetch_all(force: bool = False) -> dict:
    """Fetch + cache the latest 10-K for every ticker in config.COMPANIES.
    Returns a summary dict of what succeeded/failed — check this after
    running, don't assume silence means success."""
    results = {}
    for ticker, info in COMPANIES.items():
        out_path = RAW_FILINGS_DIR / f"{ticker}.htm"
        meta_path = RAW_FILINGS_DIR / f"{ticker}_meta.json"

        if out_path.exists() and not force:
            results[ticker] = "cached"
            continue

        try:
            meta = get_latest_10k_meta(info["cik"])
            if meta is None:
                results[ticker] = "no_10k_found"
                continue

            text = fetch_filing_text(info["cik"], meta)
            out_path.write_text(text, encoding="utf-8")
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            results[ticker] = "ok"
        except requests.HTTPError as e:
            results[ticker] = f"http_error: {e}"
        except Exception as e:  # noqa: BLE001 — log and keep going, one bad ticker shouldn't kill the run
            results[ticker] = f"error: {e}"

        time.sleep(0.15)  # SEC asks for max ~10 requests/sec; stay well under that

    return results


if __name__ == "__main__":
    summary = fetch_all()
    for ticker, status in summary.items():
        print(f"{ticker}: {status}")
