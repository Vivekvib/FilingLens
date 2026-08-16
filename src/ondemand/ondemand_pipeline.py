"""
ondemand_pipeline.py — the live "analyze any company" pipeline, run
synchronously per request from POST /analyze. Reuses the exact same
fetch/chunk/prompt logic as the offline batch pipeline
(scripts/run_pipeline.py) — only the retrieval backend differs (TF-IDF
here, ChromaDB there), and nothing here is persisted to disk. Data lives
only in the in-memory _cache dict below, for the life of the running
process — Render's free-tier disk is ephemeral anyway, so there's no
real persistence to lose. A durable cache would need a database; that's
a genuine future/funding-unlocks-this item, not solved here.
"""

import time

from src.ondemand.ticker_lookup import resolve_ticker
from src.ondemand.tfidf_retriever import TfidfRetriever
from src.ingestion import edgar_fetcher, xbrl_fetcher
from src.processing.chunker import process_filing
from src.rag.risk_memo_generator import generate_risk_memo
from src.agent.consistency_checker import check_consistency

_cache: dict[str, dict] = {}

# Simple global rate limit — max N on-demand analyses per rolling hour,
# to protect the Groq quota from a public "type any ticker" feature
# getting hammered. Not per-IP (would need request context threaded
# through); a single global cap is a blunter but much simpler safeguard,
# appropriate for a prototype's actual risk level.
_request_log: list[float] = []
MAX_REQUESTS_PER_HOUR = 20


class RateLimitExceeded(Exception):
    pass


class TickerNotFound(Exception):
    pass


def _check_rate_limit():
    now = time.time()
    global _request_log
    _request_log = [t for t in _request_log if now - t < 3600]
    if len(_request_log) >= MAX_REQUESTS_PER_HOUR:
        raise RateLimitExceeded(
            f"This demo allows {MAX_REQUESTS_PER_HOUR} live analyses per hour to protect API quota — try again later."
        )
    _request_log.append(now)


def analyze_company(ticker: str) -> dict:
    """Runs the full pipeline for one ticker: resolve CIK -> fetch 10-K ->
    fetch XBRL -> chunk -> TF-IDF index -> risk memo + consistency check.
    Raises TickerNotFound or RateLimitExceeded with a message safe to
    show directly to the user; any other exception is an unexpected
    failure the caller should treat as a 500."""
    ticker = ticker.upper().strip()

    if ticker in _cache:
        return _cache[ticker]

    _check_rate_limit()

    company = resolve_ticker(ticker)
    if company is None:
        raise TickerNotFound(f"'{ticker}' isn't a ticker SEC recognizes. Double-check the symbol.")

    meta = edgar_fetcher.get_latest_10k_meta(company["cik"])
    if meta is None:
        raise TickerNotFound(f"No 10-K filing found for {ticker} on SEC EDGAR.")
    raw_html = edgar_fetcher.fetch_filing_text(company["cik"], meta)

    xbrl_facts = xbrl_fetcher.fetch_company_facts(ticker, company["cik"])

    records = process_filing(ticker, raw_html, fiscal_year=meta.get("filing_date"))
    retriever = TfidfRetriever(records)

    memo = generate_risk_memo(ticker, retrieve_fn=retriever.retrieve)
    consistency = check_consistency(ticker, xbrl_facts, retrieve_fn=retriever.retrieve)

    result = {
        "ticker": ticker,
        "company_info": {"ticker": ticker, "cik": company["cik"], "name": company["name"], "sector": "on-demand"},
        "risk_memo": memo,
        "consistency": consistency,
    }
    _cache[ticker] = result
    return result
