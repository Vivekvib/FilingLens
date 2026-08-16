"""
test_ondemand_pipeline.py — tests for src/ondemand/ondemand_pipeline.py.

Two things are genuinely testable without live network access:
  1. The rate-limiting logic (_check_rate_limit) — pure, in-memory, no
     external calls at all.
  2. The orchestration wiring in analyze_company() — that it calls the
     right functions in the right order with the right arguments, and
     handles the TickerNotFound/caching paths correctly. We mock every
     external call (SEC, Groq) rather than skip testing this logic
     entirely, since a wiring mistake here (e.g. passing the wrong CIK,
     or not injecting the TF-IDF retrieve_fn) would be a real, silent bug.

What's NOT tested here: whether SEC/Groq actually respond as expected —
that's inherently an integration concern, verified manually against the
real deployed service, the same way the rest of this project's live
behavior was verified throughout development.
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.ondemand import ondemand_pipeline
from src.ondemand.ondemand_pipeline import (
    analyze_company,
    _check_rate_limit,
    RateLimitExceeded,
    TickerNotFound,
    MAX_REQUESTS_PER_HOUR,
)


@pytest.fixture(autouse=True)
def reset_pipeline_state():
    """Every test gets a clean cache and request log — otherwise tests
    would leak state into each other via the module-level dicts/lists."""
    ondemand_pipeline._cache.clear()
    ondemand_pipeline._request_log.clear()
    yield
    ondemand_pipeline._cache.clear()
    ondemand_pipeline._request_log.clear()


def test_rate_limit_allows_up_to_the_cap():
    for _ in range(MAX_REQUESTS_PER_HOUR):
        _check_rate_limit()  # should not raise
    assert len(ondemand_pipeline._request_log) == MAX_REQUESTS_PER_HOUR


def test_rate_limit_blocks_after_the_cap():
    for _ in range(MAX_REQUESTS_PER_HOUR):
        _check_rate_limit()
    with pytest.raises(RateLimitExceeded):
        _check_rate_limit()


def test_rate_limit_prunes_entries_older_than_an_hour():
    # Simulate MAX_REQUESTS_PER_HOUR requests from over an hour ago —
    # these should be pruned and NOT count against the current window.
    old_timestamp = time.time() - 3700  # 61+ minutes ago
    ondemand_pipeline._request_log.extend([old_timestamp] * MAX_REQUESTS_PER_HOUR)

    _check_rate_limit()  # should succeed — the old entries don't count

    assert len(ondemand_pipeline._request_log) == 1  # only the new one remains


def test_analyze_company_raises_ticker_not_found_for_unresolvable_ticker():
    with patch("src.ondemand.ondemand_pipeline.resolve_ticker", return_value=None):
        with pytest.raises(TickerNotFound):
            analyze_company("NOTAREALTICKER")


def test_analyze_company_raises_ticker_not_found_when_no_10k_exists():
    with patch("src.ondemand.ondemand_pipeline.resolve_ticker", return_value={"cik": "0000000001", "name": "Test Co"}):
        with patch("src.ondemand.ondemand_pipeline.edgar_fetcher") as mock_edgar:
            mock_edgar.get_latest_10k_meta.return_value = None
            with pytest.raises(TickerNotFound):
                analyze_company("TESTCO")


def test_analyze_company_full_flow_wiring():
    """Confirms the orchestration calls every step with the correct
    arguments and assembles the expected result shape — the actual
    external calls (SEC, Groq) are mocked, but the WIRING between them
    is real and would catch e.g. passing the wrong CIK or ticker."""
    fake_company = {"cik": "0000000001", "name": "Test Co"}
    fake_meta = {"accession_number": "123", "primary_document": "test.htm", "filing_date": "2025-01-01"}
    fake_xbrl = {"revenue": [{"fy": 2025, "val": 1000}, {"fy": 2024, "val": 900}]}
    fake_records = [{"ticker": "TESTCO", "section": "risk_factors", "method": "regex", "fiscal_year": "2025-01-01", "text": "some risk"}]
    fake_memo = {"ticker": "TESTCO", "top_risks": [], "management_tone": "", "notable_flags": []}
    fake_consistency = {"flags": [], "assessment": "ok", "computed_ratios": {"revenue_growth_pct": 11.11}}

    with patch("src.ondemand.ondemand_pipeline.resolve_ticker", return_value=fake_company) as mock_resolve, \
         patch("src.ondemand.ondemand_pipeline.edgar_fetcher") as mock_edgar, \
         patch("src.ondemand.ondemand_pipeline.xbrl_fetcher") as mock_xbrl, \
         patch("src.ondemand.ondemand_pipeline.process_filing", return_value=fake_records) as mock_process, \
         patch("src.ondemand.ondemand_pipeline.TfidfRetriever") as mock_retriever_cls, \
         patch("src.ondemand.ondemand_pipeline.generate_risk_memo", return_value=fake_memo) as mock_gen_memo, \
         patch("src.ondemand.ondemand_pipeline.check_consistency", return_value=fake_consistency) as mock_check:

        mock_edgar.get_latest_10k_meta.return_value = fake_meta
        mock_edgar.fetch_filing_text.return_value = "<html>fake filing</html>"
        mock_xbrl.fetch_company_facts.return_value = fake_xbrl
        mock_retriever_instance = MagicMock()
        mock_retriever_cls.return_value = mock_retriever_instance

        result = analyze_company("testco")  # lowercase on purpose — should be normalized to upper

        # Ticker was normalized to uppercase before being resolved
        mock_resolve.assert_called_once_with("TESTCO")

        # Filing was fetched with the CIK from resolve_ticker, not a hardcoded one
        mock_edgar.get_latest_10k_meta.assert_called_once_with("0000000001")
        mock_edgar.fetch_filing_text.assert_called_once_with("0000000001", fake_meta)

        # XBRL was fetched with the resolved CIK and the (normalized) ticker
        mock_xbrl.fetch_company_facts.assert_called_once_with("TESTCO", "0000000001")

        # Chunking got the raw HTML and the normalized ticker
        mock_process.assert_called_once()
        assert mock_process.call_args[0][0] == "TESTCO"

        # The TF-IDF retriever's bound retrieve method was injected into
        # BOTH generators — this is the actual point of the whole
        # dependency-injection refactor, so it's worth asserting directly.
        assert mock_gen_memo.call_args[1]["retrieve_fn"] == mock_retriever_instance.retrieve
        assert mock_check.call_args[1]["retrieve_fn"] == mock_retriever_instance.retrieve

        # Result shape matches what the frontend expects
        assert result["ticker"] == "TESTCO"
        assert result["risk_memo"] == fake_memo
        assert result["consistency"] == fake_consistency
        assert result["company_info"]["cik"] == "0000000001"


def test_analyze_company_uses_cache_on_second_call():
    """The second call for the same ticker should hit the in-memory cache
    and NOT call resolve_ticker again — this is what makes repeat
    requests fast and protects the rate limit budget."""
    fake_result = {"ticker": "CACHED", "company_info": {}, "risk_memo": {}, "consistency": {}}
    ondemand_pipeline._cache["CACHED"] = fake_result

    with patch("src.ondemand.ondemand_pipeline.resolve_ticker") as mock_resolve:
        result = analyze_company("CACHED")
        mock_resolve.assert_not_called()
        assert result == fake_result
