"""
test_api.py — tests the FastAPI layer itself, specifically that
POST /analyze correctly maps ondemand_pipeline's exceptions to the right
HTTP status codes. analyze_company is mocked here (patched at the
ondemand_pipeline module level, which the endpoint's lazy import picks
up correctly) — this test is about the API WIRING, not the pipeline
logic itself (covered separately in test_ondemand_pipeline.py).
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.api.main import app
from src.ondemand.ondemand_pipeline import RateLimitExceeded, TickerNotFound

client = TestClient(app)


def test_analyze_success_returns_200():
    fake_result = {
        "ticker": "TESTCO",
        "company_info": {"ticker": "TESTCO", "cik": "1", "name": "Test Co", "sector": "on-demand"},
        "risk_memo": {"ticker": "TESTCO", "top_risks": [], "management_tone": "", "notable_flags": []},
        "consistency": {"flags": [], "assessment": "ok", "computed_ratios": {}},
    }
    with patch("src.ondemand.ondemand_pipeline.analyze_company", return_value=fake_result):
        res = client.post("/analyze", json={"ticker": "TESTCO"})
    assert res.status_code == 200
    assert res.json()["ticker"] == "TESTCO"


def test_analyze_unknown_ticker_returns_404():
    with patch("src.ondemand.ondemand_pipeline.analyze_company", side_effect=TickerNotFound("not found")):
        res = client.post("/analyze", json={"ticker": "NOTREAL"})
    assert res.status_code == 404
    assert "not found" in res.json()["detail"]


def test_analyze_rate_limited_returns_429():
    with patch("src.ondemand.ondemand_pipeline.analyze_company", side_effect=RateLimitExceeded("too many")):
        res = client.post("/analyze", json={"ticker": "TESTCO"})
    assert res.status_code == 429
    assert "too many" in res.json()["detail"]


def test_companies_endpoint_still_works():
    """Sanity check that adding /analyze didn't break the existing
    cached-data endpoints."""
    res = client.get("/companies")
    assert res.status_code == 200
    assert len(res.json()) == 8
