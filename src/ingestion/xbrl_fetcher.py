"""
xbrl_fetcher.py — pulls structured financial facts (revenue, net income,
gross profit, etc.) per company from SEC's XBRL companyfacts API, resolving
each metric's tag name from an ordered candidate list (see
config.XBRL_TAG_CANDIDATES) rather than assuming one tag name fits every
filer.

This is the deterministic-numbers half of the pipeline. Nothing in here
touches an LLM — financial arithmetic should never be delegated to a model.
"""

import json
import time

import requests

from config import (
    COMPANIES,
    PROCESSED_DIR,
    SEC_COMPANYFACTS_URL,
    SEC_USER_AGENT,
    XBRL_TAG_CANDIDATES,
    XBRL_TAG_OVERRIDES,
    XBRL_FISCAL_YEARS,
)

HEADERS = {"User-Agent": SEC_USER_AGENT}


def _resolve_tag(facts_json: dict, metric: str, ticker: str) -> str | None:
    """Return the first XBRL tag (from config candidates, or a ticker
    override) that actually has data in this company's facts JSON."""
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})

    override = XBRL_TAG_OVERRIDES.get(ticker, {}).get(metric)
    if override and override in us_gaap:
        return override

    for candidate in XBRL_TAG_CANDIDATES.get(metric, []):
        if candidate in us_gaap:
            return candidate
    return None


def _extract_annual_values(facts_json: dict, tag: str) -> list[dict]:
    """Pull annual (10-K, form='10-K', fp='FY') USD values for a tag,
    most recent first."""
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
    tag_data = us_gaap.get(tag, {}).get("units", {}).get("USD", [])

    annual = [
        {"fy": entry.get("fy"), "end": entry.get("end"), "val": entry.get("val")}
        for entry in tag_data
        if entry.get("form") == "10-K" and entry.get("fp") == "FY"
    ]
    # de-dupe by fiscal year (companies sometimes restate — keep the latest filed value)
    by_fy = {a["fy"]: a for a in sorted(annual, key=lambda a: a.get("end", ""))}
    return sorted(by_fy.values(), key=lambda a: a["fy"], reverse=True)[:XBRL_FISCAL_YEARS]


def fetch_company_facts(ticker: str, cik: str) -> dict:
    """Fetch + resolve all configured metrics for one company. Returns a
    dict of {metric: [{"fy":..., "end":..., "val":...}, ...]} plus a
    "_unresolved" list flagging any metric that couldn't be matched to a
    tag, so downstream code knows to skip it rather than silently treating
    a missing metric as zero."""
    url = SEC_COMPANYFACTS_URL.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    facts_json = resp.json()

    result, unresolved = {}, []
    for metric in XBRL_TAG_CANDIDATES:
        tag = _resolve_tag(facts_json, metric, ticker)
        if tag is None:
            unresolved.append(metric)
            continue
        result[metric] = _extract_annual_values(facts_json, tag)

    result["_unresolved"] = unresolved
    result["_ticker"] = ticker
    return result


def fetch_all(force: bool = False) -> dict:
    results = {}
    for ticker, info in COMPANIES.items():
        out_path = PROCESSED_DIR / f"{ticker}_xbrl.json"
        if out_path.exists() and not force:
            results[ticker] = "cached"
            continue
        try:
            facts = fetch_company_facts(ticker, info["cik"])
            out_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
            status = "ok" if not facts["_unresolved"] else f"ok (unresolved: {facts['_unresolved']})"
            results[ticker] = status
        except requests.HTTPError as e:
            results[ticker] = f"http_error: {e}"
        except Exception as e:  # noqa: BLE001
            results[ticker] = f"error: {e}"
        time.sleep(0.15)
    return results


if __name__ == "__main__":
    summary = fetch_all()
    for ticker, status in summary.items():
        print(f"{ticker}: {status}")
