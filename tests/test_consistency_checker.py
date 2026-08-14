"""
test_consistency_checker.py — unit tests for compute_ratios(), the
deterministic (no-LLM) math at the core of the consistency checker. This
is the one part of the pipeline that must be verifiably correct, since a
wrong ratio silently discredits a "flag" the LLM raises on top of it.

Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.consistency_checker import compute_ratios, _flag_is_grounded, _enforce_grounding


def test_revenue_growth_basic():
    facts = {
        "revenue": [
            {"fy": 2025, "end": "2025-12-31", "val": 1100},
            {"fy": 2024, "end": "2024-12-31", "val": 1000},
        ]
    }
    ratios = compute_ratios(facts)
    assert ratios["revenue_growth_pct"] == 10.0


def test_revenue_growth_negative():
    facts = {
        "revenue": [
            {"fy": 2025, "end": "2025-12-31", "val": 800},
            {"fy": 2024, "end": "2024-12-31", "val": 1000},
        ]
    }
    ratios = compute_ratios(facts)
    assert ratios["revenue_growth_pct"] == -20.0


def test_missing_metric_is_skipped_not_zero():
    facts = {"revenue": [{"fy": 2025, "end": "2025-12-31", "val": 1000}]}  # only one year
    ratios = compute_ratios(facts)
    assert "revenue_growth_pct" not in ratios  # can't compute YoY with 1 data point — must not fake a 0


def test_gross_margin_delta():
    facts = {
        "revenue": [
            {"fy": 2025, "end": "2025-12-31", "val": 1000},
            {"fy": 2024, "end": "2024-12-31", "val": 1000},
        ],
        "gross_profit": [
            {"fy": 2025, "end": "2025-12-31", "val": 400},
            {"fy": 2024, "end": "2024-12-31", "val": 500},
        ],
    }
    ratios = compute_ratios(facts)
    # margin went from 50% to 40% => -10.0 pts
    assert ratios["gross_margin_delta_pts"] == -10.0


def test_operating_income_growth():
    facts = {
        "operating_income": [
            {"fy": 2025, "end": "2025-12-31", "val": 900},
            {"fy": 2024, "end": "2024-12-31", "val": 1000},
        ]
    }
    ratios = compute_ratios(facts)
    assert ratios["operating_income_growth_pct"] == -10.0


def test_cost_of_revenue_growth():
    facts = {
        "cost_of_revenue": [
            {"fy": 2025, "end": "2025-12-31", "val": 550},
            {"fy": 2024, "end": "2024-12-31", "val": 500},
        ]
    }
    ratios = compute_ratios(facts)
    assert ratios["cost_of_revenue_growth_pct"] == 10.0


def test_empty_facts_returns_empty_ratios():
    assert compute_ratios({}) == {}


def test_flag_grounded_when_it_names_a_real_ratio_key():
    ratios = {"net_income_growth_pct": 20.31}
    flag = {"claim": "...", "ratio_evidence": "net_income_growth_pct: 20.31 (an increase)", "confidence": "high"}
    assert _flag_is_grounded(flag, ratios) is True


def test_flag_not_grounded_when_no_real_ratio_is_named():
    # Modeled on the real GS output that surfaced this bug: the model
    # flagged an "investment banking revenues increased 21%" claim as
    # high confidence with no actual computed ratio for that metric —
    # only net_income_growth_pct existed in computed_ratios for GS.
    ratios = {"net_income_growth_pct": 20.31}
    flag = {
        "claim": "Investment banking revenues increased by 21%",
        "ratio_evidence": "inaccurate statement given the MD&A's narrative of increase by comparison to 2023",
        "confidence": "high",
    }
    assert _flag_is_grounded(flag, ratios) is False


def test_enforce_grounding_downgrades_ungrounded_high_confidence_flag():
    ratios = {"net_income_growth_pct": 20.31}
    result = {
        "flags": [
            {"claim": "grounded claim", "ratio_evidence": "net_income_growth_pct: 20.31", "confidence": "high"},
            {"claim": "ungrounded claim", "ratio_evidence": "vague justification, no real ratio named", "confidence": "high"},
        ]
    }
    fixed = _enforce_grounding(result, ratios)
    assert fixed["flags"][0]["confidence"] == "high"
    assert fixed["flags"][0]["grounded"] is True
    assert fixed["flags"][1]["confidence"] == "low"  # downgraded — was falsely high confidence
    assert fixed["flags"][1]["grounded"] is False


def test_enforce_grounding_ignores_the_by_year_helper_key():
    # gross_margin_by_year is a list, not a comparable metric name — must
    # not be treated as a valid "ratio key" a flag can cite.
    ratios = {"gross_margin_delta_pts": 5.97, "gross_margin_by_year": [{"fy": 2025, "margin_pct": 42.82}]}
    flag = {"claim": "...", "ratio_evidence": "gross_margin_by_year mentions 42.82", "confidence": "high"}
    assert _flag_is_grounded(flag, ratios) is False
