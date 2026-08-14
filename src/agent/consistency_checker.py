"""
consistency_checker.py — the project's core differentiator. Compares what
management SAYS in MD&A against what the XBRL numbers actually SHOW, and
flags specific inconsistencies.

Design decision (see handoff brief): all ratio math is pure deterministic
Python — the LLM never computes numbers, only reasons over numbers we hand
it. This is a single-pass synthesis call, not a multi-turn tool-calling
agent — since the whole pipeline runs offline/cached (see
scripts/run_pipeline.py), there's no latency or freshness reason for the
LLM to "go fetch" the XBRL data itself; handing it pre-computed ratios is
simpler, cheaper, and easier to unit-test than a multi-turn loop.
"""

import json

# NOTE: groq + src.rag.retriever (which pulls in chromadb via the embedder)
# are imported lazily inside check_consistency(), not at module level.
# compute_ratios() below is pure stdlib and must stay importable — and
# unit-testable — without the ML/LLM stack installed. Learned this the hard
# way: a top-level `from src.rag.retriever import retrieve` made the whole
# test suite fail to even collect when chromadb wasn't installed.


def compute_ratios(xbrl_facts: dict) -> dict:
    """Pure math, no LLM involved. Takes the per-metric annual value lists
    from xbrl_fetcher.fetch_company_facts and returns YoY deltas for the
    two most recent fiscal years available. Missing/unresolved metrics are
    skipped, not treated as zero."""
    ratios = {}

    def yoy_pct(values: list[dict]) -> float | None:
        if len(values) < 2 or values[0]["val"] is None or values[1]["val"] is None:
            return None
        prev, curr = values[1]["val"], values[0]["val"]
        if prev == 0:
            return None
        return round((curr - prev) / abs(prev) * 100, 2)

    if "revenue" in xbrl_facts:
        val = yoy_pct(xbrl_facts["revenue"])
        if val is not None:
            ratios["revenue_growth_pct"] = val

    if "revenue" in xbrl_facts and "gross_profit" in xbrl_facts:
        margins = []
        for rev, gp in zip(xbrl_facts["revenue"], xbrl_facts["gross_profit"]):
            if rev.get("val") and gp.get("val") is not None:
                margins.append({"fy": rev["fy"], "margin_pct": round(gp["val"] / rev["val"] * 100, 2)})
        if len(margins) >= 2:
            ratios["gross_margin_delta_pts"] = round(margins[0]["margin_pct"] - margins[1]["margin_pct"], 2)
            ratios["gross_margin_by_year"] = margins

    if "net_income" in xbrl_facts:
        val = yoy_pct(xbrl_facts["net_income"])
        if val is not None:
            ratios["net_income_growth_pct"] = val

    if "operating_income" in xbrl_facts:
        val = yoy_pct(xbrl_facts["operating_income"])
        if val is not None:
            ratios["operating_income_growth_pct"] = val

    if "cost_of_revenue" in xbrl_facts:
        val = yoy_pct(xbrl_facts["cost_of_revenue"])
        if val is not None:
            ratios["cost_of_revenue_growth_pct"] = val

    return ratios


CONSISTENCY_PROMPT = """You are a skeptical equity research analyst fact-checking a company's own MD&A commentary against its actual reported financials.

COMPUTED FINANCIAL RATIOS (ground truth, computed directly from SEC XBRL data):
{ratios_json}

MD&A EXCERPTS (management's own narrative from the 10-K):
{mda_text}

Identify any specific claims in the MD&A excerpts that conflict with, overstate, or understate what the COMPUTED FINANCIAL RATIOS above actually show.

STRICT RULES — follow these exactly:
1. Only compare claims against the numbers in COMPUTED FINANCIAL RATIOS above. If the MD&A mentions a metric (e.g. operating expenses, cost of revenue) that has NO matching entry in COMPUTED FINANCIAL RATIOS, do not flag it and do not estimate or compute a percentage for it yourself — skip it.
2. Before writing each flag, explicitly check the sign and direction of the computed ratio: a positive percentage is an INCREASE, a negative percentage is a DECREASE. Re-read the ratio's actual number before deciding whether it supports or contradicts the MD&A claim — do not state a direction without checking the sign first.
3. If nothing in the excerpts conflicts with the provided ratios, say so explicitly in "assessment" rather than inventing a flag.

Return ONLY a JSON object with this exact structure:
{{
  "flags": [
    {{"claim": "short paraphrase of what MD&A claims", "ratio_evidence": "the exact computed ratio value and metric name being compared, e.g. 'net_income_growth_pct: -54.93 (a decrease)'", "confidence": "high|medium|low"}}
  ],
  "assessment": "one sentence overall verdict on narrative-vs-numbers alignment"
}}"""


def _flag_is_grounded(flag: dict, ratios: dict) -> bool:
    """A flag counts as grounded only if its ratio_evidence text actually
    names one of the real computed ratio keys we handed the model — not
    just a plausible-sounding justification. We don't trust the model's
    own confidence label to reflect whether it truly cited real data; we
    check it here in code instead. The tightened prompt explicitly asks
    the model to name the exact ratio key (e.g. "net_income_growth_pct:
    -54.93"), so a genuinely grounded flag should pass this easily."""
    evidence = flag.get("ratio_evidence", "").lower()
    ratio_keys = [k for k in ratios if not k.endswith("_by_year")]
    return any(key.lower() in evidence for key in ratio_keys)


def _enforce_grounding(result: dict, ratios: dict) -> dict:
    """Downgrades any flag whose evidence doesn't name a real computed
    ratio to low confidence and tags it ungrounded, rather than shipping
    the model's self-reported confidence label at face value. Mutates and
    returns the same flags list."""
    for flag in result.get("flags", []):
        grounded = _flag_is_grounded(flag, ratios)
        flag["grounded"] = grounded
        if not grounded:
            flag["confidence"] = "low"
    return result


def check_consistency(ticker: str, xbrl_facts: dict) -> dict:
    from groq import Groq

    from config import GROQ_API_KEY, LLM_MODEL_SYNTHESIS, CONSISTENCY_CHECK_MDA_CHUNKS
    from src.rag.retriever import retrieve

    ratios = compute_ratios(xbrl_facts)
    if not ratios:
        return {"flags": [], "assessment": "Insufficient XBRL data to compute ratios for this company."}

    client = Groq(api_key=GROQ_API_KEY)

    mda_chunks = retrieve(
        "financial performance, margins, growth, and outlook commentary",
        ticker=ticker,
        section="mda",
        top_k=CONSISTENCY_CHECK_MDA_CHUNKS,
    )
    mda_text = "\n\n".join(c["text"] for c in mda_chunks) if mda_chunks else "(no MD&A excerpts retrieved)"

    prompt = CONSISTENCY_PROMPT.format(ratios_json=json.dumps(ratios, indent=2), mda_text=mda_text)

    response = client.chat.completions.create(
        model=LLM_MODEL_SYNTHESIS,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content.strip()

    result = json.loads(raw_text)
    result = _enforce_grounding(result, ratios)  # deterministic check, not another LLM call
    result["computed_ratios"] = ratios  # always ship the ground-truth numbers alongside the LLM's read of them
    return result
