"""
risk_memo_generator.py — takes retrieved Risk Factors + MD&A chunks for a
company and asks Claude (LLM_MODEL_SYNTHESIS) to produce a structured risk
memo. This is the one place in the pipeline where the LLM does synthesis
rather than lookup — everything it's given is retrieved text, and the
prompt explicitly asks it to ground every point in that text so we're not
generating from parametric memory of the company.
"""

import json

from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL_SYNTHESIS, RISK_MEMO_CHUNKS_PER_SECTION, RISK_MEMO_MAX_OUTPUT_TOKENS
from src.rag.retriever import retrieve

client = Groq(api_key=GROQ_API_KEY)

RISK_MEMO_PROMPT = """You are a credit/equity risk analyst producing a concise internal risk memo for {ticker}.

Below are excerpts retrieved from the company's most recent 10-K filing (Risk Factors and MD&A sections). Base your memo ONLY on this text — do not use outside knowledge of the company.

--- RISK FACTORS EXCERPTS ---
{risk_factors_text}

--- MD&A EXCERPTS ---
{mda_text}

Produce a JSON object with this exact structure:
{{
  "ticker": "{ticker}",
  "top_risks": [
    {{"risk": "short label", "summary": "1-2 sentence summary grounded in the excerpts", "severity": "high|medium|low"}}
  ],
  "management_tone": "one sentence characterizing how management frames the company's outlook in MD&A",
  "notable_flags": ["any risk the company discloses that seems unusually specific or severe, if present"]
}}

List at most 5 top_risks, ordered by severity. Return ONLY the JSON object, no other text."""


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "(no relevant excerpts retrieved)"
    return "\n\n".join(c["text"] for c in chunks)


def generate_risk_memo(ticker: str) -> dict:
    """Retrieves top chunks for risk_factors and mda sections, then asks
    Claude to synthesize a structured memo. Returns a parsed dict — raises
    if the model doesn't return valid JSON, on purpose, so a bad memo
    fails loudly during the pipeline run rather than silently shipping."""
    risk_chunks = retrieve(
        "key business and operational risks", ticker=ticker, section="risk_factors",
        top_k=RISK_MEMO_CHUNKS_PER_SECTION,
    )
    mda_chunks = retrieve(
        "financial performance trends and outlook", ticker=ticker, section="mda",
        top_k=RISK_MEMO_CHUNKS_PER_SECTION,
    )

    prompt = RISK_MEMO_PROMPT.format(
        ticker=ticker,
        risk_factors_text=_format_chunks(risk_chunks),
        mda_text=_format_chunks(mda_chunks),
    )

    response = client.chat.completions.create(
        model=LLM_MODEL_SYNTHESIS,
        max_tokens=RISK_MEMO_MAX_OUTPUT_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},  # Groq's JSON mode — avoids code-fence wrapping entirely
    )

    raw_text = response.choices[0].message.content.strip()

    return json.loads(raw_text)
