"""
api/main.py — FastAPI app. /companies, /risk-memo/{ticker}, and
/consistency/{ticker} serve PRE-COMPUTED, cached results (written by
scripts/run_pipeline.py) for the 8 pre-baked companies. /analyze runs a
live, on-demand pipeline for ANY other ticker (see
src/ondemand/ondemand_pipeline.py). /query is live ad-hoc Q&A against one
of the 8 pre-baked companies specifically.

Run with: uvicorn src.api.main:app --reload
Swagger docs at /docs once running — useful for a live demo even before
the frontend exists.
"""

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import COMPANIES, PROCESSED_DIR
from src.rag.retriever import retrieve

app = FastAPI(title="FilingLens API")

# Wide-open CORS for local dev / demo purposes only — tighten before any
# real deployment (restrict to your actual frontend origin).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    ticker: str
    question: str


class AnalyzeRequest(BaseModel):
    ticker: str


@app.get("/companies")
def list_companies():
    return [{"ticker": t, **info} for t, info in COMPANIES.items()]


@app.get("/risk-memo/{ticker}")
def get_risk_memo(ticker: str):
    """Serves the cached memo written during the pipeline run. Returns 404
    (not a generated-on-the-fly fallback) if the pipeline hasn't been run
    for this ticker yet — this endpoint should never silently trigger an
    expensive live LLM call."""
    ticker = ticker.upper()
    memo_path = PROCESSED_DIR / f"{ticker}_risk_memo.json"
    if not memo_path.exists():
        raise HTTPException(status_code=404, detail=f"No cached risk memo for {ticker} — run the pipeline first.")
    return json.loads(memo_path.read_text(encoding="utf-8"))


@app.get("/consistency/{ticker}")
def get_consistency(ticker: str):
    """Serves the cached narrative-vs-numbers consistency check for a
    company — same cached-only contract as /risk-memo. Each flag in the
    response includes a "grounded" boolean (set by
    consistency_checker._enforce_grounding) alongside its confidence
    level — the frontend should treat ungrounded flags as visually
    distinct (e.g. a lighter/greyed badge) rather than showing every flag
    with equal visual weight."""
    ticker = ticker.upper()
    consistency_path = PROCESSED_DIR / f"{ticker}_consistency.json"
    if not consistency_path.exists():
        raise HTTPException(
            status_code=404, detail=f"No cached consistency check for {ticker} — run the pipeline first."
        )
    return json.loads(consistency_path.read_text(encoding="utf-8"))


@app.post("/analyze")
def analyze_on_demand(req: AnalyzeRequest):
    """Live end-to-end analysis for any ticker NOT in the 8 pre-baked
    COMPANIES — fetches the filing, computes ratios, and generates a
    memo + consistency check on the spot, using TF-IDF retrieval instead
    of the semantic Chroma pipeline (see src/ondemand/tfidf_retriever.py
    for why). Takes ~10-20s per new ticker; cached in-memory after the
    first request. This is the endpoint that makes the live deployment
    genuinely interactive, not just a viewer for 8 fixed companies."""
    from src.ondemand.ondemand_pipeline import analyze_company, RateLimitExceeded, TickerNotFound

    try:
        return analyze_company(req.ticker)
    except TickerNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Live analysis isn't available in this deployment (required libraries aren't installed here).",
        )


@app.post("/query")
def ad_hoc_query(req: QueryRequest):
    """Live retrieval + generation for a user's follow-up question against
    one company's filing. This is the one endpoint allowed to call the LLM
    on demand, and the only one that needs the embedding/vector-store
    stack (sentence-transformers, chromadb) — which is deliberately NOT
    installed on the Render deployment (see requirements-render.txt) since
    it's a multi-GB dependency (PyTorch bundles the full CUDA toolkit by
    default, unneeded on a CPU-only free-tier instance) that isn't worth
    the memory footprint for the other 3 endpoints. If those libraries
    aren't installed, this degrades to a clear 503 rather than crashing
    the whole process — /companies, /risk-memo, and /consistency keep
    working regardless."""
    ticker = req.ticker.upper()
    if ticker not in COMPANIES:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")

    from config import QUERY_ENDPOINT_CHUNKS

    try:
        chunks = retrieve(req.question, ticker=ticker, top_k=QUERY_ENDPOINT_CHUNKS)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Live Q&A isn't available in this deployment (embedding libraries aren't installed here). "
            "Run this locally with the full requirements.txt to use /query, or use the cached risk memo "
            "and consistency check instead.",
        )
    if not chunks:
        return {"answer": "No relevant filing excerpts found for that question.", "sources": []}

    # Reuses the same synthesis model/pattern as the risk memo generator —
    # simple grounded Q&A doesn't need its own separate prompt module.
    from groq import Groq

    from config import GROQ_API_KEY, LLM_MODEL_SYNTHESIS

    client = Groq(api_key=GROQ_API_KEY)
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = (
        f"Answer the question below using ONLY the filing excerpts provided. "
        f"If the excerpts don't contain the answer, say so.\n\n"
        f"EXCERPTS:\n{context}\n\nQUESTION: {req.question}"
    )
    response = client.chat.completions.create(
        model=LLM_MODEL_SYNTHESIS, max_tokens=600, messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()

    return {"answer": answer, "sources": [c["metadata"] for c in chunks]}
