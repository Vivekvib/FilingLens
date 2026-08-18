"""
config.py — single source of truth for FilingLens.

Nothing in src/ should hardcode a ticker, model name, file path, or XBRL tag.
If a module needs one of those, it imports it from here. This keeps the
pipeline reproducible and makes it obvious where to look when something
needs to change (add a company, swap a model, etc.).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Loads .env into the process environment. Without this call, everything
# below silently reads empty strings via os.environ.get() — this was
# missing in the original scaffold, which is exactly why the Groq calls
# failed with a connection error even after the key was set in .env.
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RAW_FILINGS_DIR = BASE_DIR / "data" / "raw_filings"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"

for _dir in (RAW_FILINGS_DIR, PROCESSED_DIR, CHROMA_DB_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── SEC EDGAR ──────────────────────────────────────────────────────────────
# SEC requires a descriptive User-Agent on every request (name + contact
# email) or it will reject/rate-limit you. Set this before anything else —
# it's the #1 thing that silently breaks the ingestion pipeline.
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "FilingLens Student Project (replace-with-your-email@example.com)"
)
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# ── Companies in scope ───────────────────────────────────────────────────
# 8 tickers across 2 sectors, chosen so the risk memos can be compared
# sector-relative, not just company-by-company. CIK (SEC's internal company
# ID, zero-padded to 10 digits) is required for both the submissions and
# companyfacts APIs — look these up once via the EDGAR full-text search UI
# or the company_tickers.json bulk file and paste them in here. The CIKs
# below are filled in from memory of well-known large-cap tickers — verify
# each one against https://www.sec.gov/cgi-bin/browse-edgar before the
# first real pipeline run; a wrong CIK returns an empty filing list rather
# than an obvious error.
COMPANIES = {
    # ticker: {"cik": "0000000000", "name": "...", "sector": "..."}
    "JPM":  {"cik": "0000019617", "name": "JPMorgan Chase & Co.", "sector": "financials"},
    "GS":   {"cik": "0000886982", "name": "Goldman Sachs Group",   "sector": "financials"},
    "PYPL": {"cik": "0001633917", "name": "PayPal Holdings",       "sector": "fintech"},
    "XYZ":  {"cik": "0001512673", "name": "Block, Inc.",           "sector": "fintech"},  # ticker changed from SQ to XYZ, Jan 2025
    "AAPL": {"cik": "0000320193", "name": "Apple Inc.",            "sector": "tech"},
    "MSFT": {"cik": "0000789019", "name": "Microsoft Corp.",       "sector": "tech"},
    "CRM":  {"cik": "0001108524", "name": "Salesforce, Inc.",      "sector": "tech"},
    "ADBE": {"cik": "0000796343", "name": "Adobe Inc.",            "sector": "tech"},
}

# ── Embeddings / vector store ───────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers, local, free
CHROMA_COLLECTION_NAME = "filinglens_chunks"
CHUNK_SIZE_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 100

# ── LLM (Groq API — free tier, no credit card, rate-limited not spend-limited) ─
# Anthropic's API is NOT free (small one-time trial credit, then real money
# per call) — using it would have put a real cost on a "no budget" project.
# Groq's free tier is actually free, permanently: no credits system, gated
# by request-rate limits (30 req/min, ~14,400 req/day as of mid-2026 — verify
# current numbers at console.groq.com before a real run, limits do change).
# It serves open-weight models (Llama, etc.) via an OpenAI-compatible API.
# Sign up at https://console.groq.com (email or Google, no card) and create
# a key under API Keys.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Model history, for whoever reads this next:
#   1. llama-3.3-70b-versatile (original) -> hit its 100K tokens/day free
#      cap mid-build.
#   2. llama-3.1-8b-instant -> worked for months, then Groq deprecated it
#      (along with llama-3.3-70b-versatile) on 08/16/26, announced via
#      email 06/17/26. Requests started failing with
#      "model_decommissioned" errors.
#   3. A stopgap attempt at llama3-70b-8192 (no dots, old naming) also
#      failed — that model was ALREADY deprecated back on 08/30/25, over
#      a year earlier; it was never a valid fallback.
#   4. openai/gpt-oss-20b — Groq's official recommended replacement for
#      llama-3.1-8b-instant (see console.groq.com/docs/deprecations).
#      Supports response_format={"type": "json_object"}, the same JSON
#      mode already used in risk_memo_generator.py and
#      consistency_checker.py — no other code changes needed.
# Lesson: hardcoding a specific model ID is inherently a maintenance
# liability on a platform that actively deprecates models. If this
# breaks again, check console.groq.com/docs/deprecations first — the
# error message itself also names the exact recommended replacement.
LLM_MODEL_SYNTHESIS = "openai/gpt-oss-20b"

# Chunk/output budgets, sized conservatively (well under even the
# per-minute cap reported for this model, not just the daily one) so the
# full 8-company run — memo + consistency check per company — has real
# margin rather than a hairline fit. Centralized here, not hardcoded in
# each RAG module, so the whole retrieval budget can be tuned in one place
# if limits change again.
RISK_MEMO_CHUNKS_PER_SECTION = 2  # was 4, then originally 8 — trimmed further for the smaller model's tighter per-minute cap
RISK_MEMO_MAX_OUTPUT_TOKENS = 700  # was 900, then 1500
CONSISTENCY_CHECK_MDA_CHUNKS = 4  # was 8 — same reasoning
QUERY_ENDPOINT_CHUNKS = 6  # ad-hoc /query calls — already small, left as-is

# ── Target 10-K sections ────────────────────────────────────────────────
TARGET_SECTIONS = {
    "risk_factors": "Item 1A",
    "mda": "Item 7",
}
# Minimum word count for a regex-sliced section to be considered "clean".
# Below this, the chunker falls back to whole-document semantic retrieval
# for that section (see src/processing/chunker.py) instead of dropping the
# company from the pipeline.
MIN_SECTION_WORD_COUNT = 500

# ── XBRL tag resolution ─────────────────────────────────────────────────
# US-GAAP taxonomy tag names for the same underlying metric vary across
# filers. We try each candidate in order and take the first that resolves
# for a given company, rather than hardcoding one tag name per company.
XBRL_TAG_CANDIDATES = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
}

# Ticker-specific overrides for the rare company where none of the ordered
# candidates above resolve cleanly. Fill in only as needed — most tickers
# should resolve from XBRL_TAG_CANDIDATES alone. Format:
# XBRL_TAG_OVERRIDES = {"TICKER": {"revenue": "SomeUnusualTagName"}}
XBRL_TAG_OVERRIDES: dict[str, dict[str, str]] = {}

# How many fiscal years of XBRL history to pull for YoY comparisons.
XBRL_FISCAL_YEARS = 3
