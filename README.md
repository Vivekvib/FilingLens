# FilingLens

Agentic RAG system over real SEC 10-K filings. Retrieves grounded chunks
from Risk Factors / MD&A sections to generate a structured risk memo per
company, and cross-checks management's narrative claims in MD&A against
actual XBRL-reported financials to flag inconsistencies — each flag is
deterministically checked against the real computed ratios before being
labeled "verified," not just trusted at the model's own word.

> **Status:** fully working end-to-end across 8 companies (JPM, GS, PYPL,
> XYZ/Block, AAPL, MSFT, CRM, ADBE) — ingestion, XBRL, RAG, risk memos,
> consistency checks, FastAPI backend, and frontend all built and verified
> against real output, not just passing tests.

## Quickstart (local)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env       # then fill in GROQ_API_KEY (free at console.groq.com) and SEC_USER_AGENT

python scripts/run_pipeline.py     # fetches filings, builds the index, generates memos
uvicorn src.api.main:app --reload  # serves the API at http://127.0.0.1:8000/docs
```

Then open `frontend/index.html` directly in a browser (no server needed
for the frontend — it's a static file that fetches from the API above).

## Why SEC EDGAR instead of NSE/BSE

SEC EDGAR exposes filings *and* structured financial data (XBRL) through
free public APIs — that structured layer is what makes the
narrative-vs-numbers consistency check possible without brittle PDF table
parsing. NSE/BSE annual reports don't have an equivalent public structured
API, so extending to Indian filings is noted as future work rather than
in scope for this build.

## Why Groq instead of a paid API

The synthesis LLM is Groq's free tier (`llama-3.1-8b-instant`), not a
paid API — this was a deliberate zero-budget constraint, not a fallback.
Trade-off: less nuanced reasoning than a larger model, which is exactly
why the consistency checker treats the LLM's own confidence label as
unreliable and enforces a deterministic grounding check in code (see
`_enforce_grounding` in `src/agent/consistency_checker.py`) rather than
trusting it at face value.

## Architecture

See `BUILD_PLAN.md` for the full build reasoning. Short version:

```
SEC EDGAR (10-K text)  ─┐
                         ├─> chunker (section-aware, regex + semantic fallback)
SEC XBRL (companyfacts) ┘         │
        │                          ├─> embedder (sentence-transformers -> Chroma)
        │                          │
        ├─> compute_ratios()       ├─> retriever (scoped semantic search)
        │   (pure Python, no LLM)  │
        │                          ├─> risk_memo_generator (Groq)
        └──────────────┬───────────┘
                        └─> consistency_checker (ratios + narrative -> Groq
                            -> _enforce_grounding, a deterministic check)
                                    │
                                    v
        FastAPI: /companies  /risk-memo/{ticker}
                 /consistency/{ticker}  /query
                                    │
                                    v
                    frontend (case-file docket + memo UI)
```

## Known limitations

- **Bank XBRL coverage is thin.** Goldman Sachs resolved only
  `net_income_growth_pct` — banks report under different XBRL concepts
  (interest income rather than "Revenues"), so the consistency checker has
  less ground truth to work with for financial-sector filings. The
  deterministic grounding check correctly downgrades ungrounded claims to
  low confidence rather than forcing false precision, but a bank-specific
  XBRL tag mapping would close this gap — noted as future work rather
  than fixed under the build deadline.
- **`/query` is the one memory-heavy endpoint.** It's the only code path
  that loads the sentence-transformers embedding model (lazily, on first
  call) — `/companies`, `/risk-memo`, and `/consistency` only read cached
  JSON and stay lightweight. On the deployed Render instance, `/query` is
  deliberately disabled in production (see Deployment section below) —
  PyTorch's default wheel bundles the full CUDA toolkit, several GB
  unneeded on a CPU-only free tier, and installing it was OOM-killing the
  whole service before it could serve anything. Trading one endpoint for
  reliability on the other three was the right call under the deadline;
  `/query` works fully when run locally.

## Deployment

Backend on Render, frontend on Vercel — same pattern as two other shipped
projects (a Flask/Postgres marketplace app and a quant risk dashboard).

**Backend (Render):**
1. Push this repo to GitHub. `data/processed/` and `data/chroma_db/` are
   committed on purpose (not gitignored) — Render serves cached memos and
   the vector store straight from these files rather than needing to
   re-run the full SEC+Groq pipeline on the server.
2. New Render Web Service, connect the repo.
3. **Build command: `pip install -r requirements-render.txt`** (not the
   root `requirements.txt`). The full requirements file installs
   sentence-transformers/chromadb/torch, and PyTorch's default Linux
   wheel bundles the entire CUDA toolkit (multiple GB) even though
   there's no GPU on a free-tier instance — that was OOM-killing the
   deploy before it ever bound a port. `/companies`, `/risk-memo`, and
   `/consistency` never touch that stack anyway (they only read cached
   JSON), so `requirements-render.txt` deliberately excludes it.
4. Start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
5. Environment variables: `GROQ_API_KEY`, `SEC_USER_AGENT`.
6. Free tier spins down after inactivity — first request after idle can
   take 30–60s to wake up, same as the other two projects.
7. **Known trade-off:** with the slim requirements, `/query` (live ad-hoc
   Q&A) returns a `503` in this deployment instead of an answer — it's
   the one endpoint that needs the embedding stack. `/companies`,
   `/risk-memo`, and `/consistency` — the core of the demo — work fully.
   Run locally with the root `requirements.txt` for working `/query`.

**Frontend (Vercel):**
1. New Vercel project, same repo, set **Root Directory** to `frontend`.
2. No build step needed — it's a static HTML file.
3. After the Render backend URL is live, update `RENDER_API_URL` near the
   top of `frontend/index.html`'s `<script>` block and redeploy.

## License

Student project — built for internship/MBA application purposes.
