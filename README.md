# 🔍 FilingLens

[![Live Demo](https://img.shields.io/badge/Live%20Demo-filing--lens--ruby.vercel.app-14171F?style=for-the-badge)](https://filing-lens-ruby.vercel.app/)
[![API Docs](https://img.shields.io/badge/API%20Docs-filinglens--2wqv.onrender.com-2F6F62?style=for-the-badge)](https://filinglens-2wqv.onrender.com/docs)

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat&logo=fastapi&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM-F55036?style=flat&logo=groq&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-6E56CF?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Render](https://img.shields.io/badge/Backend-Render-46E3B7?style=flat&logo=render&logoColor=white)
![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?style=flat&logo=vercel&logoColor=white)

**Agentic RAG system over real SEC 10-K filings** — retrieves grounded
text from Risk Factors / MD&A sections to generate a structured risk
memo per company, then cross-checks management's narrative claims
against actual XBRL-reported financials to flag inconsistencies. Every
flag is checked in code against real computed ratios before being
labeled "verified" — never just trusted at the model's own word. A live
`/analyze` endpoint runs this whole pipeline on demand for any
US-listed ticker, not just a fixed pre-baked list.

> **Status:** fully working end-to-end — 8 pre-baked companies (JPM, GS,
> PYPL, XYZ/Block, AAPL, MSFT, CRM, ADBE) plus live on-demand analysis
> for any other ticker. Backend on Render, frontend on Vercel, both
> live. This is an early-stage prototype, built and iterated on a **$0
> infrastructure budget** — see "Known limitations" and "Roadmap" below
> for the honest trade-offs that come with that, and what a small
> hosting budget would unlock.

## What it actually does

- **Reads real SEC 10-K filings** — not summaries, not press releases —
  pulled live from SEC EDGAR's public API.
- **Generates a structured risk memo per company** — top risks ranked
  by severity, management tone, notable red flags — grounded only in
  retrieved filing text, not the model's general knowledge.
- **Independently verifies the narrative against the numbers.** A
  deterministic (non-LLM) layer computes real financial ratios from SEC
  XBRL data, and every "inconsistency" flag the model raises is checked
  against those real numbers before it's allowed to say "verified."
  Ungrounded claims get caught and downgraded automatically — see
  `_enforce_grounding` in `src/agent/consistency_checker.py`.
- **Works live for any company, not just a fixed list.** Type any
  US-listed ticker into the "Analyze a Company" tab and get a real
  memo generated in ~15 seconds, end to end.
- **Cross-company Portfolio View** — revenue growth, net income growth,
  and narrative-reliability comparisons across the whole docket on one
  screen.

## Screenshots

<img width="1889" height="872" alt="image" src="https://github.com/user-attachments/assets/02b486bb-e7a4-4f18-8969-29b8246baf21" />
<img width="1907" height="658" alt="image" src="https://github.com/user-attachments/assets/baabfa27-5b90-4548-9a8f-0c623ef5c786" />


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

## Tech stack

| Layer | Technology |
|---|---|
| Data sources | SEC EDGAR (10-K full text), SEC XBRL `companyfacts` API |
| Retrieval (pre-baked companies) | sentence-transformers embeddings + ChromaDB |
| Retrieval (on-demand companies) | scikit-learn TF-IDF, in-memory, per request |
| LLM | Groq (`llama-3.1-8b-instant`), free tier |
| Backend | Python, FastAPI |
| Frontend | Vanilla HTML/CSS/JS — no framework, no build step |
| Testing | pytest, 22 tests covering ratio math, grounding logic, API error mapping, and on-demand pipeline wiring |
| Deployment | Render (backend), Vercel (frontend) — both free tier |

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

## Two retrieval paths: semantic (pre-baked) vs. TF-IDF (on-demand)

The 8 pre-baked companies use a proper semantic search pipeline
(sentence-transformers embeddings + ChromaDB), built and run offline via
`scripts/run_pipeline.py`. The live `/analyze` endpoint — for any other
ticker a user types in — deliberately uses a lighter, keyword-based
TF-IDF index instead (`src/ondemand/tfidf_retriever.py`), built fresh
in-memory per request.

This wasn't a shortcut taken by accident — it's a direct consequence of
the same OOM lesson documented below: PyTorch's default wheel bundles
the entire CUDA toolkit, and that's what crashed the Render deployment
the first time. Keeping the live, public-facing analysis path on
scikit-learn (lightweight, no GPU baggage) instead of reintroducing
torch was the deliberate trade: slightly lower retrieval quality
(keyword matching, not meaning matching) in exchange for a feature that
actually stays up in production, for free. The pre-baked companies never
had to make that trade, since their embeddings are generated once,
offline, and just served as static cached files afterward.

## Architecture

See `BUILD_PLAN.md` for the full build reasoning. Short version:

```
                    8 PRE-BAKED COMPANIES (offline, scripts/run_pipeline.py)
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

                    ANY OTHER TICKER (live, POST /analyze)
ticker_lookup (SEC company_tickers.json) -> edgar_fetcher + xbrl_fetcher
        -> chunker (SAME code as above) -> TfidfRetriever (in-memory, no torch)
        -> risk_memo_generator + consistency_checker (SAME code, retrieve_fn injected)

                                    │
                                    v
        FastAPI: /companies  /risk-memo/{ticker}  /consistency/{ticker}
                 /analyze  /query
                                    │
                                    v
      frontend (case-file docket + memo UI + portfolio comparison + analyze tab)
```

The key design point: `risk_memo_generator.generate_risk_memo()` and
`consistency_checker.check_consistency()` both accept an optional
`retrieve_fn` — the exact same prompt/synthesis logic is reused for both
paths, only the retrieval backend underneath differs. No duplicated
LLM-calling code between the pre-baked and on-demand pipelines.

## Known limitations

- **Bank XBRL coverage is thin.** Goldman Sachs resolved only
  `net_income_growth_pct` — banks report under different XBRL concepts
  (interest income rather than "Revenues"), so the consistency checker has
  less ground truth to work with for financial-sector filings. The
  deterministic grounding check correctly downgrades ungrounded claims to
  low confidence rather than forcing false precision, but a bank-specific
  XBRL tag mapping would close this gap — noted as future work.
- **`/query` (live Q&A) only works for the 8 pre-baked companies, and
  only when running locally.** It's the one code path that needs the
  sentence-transformers embedding stack, which isn't installed on Render
  (see the OOM story below) — it returns a clean `503` in production.
  On-demand companies (via `/analyze`) don't support `/query` at all yet;
  the TF-IDF retriever underneath them isn't currently cached anywhere
  the query endpoint could reach — a natural fast-follow, not a hard
  limitation, since the retriever object could simply be kept alongside
  the cached result.
- **On-demand analysis results are not persistent.** They live in an
  in-memory dict for the life of the running Render process, and Render's
  free-tier disk is ephemeral — a restart or redeploy clears the cache,
  and the next request for that ticker just re-runs the ~15s pipeline.
  Real persistence would need a database. This is exactly the kind of
  thing a small hosting budget would unlock — see below.
- **The live `/analyze` endpoint has a simple global rate limit** (20
  analyses/hour, shared across all visitors, not per-IP) to protect the
  free Groq quota from being exhausted if the link gets shared widely.
  Simple by design for a prototype — a real deployment would want
  per-user limits.

## Roadmap — what a small hosting budget would unlock

This project is deliberately built to run entirely on free tiers, which
shapes several of the trade-offs above. With a modest budget, the
natural next steps are:
- A real database for on-demand analysis results, so they persist across
  restarts and don't need to be regenerated for every visitor.
- Semantic (not just TF-IDF) retrieval for on-demand companies too, by
  hosting the embedding stack on infrastructure with enough memory
  headroom to not OOM.
- `/query` (live ad-hoc Q&A) available in production for all companies,
  not just locally.
- Per-user rate limiting instead of a single global cap.
- Multi-year filing history per company, instead of just the latest 10-K.

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
   deploy before it ever bound a port. `requirements-render.txt` installs
   `scikit-learn` instead (needed for the live `/analyze` endpoint's
   TF-IDF retrieval) — much lighter, no GPU baggage, verified to add only
   ~100MB at actual runtime, not several GB.
4. Start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
5. Environment variables: `GROQ_API_KEY`, `SEC_USER_AGENT`.
6. Free tier spins down after inactivity — first request after idle can
   take 30–60s to wake up, same as the other two projects.
7. **Known trade-off:** with the slim requirements, `/query` (live ad-hoc
   Q&A for the 8 pre-baked companies) returns a `503` in this deployment.
   `/analyze` (live analysis for ANY ticker) works fully in production —
   it uses the lighter TF-IDF path specifically so it doesn't need the
   heavy stack that caused the original crash.

**Frontend (Vercel):**
1. New Vercel project, same repo, set **Root Directory** to `frontend`.
2. No build step needed — it's a static HTML file.
3. After the Render backend URL is live, update `RENDER_API_URL` near the
   top of `frontend/index.html`'s `<script>` block and redeploy.

## Testing

```bash
pytest tests/ -v
```

22 tests covering: deterministic ratio math (revenue/margin/net-income
growth calculations), the consistency-grounding filter (including a test
modeled directly on a real bug caught during development — see commit
history), the on-demand pipeline's orchestration wiring (mocked SEC/Groq
calls, verifying the right functions are called with the right
arguments), and API-level error mapping (404/429/503 responses).

## License

Student project — built for internship/MBA application purposes.
