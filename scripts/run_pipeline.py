"""
run_pipeline.py — the one command that takes FilingLens from nothing to
fully-populated cache:

    python scripts/run_pipeline.py

Steps: fetch 10-Ks -> fetch XBRL facts -> chunk + embed -> generate risk
memos -> run consistency checks -> write everything to data/processed/ so
the API layer can serve it without touching the LLM on every request.

Run this once after any config.py change (new ticker, changed chunk size,
etc.) — it's idempotent for the fetch steps (cached unless force=True).
Step 3 (chunking/embedding) always fully resets and rebuilds the vector
store on every run rather than appending, so re-running never leaves
orphaned embeddings behind — see reset_collection() in
src/processing/embedder.py. Steps 4-5 always regenerate memos/consistency
checks too, since those are cheap to redo and should reflect the latest
chunks.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import COMPANIES, RAW_FILINGS_DIR, PROCESSED_DIR
from src.ingestion import edgar_fetcher, xbrl_fetcher
from src.processing.chunker import process_filing
from src.processing.embedder import embed_and_store, reset_collection
from src.rag.risk_memo_generator import generate_risk_memo
from src.agent.consistency_checker import check_consistency


def run(force_fetch: bool = False):
    print("=== Step 1: Fetching 10-K filings ===")
    for ticker, status in edgar_fetcher.fetch_all(force=force_fetch).items():
        print(f"  {ticker}: {status}")

    print("\n=== Step 2: Fetching XBRL financial facts ===")
    for ticker, status in xbrl_fetcher.fetch_all(force=force_fetch).items():
        print(f"  {ticker}: {status}")

    print("\n=== Step 3: Chunking + embedding filings ===")
    # Reset first — Step 3 always reprocesses every company from scratch
    # (no per-ticker caching like Steps 1-2), so starting from an empty
    # collection each run guarantees no orphaned embeddings pile up from
    # an earlier chunker version. See reset_collection()'s docstring in
    # src/processing/embedder.py for why this matters.
    reset_collection()
    for ticker in COMPANIES:
        raw_path = RAW_FILINGS_DIR / f"{ticker}.htm"
        if not raw_path.exists():
            print(f"  {ticker}: SKIPPED (no raw filing cached — check Step 1 output)")
            continue
        records = process_filing(ticker, raw_path.read_text(encoding="utf-8"))
        n = embed_and_store(records)
        methods = {r["method"] for r in records}
        print(f"  {ticker}: {n} chunks embedded (section methods used: {methods})")

    print("\n=== Step 4: Generating risk memos ===")
    # Spaced out to stay comfortably under Groq's free-tier rate limit
    # (30 requests/min as of mid-2026 — verify current limit at
    # console.groq.com before a real run). 8 companies x 1 call each is
    # nowhere near the daily cap, this is purely about per-minute pacing.
    for ticker in COMPANIES:
        try:
            memo = generate_risk_memo(ticker)
            (PROCESSED_DIR / f"{ticker}_risk_memo.json").write_text(json.dumps(memo, indent=2), encoding="utf-8")
            print(f"  {ticker}: memo written ({len(memo.get('top_risks', []))} risks)")
        except Exception as e:  # noqa: BLE001 — one bad memo shouldn't kill the run
            print(f"  {ticker}: FAILED — {e}")
        time.sleep(3)

    print("\n=== Step 5: Running consistency checks ===")
    for ticker in COMPANIES:
        xbrl_path = PROCESSED_DIR / f"{ticker}_xbrl.json"
        if not xbrl_path.exists():
            print(f"  {ticker}: SKIPPED (no XBRL data cached)")
            continue
        try:
            xbrl_facts = json.loads(xbrl_path.read_text(encoding="utf-8"))
            result = check_consistency(ticker, xbrl_facts)
            (PROCESSED_DIR / f"{ticker}_consistency.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"  {ticker}: {len(result.get('flags', []))} flags — {result.get('assessment', '')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {ticker}: FAILED — {e}")
        time.sleep(3)

    print("\nPipeline run complete. Start the API with:\n  uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    run(force_fetch="--force" in sys.argv)
