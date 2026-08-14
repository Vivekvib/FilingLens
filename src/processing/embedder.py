"""
embedder.py — embeds chunk records (from chunker.process_filing) with a
local sentence-transformers model and writes them into a persisted Chroma
collection, one collection shared across all companies (filtered by
metadata at query time rather than one collection per ticker — simpler to
manage and still fast at this scale).

IMPORTANT: sentence_transformers and chromadb are imported LAZILY, inside
get_model()/get_collection(), not at module level. Just importing
sentence_transformers pulls all of PyTorch into memory immediately, before
any model is actually instantiated — that was happening on every process
startup (api/main.py -> retriever.py -> this module, all module-level
imports), on every request path including /companies, not just /query.
On a memory-constrained deploy that's enough on its own to get the process
OOM-killed before it ever binds a port. Keeping these imports local to the
functions that need them means importing this module stays cheap, and the
heavy stack only loads if /query (the one endpoint that actually needs
retrieval) is called.
"""

import hashlib

from config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL

_model = None
_client = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_collection():
    global _client
    if _client is None:
        import chromadb

        _client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return _client.get_or_create_collection(CHROMA_COLLECTION_NAME)


def reset_collection():
    """Deletes and recreates the Chroma collection from scratch.

    Chunk IDs are a hash of chunk text (see _chunk_id below), so upsert()
    only overwrites a chunk whose text is byte-identical to a previous
    run — if the chunker logic changes (as it did here, before/after the
    table-of-contents regex fix), the new chunks get NEW ids and the OLD
    ones are silently never removed, since upsert never deletes anything.
    Across several pipeline reruns during development this is exactly
    what bloated chroma.sqlite3 to 123MB — well over GitHub's 100MB file
    limit — from orphaned embeddings nobody was using. Called once at the
    start of scripts/run_pipeline.py's Step 3, since that step always
    reprocesses every company from scratch anyway (no per-ticker caching,
    unlike Steps 1-2), so a full reset here doesn't cost anything extra
    and keeps the collection's real size matching its real content."""
    global _client
    import chromadb

    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        _client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet — fine, get_or_create below handles that
    return _client.get_or_create_collection(CHROMA_COLLECTION_NAME)


def _chunk_id(record: dict, idx: int) -> str:
    """Deterministic ID so re-running embedding on the same chunk text
    upserts instead of duplicating."""
    key = f"{record['ticker']}|{record['section']}|{idx}|{record['text'][:80]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def embed_and_store(records: list[dict]) -> int:
    """Embeds a list of chunk records and upserts them into Chroma.
    Returns the number of chunks written."""
    if not records:
        return 0

    model = get_model()
    collection = get_collection()

    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32).tolist()

    ids = [_chunk_id(r, i) for i, r in enumerate(records)]
    metadatas = [
        {
            "ticker": r["ticker"],
            "section": r["section"],
            "method": r["method"],
            "fiscal_year": r.get("fiscal_year") or "unknown",
        }
        for r in records
    ]

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(records)
