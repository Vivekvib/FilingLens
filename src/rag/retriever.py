"""
retriever.py — semantic search over the embedded filing chunks. Every
retrieval call is scoped to a ticker (and optionally a section), so a
question about JPM never accidentally pulls in AAPL chunks — retrieval
should be grounded and scoped, not a free-for-all similarity search across
every company at once.
"""

from src.processing.embedder import get_collection, get_model


def retrieve(query: str, ticker: str, section: str | None = None, top_k: int = 6) -> list[dict]:
    """Returns the top_k most relevant chunks for `query`, scoped to
    `ticker` (and `section` if given). Each result includes the chunk
    text, metadata, and similarity distance, so callers can decide their
    own relevance cutoff rather than trusting top_k blindly."""
    model = get_model()
    collection = get_collection()

    where = {"ticker": ticker} if section is None else {"$and": [{"ticker": ticker}, {"section": section}]}

    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where,
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
