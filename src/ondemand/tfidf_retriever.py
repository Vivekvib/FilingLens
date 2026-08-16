"""
tfidf_retriever.py — in-memory TF-IDF retrieval for the on-demand
"analyze any company" feature. Deliberately NOT semantic (no
sentence-transformers/torch) — this keeps the live production deployment
free of the multi-GB ML stack that caused the Render OOM crash (see
src/processing/embedder.py's docstring for that history). Quality
trade-off: keyword-based matching, not meaning-based. Acceptable for a
single ad-hoc request; the 8 pre-baked companies keep using the
higher-quality Chroma + sentence-transformers pipeline offline,
completely unaffected by this.

Unlike src/rag/retriever.py (which queries a persistent, shared Chroma
collection across all companies), this builds a fresh, ephemeral TF-IDF
index over ONE company's chunks per request — no persistence, and no
cross-company leakage risk by construction (there's only ever one
company's chunks in memory at a time).

IMPORTANT: scikit-learn is imported LAZILY, inside __init__, not at
module level — same lesson learned from embedder.py's sentence_transformers
bug. Importing this module (which api/main.py does at startup to register
the /analyze route) must not eagerly pull in numpy/scipy/scikit-learn;
only an actual POST /analyze call should pay that cost.
"""


class TfidfRetriever:
    """Wraps one company's chunk records (from chunker.process_filing) in
    a fresh TF-IDF index. Exposes the same retrieve(query, ticker=,
    section=, top_k=) shape as src.rag.retriever.retrieve, so
    risk_memo_generator and consistency_checker can use either
    interchangeably via dependency injection — see generate_risk_memo's
    and check_consistency's retrieve_fn parameter."""

    def __init__(self, chunk_records: list[dict]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._records = chunk_records
        texts = [r["text"] for r in chunk_records]
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self._matrix = self._vectorizer.fit_transform(texts) if texts else None

    def retrieve(self, query: str, ticker: str | None = None, section: str | None = None, top_k: int = 6) -> list[dict]:
        """`ticker` is accepted for interface compatibility with the real
        retriever but unused here — there's only ever one company's
        chunks in this index, so no cross-company filtering is needed."""
        if self._matrix is None:
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        candidate_indices = [
            i for i, r in enumerate(self._records) if section is None or r["section"] == section
        ]
        if not candidate_indices:
            return []

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix[candidate_indices])[0]
        ranked = sorted(zip(candidate_indices, sims), key=lambda pair: pair[1], reverse=True)[:top_k]

        return [
            {
                "text": self._records[i]["text"],
                "metadata": {
                    "ticker": self._records[i]["ticker"],
                    "section": self._records[i]["section"],
                    "method": self._records[i]["method"],
                    "fiscal_year": self._records[i].get("fiscal_year") or "unknown",
                },
                # 1 - cosine_similarity, matching the "lower = more similar"
                # convention of Chroma's "distance" field in the real
                # retriever, so any code reading .distance behaves the same.
                "distance": 1 - float(score),
            }
            for i, score in ranked
        ]
