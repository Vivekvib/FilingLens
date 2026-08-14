"""
chunker.py — slices a raw 10-K HTML/text file into the two sections we
care about (Item 1A Risk Factors, Item 7 MD&A) and splits each into
overlapping chunks for embedding.

Two-tier strategy (see BUILD_PLAN.md / handoff brief for the reasoning):
  Tier 1 — regex match on section headers, with a few pattern variants
           since filers aren't consistent about spacing/punctuation.
  Tier 2 — if regex fails, or the matched section is suspiciously short
           (< config.MIN_SECTION_WORD_COUNT, usually a sign it sliced
           wrong), fall back to chunking the WHOLE document and tagging
           those chunks "section: auto-detected" instead of dropping the
           company from the pipeline.
"""

import re

from bs4 import BeautifulSoup

from config import MIN_SECTION_WORD_COUNT, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

# Item 1A / Item 7 header variants seen across real filers. Ordered
# roughly by how common they are; first match wins.
SECTION_PATTERNS = {
    "risk_factors": [
        r"item\s*1a\.?\s*[-–—]?\s*risk\s*factors",
    ],
    "mda": [
        r"item\s*7\.?\s*[-–—]?\s*management.?s\s*discussion\s*and\s*analysis",
    ],
    # The section that follows each target section, used as the "stop"
    # boundary when slicing. Item 1B (Unresolved Staff Comments) follows
    # 1A; Item 7A (Quantitative Disclosures About Market Risk) follows 7.
    "_stop_risk_factors": [r"item\s*1b\.?"],
    "_stop_mda": [r"item\s*7a\.?"],
}


def html_to_text(raw_html: str) -> str:
    """Strip HTML down to plain text, collapsing whitespace. 10-Ks are
    HTML with heavy inline styling — BeautifulSoup's get_text is enough,
    we don't need layout fidelity for RAG purposes."""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def _find_section(text_lower: str, start_patterns: list[str], stop_patterns: list[str]) -> tuple[int, int] | None:
    # 10-Ks list every Item header in a table of contents before the actual
    # section body — re.search() (first match) lands on that ToC entry,
    # which slices to almost nothing before the next ToC line and forces
    # the Tier 2 fallback on nearly every filing. Taking the LAST match
    # instead reliably skips past the ToC and lands on the real section,
    # since a 10-K has exactly one ToC entry + one body occurrence of each
    # header in the overwhelming majority of cases.
    start_match = None
    for pattern in start_patterns:
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            start_match = matches[-1]
            break
    if start_match is None:
        return None

    stop_match = None
    for pattern in stop_patterns:
        m = re.search(pattern, text_lower[start_match.end():])
        if m:
            stop_match = m
            break

    start_idx = start_match.end()
    end_idx = start_idx + stop_match.start() if stop_match else len(text_lower)
    return start_idx, end_idx


def extract_sections(raw_html: str) -> dict:
    """Returns {"risk_factors": {"text": ..., "method": "regex"|"fallback"},
    "mda": {...}}."""
    full_text = html_to_text(raw_html)
    text_lower = full_text.lower()
    sections = {}

    for key, stop_key in (("risk_factors", "_stop_risk_factors"), ("mda", "_stop_mda")):
        offsets = _find_section(text_lower, SECTION_PATTERNS[key], SECTION_PATTERNS[stop_key])
        word_count = 0
        if offsets is not None:
            start_idx, end_idx = offsets
            word_count = len(text_lower[start_idx:end_idx].split())

        if offsets is not None and word_count >= MIN_SECTION_WORD_COUNT:
            start_idx, end_idx = offsets
            # Slice the ORIGINAL (non-lowered) text at the same offsets so
            # casing is preserved for the LLM-facing chunks — carrying the
            # exact positions through means no fragile re-search for the
            # sliced text is needed (the previous version re-located the
            # slice by searching for its first 50 chars, which could in
            # principle land on a different occurrence than the one we
            # actually chose).
            sections[key] = {
                "text": full_text[start_idx:end_idx],
                "method": "regex",
            }
        else:
            # Tier 2 fallback: hand back the whole document, tagged so the
            # retriever knows to lean on semantic search rather than trust
            # this as a clean section slice.
            sections[key] = {"text": full_text, "method": "fallback_full_document"}

    return sections


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Naive whitespace-token chunking with overlap. Good enough for a
    2-day build — a tokenizer-aware splitter would be more precise but
    isn't worth the extra dependency here."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def process_filing(ticker: str, raw_html: str, fiscal_year: str | None = None) -> list[dict]:
    """Returns a flat list of chunk records ready for embedding:
    [{"ticker":..., "section":..., "method":..., "fiscal_year":..., "text":...}, ...]"""
    sections = extract_sections(raw_html)
    records = []
    for section_key, section_data in sections.items():
        for chunk in chunk_text(section_data["text"]):
            records.append({
                "ticker": ticker,
                "section": section_key,
                "method": section_data["method"],
                "fiscal_year": fiscal_year,
                "text": chunk,
            })
    return records
