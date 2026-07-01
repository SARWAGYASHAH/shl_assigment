"""
Embedding generation and FAISS-based vector search for semantic retrieval.
Uses sentence-transformers for local embedding — no API key needed.
"""

import numpy as np
from typing import Optional
from pathlib import Path

from app.catalog import catalog, Assessment


# Lazy imports to handle missing dependencies gracefully
_model = None
_index = None
_embeddings = None


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Loaded sentence-transformers model: all-MiniLM-L6-v2")
    return _model


def build_index():
    """Build FAISS index from catalog assessments."""
    global _index, _embeddings

    if not catalog.is_loaded:
        raise RuntimeError("Catalog must be loaded before building index")

    import faiss

    model = _get_model()

    # Generate embeddings for all assessments
    texts = [a.search_text for a in catalog.assessments]
    print(f"Generating embeddings for {len(texts)} assessments...")
    _embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    # Build FAISS index (cosine similarity via inner product on normalized vectors)
    dimension = _embeddings.shape[1]
    _index = faiss.IndexFlatIP(dimension)
    _index.add(_embeddings.astype(np.float32))

    print(f"Built FAISS index: {_index.ntotal} vectors, {dimension} dimensions")


def semantic_search(query: str, top_k: int = 20) -> list[tuple[Assessment, float]]:
    """
    Search for assessments semantically similar to the query.
    Returns list of (Assessment, similarity_score) tuples.
    """
    if _index is None:
        build_index()

    model = _get_model()

    # Encode query
    query_embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)

    # Search
    scores, indices = _index.search(query_embedding, min(top_k, _index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(catalog.assessments) and idx >= 0:
            results.append((catalog.assessments[idx], float(score)))

    return results


def hybrid_search(query: str, top_k: int = 15) -> list[Assessment]:
    """
    Hybrid search combining semantic and keyword search.
    Merges results from both approaches with score normalization.
    """
    # Semantic search
    semantic_results = semantic_search(query, top_k=top_k * 2)

    # Keyword search
    keyword_results = catalog.keyword_search(query, top_k=top_k * 2)

    # Combine scores
    assessment_scores: dict[str, float] = {}

    # Add semantic scores (already normalized to 0-1 for cosine similarity)
    for assessment, score in semantic_results:
        assessment_scores[assessment.name] = score * 0.7  # 70% weight for semantic

    # Add keyword scores
    query_lower = query.lower()
    for assessment in keyword_results:
        kw_score = assessment.matches_keywords(query_lower)
        existing = assessment_scores.get(assessment.name, 0.0)
        assessment_scores[assessment.name] = existing + kw_score * 0.3  # 30% weight

    # Sort by combined score
    sorted_names = sorted(assessment_scores, key=assessment_scores.get, reverse=True)

    # Return top-k unique assessments
    results = []
    for name in sorted_names[:top_k]:
        assessment = catalog.get_by_name(name)
        if assessment:
            results.append(assessment)

    return results


def get_relevant_assessments(
    query: str,
    test_types: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    top_k: int = 10,
) -> list[Assessment]:
    """
    Get relevant assessments with optional type/category filtering.
    This is the main retrieval function used by the agent.
    """
    # Start with hybrid search
    candidates = hybrid_search(query, top_k=top_k * 3)

    # Apply filters if specified
    if test_types:
        type_set = {t.upper() for t in test_types}
        candidates = [a for a in candidates if a.test_type in type_set]

    if categories:
        cat_lower = {c.lower() for c in categories}
        candidates = [
            a for a in candidates
            if any(c in a.category.lower() for c in cat_lower)
        ]

    return candidates[:top_k]
