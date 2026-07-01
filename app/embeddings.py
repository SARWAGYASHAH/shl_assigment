"""
Lightweight, memory-optimized retrieval engine for the SHL catalog.
Uses a custom TF-IDF Vectorizer with synonym expansion to mimic semantic search
without loading heavy PyTorch / Sentence-Transformers models.
Uses under 1MB of memory, ensuring the container easily fits Render's 512MB limit.
"""

import math
import re
from typing import Optional, Dict, List, Tuple
import numpy as np

from app.catalog import catalog, Assessment

# Globals for TF-IDF
_catalog_texts: List[str] = []
_vocabulary: List[str] = []
_word_to_idx: Dict[str, int] = {}
_idf: Dict[str, float] = {}
_doc_vectors: List[np.ndarray] = []
_is_initialized = False

# Stopwords to filter out noise
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves", "hiring", "hire", "need", "assessment",
    "test", "candidate", "role", "position", "looking", "assess", "evaluate"
}

# Synonym expansion maps user concept intents to catalog keywords
SYNONYMS = {
    "java": ["java", "coding", "simulation", "technical"],
    "python": ["python", "data science", "machine learning", "coding", "simulation", "technical"],
    "javascript": ["javascript", "js", "web", "frontend", "coding", "simulation", "technical"],
    "js": ["javascript", "js", "web", "frontend", "coding", "simulation", "technical"],
    "c#": ["c#", "csharp", "coding", "simulation", "technical", "net"],
    "c++": ["c++", "cpp", "coding", "simulation", "technical"],
    "sql": ["sql", "database", "query", "data", "technical"],
    "react": ["react", "frontend", "web", "javascript", "technical"],
    "angular": ["angular", "frontend", "web", "javascript", "technical"],
    "html": ["html", "css", "web", "frontend"],
    "css": ["html", "css", "web", "frontend"],
    "coding": ["coding", "simulation", "developer", "programmer", "engineer"],
    "developer": ["coding", "simulation", "developer", "programmer", "engineer"],
    "engineer": ["coding", "simulation", "developer", "programmer", "engineer"],
    "programmer": ["coding", "simulation", "developer", "programmer", "engineer"],
    "sales": ["sales", "customer", "contact", "telesales", "retail", "negotiation", "scenarios"],
    "marketing": ["marketing", "sales", "strategy", "campaign"],
    "customer": ["customer", "contact", "support", "service", "retail", "scenarios", "complaints"],
    "support": ["customer", "contact", "support", "service", "scenarios"],
    "service": ["customer", "contact", "support", "service", "scenarios"],
    "management": ["supervisory", "scenarios", "management", "leadership", "graduate", "decisions", "business"],
    "manager": ["supervisory", "scenarios", "management", "leadership", "graduate", "decisions", "business"],
    "leadership": ["supervisory", "scenarios", "management", "leadership", "graduate", "decisions", "business"],
    "supervisor": ["supervisory", "scenarios", "management", "leadership", "graduate", "decisions", "business"],
    "supervisory": ["supervisory", "scenarios", "management", "leadership", "graduate", "decisions", "business"],
    "stakeholder": ["supervisory", "scenarios", "management", "graduate", "negotiation", "influence", "communication"],
    "communication": ["communication", "influence", "negotiation", "relationship"],
    "data science": ["data science", "machine learning", "python", "verify", "numerical"],
    "machine learning": ["machine learning", "data science", "python", "technical"],
    "statistics": ["statistics", "data science", "numerical", "verify"],
    "financial": ["finance", "financial", "accounting", "verify", "numerical"],
    "accounting": ["accounting", "finance", "financial", "verify", "numerical"],
    "cognitive": ["verify", "reasoning", "aptitude", "cognitive", "ability", "inductive", "deductive", "numerical", "verbal", "calculation", "checking"],
    "ability": ["verify", "reasoning", "aptitude", "cognitive", "ability", "inductive", "deductive", "numerical", "verbal", "calculation", "checking"],
    "reasoning": ["verify", "reasoning", "aptitude", "cognitive", "ability", "inductive", "deductive", "numerical", "verbal", "calculation", "checking"],
    "aptitude": ["verify", "reasoning", "aptitude", "cognitive", "ability", "inductive", "deductive", "numerical", "verbal", "calculation", "checking"],
    "numerical": ["verify", "numerical", "reasoning", "math", "calculation", "ability"],
    "verbal": ["verify", "verbal", "reasoning", "comprehension", "reading", "ability"],
    "checking": ["verify", "checking", "accuracy", "attention", "detail"],
    "personality": ["opq", "opq32", "occupational", "personality", "behavioral", "work style", "motivation"],
    "behavioral": ["behavioral", "behavioural", "scenarios", "sjt", "situational", "judgment", "personality"],
    "behavioural": ["behavioral", "behavioural", "scenarios", "sjt", "situational", "judgment", "personality"],
    "sjt": ["behavioral", "behavioural", "scenarios", "sjt", "situational", "judgment"],
    "stress": ["resilience", "pressure", "coping", "personality", "motivation"],
    "pressure": ["resilience", "pressure", "coping", "personality", "motivation"],
    "resilience": ["resilience", "pressure", "coping", "personality", "motivation"],
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, and split into tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s\-\#\+]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _expand_query(tokens: List[str]) -> List[str]:
    """Expand tokens with synonyms to capture semantic concepts."""
    expanded = list(tokens)
    # Check for phrases first
    joined = " ".join(tokens)
    for phrase, syns in SYNONYMS.items():
        if len(phrase.split()) > 1 and phrase in joined:
            expanded.extend(syns)

    # Check for individual words
    for token in tokens:
        if token in SYNONYMS:
            expanded.extend(SYNONYMS[token])

    return list(set(expanded))


def build_index():
    """Build the custom TF-IDF matrix for catalog assessments."""
    global _catalog_texts, _vocabulary, _word_to_idx, _idf, _doc_vectors, _is_initialized

    if not catalog.is_loaded:
        raise RuntimeError("Catalog must be loaded before building index")

    # 1. Tokenize all documents
    doc_tokens: List[List[str]] = []
    word_doc_counts: Dict[str, int] = {}

    for a in catalog.assessments:
        tokens = _tokenize(a.search_text)
        doc_tokens.append(tokens)
        
        # Count document frequency for IDF
        seen_in_doc = set(tokens)
        for w in seen_in_doc:
            word_doc_counts[w] = word_doc_counts.get(w, 0) + 1

    # 2. Build vocabulary
    _vocabulary = sorted(list(word_doc_counts.keys()))
    _word_to_idx = {word: idx for idx, word in enumerate(_vocabulary)}

    # 3. Compute IDF
    num_docs = len(catalog.assessments)
    for word, doc_count in word_doc_counts.items():
        # Smooth IDF formulation
        _idf[word] = math.log((1 + num_docs) / (1 + doc_count)) + 1.0

    # 4. Generate document vectors
    _doc_vectors = []
    for tokens in doc_tokens:
        vector = np.zeros(len(_vocabulary), dtype=np.float32)
        # Term Frequency
        for w in tokens:
            if w in _word_to_idx:
                vector[_word_to_idx[w]] += 1.0
        
        # Apply TF-IDF formula
        for w in set(tokens):
            if w in _word_to_idx:
                idx = _word_to_idx[w]
                # Log-scaling TF
                tf = 1.0 + math.log(vector[idx])
                vector[idx] = tf * _idf[w]

        # Normalize document vector (L2 norm)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        _doc_vectors.append(vector)

    _is_initialized = True
    print(f"Built custom TF-IDF index: {len(_doc_vectors)} vectors, {len(_vocabulary)} terms")


def semantic_search(query: str, top_k: int = 20) -> List[Tuple[Assessment, float]]:
    """
    Search for assessments semantically similar to the query using TF-IDF cosine similarity.
    Synonym expansion is applied to bridge semantic concepts.
    """
    global _is_initialized
    if not _is_initialized:
        build_index()

    # Tokenize and expand query
    query_tokens = _tokenize(query)
    expanded_tokens = _expand_query(query_tokens)

    if not expanded_tokens:
        # Fallback to general search if no valid tokens
        return [(a, 0.0) for a in catalog.assessments[:top_k]]

    # Generate query TF-IDF vector
    query_vector = np.zeros(len(_vocabulary), dtype=np.float32)
    for w in expanded_tokens:
        if w in _word_to_idx:
            query_vector[_word_to_idx[w]] += 1.0

    # Apply TF-IDF to query
    for w in set(expanded_tokens):
        if w in _word_to_idx:
            idx = _word_to_idx[w]
            tf = 1.0 + math.log(query_vector[idx])
            query_vector[idx] = tf * _idf[w]

    # Normalize query vector
    q_norm = np.linalg.norm(query_vector)
    if q_norm > 0:
        query_vector = query_vector / q_norm

    # Calculate cosine similarity with all documents (inner product on normalized vectors)
    results: List[Tuple[Assessment, float]] = []
    for idx, doc_vector in enumerate(_doc_vectors):
        score = float(np.dot(query_vector, doc_vector))
        results.append((catalog.assessments[idx], score))

    # Sort by similarity score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def hybrid_search(query: str, top_k: int = 15) -> List[Assessment]:
    """
    Hybrid search combining concept-expanded TF-IDF and direct keyword scoring.
    """
    # 1. TF-IDF search (mimics semantic search)
    semantic_results = semantic_search(query, top_k=top_k * 2)

    # 2. Direct keyword score
    keyword_results = catalog.keyword_search(query, top_k=top_k * 2)

    # 3. Combine scores
    assessment_scores: Dict[str, float] = {}

    # 70% weight for TF-IDF cosine similarity
    for assessment, score in semantic_results:
        assessment_scores[assessment.name] = score * 0.7

    # 30% weight for direct keyword matches
    query_lower = query.lower()
    for assessment in keyword_results:
        kw_score = assessment.matches_keywords(query_lower)
        existing = assessment_scores.get(assessment.name, 0.0)
        assessment_scores[assessment.name] = existing + kw_score * 0.3

    # Sort by combined score descending
    sorted_names = sorted(assessment_scores, key=assessment_scores.get, reverse=True)

    # Return matching assessment instances
    results = []
    for name in sorted_names[:top_k]:
        assessment = catalog.get_by_name(name)
        if assessment:
            results.append(assessment)

    return results


def get_relevant_assessments(
    query: str,
    test_types: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    top_k: int = 10,
) -> List[Assessment]:
    """
    Get relevant assessments with optional type/category filtering.
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
