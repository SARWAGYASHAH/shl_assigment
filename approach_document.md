# SHL Assessment Recommender — Approach Document

## Design Overview

This system is a **conversational RAG (Retrieval-Augmented Generation) agent** that recommends SHL Individual Test Solutions through multi-turn dialogue. It combines **semantic vector search** with an **LLM reasoning layer** to handle the four required conversational behaviors: clarify, recommend, refine, and compare.

### Architecture

```
User Message → Guardrails Check → Context Extraction (LLM) → Retrieval (FAISS + Keywords) → Response Generation (LLM) → Schema Validation → Response
```

**Key design decisions:**

1. **Stateless by design**: Every `/chat` call carries full conversation history. No server-side session state. Context extraction runs on each request, making the system resilient to restarts.

2. **Hybrid retrieval**: Combines FAISS cosine similarity search (70% weight) with keyword matching (30% weight). Semantic search handles intent ("I need someone who handles pressure" → personality tests), while keyword search ensures exact matches ("Java 8" → Java 8 test).

3. **LLM as router + generator**: The LLM (Gemini 2.0 Flash) serves dual purpose — it extracts structured context from conversation history AND generates natural language responses. This avoids hand-crafted intent classification.

4. **Guardrails before LLM**: Off-topic detection and vague-query handling run as fast regex/heuristic checks before any LLM call, saving latency and API quota.

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **LLM** | Gemini 2.0 Flash | Free tier (15 RPM), fast inference (<3s), good instruction following |
| **Embeddings** | all-MiniLM-L6-v2 | Local, 384-dim, no API key needed, good semantic quality |
| **Vector Store** | FAISS (IndexFlatIP) | In-memory, zero config, fast cosine similarity search |
| **Framework** | FastAPI | Assignment requirement, async support |
| **Deployment** | Render (Docker) | Free tier, Docker support, health check endpoint |

## Retrieval Setup

- **Catalog**: 100+ Individual Test Solutions scraped from the SHL online catalog, stored as JSON with name, URL, description, test type, category, duration, keywords
- **Embedding strategy**: Each assessment's `search_text` combines name + description + category + test type + duration + keywords into a single document for embedding
- **Search**: `hybrid_search()` merges FAISS results with keyword results, then `get_relevant_assessments()` applies type/category filters before returning top-K

## Prompt Design

The system prompt enforces:
- **Scope constraints**: Only SHL assessments, refuse off-topic
- **Turn budget awareness**: Max 8 turns, gather context in 1-2 turns, recommend by turn 3-4
- **Test type vocabulary**: K/P/A/S/B codes explained
- **Grounding requirement**: Only recommend from catalog, no hallucinated URLs

Two-stage prompting:
1. **Context extraction**: Structured JSON extraction of job role, seniority, skills, preferences, and intent (clarify vs recommend vs refine vs compare)
2. **Response generation**: Augmented prompt with retrieved assessments + conversation history

## Evaluation Approach

**What worked:**
- Hybrid search dramatically improved recall vs pure keyword or pure semantic search
- Adding keyword lists to assessment metadata (e.g., "java", "developer", "coding") closed the vocabulary gap between user queries and catalog entries
- Keeping prompts specific about the 4 behaviors prevented the LLM from over-clarifying

**What didn't work initially:**
- Pure semantic search missed exact technology matches (e.g., "Java 8" matched general ability tests before Java tests)
- Without turn budget in the prompt, the agent asked 5+ clarifying questions before recommending
- Single-stage prompting (no context extraction) led to inconsistent recommendations across conversation turns

**Measurement:**
- Tested against all 10 public conversation traces
- Measured Recall@10 on expected shortlists
- Verified schema compliance, turn cap, and scope enforcement on every response

## AI Tools Used

- Gemini API for runtime LLM inference (not for code generation)
- Cursor/Gemini Code Assist for code scaffolding and debugging
