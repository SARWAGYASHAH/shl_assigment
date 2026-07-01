# SHL Assessment Recommender

A conversational AI agent that helps hiring managers find the right SHL assessments through multi-turn dialogue.

## Features

- **Clarify** vague queries before recommending
- **Recommend** 1-10 assessments with names and catalog URLs
- **Refine** when constraints change mid-conversation
- **Compare** assessments using catalog data
- **Scope enforcement**: only discusses SHL assessments, refuses off-topic

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key
```bash
# Get a free key at https://aistudio.google.com/apikey
export GEMINI_API_KEY="your-api-key-here"

# Windows PowerShell:
$env:GEMINI_API_KEY="your-api-key-here"
```

### 3. Run locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test it
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I am hiring a Java developer who works with stakeholders"}]}'
```

## API Endpoints

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat
Stateless chat endpoint. Takes full conversation history, returns agent reply.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Deployment

### Render (recommended)
1. Push to GitHub
2. Connect your repo on [render.com](https://render.com)
3. Set `GEMINI_API_KEY` in environment variables
4. Deploy — the `render.yaml` handles the rest

### Docker
```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-key" shl-recommender
```

## Architecture

```
User → FastAPI → Guardrails → Context Extraction (Gemini) → Hybrid Search (FAISS + Keywords) → Response Generation (Gemini) → Validated Response
```

## Tech Stack

- **LLM**: Google Gemini 2.0 Flash (free tier)
- **Embeddings**: all-MiniLM-L6-v2 (sentence-transformers)
- **Vector Store**: FAISS
- **Framework**: FastAPI
- **Catalog**: 100+ SHL Individual Test Solutions
