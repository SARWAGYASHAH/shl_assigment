"""
FastAPI service for the SHL Assessment Recommender.
Exposes two endpoints:
  - GET /health → readiness check
  - POST /chat → stateless conversational agent
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.catalog import catalog
from app.embeddings import build_index
from app.agent import process_chat
from app.keep_alive import start_keep_alive

# Resolve project root for static files
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for the FastAPI app."""
    print("=" * 60)
    print("SHL Assessment Recommender — Starting up...")
    print("=" * 60)

    start = time.time()

    # 1. Load catalog
    try:
        catalog.load()
        print(f"[OK] Catalog loaded: {len(catalog.assessments)} assessments")
    except Exception as e:
        print(f"[FAIL] Failed to load catalog: {e}")
        raise

    # 2. Build FAISS index
    try:
        build_index()
        print("[OK] FAISS index built")
    except Exception as e:
        print(f"[FAIL] Failed to build index: {e}")
        raise

    elapsed = time.time() - start
    print(f"\n[OK] Startup complete in {elapsed:.1f}s")
    print("=" * 60)

    # Start Render keep-alive if configured
    try:
        start_keep_alive()
    except Exception as e:
        print(f"[FAIL] Failed to start keep-alive background task: {e}", flush=True)

    yield  # App is running

    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that recommends SHL assessments to hiring managers "
        "through multi-turn dialogue. Clarifies vague queries, recommends 1-10 "
        "assessments, refines on constraint changes, and compares assessments."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = PROJECT_ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the chat UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "SHL Assessment Recommender API. See /docs for API documentation."}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Readiness check endpoint.
    Returns {"status": "ok"} with HTTP 200.
    The evaluator allows up to 2 minutes for cold start.
    """
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless chat endpoint.
    Takes full conversation history, returns agent reply with optional recommendations.

    The agent handles 4 conversational behaviors:
    - Clarify: asks targeted questions for vague queries
    - Recommend: returns 1-10 assessments with names and catalog URLs
    - Refine: updates shortlist when constraints change
    - Compare: produces grounded comparison from catalog data
    """
    try:
        response = await process_chat(request)
        return response
    except Exception as e:
        print(f"Error processing chat: {e}")
        # Return a graceful error response instead of crashing
        return ChatResponse(
            reply="I encountered an issue processing your request. Could you please rephrase your question about the role you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
