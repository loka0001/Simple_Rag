# api/app.py
"""
FastAPI interface layer for the Agentic RAG system.

Endpoints:
    POST /query        — Run the agent on a user question (Arabic or English).
    POST /ingest       — Trigger document ingestion from the data directory.
    GET  /health       — Liveness probe.
    GET  /ollama/check — Check if local Ollama server is reachable.
    GET  /config       — Show current (non-secret) configuration.
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import build_agent, run_agent
from src.ingest import run_ingestion_pipeline
from src.llm_factory import check_ollama_connection
from src.logging_config import setup_logging
from src.config import get_settings

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic RAG API",
    description=(
        "Production-ready multilingual (Arabic + English) Retrieval-Augmented Generation "
        "system with agentic routing, local Ollama support, and transparent reasoning."
    ),
    version="2.0.0",
)

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent_executor = None


@app.on_event("startup")
async def startup_event():
    global _agent_executor
    cfg = get_settings()
    logger.info("Starting up | LLM provider: %s", cfg.llm_provider)
    if cfg.llm_provider == "ollama":
        status = check_ollama_connection()
        if not status["reachable"]:
            logger.warning(
                "Ollama not reachable at %s. Make sure 'ollama serve' is running.",
                cfg.ollama_base_url,
            )
        else:
            logger.info("Ollama ready. Available models: %s", status["models"])
    try:
        _agent_executor = build_agent()
        logger.info("Agent executor ready.")
    except Exception as exc:
        logger.error("Failed to initialize agent: %s", exc)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Input schema for the /query endpoint."""
    question: str = Field(..., min_length=2, description="Question in Arabic or English.")
    reformulate: bool = Field(False, description="Reformulate the query with LLM before retrieval.")
    with_reasoning: Optional[bool] = Field(
        None,
        description="Override ENABLE_REASONING setting. True = include step-by-step reasoning.",
    )


class QueryResponse(BaseModel):
    """Output schema for the /query endpoint."""
    question: str
    language: str
    answer: str


class IngestRequest(BaseModel):
    """Input schema for the /ingest endpoint."""
    data_dir: str = Field("./data/raw", description="Path to directory with source documents.")
    use_elasticsearch: bool = Field(False, description="Also index into Elasticsearch.")


class IngestResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok", "version": "2.0.0"}


@app.get("/config")
async def get_config():
    """Return current non-secret configuration for debugging."""
    cfg = get_settings()
    return {
        "llm_provider": cfg.llm_provider,
        "ollama_model": cfg.ollama_model,
        "ollama_reasoning_model": cfg.ollama_reasoning_model,
        "ollama_base_url": cfg.ollama_base_url,
        "groq_model": cfg.groq_model,
        "embedding_model": cfg.embedding_model,
        "rerank_model": cfg.rerank_model,
        "enable_reasoning": cfg.enable_reasoning,
        "reasoning_language": cfg.reasoning_language,
        "normalize_arabic": cfg.normalize_arabic,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "agent_max_iterations": cfg.agent_max_iterations,
    }


@app.get("/ollama/check")
async def ollama_check():
    """Check Ollama server connectivity and list available local models."""
    return check_ollama_connection()


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Run the agentic RAG pipeline on a question in Arabic or English.

    Automatically detects language, routes to appropriate retrieval strategy,
    and optionally returns a structured reasoning trace with confidence score.
    """
    if _agent_executor is None:
        raise HTTPException(status_code=503, detail="Agent is not yet initialized.")

    from src.arabic_utils import detect_language
    language = detect_language(request.question)

    try:
        answer = run_agent(
            query=request.question,
            reformulate=request.reformulate,
            with_reasoning=request.with_reasoning,
            agent_executor=_agent_executor,
        )
        return QueryResponse(question=request.question, language=language, answer=answer)
    except Exception as exc:
        logger.error("Query endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest):
    """
    Trigger the ingestion pipeline to load, chunk, embed, and index documents.

    Arabic files (.txt, .pdf, .docx) are automatically detected and processed
    with encoding detection and text normalization.
    """
    try:
        run_ingestion_pipeline(
            data_dir=request.data_dir,
            use_elasticsearch=request.use_elasticsearch,
        )
        return IngestResponse(
            status="success",
            message=f"Documents from '{request.data_dir}' ingested successfully.",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Ingest endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
