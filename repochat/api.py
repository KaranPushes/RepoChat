"""FastAPI application exposing /health, /index, and /ask endpoints with SQLite logging."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from repochat.config import Settings, get_settings
from repochat.db import QueryLogger
from repochat.generator import AnswerGenerator
from repochat.indexer import RepoIndexer
from repochat.retriever import QueryRewriter, get_retriever

logger = logging.getLogger("repochat.api")

app = FastAPI(
    title="RepoChat API",
    description="Grounded codebase question-answering with file and line citations.",
    version="0.1.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()
query_logger = QueryLogger()


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_provider: str
    generation_model: str
    indexed_collections: List[str]


class IndexRequest(BaseModel):
    repo_url: Optional[str] = Field(default=None, description="GitHub repository URL e.g. https://github.com/owner/repo")
    repo_path: Optional[str] = Field(default=None, description="Local path to repository e.g. ./target_repo")
    variant: Literal["v1", "v2"] = Field(default="v2", description="Chunking variant")
    reindex: bool = Field(default=False, description="Re-embed and replace existing collection")


class IndexResponse(BaseModel):
    status: str
    message: str
    summary: Dict[str, Any]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Question about the indexed codebase")
    repo_id: Optional[str] = Field(default=None, description="Repository ID to query (e.g. StreamNova)")
    retriever: Literal["vector", "bm25", "hybrid"] = Field(
        default="hybrid", description="Retrieval strategy"
    )
    variant: Literal["v1", "v2"] = Field(default="v2", description="Index chunking variant")
    k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    rewrite_query: bool = Field(default=False, description="Use LLM query rewriting before retrieval")


class CitationModel(BaseModel):
    raw_citation: str
    file: str
    start: int
    end: int
    verified: bool
    exact_contained: bool
    reason: str
    snippet: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    citations: List[CitationModel]
    retrieved_chunks: List[Dict[str, Any]]
    latency_ms: float
    verification_rate: float
    log_id: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return system operational status and active models."""
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    collections = [c.name for c in client.list_collections()]

    return HealthResponse(
        status="ok",
        version="0.1.0",
        embedding_provider=settings.embedding_provider,
        generation_model=settings.generation_model,
        indexed_collections=collections,
    )


@app.get("/repos", tags=["Repositories"])
def list_repositories_endpoint() -> List[Dict[str, Any]]:
    """List all indexed repositories available for querying."""
    return query_logger.list_repos()


@app.post("/index", response_model=IndexResponse, tags=["Indexing"])
def index_repository_endpoint(request: IndexRequest) -> IndexResponse:
    """Index or re-index a repository (local path or GitHub URL) into ChromaDB vector store."""
    target = request.repo_url or request.repo_path
    if not target:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'repo_url' (e.g. https://github.com/...) or 'repo_path'.",
        )

    try:
        indexer = RepoIndexer(settings=settings)
        summary = indexer.index_repository(
            repo_path=target,
            variant=request.variant,
            reindex=request.reindex,
        )
        return IndexResponse(
            status="success",
            message=f"Indexed {summary.total_chunks} chunks for {summary.repo_name} into {summary.collection_name}",
            summary={
                "repo_id": summary.repo_id,
                "repo_name": summary.repo_name,
                "repo_path": summary.repo_path,
                "variant": summary.variant,
                "total_files_scanned": summary.total_files_scanned,
                "total_files_indexed": summary.total_files_indexed,
                "total_chunks": summary.total_chunks,
                "chunks_cached": summary.chunks_cached,
                "chunks_embedded": summary.chunks_embedded,
                "elapsed_time_sec": round(summary.elapsed_time_sec, 2),
            },
        )
    except Exception as e:
        logger.error(f"Indexing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}",
        )


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask_question_endpoint(request: AskRequest) -> AskResponse:
    """
    Answer a question grounded in the indexed codebase.
    Retrieves chunks with vector/BM25/hybrid, generates answer with Gemini Flash,
    verifies citations, and logs query details to SQLite.
    """
    total_start = time.perf_counter()

    # 1. Query Rewriting (optional)
    search_query = request.question
    if request.rewrite_query:
        rewriter = QueryRewriter(settings=settings)
        search_query = rewriter.rewrite(request.question)

    # 2. Retrieval
    try:
        retriever = get_retriever(
            strategy=request.retriever,
            variant=request.variant,
            settings=settings,
            repo_id=request.repo_id,
        )
        chunks = retriever.retrieve(search_query, k=request.k)
    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {str(e)}",
        )

    # 3. Answer Generation
    try:
        generator = AnswerGenerator(settings=settings, prompt_version="v1")
        gen_res = generator.generate(
            question=request.question,
            retrieved_chunks=chunks,
        )
    except Exception as e:
        logger.error(f"Answer generation error: {e}", exc_info=True)
        err_msg = str(e)
        if "429" in err_msg or "quota" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Gemini API rate limit reached. Please wait a moment before trying again.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {err_msg}",
        )

    total_latency_ms = (time.perf_counter() - total_start) * 1000.0

    # 4. Convert Citations
    citation_models = [
        CitationModel(
            raw_citation=c.raw_citation,
            file=c.file,
            start=c.start,
            end=c.end,
            verified=c.verified,
            exact_contained=c.exact_contained,
            reason=c.reason or "",
            snippet=c.snippet,
        )
        for c in gen_res.citations
    ]

    retrieved_chunk_dicts = [c.to_dict() for c in chunks]

    # 5. Log to SQLite
    log_id = query_logger.log_query(
        question=request.question,
        retriever_type=request.retriever,
        variant=request.variant,
        k=request.k,
        answer=gen_res.answer,
        citations=[c.model_dump() for c in citation_models],
        retrieved_chunks=retrieved_chunk_dicts,
        verification_rate=gen_res.citation_report.verification_rate,
        latency_ms=total_latency_ms,
        prompt_version=gen_res.prompt_version,
    )

    return AskResponse(
        answer=gen_res.answer,
        citations=citation_models,
        retrieved_chunks=retrieved_chunk_dicts,
        latency_ms=total_latency_ms,
        verification_rate=gen_res.citation_report.verification_rate,
        log_id=log_id,
    )


@app.get("/logs", tags=["Audit"])
def get_logs_endpoint(limit: int = Query(default=20, ge=1, le=100)) -> Dict[str, Any]:
    """Retrieve audit history of logged /ask queries and latency stats."""
    logs = query_logger.get_recent_logs(limit=limit)
    stats = query_logger.get_stats()
    return {
        "stats": stats,
        "logs": logs,
    }


# Serve Single-Page Chat UI at root (dist if built, or frontend source)
frontend_dist = Path("frontend/dist")
frontend_static = frontend_dist if frontend_dist.exists() else Path("frontend")
if frontend_static.exists():
    app.mount("/", StaticFiles(directory=str(frontend_static), html=True), name="frontend")
