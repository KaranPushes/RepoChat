"""Tests for Vector, BM25, and Hybrid RRF retrievers."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from repochat.chunker import Chunker, RepositoryWalker
from repochat.config import Settings
from repochat.retriever import (
    BaseRetriever,
    BM25Retriever,
    HybridRetriever,
    RetrievedChunk,
    VectorRetriever,
    get_retriever,
    tokenize_code,
)


def test_tokenize_code():
    """Verify code tokenizer splits camelCase, snake_case, and extracts subwords."""
    tokens = tokenize_code("JwtAuthenticationFilter.validateToken(user_id)")
    assert "jwtauthenticationfilter" in tokens
    assert "jwt" in tokens
    assert "authentication" in tokens
    assert "filter" in tokens
    assert "validatetoken" in tokens
    assert "validate" in tokens
    assert "token" in tokens
    assert "user_id" in tokens
    assert "user" in tokens
    assert "id" in tokens


def test_bm25_retriever_toy_repo():
    """Verify BM25 keyword retriever correctly identifies expected files in toy repo."""
    toy_repo = Path("tests/toy_repo")
    walker = RepositoryWalker(toy_repo)
    chunker = Chunker()

    chunks = []
    for rel_path, abs_path in walker.walk():
        content = abs_path.read_text()
        chunks.extend(chunker.chunk_file(rel_path, content, variant="v2"))

    retriever = BM25Retriever.__new__(BM25Retriever)
    retriever.variant = "v2"
    retriever.settings = Settings()
    retriever.chunks = [c.to_dict() for c in chunks]
    retriever.tokenized_corpus = [
        tokenize_code(c["text"] + " " + c["file_path"])
        for c in retriever.chunks
    ]
    from rank_bm25 import BM25Okapi
    retriever.bm25 = BM25Okapi(retriever.tokenized_corpus)

    # 1. JWT auth query
    results_auth = retriever.retrieve("How does JWT token generation and validation work?", k=3)
    assert len(results_auth) > 0
    assert "JwtAuth.java" in results_auth[0].file_path
    assert results_auth[0].retriever_type == "bm25"

    # 2. Stream scheduler query
    results_stream = retriever.retrieve("Where is stream scheduling and @Scheduled handled?", k=3)
    assert len(results_stream) > 0
    assert "StreamScheduler.java" in results_stream[0].file_path

    # 3. Chat broadcast query
    results_chat = retriever.retrieve("Where is chat message broadcasting handled?", k=3)
    assert len(results_chat) > 0
    assert "ChatService.java" in results_chat[0].file_path


def test_hybrid_rrf_scoring():
    """Verify Reciprocal Rank Fusion calculation merges vector and BM25 results accurately."""
    mock_chunk_a = RetrievedChunk(
        chunk_id="fileA.java:1-50",
        file_path="fileA.java",
        start_line=1,
        end_line=50,
        language="java",
        text="code A",
        raw_text="code A",
        score=0.9,
        retriever_type="vector",
        rank=1,
    )
    mock_chunk_b = RetrievedChunk(
        chunk_id="fileB.java:1-50",
        file_path="fileB.java",
        start_line=1,
        end_line=50,
        language="java",
        text="code B",
        raw_text="code B",
        score=0.8,
        retriever_type="vector",
        rank=2,
    )
    mock_chunk_c = RetrievedChunk(
        chunk_id="fileC.java:1-50",
        file_path="fileC.java",
        start_line=1,
        end_line=50,
        language="java",
        text="code C",
        raw_text="code C",
        score=0.7,
        retriever_type="bm25",
        rank=1,
    )

    mock_vec = MagicMock(spec=VectorRetriever)
    mock_bm25 = MagicMock(spec=BM25Retriever)

    # Vector returns: [A (rank 1), B (rank 2)]
    mock_vec.retrieve.return_value = [mock_chunk_a, mock_chunk_b]
    # BM25 returns: [A (rank 1), C (rank 2)]  -- A appears at rank 1 in both!
    mock_chunk_a_bm25 = RetrievedChunk(
        chunk_id="fileA.java:1-50",
        file_path="fileA.java",
        start_line=1,
        end_line=50,
        language="java",
        text="code A",
        raw_text="code A",
        score=1.0,
        retriever_type="bm25",
        rank=1,
    )
    mock_bm25.retrieve.return_value = [mock_chunk_a_bm25, mock_chunk_c]

    hybrid = HybridRetriever(
        variant="v2",
        rrf_k=60,
        vector_retriever=mock_vec,
        bm25_retriever=mock_bm25,
    )

    results = hybrid.retrieve("query", k=3)
    assert len(results) == 3

    # File A was rank 1 in vector and rank 1 in BM25:
    # Expected score = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0327868
    expected_score_a = (1.0 / 61.0) + (1.0 / 61.0)
    assert results[0].file_path == "fileA.java"
    assert pytest.approx(results[0].score, rel=1e-4) == expected_score_a
    assert results[0].retriever_type == "hybrid"


def test_get_retriever_factory():
    """Verify factory returns appropriate retriever instances."""
    with pytest.raises(ValueError):
        get_retriever("invalid_strategy")

