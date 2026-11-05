"""Unit tests for FastAPI endpoints (/health, /ask) with mocked LLM client."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from repochat.api import app
from repochat.retriever import RetrievedChunk

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns 200 and system metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "generation_model" in data
    assert isinstance(data["indexed_collections"], list)


def test_ask_endpoint_with_mocked_gemini():
    """Verify /ask endpoint retrieves chunks, calls generator, verifies citations, and logs to SQLite."""
    mock_chunk = RetrievedChunk(
        chunk_id="auth-service/AuthFilter.java:10-50",
        file_path="auth-service/AuthFilter.java",
        start_line=10,
        end_line=50,
        language="java",
        text="public class AuthFilter {}",
        raw_text="public class AuthFilter {}",
        score=0.95,
        retriever_type="hybrid",
    )

    mock_llm_response = MagicMock()
    mock_llm_response.text = (
        "Authentication is handled by AuthFilter "
        "[auth-service/AuthFilter.java:12-30]."
    )

    with patch("repochat.api.get_retriever") as mock_get_retriever, \
         patch("google.genai.Client") as mock_client_cls:

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_chunk]
        mock_get_retriever.return_value = mock_retriever

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_llm_response
        mock_client_cls.return_value = mock_client

        payload = {
            "question": "How does authentication work?",
            "retriever": "hybrid",
            "variant": "v2",
            "k": 3,
            "rewrite_query": False,
        }

        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "AuthFilter" in data["answer"]
        assert len(data["citations"]) == 1
        assert data["citations"][0]["file"] == "auth-service/AuthFilter.java"
        assert data["citations"][0]["verified"] is True
        assert data["verification_rate"] == 1.0
        assert len(data["retrieved_chunks"]) == 1
        assert data["log_id"] > 0
        assert data["latency_ms"] >= 0


def test_ask_endpoint_gemini_rate_limit_handling():
    """Verify API handles Gemini rate limit error with 429 status."""
    from google.genai import errors

    with patch("repochat.api.get_retriever") as mock_get_retriever, \
         patch("google.genai.Client") as mock_client_cls:

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_get_retriever.return_value = mock_retriever

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = errors.APIError(
            code=429,
            response_json={"error": {"message": "Resource exhausted"}},
        )
        mock_client_cls.return_value = mock_client

        payload = {
            "question": "Test question",
            "retriever": "hybrid",
            "variant": "v2",
            "k": 3,
        }

        response = client.post("/ask", json=payload)
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()


def test_repos_endpoint():
    """Verify /repos returns a list of indexed repositories."""
    response = client.get("/repos")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        assert "repo_id" in data[0]
        assert "repo_name" in data[0]
        assert "total_chunks" in data[0]


