"""Unit tests for citation verification and hallucination detection."""

import pytest

from repochat.citations import Citation, CitationReport, parse_and_verify_citations
from repochat.retriever import RetrievedChunk


def make_chunk(file_path: str, start: int, end: int, text: str = "sample code") -> RetrievedChunk:
    """Helper to construct RetrievedChunk objects for testing."""
    return RetrievedChunk(
        chunk_id=f"{file_path}:{start}-{end}",
        file_path=file_path,
        start_line=start,
        end_line=end,
        language="java",
        text=text,
        raw_text=text,
        score=1.0,
        retriever_type="vector",
    )


def test_citation_verifier_catches_valid_citation():
    """Verify valid citation inside retrieved chunk bounds is marked verified."""
    chunks = [
        make_chunk("backend/services/chat-service/AuthFilter.java", 40, 80, "line40\nline41\nline42\nline78\nline80")
    ]
    answer = (
        "JWT token validation is checked in "
        "[backend/services/chat-service/AuthFilter.java:42-78]."
    )

    report = parse_and_verify_citations(answer, chunks)

    assert report.total_citations == 1
    assert report.verified_citations == 1
    assert report.hallucinated_citations == 0
    assert report.verification_rate == 1.0

    cit = report.citations[0]
    assert cit.verified is True
    assert cit.file == "backend/services/chat-service/AuthFilter.java"
    assert cit.start == 42
    assert cit.end == 78


def test_citation_verifier_catches_fabricated_file():
    """Verify citation pointing to a non-existent or unretrieved file is flagged as hallucinated."""
    chunks = [
        make_chunk("backend/services/chat-service/AuthFilter.java", 1, 50)
    ]
    # LLM hallucinates a file that was never retrieved
    answer = "Database encryption is in [com/security/CryptoVault.java:10-25]."

    report = parse_and_verify_citations(answer, chunks)

    assert report.total_citations == 1
    assert report.verified_citations == 0
    assert report.hallucinated_citations == 1
    assert report.verification_rate == 0.0

    cit = report.citations[0]
    assert cit.verified is False
    assert "not found in any retrieved chunks" in cit.reason


def test_citation_verifier_catches_out_of_bounds_lines():
    """Verify citation referencing line numbers outside the retrieved chunk is flagged."""
    chunks = [
        # Only lines 1-50 were retrieved
        make_chunk("backend/services/stream-service/StreamService.java", 1, 50)
    ]
    # LLM hallucinates lines 200-250
    answer = (
        "Transcoding trigger is configured at "
        "[backend/services/stream-service/StreamService.java:200-250]."
    )

    report = parse_and_verify_citations(answer, chunks)

    assert report.total_citations == 1
    assert report.verified_citations == 0
    assert report.hallucinated_citations == 1
    assert report.verification_rate == 0.0

    cit = report.citations[0]
    assert cit.verified is False
    assert "falls outside retrieved chunks" in cit.reason


def test_mixed_citations():
    """Verify report with both verified and hallucinated citations computes correct rate."""
    chunks = [
        make_chunk("service/Auth.java", 10, 60),
    ]
    answer = (
        "Token is created in [service/Auth.java:15-30], but stored in [storage/Redis.java:5-10]."
    )

    report = parse_and_verify_citations(answer, chunks)

    assert report.total_citations == 2
    assert report.verified_citations == 1
    assert report.hallucinated_citations == 1
    assert report.verification_rate == 0.5

