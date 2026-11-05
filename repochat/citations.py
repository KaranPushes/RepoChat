"""Citation extraction, verification, and hallucination detection for generated answers."""

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from repochat.retriever import RetrievedChunk


@dataclass
class Citation:
    """Represents a single parsed code citation with verification status."""

    raw_citation: str
    file: str
    start: int
    end: int
    verified: bool
    exact_contained: bool
    matched_chunk_id: Optional[str] = None
    reason: Optional[str] = None
    snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "raw_citation": self.raw_citation,
            "file": self.file,
            "start": self.start,
            "end": self.end,
            "verified": self.verified,
            "exact_contained": self.exact_contained,
            "matched_chunk_id": self.matched_chunk_id,
            "reason": self.reason or ("Verified in retrieved chunks" if self.verified else "Unverified/Hallucinated citation"),
            "snippet": self.snippet,
        }


@dataclass
class CitationReport:
    """Aggregated verification results for all citations in an answer."""

    citations: List[Citation]
    total_citations: int
    verified_citations: int
    hallucinated_citations: int
    verification_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report."""
        return {
            "citations": [c.to_dict() for c in self.citations],
            "total_citations": self.total_citations,
            "verified_citations": self.verified_citations,
            "hallucinated_citations": self.hallucinated_citations,
            "verification_rate": round(self.verification_rate, 4),
        }


# Matches patterns like [path/to/File.java:42-78] or [path/to/File.java:42]
CITATION_PATTERN = re.compile(
    r"\[([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9_]+):(\d+)(?:-(\d+))?\]"
)


def normalize_path(path_str: str) -> str:
    """Normalize file path for consistent matching across operating systems."""
    cleaned = path_str.strip().replace("\\", "/").lstrip("./")
    return cleaned.lower()


def paths_match(cited_path: str, chunk_path: str) -> bool:
    """Check if cited path matches or is a valid suffix/prefix of the chunk file path."""
    c_cited = normalize_path(cited_path)
    c_chunk = normalize_path(chunk_path)

    if c_cited == c_chunk:
        return True
    # Match suffix (e.g. "AuthFilter.java" matches "backend/services/chat-service/.../AuthFilter.java")
    if c_chunk.endswith(c_cited) or c_cited.endswith(c_chunk):
        return True
    return False


def extract_code_snippet(
    raw_text: str,
    chunk_start: int,
    chunk_end: int,
    cited_start: int,
    cited_end: int,
) -> Optional[str]:
    """Extract the specific lines cited from raw chunk text if available."""
    lines = raw_text.splitlines()
    # Map line numbers to 0-based index in lines list
    rel_start = max(0, cited_start - chunk_start)
    rel_end = min(len(lines), cited_end - chunk_start + 1)

    if rel_start < len(lines):
        snippet_lines = lines[rel_start:rel_end]
        return "\n".join(snippet_lines)
    return None


def parse_and_verify_citations(
    answer: str,
    retrieved_chunks: List[RetrievedChunk],
) -> CitationReport:
    """
    Extract all bracketed citations from generated answer text and verify
    whether the cited file and line range exist in the provided retrieved chunks.
    """
    matches = list(CITATION_PATTERN.finditer(answer))
    citations: List[Citation] = []

    verified_count = 0

    for match in matches:
        raw_text = match.group(0)
        file_part = match.group(1)
        start_line = int(match.group(2))
        end_line = int(match.group(3)) if match.group(3) else start_line

        # Ensure start <= end
        if start_line > end_line:
            start_line, end_line = end_line, start_line

        verified = False
        exact_contained = False
        matched_chunk_id: Optional[str] = None
        failure_reason: Optional[str] = None
        snippet: Optional[str] = None

        # Check against each retrieved chunk
        candidate_file_matches: List[RetrievedChunk] = []
        for chunk in retrieved_chunks:
            if paths_match(file_part, chunk.file_path):
                candidate_file_matches.append(chunk)

                # Check if cited range overlaps with chunk range
                overlaps = (max(start_line, chunk.start_line) <= min(end_line, chunk.end_line))
                is_contained = (chunk.start_line <= start_line and end_line <= chunk.end_line)

                if overlaps:
                    verified = True
                    matched_chunk_id = chunk.chunk_id
                    exact_contained = is_contained
                    snippet = extract_code_snippet(
                        raw_text=chunk.raw_text,
                        chunk_start=chunk.start_line,
                        chunk_end=chunk.end_line,
                        cited_start=start_line,
                        cited_end=end_line,
                    )
                    break

        if not candidate_file_matches:
            failure_reason = f"File '{file_part}' was not found in any retrieved chunks."
        elif not verified:
            ranges_str = ", ".join(f"[{c.start_line}-{c.end_line}]" for c in candidate_file_matches)
            failure_reason = (
                f"File '{file_part}' was retrieved, but cited range [{start_line}-{end_line}] "
                f"falls outside retrieved chunks (available ranges: {ranges_str})."
            )

        if verified:
            verified_count += 1

        citations.append(
            Citation(
                raw_citation=raw_text,
                file=file_part,
                start=start_line,
                end=end_line,
                verified=verified,
                exact_contained=exact_contained,
                matched_chunk_id=matched_chunk_id,
                reason=failure_reason or "Verified: cited code range exists in retrieved context.",
                snippet=snippet,
            )
        )

    total = len(citations)
    hallucinated = total - verified_count
    rate = (verified_count / total) if total > 0 else 1.0

    return CitationReport(
        citations=citations,
        total_citations=total,
        verified_citations=verified_count,
        hallucinated_citations=hallucinated,
        verification_rate=rate,
    )

