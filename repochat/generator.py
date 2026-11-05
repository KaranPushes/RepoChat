"""Grounded answer generation with prompt defense, citation verification, and latency measurement."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from google import genai
from google.genai import errors

from repochat.citations import Citation, CitationReport, parse_and_verify_citations
from repochat.config import Settings, get_settings
from repochat.retriever import RetrievedChunk


@dataclass
class GenerationResponse:
    """Complete answer generation output with verification metrics and retrieved chunks."""

    answer: str
    citations: List[Citation]
    citation_report: CitationReport
    retrieved_chunks: List[RetrievedChunk]
    latency_ms: float
    prompt_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for API endpoints."""
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "citation_report": self.citation_report.to_dict(),
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "latency_ms": round(self.latency_ms, 2),
            "prompt_version": self.prompt_version,
        }


class AnswerGenerator:
    """Generates answers strictly grounded in retrieved code chunks with anti-injection defense."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        prompt_version: str = "v1",
    ):
        self.settings = settings or get_settings()
        self.prompt_version = prompt_version
        self.prompt_path = Path(f"prompts/answer_{prompt_version}.txt")
        self._load_prompt_template()

        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured in .env or environment")
        self.client = genai.Client(api_key=self.settings.gemini_api_key)

    def _load_prompt_template(self) -> None:
        """Load prompt template from file."""
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt template file not found: {self.prompt_path}")
        self.template = self.prompt_path.read_text(encoding="utf-8")

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved code chunks inside structured delimiters as untrusted data."""
        if not chunks:
            return "NO CODE CHUNKS RETRIEVED."

        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            chunk_header = (
                f'<CHUNK index="{i}" file="{chunk.file_path}" '
                f'lines="{chunk.start_line}-{chunk.end_line}" language="{chunk.language}">'
            )
            chunk_footer = "</CHUNK>"
            formatted_chunks.append(f"{chunk_header}\n{chunk.raw_text}\n{chunk_footer}")

        return "\n\n".join(formatted_chunks)

    def generate(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> GenerationResponse:
        """
        Generate answer for question grounded in retrieved code chunks.
        Measures latency, verifies citations, and detects hallucinations.
        """
        start_time = time.perf_counter()

        context_str = self._build_context(retrieved_chunks)
        full_prompt = self.template.format(
            context=context_str,
            question=question,
        )

        try:
            response = self.client.models.generate_content(
                model=self.settings.generation_model,
                contents=full_prompt,
            )
            answer_text = response.text.strip() if response.text else "I could not find this in the indexed code."
        except errors.APIError as e:
            raise RuntimeError(f"Gemini API error during generation (code {getattr(e, 'code', 'unknown')}): {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {e}") from e

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Post-process and verify citations
        citation_report = parse_and_verify_citations(answer_text, retrieved_chunks)

        return GenerationResponse(
            answer=answer_text,
            citations=citation_report.citations,
            citation_report=citation_report,
            retrieved_chunks=retrieved_chunks,
            latency_ms=elapsed_ms,
            prompt_version=self.prompt_version,
        )

    def generate_stream(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Iterator[str]:
        """Stream generated tokens incrementally."""
        context_str = self._build_context(retrieved_chunks)
        full_prompt = self.template.format(
            context=context_str,
            question=question,
        )

        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.settings.generation_model,
                contents=full_prompt,
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n[Generation error: {e}]"

