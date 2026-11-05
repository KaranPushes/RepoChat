"""Retrieval engine: Vector, BM25, and Hybrid (Reciprocal Rank Fusion) retrievers."""

import json
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
from rank_bm25 import BM25Okapi

from repochat.config import Settings, get_settings
from repochat.indexer import Embedder

logger = logging.getLogger("repochat.retriever")


@dataclass
class RetrievedChunk:
    """Represents a code chunk retrieved for a given query with scoring metadata."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    text: str
    raw_text: str
    score: float
    retriever_type: str  # "vector", "bm25", or "hybrid"
    scope: Optional[str] = None
    variant: str = "v2"
    rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "text": self.text,
            "raw_text": self.raw_text,
            "score": round(self.score, 6),
            "retriever_type": self.retriever_type,
            "scope": self.scope or "",
            "variant": self.variant,
            "rank": self.rank,
        }

    @property
    def citation_label(self) -> str:
        """Formatted citation tag, e.g. [auth-service/AuthFilter.java:42-78]."""
        return f"[{self.file_path}:{self.start_line}-{self.end_line}]"


def tokenize_code(text: str) -> List[str]:
    """
    Code-aware tokenizer for BM25.
    Splits on punctuation, snake_case, and camelCase while keeping compound tokens.
    """
    words = re.findall(r"[a-zA-Z0-9_]+", text)
    tokens: List[str] = []

    for word in words:
        low = word.lower()
        tokens.append(low)

        # Split camelCase: e.g. JwtAuthenticationFilter -> [jwt, authentication, filter]
        camel_parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z]|\b)", word)
        if len(camel_parts) > 1:
            for p in camel_parts:
                tokens.append(p.lower())

        # Split snake_case: e.g. stream_scheduler -> [stream, scheduler]
        if "_" in word:
            for p in word.split("_"):
                if p:
                    tokens.append(p.lower())

    return tokens


class BaseRetriever(ABC):
    """Abstract base class for all code retrieval implementations."""

    @abstractmethod
    def retrieve(self, question: str, k: int = 5) -> List[RetrievedChunk]:
        """Retrieve the top-k most relevant code chunks for a question."""
        pass


class VectorRetriever(BaseRetriever):
    """Dense vector retriever using ChromaDB collection embeddings."""

    def __init__(
        self,
        variant: str = "v2",
        settings: Optional[Settings] = None,
        embedder: Optional[Embedder] = None,
        repo_id: Optional[str] = None,
    ):
        self.variant = variant
        self.repo_id = repo_id or "default"
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings)
        self.chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_persist_dir))

        clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", self.repo_id)
        candidates = [f"repochat_{clean_id}_{variant}"]
        if clean_id in ("default", "StreamNova", "target_repo", "KaranPushes_StreamNova"):
            candidates.append(f"repochat_{variant}")

        self.collection = None
        for cname in candidates:
            try:
                self.collection = self.chroma_client.get_collection(cname)
                self.collection_name = cname
                break
            except Exception:
                pass

        if self.collection is None:
            self.collection_name = candidates[0]
            self.collection = self.chroma_client.get_or_create_collection(self.collection_name)

    def retrieve(self, question: str, k: int = 5) -> List[RetrievedChunk]:
        """Query ChromaDB by embedding the question and returning top-k chunks."""
        query_vector = self.embedder.embed_query(question)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return retrieved

        ids = results["ids"][0]
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metadatas, distances), 1):
            # Normalize distance to similarity score in (0, 1]
            similarity = 1.0 / (1.0 + max(0.0, float(dist)))
            retrieved.append(
                RetrievedChunk(
                    chunk_id=cid,
                    file_path=meta.get("file_path", ""),
                    start_line=int(meta.get("start_line", 1)),
                    end_line=int(meta.get("end_line", 1)),
                    language=meta.get("language", "text"),
                    text=doc,
                    raw_text=meta.get("raw_text", doc),
                    score=similarity,
                    retriever_type="vector",
                    scope=meta.get("scope") or None,
                    variant=meta.get("variant", self.variant),
                    rank=rank,
                )
            )

        return retrieved


class BM25Retriever(BaseRetriever):
    """Sparse keyword retriever using BM25Okapi over code chunks."""

    def __init__(
        self,
        variant: str = "v2",
        settings: Optional[Settings] = None,
        repo_id: Optional[str] = None,
    ):
        self.variant = variant
        self.repo_id = repo_id or "default"
        self.settings = settings or get_settings()
        self.chunks: List[Dict[str, Any]] = []
        self._load_corpus()

        # Tokenize corpus for BM25
        self.tokenized_corpus = [
            tokenize_code(c.get("text", "") + " " + c.get("file_path", ""))
            for c in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _load_corpus(self) -> None:
        """Load chunk registry from disk cache, falling back to ChromaDB if needed."""
        clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", self.repo_id)
        candidates = [self.settings.embedding_cache_dir / f"chunks_{clean_id}_{self.variant}.json"]
        if clean_id in ("default", "StreamNova", "target_repo", "KaranPushes_StreamNova"):
            candidates.append(self.settings.embedding_cache_dir / f"chunks_{self.variant}.json")

        for p in candidates:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                return

        # Fallback to loading all documents from ChromaDB collection
        client = chromadb.PersistentClient(path=str(self.settings.chroma_persist_dir))
        cand_coll_names = [f"repochat_{clean_id}_{self.variant}"]
        if clean_id in ("default", "StreamNova", "target_repo", "KaranPushes_StreamNova"):
            cand_coll_names.append(f"repochat_{self.variant}")

        coll = None
        for cname in cand_coll_names:
            try:
                coll = client.get_collection(cname)
                break
            except Exception:
                pass

        if coll:
            all_data = coll.get(include=["documents", "metadatas"])
            for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
                self.chunks.append(
                    {
                        "chunk_id": cid,
                        "file_path": meta.get("file_path", ""),
                        "start_line": int(meta.get("start_line", 1)),
                        "end_line": int(meta.get("end_line", 1)),
                        "language": meta.get("language", "text"),
                        "text": doc,
                        "raw_text": meta.get("raw_text", doc),
                        "scope": meta.get("scope", ""),
                        "variant": meta.get("variant", self.variant),
                    }
                )

    def retrieve(self, question: str, k: int = 5) -> List[RetrievedChunk]:
        """Compute BM25 scores for the question and return top-k chunks."""
        tokenized_query = tokenize_code(question)
        if not tokenized_query or not self.chunks:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Rank indices by score descending
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )

        max_score = float(max(scores)) if len(scores) > 0 else 1.0
        retrieved: List[RetrievedChunk] = []

        for rank, idx in enumerate(ranked_indices[:k], 1):
            raw_score = float(scores[idx])
            # Normalize BM25 score
            norm_score = (raw_score / max_score) if max_score > 0 else 0.0

            chunk_data = self.chunks[idx]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_data["chunk_id"],
                    file_path=chunk_data["file_path"],
                    start_line=chunk_data["start_line"],
                    end_line=chunk_data["end_line"],
                    language=chunk_data["language"],
                    text=chunk_data["text"],
                    raw_text=chunk_data.get("raw_text", chunk_data["text"]),
                    score=norm_score,
                    retriever_type="bm25",
                    scope=chunk_data.get("scope") or None,
                    variant=chunk_data.get("variant", self.variant),
                    rank=rank,
                )
            )

        return retrieved


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining Vector and BM25 search via Reciprocal Rank Fusion (RRF).
    RRF score = sum(1.0 / (rrf_k + rank_i)) for each retriever system.
    """

    def __init__(
        self,
        variant: str = "v2",
        settings: Optional[Settings] = None,
        rrf_k: int = 60,
        vector_retriever: Optional[VectorRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        repo_id: Optional[str] = None,
    ):
        self.variant = variant
        self.repo_id = repo_id or "default"
        self.settings = settings or get_settings()
        self.rrf_k = rrf_k
        self.vector_retriever = vector_retriever or VectorRetriever(
            variant=variant, settings=self.settings, repo_id=self.repo_id
        )
        self.bm25_retriever = bm25_retriever or BM25Retriever(
            variant=variant, settings=self.settings, repo_id=self.repo_id
        )

    def retrieve(self, question: str, k: int = 5) -> List[RetrievedChunk]:
        """
        Retrieve top candidates from Vector and BM25, then fuse rankings with RRF.
        Candidate pool size N = max(2 * k, 25).
        """
        candidate_k = max(2 * k, 25)

        vector_results = self.vector_retriever.retrieve(question, k=candidate_k)
        bm25_results = self.bm25_retriever.retrieve(question, k=candidate_k)

        # Store chunk objects by chunk_id
        chunk_map: Dict[str, RetrievedChunk] = {}
        rrf_scores: Dict[str, float] = {}

        # 1. Accumulate Vector ranks
        for rank, chunk in enumerate(vector_results, 1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # 2. Accumulate BM25 ranks
        for rank, chunk in enumerate(bm25_results, 1):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # 3. Sort by RRF score descending
        sorted_cids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        fused_results: List[RetrievedChunk] = []
        for rank, cid in enumerate(sorted_cids[:k], 1):
            original = chunk_map[cid]
            fused_results.append(
                RetrievedChunk(
                    chunk_id=original.chunk_id,
                    file_path=original.file_path,
                    start_line=original.start_line,
                    end_line=original.end_line,
                    language=original.language,
                    text=original.text,
                    raw_text=original.raw_text,
                    score=rrf_scores[cid],
                    retriever_type="hybrid",
                    scope=original.scope,
                    variant=original.variant,
                    rank=rank,
                )
            )

        return fused_results


class QueryRewriter:
    """Optional LLM-based query rewriter for expanding code search terms."""

    REWRITE_PROMPT = """You are a senior software engineer assistant.
Given a user's question about a codebase, output an optimized search query that includes relevant programming keywords, typical class/method patterns, and synonyms (e.g. JWT, AuthFilter, SecurityConfig, @Scheduled).
Output ONLY the rewritten search keywords on a single line, nothing else.

User Question: {question}
Optimized Query:"""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        api_key = self.settings.gemini_api_key
        if api_key:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    def rewrite(self, question: str) -> str:
        """Rewrite question for higher retrieval recall."""
        if not self.client:
            return question

        try:
            response = self.client.models.generate_content(
                model=self.settings.generation_model,
                contents=self.REWRITE_PROMPT.format(question=question),
            )
            rewritten = response.text.strip().replace('"', "")
            return f"{question} {rewritten}" if rewritten else question
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return question


def get_retriever(
    strategy: str = "hybrid",
    variant: str = "v2",
    settings: Optional[Settings] = None,
    repo_id: Optional[str] = None,
) -> BaseRetriever:
    """
    Factory function to instantiate a retriever by strategy name.
    Allowed strategies: 'vector', 'bm25', 'hybrid'.
    """
    strat = strategy.lower().strip()
    s = settings or get_settings()

    if strat == "vector":
        return VectorRetriever(variant=variant, settings=s, repo_id=repo_id)
    elif strat == "bm25":
        return BM25Retriever(variant=variant, settings=s, repo_id=repo_id)
    elif strat == "hybrid":
        return HybridRetriever(variant=variant, settings=s, repo_id=repo_id)
    else:
        raise ValueError(
            f"Unknown retriever strategy '{strategy}'. Expected one of: 'vector', 'bm25', 'hybrid'."
        )
