"""Indexing pipeline: batch embedding with retry/backoff, persistent caching, and ChromaDB storage."""

import hashlib
import json
import logging
import random
import re
import sqlite3
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
import httpx
from google import genai
from google.genai import errors

from repochat.chunker import Chunker, CodeChunk, RepositoryWalker
from repochat.config import Settings, get_settings

logger = logging.getLogger("repochat.indexer")


class EmbeddingCache:
    """Local SQLite-backed cache for text embeddings, keyed by content hash and model."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "embeddings_cache.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the cache table and indices."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    content_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (content_hash, model)
                )
                """
            )
            conn.commit()

    def get_batch(self, hashes: List[str], model: str) -> Dict[str, List[float]]:
        """Retrieve cached embeddings for a list of hashes."""
        if not hashes:
            return {}

        results: Dict[str, List[float]] = {}
        chunk_size = 500
        with sqlite3.connect(self.db_path) as conn:
            for i in range(0, len(hashes), chunk_size):
                sub_hashes = hashes[i : i + chunk_size]
                placeholders = ",".join("?" for _ in sub_hashes)
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT content_hash, embedding FROM embeddings WHERE model = ? AND content_hash IN ({placeholders})",
                    [model] + sub_hashes,
                )
                for h, emb_json in cursor.fetchall():
                    try:
                        results[h] = json.loads(emb_json)
                    except json.JSONDecodeError:
                        continue
        return results

    def set_batch(self, entries: List[Tuple[str, str, List[float]]]) -> None:
        """Cache newly computed embeddings."""
        if not entries:
            return

        now = time.time()
        records = [
            (h, model, json.dumps(emb), now)
            for h, model, emb in entries
        ]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings (content_hash, model, embedding, created_at)
                VALUES (?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()


class Embedder:
    """Unified embedder supporting Mistral and Gemini with automatic backoff and caching."""

    def __init__(self, settings: Optional[Settings] = None, cache: Optional[EmbeddingCache] = None):
        self.settings = settings or get_settings()
        self.cache = cache or EmbeddingCache(self.settings.embedding_cache_dir)
        self.provider = self.settings.embedding_provider.lower()
        self.model = self.settings.embedding_model

        if self.provider == "mistral":
            if not self.settings.mistral_api_key:
                raise ValueError("MISTRAL_API_KEY is not configured in .env or environment")
            self.http_client = httpx.Client(
                headers={
                    "Authorization": f"Bearer {self.settings.mistral_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        else:
            if not self.settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not configured in .env or environment")
            self.gemini_client = genai.Client(api_key=self.settings.gemini_api_key)

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Generate SHA-256 hash of chunk text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_chunks(
        self,
        chunks: List[CodeChunk],
        batch_size: Optional[int] = None,
    ) -> Tuple[List[List[float]], int, int]:
        """Embed a list of CodeChunks, returning (embeddings, cached_count, newly_embedded_count)."""
        if not chunks:
            return [], 0, 0

        bs = batch_size or self.settings.embedding_batch_size
        hashes = [self.compute_content_hash(c.text) for c in chunks]
        cached_map = self.cache.get_batch(hashes, self.model)

        cached_count = len(cached_map)
        missing_indices = [i for i, h in enumerate(hashes) if h not in cached_map]
        newly_embedded_count = len(missing_indices)

        if missing_indices:
            print(
                f"[*] Embedding {newly_embedded_count} chunks using {self.provider.upper()} ({self.model}) "
                f"[{cached_count} loaded from cache]..."
            )

            total_batches = (len(missing_indices) + bs - 1) // bs
            for b_idx in range(0, len(missing_indices), bs):
                batch_subset_indices = missing_indices[b_idx : b_idx + bs]
                batch_texts = [chunks[i].text for i in batch_subset_indices]
                batch_hashes = [hashes[i] for i in batch_subset_indices]

                batch_vectors = self._embed_batch_with_retry(batch_texts)

                cache_entries = [
                    (batch_hashes[i], self.model, batch_vectors[i])
                    for i in range(len(batch_vectors))
                ]
                self.cache.set_batch(cache_entries)

                for h, vec in zip(batch_hashes, batch_vectors):
                    cached_map[h] = vec

                current_batch = (b_idx // bs) + 1
                print(f"  -> Batch {current_batch}/{total_batches} embedded and cached ({len(batch_vectors)} chunks).")

                # Small sleep between batches
                time.sleep(0.2)

        ordered_embeddings = [cached_map[h] for h in hashes]
        return ordered_embeddings, cached_count, newly_embedded_count

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query."""
        hashes = [self.compute_content_hash(query)]
        cached = self.cache.get_batch(hashes, self.model)
        if hashes[0] in cached:
            return cached[hashes[0]]

        vectors = self._embed_batch_with_retry([query])
        self.cache.set_batch([(hashes[0], self.model, vectors[0])])
        return vectors[0]

    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed batch texts with retry logic based on active provider."""
        if self.provider == "mistral":
            return self._embed_mistral_with_retry(texts)
        return self._embed_gemini_with_retry(texts)

    def _embed_mistral_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Call Mistral v1/embeddings with backoff retry."""
        max_retries = 5
        backoff = 2.0

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.http_client.post(
                    "https://api.mistral.ai/v1/embeddings",
                    json={"model": self.model, "input": texts},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return [item["embedding"] for item in data["data"]]

                if resp.status_code == 429 and attempt < max_retries:
                    sleep_time = backoff + random.uniform(0.5, 1.5)
                    print(f"  [!] Mistral rate limit. Retrying in {sleep_time:.1f}s (attempt {attempt}/{max_retries})...")
                    time.sleep(sleep_time)
                    backoff *= 2.0
                    continue

                raise RuntimeError(f"Mistral embedding error {resp.status_code}: {resp.text}")

            except httpx.RequestError as e:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    raise RuntimeError(f"Mistral network error: {e}") from e

        raise RuntimeError("Failed to embed batch with Mistral.")

    def _embed_gemini_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Call Gemini API with dynamic retry delay parsing on rate limits."""
        max_retries = 8
        backoff = 5.0

        for attempt in range(1, max_retries + 1):
            try:
                response = self.gemini_client.models.embed_content(
                    model=self.model,
                    contents=texts,
                )
                vectors: List[List[float]] = []
                for item in response.embeddings:
                    if hasattr(item, "values"):
                        vectors.append(list(item.values))
                    elif isinstance(item, dict) and "values" in item:
                        vectors.append(list(item["values"]))
                    else:
                        vectors.append(list(item))
                return vectors

            except errors.APIError as e:
                err_str = str(e)
                is_rate_limit = getattr(e, "code", None) == 429 or "rate" in err_str.lower() or "quota" in err_str.lower()
                if is_rate_limit and attempt < max_retries:
                    delay_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                    if not delay_match:
                        delay_match = re.search(r"retryDelay[\'\":\s]+(\d+)", err_str, re.IGNORECASE)
                    sleep_time = (float(delay_match.group(1)) + 2.0) if delay_match else (backoff + 1.0)
                    print(f"  [!] Gemini quota pause: sleeping {sleep_time:.1f}s (attempt {attempt}/{max_retries})...")
                    time.sleep(sleep_time)
                    backoff *= 1.8
                else:
                    raise RuntimeError(f"Gemini embedding API failed: {e}") from e
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    raise RuntimeError(f"Gemini embedding error: {e}") from e

        raise RuntimeError("Failed to embed batch with Gemini.")


# Backward compatibility alias
GeminiEmbedder = Embedder


@dataclass
class IndexingSummary:
    """Summary statistics for an indexing run."""

    repo_id: str
    repo_name: str
    repo_path: str
    variant: str
    total_files_scanned: int
    total_files_indexed: int
    total_chunks: int
    chunks_cached: int
    chunks_embedded: int
    collection_name: str
    elapsed_time_sec: float

    def print_summary(self) -> None:
        """Print formatted indexing statistics."""
        print("\n" + "=" * 50)
        print(f"RepoChat Indexing Complete ({self.variant.upper()})")
        print("=" * 50)
        print(f"Repo ID:              {self.repo_id}")
        print(f"Repo Name:            {self.repo_name}")
        print(f"Repository Path:      {self.repo_path}")
        print(f"Files Scanned:        {self.total_files_scanned}")
        print(f"Files Indexed:        {self.total_files_indexed}")
        print(f"Total Chunks:         {self.total_chunks}")
        print(f"Chunks from Cache:    {self.chunks_cached}")
        print(f"Chunks Newly Embedded:{self.chunks_embedded}")
        print(f"Chroma Collection:    {self.collection_name}")
        print(f"Elapsed Time:         {self.elapsed_time_sec:.2f}s")
        print("=" * 50 + "\n")


def clone_or_resolve_repo(target: str, base_dir: Path = Path("repos")) -> Tuple[str, str, Path]:
    """
    Resolve local path or clone GitHub repository.
    Returns: (repo_id, repo_display_name, local_repo_path)
    """
    target = target.strip()
    is_url = (
        target.startswith("http://")
        or target.startswith("https://")
        or target.startswith("git@")
        or "github.com/" in target
    )
    if is_url:
        if not (target.startswith("http://") or target.startswith("https://") or target.startswith("git@")):
            target = f"https://{target}"
        clean_url = target.rstrip("/")
        if clean_url.endswith(".git"):
            clean_url = clean_url[:-4]
        parts = clean_url.split("/")
        repo_name = parts[-1]
        owner = parts[-2] if len(parts) >= 2 else "repo"
        repo_id = f"{owner}_{repo_name}"

        base_dir.mkdir(parents=True, exist_ok=True)
        local_path = (base_dir / repo_id).resolve()

        if not local_path.exists():
            print(f"[*] Cloning repository from {target} into {local_path} (shallow clone)...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", target, str(local_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone git repository {target}: {e.stderr}") from e
        else:
            print(f"[*] Repository already exists at {local_path}, using clone.")

        return repo_id, f"{owner}/{repo_name}", local_path

    # Local path resolution
    p = Path(target).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Local repository path '{target}' does not exist.")
    name = p.name
    return name, name, p


class RepoIndexer:
    """Indexes a repository by chunking files, computing embeddings, and storing in ChromaDB."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.chunker = Chunker(self.settings)
        self.cache = EmbeddingCache(self.settings.embedding_cache_dir)
        self.embedder = Embedder(self.settings, self.cache)
        self.chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_persist_dir))

    def get_collection_name(self, variant: str, repo_id: Optional[str] = None) -> str:
        """Return standardized collection name for variant and repo."""
        if not repo_id or repo_id in ("target_repo", "StreamNova", "KaranPushes_StreamNova"):
            # Backwards compatibility check
            try:
                self.chroma_client.get_collection(f"repochat_{variant}")
                return f"repochat_{variant}"
            except Exception:
                pass
        clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", repo_id or "default")
        return f"repochat_{clean_id}_{variant}"

    def index_repository(
        self,
        repo_path: Union[str, Path],
        variant: str = "v2",
        reindex: bool = False,
        batch_size: Optional[int] = None,
        custom_repo_id: Optional[str] = None,
    ) -> IndexingSummary:
        """
        Index a local or remote repository into ChromaDB under the given variant.
        Also persists chunk metadata to disk for BM25 retrieval.
        """
        start_time = time.time()
        repo_id, repo_name, local_path = clone_or_resolve_repo(str(repo_path))
        if custom_repo_id:
            repo_id = custom_repo_id

        collection_name = self.get_collection_name(variant=variant, repo_id=repo_id)

        if reindex:
            try:
                self.chroma_client.delete_collection(collection_name)
                print(f"[*] Deleted existing collection: {collection_name}")
            except Exception:
                pass

        collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"RepoChat code embeddings for {repo_name} variant {variant}"},
        )

        walker = RepositoryWalker(local_path, self.settings)
        files = list(walker.walk())
        total_files = len(files)

        all_chunks: List[CodeChunk] = []
        files_indexed = 0

        print(f"[*] Discovered {total_files} indexable source files in {repo_name}")
        for rel_path, abs_path in files:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                chunks = self.chunker.chunk_file(rel_path, content, variant=variant)
                if chunks:
                    all_chunks.extend(chunks)
                    files_indexed += 1
            except Exception as e:
                logger.warning(f"Error reading {rel_path}: {e}")

        total_chunks = len(all_chunks)
        print(f"[*] Generated {total_chunks} chunks from {files_indexed} files.")

        # Embed all chunks
        embeddings, cached_count, newly_embedded_count = self.embedder.embed_chunks(
            all_chunks, batch_size=batch_size
        )

        # Store in ChromaDB in batches
        chroma_batch_size = 200
        print(f"[*] Storing {total_chunks} chunks in ChromaDB collection '{collection_name}'...")
        for i in range(0, total_chunks, chroma_batch_size):
            chunk_slice = all_chunks[i : i + chroma_batch_size]
            emb_slice = embeddings[i : i + chroma_batch_size]

            ids = [c.chunk_id for c in chunk_slice]
            documents = [c.text for c in chunk_slice]
            metadatas = [
                {
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "language": c.language,
                    "scope": c.scope or "",
                    "variant": c.variant,
                    "raw_text": c.raw_text,
                    "repo_id": repo_id,
                }
                for c in chunk_slice
            ]

            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=emb_slice,
                metadatas=metadatas,
            )

        # Persist chunk registry for BM25 retrieval caching
        self._save_chunks_registry(all_chunks, variant, repo_id)

        # Register in SQLite database
        try:
            from repochat.db import QueryLogger
            QueryLogger(self.settings.sqlite_db_path).register_repo(
                repo_id=repo_id,
                repo_name=repo_name,
                repo_source=str(repo_path),
                variant=variant,
                total_chunks=total_chunks,
                total_files=files_indexed,
            )
        except Exception as e:
            logger.warning(f"Could not register repository in SQLite: {e}")

        elapsed = time.time() - start_time
        summary = IndexingSummary(
            repo_id=repo_id,
            repo_name=repo_name,
            repo_path=str(local_path),
            variant=variant,
            total_files_scanned=total_files,
            total_files_indexed=files_indexed,
            total_chunks=total_chunks,
            chunks_cached=cached_count,
            chunks_embedded=newly_embedded_count,
            collection_name=collection_name,
            elapsed_time_sec=elapsed,
        )
        return summary

    def _save_chunks_registry(self, chunks: List[CodeChunk], variant: str, repo_id: str = "default") -> None:
        """Persist chunk list to disk so BM25 retriever can load without reparsing."""
        cache_dir = self.settings.embedding_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", repo_id)
        registry_path = cache_dir / f"chunks_{clean_id}_{variant}.json"
        data = [c.to_dict() for c in chunks]
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Also write to generic filename if default/StreamNova for backward compatibility
        if clean_id in ("StreamNova", "target_repo", "default", "KaranPushes_StreamNova"):
            fallback_path = cache_dir / f"chunks_{variant}.json"
            with open(fallback_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
