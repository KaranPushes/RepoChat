"""SQLite database for query logging, audit trails, and latency tracking."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from repochat.config import Settings, get_settings


class QueryLogger:
    """Manages SQLite storage for all /ask queries, answers, citations, and latencies."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_settings().sqlite_db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create query_logs and indexed_repos tables with performance indices."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    question TEXT NOT NULL,
                    retriever_type TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    k INTEGER NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    retrieved_chunks_json TEXT NOT NULL,
                    verification_rate REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    prompt_version TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_repos (
                    repo_id TEXT PRIMARY KEY,
                    repo_name TEXT NOT NULL,
                    repo_source TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    total_files INTEGER NOT NULL,
                    indexed_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_created_at ON query_logs(created_at)"
            )
            conn.commit()

    def register_repo(
        self,
        repo_id: str,
        repo_name: str,
        repo_source: str,
        variant: str,
        total_chunks: int,
        total_files: int,
    ) -> None:
        """Register or update an indexed repository record."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO indexed_repos (
                    repo_id, repo_name, repo_source, variant, total_chunks, total_files, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (repo_id, repo_name, repo_source, variant, total_chunks, total_files, now),
            )
            conn.commit()

    def list_repos(self) -> List[Dict[str, Any]]:
        """Return all indexed repositories sorted by last indexed timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM indexed_repos ORDER BY indexed_at DESC"
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_repo(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """Fetch details for a single repository."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM indexed_repos WHERE repo_id = ?",
                (repo_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def log_query(
        self,
        question: str,
        retriever_type: str,
        variant: str,
        k: int,
        answer: str,
        citations: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
        verification_rate: float,
        latency_ms: float,
        prompt_version: str = "v1",
    ) -> int:
        """Log a completed query to SQLite and return the new log ID."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO query_logs (
                    created_at, question, retriever_type, variant, k,
                    answer, citations_json, retrieved_chunks_json,
                    verification_rate, latency_ms, prompt_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    question,
                    retriever_type,
                    variant,
                    k,
                    answer,
                    json.dumps(citations),
                    json.dumps(retrieved_chunks),
                    verification_rate,
                    latency_ms,
                    prompt_version,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch the most recent query logs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM query_logs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                try:
                    item["citations"] = json.loads(item.pop("citations_json", "[]"))
                except Exception:
                    item["citations"] = []
                try:
                    item["retrieved_chunks"] = json.loads(item.pop("retrieved_chunks_json", "[]"))
                except Exception:
                    item["retrieved_chunks"] = []
                results.append(item)
            return results

    def get_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for logged queries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    COUNT(*),
                    AVG(latency_ms),
                    AVG(verification_rate)
                FROM query_logs
                """
            )
            count, avg_lat, avg_ver = cursor.fetchone()
            return {
                "total_queries": count or 0,
                "avg_latency_ms": round(avg_lat or 0.0, 2),
                "avg_verification_rate": round(avg_ver or 0.0, 4),
            }

