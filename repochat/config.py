"""Configuration management for RepoChat."""

import os
from pathlib import Path
from typing import Set
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    """Application settings, thresholds, and model configurations."""

    # API Keys
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    mistral_api_key: str = Field(
        default_factory=lambda: os.getenv("MISTRAL_API_KEY", "")
    )

    # Embedding provider: "mistral" or "gemini"
    embedding_provider: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_PROVIDER",
            "mistral" if os.getenv("MISTRAL_API_KEY") else "gemini",
        )
    )

    # Models
    embedding_model: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL",
            "mistral-embed" if os.getenv("MISTRAL_API_KEY") else "gemini-embedding-001",
        )
    )
    generation_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")
    )

    # Storage paths
    chroma_persist_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIR", ".chroma"))
    )
    embedding_cache_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings"))
    )
    sqlite_db_path: Path = Field(
        default_factory=lambda: Path(os.getenv("SQLITE_DB_PATH", "repochat.db"))
    )

    # Chunker settings (~40-60 line chunks with ~10 line overlap)
    chunk_min_lines: int = 40
    chunk_max_lines: int = 60
    chunk_target_lines: int = 50
    chunk_overlap_lines: int = 10
    max_file_size_bytes: int = 200 * 1024  # 200 KB limit per specification

    # Indexer batching and retries
    embedding_batch_size: int = 40
    max_retries: int = 5
    initial_backoff_sec: float = 2.0
    backoff_multiplier: float = 2.0

    # Directory and file filters
    ignored_directories: Set[str] = {
        ".git",
        ".github",
        ".next",
        "node_modules",
        "build",
        "target",
        "dist",
        "out",
        "bin",
        "obj",
        ".idea",
        ".vscode",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "coverage",
        ".cache",
        ".chroma",
    }

    ignored_files: Set[str] = {
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "bun.lockb",
        "cargo.lock",
        "poetry.lock",
        "composer.lock",
        "go.sum",
        ".ds_store",
        "thumbs.db",
    }

    ignored_extensions: Set[str] = {
        # Binaries, archives & bytecode
        ".jar", ".war", ".class", ".pyc", ".pyo", ".so", ".dylib", ".dll",
        ".exe", ".bin", ".o", ".a", ".zip", ".tar", ".gz", ".7z", ".rar",
        # Media & images
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".bmp",
        ".mp3", ".mp4", ".mov", ".wav", ".avi", ".mkv",
        # Documents & fonts
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".woff", ".woff2", ".ttf", ".eot", ".otf",
    }


def get_settings() -> Settings:
    """Return an initialized Settings instance."""
    return Settings()
