"""Codebase file discovery, language detection, scope extraction, and chunking."""

import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from repochat.config import Settings, get_settings


@dataclass
class CodeChunk:
    """Represents a chunk of code extracted from a source file."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    text: str
    raw_text: str
    scope: Optional[str] = None
    variant: str = "v2"

    def to_dict(self) -> dict:
        """Convert chunk metadata and content to a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "text": self.text,
            "raw_text": self.raw_text,
            "scope": self.scope or "",
            "variant": self.variant,
        }


# File extension to language mapping
EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".py": "python",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".sass": "css",
    ".less": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".md": "markdown",
    ".markdown": "markdown",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".properties": "properties",
    ".env": "config",
    ".toml": "toml",
    ".graphql": "graphql",
    ".proto": "protobuf",
}

# Comment syntax for context headers
LANGUAGE_COMMENT_PREFIX: Dict[str, Tuple[str, str]] = {
    "java": ("//", ""),
    "javascript": ("//", ""),
    "typescript": ("//", ""),
    "python": ("#", ""),
    "shell": ("#", ""),
    "yaml": ("#", ""),
    "toml": ("#", ""),
    "properties": ("#", ""),
    "config": ("#", ""),
    "sql": ("--", ""),
    "html": ("<!--", "-->"),
    "markdown": ("<!--", "-->"),
    "xml": ("<!--", "-->"),
    "css": ("/*", "*/"),
}


def detect_language(file_path: str) -> str:
    """Detect language of a file based on filename or extension."""
    filename = Path(file_path).name.lower()
    if filename in ("dockerfile", "containerfile"):
        return "dockerfile"
    if filename in ("makefile", "gnumakefile"):
        return "makefile"

    ext = Path(file_path).suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext, "text")


def is_binary_file(file_path: Path) -> bool:
    """Check if a file appears to be binary by scanning the first 1024 bytes."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return True
    except (OSError, IOError):
        return True
    return False


class GitignoreParser:
    """Simple parser for gitignore rules when git binary or .git directory is not used."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir.resolve()
        self.rules: List[Tuple[Path, str, bool]] = []
        self._load_all_gitignores()

    def _load_all_gitignores(self) -> None:
        """Scan and collect all .gitignore files in the hierarchy."""
        for dirpath, _, filenames in os.walk(self.root_dir):
            if ".gitignore" in filenames:
                gitignore_path = Path(dirpath) / ".gitignore"
                self._parse_file(gitignore_path)

    def _parse_file(self, gitignore_path: Path) -> None:
        """Parse individual .gitignore file rules."""
        base_dir = gitignore_path.parent.resolve()
        try:
            with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    negated = line.startswith("!")
                    if negated:
                        pattern = line[1:].strip()
                    else:
                        pattern = line
                    self.rules.append((base_dir, pattern, negated))
        except (OSError, IOError):
            pass

    def is_ignored(self, path: Path) -> bool:
        """Test if a path is ignored according to parsed rules."""
        resolved = path.resolve()
        ignored = False
        for base_dir, pattern, negated in self.rules:
            try:
                rel = resolved.relative_to(base_dir).as_posix()
            except ValueError:
                continue

            matches = False
            # Directory-only match
            if pattern.endswith("/"):
                dir_pattern = pattern.rstrip("/")
                if path.is_dir() and (
                    fnmatch.fnmatch(rel, dir_pattern)
                    or fnmatch.fnmatch(rel, f"**/{dir_pattern}")
                    or fnmatch.fnmatch(rel.split("/")[0], dir_pattern)
                ):
                    matches = True
            elif "/" in pattern:
                # Anchored pattern
                pat = pattern.lstrip("/")
                if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, f"**/{pat}"):
                    matches = True
            else:
                # Matches filename anywhere
                name = resolved.name
                if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, f"**/{pattern}"):
                    matches = True

            if matches:
                ignored = not negated

        return ignored


class RepositoryWalker:
    """Walks a repository and yields indexable files respecting gitignore and filters."""

    def __init__(self, repo_path: Path, settings: Optional[Settings] = None):
        self.repo_path = repo_path.resolve()
        self.settings = settings or get_settings()

    def _should_skip_path(self, rel_path: str, abs_path: Path) -> bool:
        """Check if a file should be skipped based on blacklists and limits."""
        parts = [p.lower() for p in rel_path.split("/")]
        filename = parts[-1]

        # Skip ignored directory names anywhere in path
        for part in parts[:-1]:
            if part in self.settings.ignored_directories:
                return True

        # Skip ignored specific files
        if filename in self.settings.ignored_files:
            return True

        # Skip ignored extensions
        ext = abs_path.suffix.lower()
        if ext in self.settings.ignored_extensions:
            return True

        # Check file existence and size
        if not abs_path.is_file():
            return True

        try:
            size = abs_path.stat().st_size
            if size > self.settings.max_file_size_bytes or size == 0:
                return True
        except OSError:
            return True

        # Check binary
        if is_binary_file(abs_path):
            return True

        return False

    def walk(self) -> Iterator[Tuple[str, Path]]:
        """
        Yield (relative_path, absolute_path) for every indexable source file.
        Uses git ls-files if a git repo is present, otherwise falls back to GitignoreParser.
        """
        git_dir = self.repo_path / ".git"
        if git_dir.exists() and git_dir.is_dir():
            # Use git ls-files to accurately respect all gitignore rules
            try:
                proc = subprocess.run(
                    ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                    cwd=str(self.repo_path),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                for line in proc.stdout.splitlines():
                    rel_path = line.strip()
                    if not rel_path:
                        continue
                    abs_path = self.repo_path / rel_path
                    if not self._should_skip_path(rel_path, abs_path):
                        yield rel_path, abs_path
                return
            except (subprocess.SubprocessError, OSError):
                # Fall back to manual walk if git execution fails
                pass

        # Standalone manual walk using GitignoreParser
        gitignore_parser = GitignoreParser(self.repo_path)
        for root, dirs, files in os.walk(self.repo_path):
            # Prune ignored directories in-place
            dirs[:] = [
                d
                for d in dirs
                if d.lower() not in self.settings.ignored_directories
                and not gitignore_parser.is_ignored(Path(root) / d)
            ]

            for file in files:
                abs_path = Path(root) / file
                try:
                    rel_path = abs_path.relative_to(self.repo_path).as_posix()
                except ValueError:
                    continue

                if gitignore_parser.is_ignored(abs_path):
                    continue

                if not self._should_skip_path(rel_path, abs_path):
                    yield rel_path, abs_path


class CodeScopeDetector:
    """Extracts enclosing class and function/method names and identifies line boundaries."""

    JAVA_CLASS_PATTERN = re.compile(
        r"\b(?:class|interface|enum|record)\s+([a-zA-Z0-9_]+)"
    )
    JAVA_METHOD_PATTERN = re.compile(
        r"(?:public|protected|private|static|final|synchronized|abstract|default)\s+(?:[\w<>\[\], ?]+\s+)?([a-zA-Z0-9_]+)\s*\("
    )
    JAVA_CONTROL_KEYWORDS = {"if", "for", "while", "switch", "catch", "synchronized"}

    PYTHON_CLASS_PATTERN = re.compile(r"^\s*class\s+([a-zA-Z0-9_]+)")
    PYTHON_DEF_PATTERN = re.compile(r"^\s*(?:async\s+)?def\s+([a-zA-Z0-9_]+)\s*\(")

    JS_CLASS_PATTERN = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([a-zA-Z0-9_]+)")
    JS_FUNC_PATTERN = re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_]+)\s*\("
    )
    JS_ARROW_PATTERN = re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>"
    )

    @classmethod
    def analyze(cls, language: str, lines: List[str]) -> Tuple[Dict[int, Optional[str]], Set[int]]:
        """
        Analyze lines to produce:
        1. A mapping from line number (1-indexed) to enclosing scope string (e.g. 'Class > method').
        2. A set of boundary line numbers (1-indexed) where classes or functions start.
        """
        line_scopes: Dict[int, Optional[str]] = {}
        boundaries: Set[int] = set()

        if language == "java":
            current_class: Optional[str] = None
            current_method: Optional[str] = None
            for idx, line in enumerate(lines, 1):
                clean = line.strip()
                m_cls = cls.JAVA_CLASS_PATTERN.search(clean)
                m_meth = cls.JAVA_METHOD_PATTERN.search(clean)

                if m_cls:
                    current_class = m_cls.group(1)
                    current_method = None
                    boundaries.add(idx)
                elif m_meth:
                    name = m_meth.group(1)
                    if name not in cls.JAVA_CONTROL_KEYWORDS:
                        current_method = name
                        boundaries.add(idx)

                parts = []
                if current_class:
                    parts.append(current_class)
                if current_method:
                    parts.append(current_method)
                line_scopes[idx] = " > ".join(parts) if parts else None

        elif language == "python":
            current_class: Optional[str] = None
            current_func: Optional[str] = None
            for idx, line in enumerate(lines, 1):
                m_cls = cls.PYTHON_CLASS_PATTERN.search(line)
                m_def = cls.PYTHON_DEF_PATTERN.search(line)

                if m_cls:
                    current_class = m_cls.group(1)
                    current_func = None
                    boundaries.add(idx)
                elif m_def:
                    current_func = m_def.group(1)
                    boundaries.add(idx)

                parts = []
                if current_class:
                    parts.append(current_class)
                if current_func:
                    parts.append(current_func)
                line_scopes[idx] = " > ".join(parts) if parts else None

        elif language in ("javascript", "typescript"):
            current_class: Optional[str] = None
            current_func: Optional[str] = None
            for idx, line in enumerate(lines, 1):
                m_cls = cls.JS_CLASS_PATTERN.search(line)
                m_func = cls.JS_FUNC_PATTERN.search(line)
                m_arrow = cls.JS_ARROW_PATTERN.search(line)

                if m_cls:
                    current_class = m_cls.group(1)
                    current_func = None
                    boundaries.add(idx)
                elif m_func:
                    current_func = m_func.group(1)
                    boundaries.add(idx)
                elif m_arrow:
                    current_func = m_arrow.group(1)
                    boundaries.add(idx)

                parts = []
                if current_class:
                    parts.append(current_class)
                if current_func:
                    parts.append(current_func)
                line_scopes[idx] = " > ".join(parts) if parts else None

        else:
            # Fallback for other languages / documents
            for idx in range(1, len(lines) + 1):
                line_scopes[idx] = None

        return line_scopes, boundaries


class Chunker:
    """Splits source code files into overlapping chunks with scope headers."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def chunk_file(
        self,
        file_path: str,
        content: str,
        variant: str = "v2",
    ) -> List[CodeChunk]:
        """
        Split file content into ~40-60 line chunks with ~10 line overlap.
        variant: 'v1' (plain chunk text) or 'v2' (prefixed with context header).
        """
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        if total_lines == 0:
            return []

        language = detect_language(file_path)
        line_scopes, boundaries = CodeScopeDetector.analyze(language, lines)

        min_lines = self.settings.chunk_min_lines
        max_lines = self.settings.chunk_max_lines
        target_lines = self.settings.chunk_target_lines
        overlap = self.settings.chunk_overlap_lines

        chunks: List[CodeChunk] = []
        start = 1

        while start <= total_lines:
            # If remaining lines fit comfortably in a single chunk, emit final chunk
            if total_lines - start + 1 <= max_lines:
                chunks.append(
                    self._create_chunk(
                        file_path=file_path,
                        lines=lines,
                        start_line=start,
                        end_line=total_lines,
                        language=language,
                        line_scopes=line_scopes,
                        variant=variant,
                    )
                )
                break

            # Search window for natural boundary
            win_start = start + min_lines - 1
            win_end = min(total_lines, start + max_lines - 1)

            chosen_end: Optional[int] = None
            # Check boundaries from end to start of window to maximize chunk size
            for b in sorted(boundaries):
                if win_start <= b <= win_end:
                    chosen_end = b - 1
                    break

            # Fallback to target lines if no boundary is found
            if chosen_end is None or chosen_end < start + 20:
                chosen_end = min(total_lines, start + target_lines - 1)

            chunks.append(
                self._create_chunk(
                    file_path=file_path,
                    lines=lines,
                    start_line=start,
                    end_line=chosen_end,
                    language=language,
                    line_scopes=line_scopes,
                    variant=variant,
                )
            )

            # Advance with overlap
            next_start = chosen_end - overlap + 1
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks

    def _create_chunk(
        self,
        file_path: str,
        lines: List[str],
        start_line: int,
        end_line: int,
        language: str,
        line_scopes: Dict[int, Optional[str]],
        variant: str,
    ) -> CodeChunk:
        """Construct a CodeChunk object with appropriate headers based on variant."""
        chunk_lines = lines[start_line - 1 : end_line]
        raw_text = "".join(chunk_lines).rstrip()

        # Find the most representative scope in this chunk
        scope: Optional[str] = None
        for line_num in range(start_line, end_line + 1):
            s = line_scopes.get(line_num)
            if s:
                scope = s
                break

        if variant == "v2":
            prefix, suffix = LANGUAGE_COMMENT_PREFIX.get(language, ("//", ""))
            header_parts = [
                f"{prefix} File: {file_path} (lines {start_line}-{end_line}){suffix}".strip()
            ]
            if scope:
                header_parts.append(f"{prefix} Scope: {scope}{suffix}".strip())
            header = "\n".join(header_parts)
            text = f"{header}\n\n{raw_text}"
        else:
            text = raw_text

        chunk_id = f"{file_path}:{start_line}-{end_line}"
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            language=language,
            text=text,
            raw_text=raw_text,
            scope=scope,
            variant=variant,
        )
