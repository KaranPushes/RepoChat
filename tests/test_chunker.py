"""Unit tests for code chunker, gitignore walker, and scope detector."""

import tempfile
from pathlib import Path

import pytest

from repochat.chunker import (
    Chunker,
    CodeScopeDetector,
    GitignoreParser,
    RepositoryWalker,
    detect_language,
)
from repochat.config import Settings


def test_detect_language():
    """Verify language detection across various extensions."""
    assert detect_language("foo.java") == "java"
    assert detect_language("bar.py") == "python"
    assert detect_language("baz.ts") == "typescript"
    assert detect_language("baz.tsx") == "typescript"
    assert detect_language("app.jsx") == "javascript"
    assert detect_language("Dockerfile") == "dockerfile"
    assert detect_language("schema.sql") == "sql"
    assert detect_language("unknown.xyz") == "text"


def test_chunker_line_numbers_and_overlap():
    """Verify chunk boundaries, 1-indexed line numbers, and overlap between consecutive chunks."""
    # Create 120 lines of synthetic code
    lines = [f"int line_{i} = {i};\n" for i in range(1, 121)]
    content = "".join(lines)

    settings = Settings(
        chunk_min_lines=40,
        chunk_max_lines=60,
        chunk_target_lines=50,
        chunk_overlap_lines=10,
    )
    chunker = Chunker(settings=settings)
    chunks_v1 = chunker.chunk_file("Test.java", content, variant="v1")

    assert len(chunks_v1) >= 2
    # Verify 1-indexed line numbers
    assert chunks_v1[0].start_line == 1
    assert chunks_v1[0].end_line == 50
    assert chunks_v1[0].file_path == "Test.java"
    assert chunks_v1[0].language == "java"
    assert chunks_v1[0].variant == "v1"

    # Verify overlap: second chunk starts at chunk1.end_line - overlap + 1 = 50 - 10 + 1 = 41
    assert chunks_v1[1].start_line == 41
    assert chunks_v1[1].start_line < chunks_v1[0].end_line

    # Verify final chunk reaches the end of the file
    assert chunks_v1[-1].end_line == 120

    # Ensure every chunk text is not empty and matches raw_text for v1
    for c in chunks_v1:
        assert c.text == c.raw_text
        assert len(c.text) > 0


def test_chunker_variant_v2_header():
    """Verify variant v2 prepends context header with scope and line range."""
    sample_java = (
        "package com.example;\n"
        "public class AuthService {\n"
        "    public boolean validateToken(String token) {\n"
        "        return token != null;\n"
        "    }\n"
        "}\n"
    )
    settings = Settings()
    chunker = Chunker(settings=settings)
    chunks_v2 = chunker.chunk_file("AuthService.java", sample_java, variant="v2")

    assert len(chunks_v2) == 1
    chunk = chunks_v2[0]
    assert chunk.variant == "v2"
    assert "// File: AuthService.java (lines 1-6)" in chunk.text
    assert "Scope: AuthService" in chunk.text
    assert "public class AuthService" in chunk.raw_text


def test_scope_detection_java():
    """Verify enclosing class and method identification for Java."""
    lines = [
        "package com.example;\n",
        "public class UserController {\n",
        "    private String name;\n",
        "    public String getUser(String id) {\n",
        "        return id;\n",
        "    }\n",
        "}\n",
    ]
    scopes, boundaries = CodeScopeDetector.analyze("java", lines)
    assert 2 in boundaries  # class line
    assert 4 in boundaries  # method line
    assert scopes[2] == "UserController"
    assert scopes[4] == "UserController > getUser"
    assert scopes[5] == "UserController > getUser"


def test_scope_detection_python():
    """Verify class and def identification for Python."""
    lines = [
        "class ModelRunner:\n",
        "    def run_inference(self, prompt):\n",
        "        return prompt\n",
    ]
    scopes, boundaries = CodeScopeDetector.analyze("python", lines)
    assert 1 in boundaries
    assert 2 in boundaries
    assert scopes[1] == "ModelRunner"
    assert scopes[2] == "ModelRunner > run_inference"
    assert scopes[3] == "ModelRunner > run_inference"


def test_repository_walker_ignores():
    """Verify walker skips node_modules, build outputs, lockfiles, images, and files over 200 KB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # 1. Valid code file
        valid_file = root / "App.java"
        valid_file.write_text("public class App {}")

        # 2. node_modules file (must be skipped)
        nm_dir = root / "node_modules" / "pkg"
        nm_dir.mkdir(parents=True)
        (nm_dir / "index.js").write_text("console.log('hi')")

        # 3. build output file (must be skipped)
        target_dir = root / "target" / "classes"
        target_dir.mkdir(parents=True)
        (target_dir / "App.class").write_bytes(b"\xca\xfe\xba\xbe")

        # 4. Lockfile (must be skipped)
        (root / "package-lock.json").write_text("{}")

        # 5. Image file (must be skipped)
        (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        # 6. File over 200 KB (must be skipped)
        large_file = root / "large.txt"
        large_file.write_text("a" * (201 * 1024))

        # 7. Gitignored file
        gitignore = root / ".gitignore"
        gitignore.write_text("*.secret\nignored_folder/\n")
        (root / "my.secret").write_text("secret")

        ignored_sub = root / "ignored_folder"
        ignored_sub.mkdir()
        (ignored_sub / "secret.txt").write_text("secret")

        walker = RepositoryWalker(root)
        discovered_rel_paths = [rel for rel, _ in walker.walk()]

        assert "App.java" in discovered_rel_paths
        assert "package-lock.json" not in discovered_rel_paths
        assert "logo.png" not in discovered_rel_paths
        assert "large.txt" not in discovered_rel_paths
        assert not any("node_modules" in p for p in discovered_rel_paths)
        assert not any("target" in p for p in discovered_rel_paths)
        assert not any("secret" in p for p in discovered_rel_paths)
