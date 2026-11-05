"""Helper tool to sample files, classes, and methods from target_repo to assist question drafting."""

import argparse
import random
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from repochat.chunker import Chunker, CodeScopeDetector, RepositoryWalker, detect_language
from repochat.config import get_settings


def sample_repository(repo_path: Path, count: int = 15) -> None:
    """Print a random sample of files with detected classes, methods, and line ranges."""
    settings = get_settings()
    walker = RepositoryWalker(repo_path, settings)
    files = list(walker.walk())

    if not files:
        print(f"No indexable files found in {repo_path}")
        return

    # Focus on core code files (Java, TypeScript, SQL, Markdown)
    code_files = [
        (rel, abs_p)
        for rel, abs_p in files
        if any(rel.endswith(ext) for ext in [".java", ".ts", ".tsx", ".sql", ".md"])
    ]

    selected = random.sample(code_files, min(count, len(code_files)))

    print("\n" + "=" * 70)
    print(f"RepoChat Candidate Sampling for eval/questions.csv ({len(selected)} samples)")
    print("Use these files and line ranges to draft ground-truth questions.")
    print("=" * 70 + "\n")

    for i, (rel_path, abs_path) in enumerate(selected, 1):
        try:
            content = abs_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            lang = detect_language(rel_path)
            scopes, boundaries = CodeScopeDetector.analyze(lang, lines)

            # Find interesting methods/scopes
            detected_scopes = sorted(set(s for s in scopes.values() if s))

            print(f"[{i}] File: {rel_path} ({len(lines)} lines)")
            if detected_scopes:
                print(f"    Detected Scopes: {', '.join(detected_scopes[:4])}")

            # Print first non-empty snippet preview
            preview_lines = [l for l in lines if l.strip() and not l.strip().startswith("import")][:4]
            if preview_lines:
                print("    Preview:")
                for pl in preview_lines[:3]:
                    print(f"      {pl[:80]}")
            print("-" * 70)
        except Exception as e:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample candidate functions/files from codebase.")
    parser.add_argument("--repo", default="./target_repo", help="Path to repository")
    parser.add_argument("--count", type=int, default=15, help="Number of files to sample")
    args = parser.parse_args()

    sample_repository(Path(args.repo), count=args.count)
