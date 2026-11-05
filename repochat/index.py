"""CLI entry point for indexing a repository."""

import argparse
import sys
from pathlib import Path

from repochat.indexer import RepoIndexer


def main() -> None:
    """Parse CLI arguments and run repo indexing."""
    parser = argparse.ArgumentParser(
        prog="python -m repochat.index",
        description="Index a repository into RepoChat ChromaDB vector store.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="./target_repo",
        help="Path to the repository to index (default: ./target_repo)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["v1", "v2"],
        default="v2",
        help="Chunking variant: v1 (plain text) or v2 (context header with scope/file)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing and overwrite existing collection in ChromaDB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )

    args = parser.parse_args()

    repo_target = args.repo.strip()
    is_url = (
        repo_target.startswith("http://")
        or repo_target.startswith("https://")
        or repo_target.startswith("git@")
        or "github.com/" in repo_target
    )

    if not is_url:
        repo_path = Path(repo_target)
        if not repo_path.exists():
            print(f"Error: Local repository path '{repo_path}' does not exist.", file=sys.stderr)
            sys.exit(1)

    print(f"=== Starting RepoChat Indexer ===")
    print(f"Repository: {repo_target}")
    print(f"Variant:    {args.variant}")
    print(f"Reindex:    {args.reindex}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 33)

    indexer = RepoIndexer()
    try:
        summary = indexer.index_repository(
            repo_path=repo_target,
            variant=args.variant,
            reindex=args.reindex,
            batch_size=args.batch_size,
        )
        summary.print_summary()
    except KeyboardInterrupt:
        print("\n[!] Indexing interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[!] Error during indexing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
