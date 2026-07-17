"""Repository-hygiene check: generated ChromaDB files must stay git-ignored.

This does not touch the real `data/vector_store/` directory -- it only
statically verifies the `.gitignore` rule that keeps it ignored is present,
catching an accidental regression (e.g. someone "cleaning up" .gitignore)
before generated Chroma database files could ever be staged.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_gitignore_excludes_vector_store_contents():
    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "data/vector_store/*" in gitignore_text
    assert "!data/vector_store/.gitkeep" in gitignore_text


def test_no_chroma_files_are_tracked_by_git():
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "data/vector_store"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    generated_files = [f for f in tracked if not f.endswith(".gitkeep")]
    assert generated_files == []
