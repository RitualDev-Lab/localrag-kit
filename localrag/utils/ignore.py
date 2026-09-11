"""Ignore pattern matcher supporting .gitignore and .ragignore with built-in defaults."""

import os
from pathlib import Path
from typing import List, Optional
import pathspec


DEFAULT_IGNORE_PATTERNS = [
    # VCS & Package Managers
    ".git/",
    ".git/**",
    ".github/",
    ".svn/",
    ".hg/",
    "node_modules/",
    "node_modules/**",
    "__pycache__/",
    "__pycache__/**",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".venv/",
    ".venv/**",
    "venv/",
    "venv/**",
    "env/",
    ".env",
    ".env.*",
    # LocalRAG Index Directory
    ".localrag/",
    ".localrag/**",
    # Build & Distribution outputs
    "dist/",
    "build/",
    "*.egg-info/",
    ".eggs/",
    ".tox/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".coverage",
    "htmlcov/",
    # IDE & OS Artifacts
    ".vscode/",
    ".idea/",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    # Binaries, Archives & Large Blobs
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.rar",
    "*.7z",
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    "*.bin",
    "*.iso",
    # Media files (non-textual)
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.svg",
    "*.ico",
    "*.mp3",
    "*.mp4",
    "*.avi",
    "*.mov",
    "*.wav",
    "*.ttf",
    "*.woff",
    "*.woff2",
    "*.eot",
]


class IgnoreFilter:
    """Evaluates whether a relative file path should be ignored during indexing."""

    def __init__(self, root_dir: str | Path, custom_patterns: Optional[List[str]] = None):
        self.root_dir = Path(root_dir).resolve()
        patterns = list(DEFAULT_IGNORE_PATTERNS)

        if custom_patterns:
            patterns.extend(custom_patterns)

        # Load .gitignore if present in root
        gitignore_path = self.root_dir / ".gitignore"
        if gitignore_path.is_file():
            try:
                with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                    patterns.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
            except Exception:
                pass

        # Load .ragignore if present in root
        ragignore_path = self.root_dir / ".ragignore"
        if ragignore_path.is_file():
            try:
                with open(ragignore_path, "r", encoding="utf-8", errors="ignore") as f:
                    patterns.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
            except Exception:
                pass

        self.spec = pathspec.PathSpec.from_lines("gitignore", patterns)

    def is_ignored(self, relative_path: str) -> bool:
        """Check if relative path matches any ignore rule."""
        # Normalize to POSIX format for pathspec
        normalized = relative_path.replace("\\", "/")
        if not normalized:
            return False
        
        # Check parent folder segments
        parts = normalized.split("/")
        for i in range(1, len(parts)):
            sub_dir = "/".join(parts[:i]) + "/"
            if self.spec.match_file(sub_dir):
                return True

        return bool(self.spec.match_file(normalized))
