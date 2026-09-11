"""Directory crawler and file harvester with format classification."""

import os
from collections.abc import Iterator
from pathlib import Path

from localrag.core.models import FileMetadata, FileType
from localrag.utils.hasher import compute_file_sha256
from localrag.utils.ignore import IgnoreFilter

# Mapping of file extensions to FileType categories
EXTENSION_MAP: dict[str, FileType] = {
    # Markdown
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".mdown": FileType.MARKDOWN,
    ".mdx": FileType.MARKDOWN,
    # Plain Text
    ".txt": FileType.TEXT,
    ".text": FileType.TEXT,
    ".rst": FileType.TEXT,
    ".asciidoc": FileType.TEXT,
    ".adoc": FileType.TEXT,
    ".csv": FileType.TEXT,
    ".tsv": FileType.TEXT,
    ".log": FileType.TEXT,
    # HTML
    ".html": FileType.HTML,
    ".htm": FileType.HTML,
    ".xhtml": FileType.HTML,
    # PDF
    ".pdf": FileType.PDF,
    # Code: Python
    ".py": FileType.CODE,
    ".pyi": FileType.CODE,
    # Code: JavaScript / TypeScript
    ".js": FileType.CODE,
    ".jsx": FileType.CODE,
    ".mjs": FileType.CODE,
    ".cjs": FileType.CODE,
    ".ts": FileType.CODE,
    ".tsx": FileType.CODE,
    # Code: Systems & Backend
    ".go": FileType.CODE,
    ".rs": FileType.CODE,
    ".java": FileType.CODE,
    ".c": FileType.CODE,
    ".cpp": FileType.CODE,
    ".cc": FileType.CODE,
    ".cxx": FileType.CODE,
    ".h": FileType.CODE,
    ".hpp": FileType.CODE,
    ".cs": FileType.CODE,
    ".php": FileType.CODE,
    ".rb": FileType.CODE,
    ".swift": FileType.CODE,
    ".kt": FileType.CODE,
    ".scala": FileType.CODE,
    # Config & Markup
    ".json": FileType.CODE,
    ".jsonc": FileType.CODE,
    ".yaml": FileType.CODE,
    ".yml": FileType.CODE,
    ".toml": FileType.CODE,
    ".xml": FileType.CODE,
    ".sql": FileType.CODE,
    ".sh": FileType.CODE,
    ".bash": FileType.CODE,
    ".zsh": FileType.CODE,
    ".ps1": FileType.CODE,
    ".dockerfile": FileType.CODE,
}

LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sql": "sql",
    ".sh": "bash",
    ".ps1": "powershell",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".html": "html",
    ".md": "markdown",
}


class FileHarvester:
    """Discovers and filters indexable files within a target directory."""

    def __init__(
        self,
        root_dir: str | Path,
        ignore_filter: IgnoreFilter | None = None,
        max_file_size_bytes: int = 5 * 1024 * 1024,  # 5MB per file limit
        allowed_types: set[FileType] | None = None,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.ignore_filter = ignore_filter or IgnoreFilter(self.root_dir)
        self.max_file_size_bytes = max_file_size_bytes
        self.allowed_types = allowed_types

    def harvest(self) -> list[FileMetadata]:
        """Traverse directory and return metadata for all valid indexable files."""
        collected: list[FileMetadata] = []
        for meta in self.iter_files():
            collected.append(meta)
        return collected

    def iter_files(self) -> Iterator[FileMetadata]:
        """Iterate lazily over valid files in workspace."""
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            return

        for root, dirs, files in os.walk(self.root_dir):
            rel_root = os.path.relpath(root, self.root_dir)
            if rel_root == ".":
                rel_root = ""

            # Filter out ignored directories in-place to avoid traversing subtrees
            dirs[:] = [
                d for d in dirs if not self.ignore_filter.is_ignored(os.path.join(rel_root, d).replace("\\", "/") + "/")
            ]

            for file_name in files:
                file_rel_path = os.path.join(rel_root, file_name).replace("\\", "/")
                if self.ignore_filter.is_ignored(file_rel_path):
                    continue

                abs_path = Path(root) / file_name
                file_meta = self._inspect_file(abs_path, file_rel_path)
                if file_meta:
                    yield file_meta

    def _inspect_file(self, abs_path: Path, rel_path: str) -> FileMetadata | None:
        """Inspect and categorize a single file."""
        try:
            stat = abs_path.stat()
        except (OSError, PermissionError):
            return None

        # Check size threshold
        if stat.st_size > self.max_file_size_bytes or stat.st_size == 0:
            return None

        # Determine file type
        suffix = abs_path.suffix.lower()
        if not suffix and abs_path.name.lower() in ("dockerfile", "makefile", "license", "readme"):
            file_type = FileType.TEXT
            language = abs_path.name.lower()
        else:
            file_type = EXTENSION_MAP.get(suffix, FileType.UNKNOWN)
            language = LANGUAGE_BY_EXT.get(suffix)

        if file_type == FileType.UNKNOWN:
            return None

        if self.allowed_types and file_type not in self.allowed_types:
            return None

        try:
            sha256 = compute_file_sha256(abs_path)
        except Exception:
            return None

        line_count = 0
        if file_type != FileType.PDF:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    line_count = sum(1 for _ in f)
            except Exception:
                pass

        return FileMetadata(
            file_path=str(abs_path),
            relative_path=rel_path,
            file_type=file_type,
            file_size_bytes=stat.st_size,
            sha256=sha256,
            line_count=line_count,
            last_modified=stat.st_mtime,
            language=language,
        )
