"""Parser for source code and configuration files."""

from pathlib import Path

from localrag.core.models import Document, FileMetadata, FileType
from localrag.core.parsers.base import BaseParser


class CodeParser(BaseParser):
    """Parses source code files preserving precise formatting and indentation."""

    def can_handle(self, file_type: FileType) -> bool:
        return file_type == FileType.CODE

    def parse(self, metadata: FileMetadata) -> Document:
        file_path = Path(metadata.file_path)
        content = ""
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="latin-1")
            except Exception:
                content = file_path.read_text(encoding="utf-8", errors="replace")

        metadata.line_count = len(content.splitlines()) if content else 0

        return Document(
            metadata=metadata,
            content=content,
            chunks=[],
        )
