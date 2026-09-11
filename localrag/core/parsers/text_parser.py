"""Parser for Markdown, HTML, and Plain Text files."""

import html
import re
from pathlib import Path
from localrag.core.models import Document, FileMetadata, FileType
from localrag.core.parsers.base import BaseParser


class TextParser(BaseParser):
    """Parses text, markdown, and lightweight HTML files."""

    def can_handle(self, file_type: FileType) -> bool:
        return file_type in (FileType.TEXT, FileType.MARKDOWN, FileType.HTML)

    def parse(self, metadata: FileMetadata) -> Document:
        file_path = Path(metadata.file_path)
        content = ""
        # Try UTF-8 with fallback to Latin-1
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="latin-1")
            except Exception:
                content = file_path.read_text(encoding="utf-8", errors="replace")

        # Basic HTML tag stripping if HTML format
        if metadata.file_type == FileType.HTML:
            content = self._clean_html(content)

        # Update actual line count
        metadata.line_count = len(content.splitlines()) if content else 0

        return Document(
            metadata=metadata,
            content=content,
            chunks=[],
        )

    def _clean_html(self, raw_html: str) -> str:
        """Strip script, style, and HTML tags while keeping readable text."""
        # Remove script and style tags and contents
        cleaned = re.sub(r"<(script|style).*?>.*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
        # Remove comments
        cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
        # Convert break and paragraph tags to newlines
        cleaned = re.sub(r"<(br|p|div|tr|h[1-6]).*?>", "\n", cleaned, flags=re.IGNORECASE)
        # Strip all remaining tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        # Unescape HTML entities
        cleaned = html.unescape(cleaned)
        # Clean extra whitespace
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()
