"""Parser for PDF documents extracting textual content with page tracking."""

from pathlib import Path

from pypdf import PdfReader

from localrag.core.models import Document, FileMetadata, FileType
from localrag.core.parsers.base import BaseParser


class PDFParser(BaseParser):
    """Extracts text from PDF documents preserving page divisions."""

    def can_handle(self, file_type: FileType) -> bool:
        return file_type == FileType.PDF

    def parse(self, metadata: FileMetadata) -> Document:
        file_path = Path(metadata.file_path)
        pages_text = []

        try:
            reader = PdfReader(str(file_path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(f"--- Page {i + 1} ---\n{text.strip()}")
        except Exception as e:
            pages_text.append(f"[Error reading PDF: {e}]")

        full_content = "\n\n".join(pages_text)
        metadata.line_count = len(full_content.splitlines())

        return Document(
            metadata=metadata,
            content=full_content,
            chunks=[],
        )
