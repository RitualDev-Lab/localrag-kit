"""Parser registry and dispatcher."""

from typing import Dict, List, Optional

from localrag.core.models import Document, FileMetadata, FileType
from localrag.core.parsers.base import BaseParser
from localrag.core.parsers.code_parser import CodeParser
from localrag.core.parsers.pdf_parser import PDFParser
from localrag.core.parsers.text_parser import TextParser


class ParserRegistry:
    """Dispatches files to their registered specialized parser."""

    def __init__(self, parsers: list[BaseParser] | None = None):
        self.parsers: list[BaseParser] = parsers or [
            TextParser(),
            CodeParser(),
            PDFParser(),
        ]

    def get_parser(self, file_type: FileType) -> BaseParser | None:
        """Find the first matching parser for a FileType."""
        for parser in self.parsers:
            if parser.can_handle(file_type):
                return parser
        return None

    def parse(self, metadata: FileMetadata) -> Document:
        """Parse metadata into a Document using appropriate parser."""
        parser = self.get_parser(metadata.file_type)
        if not parser:
            raise ValueError(f"No parser registered for file type: {metadata.file_type}")
        return parser.parse(metadata)


__all__ = ["BaseParser", "CodeParser", "PDFParser", "ParserRegistry", "TextParser"]
