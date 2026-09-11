"""Chunker registry and automatic format dispatching."""

from typing import Dict, List, Optional
from localrag.core.chunkers.base import BaseChunker
from localrag.core.chunkers.code import CodeChunker
from localrag.core.chunkers.markdown import MarkdownChunker
from localrag.core.chunkers.sliding_window import SlidingWindowChunker
from localrag.core.models import Chunk, Document, FileType


class ChunkerRegistry:
    """Selects and applies the appropriate chunker for each document type."""

    def __init__(self):
        self.markdown_chunker = MarkdownChunker()
        self.code_chunker = CodeChunker()
        self.default_chunker = SlidingWindowChunker()

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a document according to its file type."""
        file_type = document.metadata.file_type

        if file_type == FileType.MARKDOWN:
            chunks = self.markdown_chunker.chunk(document)
        elif file_type == FileType.CODE:
            chunks = self.code_chunker.chunk(document)
        else:
            chunks = self.default_chunker.chunk(document)

        document.chunks = chunks
        return chunks


__all__ = [
    "BaseChunker",
    "SlidingWindowChunker",
    "MarkdownChunker",
    "CodeChunker",
    "ChunkerRegistry",
]
