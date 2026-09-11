"""Unified ingestion orchestrator coordinating file harvesting, parsing, and chunking."""

from pathlib import Path
from typing import Callable, Iterator, List, Optional
from pydantic import BaseModel, Field
from localrag.core.chunkers import ChunkerRegistry
from localrag.core.harvester import FileHarvester
from localrag.core.models import Chunk, Document, FileMetadata
from localrag.core.parsers import ParserRegistry
from localrag.utils.ignore import IgnoreFilter


class IngestionStats(BaseModel):
    """Aggregate metrics of an ingestion run."""
    total_files_scanned: int = 0
    total_files_parsed: int = 0
    total_chunks_produced: int = 0
    total_characters: int = 0
    total_estimated_tokens: int = 0
    skipped_files: int = 0
    failed_files: int = 0


class IngestionPipeline:
    """End-to-end pipeline: Directory -> Discovered Files -> Documents -> Chunks."""

    def __init__(
        self,
        root_dir: str | Path,
        custom_ignore_patterns: Optional[List[str]] = None,
        max_file_size_bytes: int = 5 * 1024 * 1024,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.ignore_filter = IgnoreFilter(self.root_dir, custom_patterns=custom_ignore_patterns)
        self.harvester = FileHarvester(
            root_dir=self.root_dir,
            ignore_filter=self.ignore_filter,
            max_file_size_bytes=max_file_size_bytes,
        )
        self.parser_registry = ParserRegistry()
        self.chunker_registry = ChunkerRegistry()

    def process_directory(
        self,
        progress_callback: Optional[Callable[[FileMetadata, int, int], None]] = None,
    ) -> tuple[List[Document], IngestionStats]:
        """Harvest, parse, and chunk all indexable files in the directory."""
        files = self.harvester.harvest()
        stats = IngestionStats(total_files_scanned=len(files))
        documents: List[Document] = []

        for i, meta in enumerate(files, start=1):
            if progress_callback:
                progress_callback(meta, i, len(files))

            try:
                doc = self.parser_registry.parse(meta)
                chunks = self.chunker_registry.chunk_document(doc)

                stats.total_files_parsed += 1
                stats.total_chunks_produced += len(chunks)
                stats.total_characters += sum(c.metadata.char_count for c in chunks)
                stats.total_estimated_tokens += sum(c.metadata.estimated_tokens for c in chunks)
                documents.append(doc)
            except Exception:
                stats.failed_files += 1

        return documents, stats

    def iter_chunks(self) -> Iterator[Chunk]:
        """Stream chunks lazily one file at a time to keep memory minimal."""
        for meta in self.harvester.iter_files():
            try:
                doc = self.parser_registry.parse(meta)
                chunks = self.chunker_registry.chunk_document(doc)
                for chunk in chunks:
                    yield chunk
            except Exception:
                continue
