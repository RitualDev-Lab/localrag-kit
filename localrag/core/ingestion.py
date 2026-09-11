"""Unified ingestion orchestrator coordinating file harvesting, parsing, chunking, and database indexing."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from localrag.providers.base import BaseEmbeddingProvider

from pydantic import BaseModel

from localrag.core.chunkers import ChunkerRegistry
from localrag.core.harvester import FileHarvester
from localrag.core.models import Chunk, Document, FileMetadata
from localrag.core.parsers import ParserRegistry
from localrag.storage.sqlite_store import SQLiteStore
from localrag.utils.ignore import IgnoreFilter


class IngestionStats(BaseModel):
    """Aggregate metrics of an in-memory ingestion run."""

    total_files_scanned: int = 0
    total_files_parsed: int = 0
    total_chunks_produced: int = 0
    total_characters: int = 0
    total_estimated_tokens: int = 0
    skipped_files: int = 0
    failed_files: int = 0


class IndexingStats(BaseModel):
    """Aggregate metrics of a database indexing run."""

    total_files_scanned: int = 0
    files_added_or_modified: int = 0
    files_unchanged: int = 0
    files_deleted: int = 0
    chunks_indexed: int = 0
    failed_files: int = 0


class IngestionPipeline:
    """End-to-end pipeline: Directory -> Discovered Files -> Documents -> Chunks -> Storage."""

    def __init__(
        self,
        root_dir: str | Path,
        custom_ignore_patterns: list[str] | None = None,
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
        progress_callback: Callable[[FileMetadata, int, int], None] | None = None,
    ) -> tuple[list[Document], IngestionStats]:
        """Harvest, parse, and chunk all indexable files in the directory in memory."""
        files = self.harvester.harvest()
        stats = IngestionStats(total_files_scanned=len(files))
        documents: list[Document] = []

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

    def index_to_store(
        self,
        store: SQLiteStore,
        incremental: bool = True,
        embedding_provider: Optional["BaseEmbeddingProvider"] = None,
        progress_callback: Callable[[FileMetadata, int, int], None] | None = None,
    ) -> IndexingStats:
        """
        Harvest files and write them to SQLiteStore with FTS5 and vector tables.
        If incremental=True, only modified/new files are re-chunked, and deleted files are purged.
        If embedding_provider is provided, generates dense float vectors for chunks.
        """
        files = self.harvester.harvest()
        stats = IndexingStats(total_files_scanned=len(files))

        if incremental:
            to_index, to_delete = store.get_changed_files(files)
            stats.files_deleted = len(to_delete)
            for del_path in to_delete:
                store.delete_file(del_path)

            stats.files_unchanged = len(files) - len(to_index)
            stats.files_added_or_modified = len(to_index)
        else:
            to_index = files
            stats.files_added_or_modified = len(files)

        total_to_process = len(to_index)

        for i, meta in enumerate(to_index, start=1):
            if progress_callback:
                progress_callback(meta, i, total_to_process)

            try:
                doc = self.parser_registry.parse(meta)
                chunks = self.chunker_registry.chunk_document(doc)

                # Compute dense embeddings if provider enabled
                if embedding_provider and chunks:
                    chunk_texts = [c.text for c in chunks]
                    embeddings = embedding_provider.embed_batch(chunk_texts)
                    for c, emb in zip(chunks, embeddings):
                        c.embedding = emb

                # Upsert file record
                store.upsert_file(meta)
                # Clear previous chunks for this file if re-indexing
                store.delete_chunks_for_file(meta.relative_path)
                # Insert new chunks
                store.upsert_chunks(chunks)

                stats.chunks_indexed += len(chunks)
            except Exception:
                stats.failed_files += 1

        return stats

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
