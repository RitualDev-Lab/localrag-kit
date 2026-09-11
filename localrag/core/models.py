"""Data models for LocalRAG-Kit core ingestion, chunking, and search."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FileType(str, Enum):
    """Supported file categories for specialized chunking."""
    MARKDOWN = "markdown"
    CODE = "code"
    PDF = "pdf"
    TEXT = "text"
    HTML = "html"
    UNKNOWN = "unknown"


class FileMetadata(BaseModel):
    """Metadata describing an ingested physical file."""
    file_path: str = Field(description="Absolute path to the file on disk")
    relative_path: str = Field(description="Relative path from workspace root")
    file_type: FileType = Field(default=FileType.UNKNOWN, description="Detected category")
    file_size_bytes: int = Field(default=0, description="Size in bytes")
    sha256: str = Field(description="SHA-256 hash of file content for change detection")
    line_count: int = Field(default=0, description="Total lines in file")
    last_modified: float = Field(default=0.0, description="POSIX timestamp of modification")
    language: Optional[str] = Field(default=None, description="Programming language if code")


class ChunkMetadata(BaseModel):
    """Metadata tracking the origin and structural context of a chunk."""
    chunk_id: str = Field(description="Unique deterministic ID of this chunk")
    file_path: str = Field(description="Absolute path of parent file")
    relative_path: str = Field(description="Relative path of parent file")
    file_type: FileType = Field(description="Category of parent file")
    start_line: int = Field(default=1, description="1-indexed starting line")
    end_line: int = Field(default=1, description="1-indexed ending line")
    char_count: int = Field(default=0, description="Total characters in chunk text")
    estimated_tokens: int = Field(default=0, description="Approximate token count")
    section_title: Optional[str] = Field(default=None, description="Markdown header or symbol")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


class Chunk(BaseModel):
    """A discrete searchable text unit with full provenance."""
    text: str = Field(description="Extracted chunk text content")
    metadata: ChunkMetadata = Field(description="Line numbers, file path, and context")
    embedding: Optional[List[float]] = Field(default=None, description="Dense vector embedding")

    @property
    def id(self) -> str:
        return self.metadata.chunk_id


class Document(BaseModel):
    """A parsed file containing raw text and optional pre-chunked segments."""
    metadata: FileMetadata = Field(description="File system and identity metadata")
    content: str = Field(description="Full text content of document")
    chunks: List[Chunk] = Field(default_factory=list, description="Derived chunks")
