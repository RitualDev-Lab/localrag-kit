"""Pydantic request and response schemas for LocalRAG-Kit REST & SSE API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from localrag.core.models import Chunk, FileMetadata
from localrag.retrieval.synthesizer import SourceCitation
from localrag.storage.sqlite_store import SearchResult, StoreStats


class StatusResponse(BaseModel):
    workspace_path: str
    db_path: str
    stats: StoreStats
    embedding_provider: str
    llm_provider: str


class SearchRequest(BaseModel):
    query: str
    mode: str = Field(default="hybrid", description="'hybrid', 'bm25', or 'vector'")
    limit: int = Field(default=8, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    mode: str
    count: int
    results: List[SearchResult]


class ChatRequest(BaseModel):
    query: str
    mode: str = Field(default="hybrid", description="'hybrid', 'bm25', or 'vector'")
    top_k: int = Field(default=6, ge=1, le=20)
    temperature: float = Field(default=0.5, ge=0.0, le=1.5)


class ReindexRequest(BaseModel):
    full: bool = Field(default=False, description="Whether to rebuild from scratch")
    embed: bool = Field(default=True, description="Whether to compute dense embeddings")


class ReindexResponse(BaseModel):
    files_scanned: int
    files_updated: int
    files_unchanged: int
    files_deleted: int
    chunks_indexed: int
    time_taken_ms: float


class ChunkDetailResponse(BaseModel):
    chunk: Chunk
    parent_file: Optional[FileMetadata] = None
    file_content_snippet: Optional[str] = None
