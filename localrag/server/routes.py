"""API route definitions for LocalRAG-Kit."""

import json
import time
from pathlib import Path
from typing import AsyncIterator, List, Optional
from fastapi import APIRouter, HTTPException, Query, Request
from starlette.responses import StreamingResponse

from localrag.core.ingestion import IngestionPipeline
from localrag.providers import BaseEmbeddingProvider, BaseLLMProvider
from localrag.retrieval.orchestrator import RAGOrchestrator
from localrag.server.schemas import (
    ChatRequest,
    ChunkDetailResponse,
    ReindexRequest,
    ReindexResponse,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)
from localrag.storage.sqlite_store import SQLiteStore


def create_router(
    workspace_path: Path,
    store: SQLiteStore,
    embed_provider: BaseEmbeddingProvider,
    llm_provider: BaseLLMProvider,
) -> APIRouter:
    """Build and return configured FastAPI router."""
    router = APIRouter(prefix="/api")
    orchestrator = RAGOrchestrator(
        store=store,
        embedding_provider=embed_provider,
        llm_provider=llm_provider,
    )
    pipeline = IngestionPipeline(workspace_path)

    @router.get("/status", response_model=StatusResponse)
    def get_status():
        """Fetch system statistics, index status, and active provider metadata."""
        stats = store.get_stats()
        return StatusResponse(
            workspace_path=str(workspace_path),
            db_path=str(store.db_path),
            stats=stats,
            embedding_provider=embed_provider.name,
            llm_provider=llm_provider.model_name,
        )

    @router.get("/files")
    def list_files():
        """List all indexed physical files in the target directory."""
        return {"files": store.get_all_files()}

    @router.post("/reindex", response_model=ReindexResponse)
    def reindex_workspace(req: ReindexRequest):
        """Trigger an incremental or full re-indexing of the workspace."""
        t0 = time.perf_counter()
        active_embedder = embed_provider if req.embed else None

        stats = pipeline.index_to_store(
            store=store,
            incremental=not req.full,
            embedding_provider=active_embedder,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return ReindexResponse(
            files_scanned=stats.total_files_scanned,
            files_updated=stats.files_added_or_modified,
            files_unchanged=stats.files_unchanged,
            files_deleted=stats.files_deleted,
            chunks_indexed=stats.chunks_indexed,
            time_taken_ms=round(elapsed_ms, 2),
        )

    @router.post("/search", response_model=SearchResponse)
    def execute_search(req: SearchRequest):
        """Execute hybrid RRF, BM25, or vector similarity search."""
        results = orchestrator.retrieve(
            query=req.query,
            top_k=req.limit,
            mode=req.mode,
        )
        return SearchResponse(
            query=req.query,
            mode=req.mode,
            count=len(results),
            results=results,
        )

    @router.post("/chat")
    def stream_chat(req: ChatRequest):
        """
        Stream grounded AI responses with real-time SSE tokens and citation metadata.
        Sends events in format:
          data: {"type": "citations", "citations": [...]}
          data: {"type": "token", "token": "..."}
          data: {"type": "done"}
        """
        def event_generator():
            try:
                # 1. First retrieve & send citations
                results = orchestrator.retrieve(req.query, top_k=req.top_k, mode=req.mode)
                ctx = orchestrator.synthesizer.synthesize(results)

                # Send citations event
                citations_payload = [c.model_dump() for c in ctx.citations]
                yield f"data: {json.dumps({'type': 'citations', 'citations': citations_payload})}\n\n"

                if not ctx.formatted_context:
                    yield f"data: {json.dumps({'type': 'token', 'token': 'No relevant documents or code chunks were found in the index.'})}\n\n"
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return

                # 2. Stream LLM tokens
                from localrag.retrieval.orchestrator import SYSTEM_PROMPT
                prompt = (
                    f"Context from local files:\n\n{ctx.formatted_context}\n\n"
                    f"User Question:\n{req.query}\n\nAnswer:"
                )

                for token in llm_provider.stream_generate(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=req.temperature,
                ):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/chunk/{chunk_id}", response_model=ChunkDetailResponse)
    def get_chunk_detail(chunk_id: str):
        """Retrieve chunk details and surrounding file context."""
        chunk = store.get_chunk(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")

        parent_file = store.get_file(chunk.metadata.relative_path)
        snippet = None

        # Read surrounding lines from physical file if exists
        abs_path = workspace_path / chunk.metadata.relative_path
        if abs_path.is_file():
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    file_lines = f.readlines()
                start = max(0, chunk.metadata.start_line - 5)
                end = min(len(file_lines), chunk.metadata.end_line + 5)
                snippet = "".join(file_lines[start:end])
            except Exception:
                pass

        return ChunkDetailResponse(
            chunk=chunk,
            parent_file=parent_file,
            file_content_snippet=snippet,
        )

    @router.get("/file-content")
    def get_file_content(path: str = Query(..., description="Relative file path")):
        """Read raw text of a physical file in the workspace."""
        abs_path = (workspace_path / path).resolve()
        # Security: prevent directory traversal outside workspace
        if not str(abs_path).startswith(str(workspace_path.resolve())):
            raise HTTPException(status_code=403, detail="Access denied outside workspace")

        if not abs_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
            return {"path": path, "content": content}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
