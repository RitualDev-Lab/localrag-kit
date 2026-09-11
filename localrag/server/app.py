"""FastAPI application factory for LocalRAG-Kit."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from localrag import __version__
from localrag.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    get_embedding_provider,
    get_llm_provider,
)
from localrag.server.routes import create_router
from localrag.storage.sqlite_store import SQLiteStore


def create_app(
    workspace_path: str | Path,
    db_path: str | Path | None = None,
    embedding_provider: BaseEmbeddingProvider | None = None,
    llm_provider: BaseLLMProvider | None = None,
    embed_provider_name: str = "fast",
    llm_provider_name: str = "auto",
) -> FastAPI:
    """Create and configure FastAPI application for a specific target workspace."""
    ws_path = Path(workspace_path).resolve()
    database_path = Path(db_path).resolve() if db_path else ws_path / ".localrag" / "index.db"

    # Resolve store & providers
    store = SQLiteStore(database_path)
    embedder = embedding_provider or get_embedding_provider(embed_provider_name)
    llm = llm_provider or get_llm_provider(llm_provider_name)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        store.close()

    app = FastAPI(
        title="LocalRAG-Kit API",
        version=__version__,
        description="100% offline, privacy-first local RAG search & chat engine",
        lifespan=lifespan,
    )

    # Enable CORS for local web interfaces
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach router
    router = create_router(
        workspace_path=ws_path,
        store=store,
        embed_provider=embedder,
        llm_provider=llm,
    )
    app.include_router(router)

    # Mount static assets if web UI build exists (Phase 6)
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
