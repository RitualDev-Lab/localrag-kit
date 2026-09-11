"""Tests for FastAPI server, REST endpoints, and SSE streaming chat."""

import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from localrag.providers import FastFeatureEmbeddingProvider, MockLLMProvider
from localrag.server.app import create_app


def test_server_endpoints_and_streaming():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "README.md").write_text("# Test Docs\nExplaining FastAPI integration.")
        (root / "main.py").write_text("def run():\n    return 'hello world'\n")

        embedder = FastFeatureEmbeddingProvider()
        llm = MockLLMProvider()

        app = create_app(
            workspace_path=root,
            embedding_provider=embedder,
            llm_provider=llm,
        )

        with TestClient(app) as client:
            # 1. Test status (initially 0 files indexed)
            res_status = client.get("/api/status")
            assert res_status.status_code == 200
            data_status = res_status.json()
            assert data_status["embedding_provider"] == "fast-local"
            assert data_status["llm_provider"] == "mock-agent"
            assert data_status["stats"]["total_files"] == 0

            # 2. Trigger reindex
            res_reindex = client.post("/api/reindex", json={"full": False, "embed": True})
            assert res_reindex.status_code == 200
            data_reindex = res_reindex.json()
            assert data_reindex["files_scanned"] == 2
            assert data_reindex["files_updated"] == 2
            assert data_reindex["chunks_indexed"] >= 2

            # 3. List files
            res_files = client.get("/api/files")
            assert res_files.status_code == 200
            data_files = res_files.json()["files"]
            assert len(data_files) == 2
            assert any(f["relative_path"] == "README.md" for f in data_files)

            # 4. Search endpoint (hybrid)
            res_search = client.post(
                "/api/search",
                json={"query": "FastAPI integration", "mode": "hybrid", "limit": 5},
            )
            assert res_search.status_code == 200
            data_search = res_search.json()
            assert data_search["count"] >= 1
            assert data_search["results"][0]["chunk"]["metadata"]["relative_path"] == "README.md"

            # 5. Fetch chunk detail
            chunk_id = data_search["results"][0]["chunk"]["metadata"]["chunk_id"]
            res_chunk = client.get(f"/api/chunk/{chunk_id}")
            assert res_chunk.status_code == 200
            assert res_chunk.json()["chunk"]["metadata"]["chunk_id"] == chunk_id

            # 6. Fetch raw file content
            res_content = client.get("/api/file-content", params={"path": "README.md"})
            assert res_content.status_code == 200
            assert "Explaining FastAPI integration" in res_content.json()["content"]

            # 7. SSE streaming chat endpoint
            res_chat = client.post(
                "/api/chat",
                json={"query": "What does main.py do?", "mode": "hybrid", "top_k": 3},
            )
            assert res_chat.status_code == 200
            assert "text/event-stream" in res_chat.headers["content-type"]

            # Parse SSE events
            lines = res_chat.text.strip().split("\n\n")
            events = []
            for line in lines:
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

            event_types = [e["type"] for e in events]
            assert "citations" in event_types
            assert "token" in event_types
            assert "done" in event_types
