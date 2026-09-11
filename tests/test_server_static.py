"""Tests for web UI static assets serving and integration."""

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from localrag.providers import FastFeatureEmbeddingProvider, MockLLMProvider
from localrag.server.app import create_app


def test_static_assets_serving():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "doc.txt").write_text("LocalRAG offline documentation.")

        app = create_app(
            workspace_path=root,
            embedding_provider=FastFeatureEmbeddingProvider(),
            llm_provider=MockLLMProvider(),
        )

        with TestClient(app) as client:
            # 1. Root index.html
            res_root = client.get("/")
            assert res_root.status_code == 200
            assert "text/html" in res_root.headers.get("content-type", "")
            assert "LocalRAG-Kit" in res_root.text
            assert "100% Offline" in res_root.text

            # 2. Stylesheet
            res_css = client.get("/style.css")
            assert res_css.status_code == 200
            assert "custom-scrollbar" in res_css.text

            # 3. JavaScript frontend bundle
            res_js = client.get("/app.js")
            assert res_js.status_code == 200
            assert "LocalRAG-Kit Frontend Application" in res_js.text

            # 4. API endpoints still cleanly accessible alongside static mount
            res_status = client.get("/api/status")
            assert res_status.status_code == 200
            assert res_status.json()["workspace_path"] == str(root.resolve())
