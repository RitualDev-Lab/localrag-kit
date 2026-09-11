"""Tests for the unified IngestionPipeline."""

import tempfile
from pathlib import Path
from localrag.core.ingestion import IngestionPipeline


def test_ingestion_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Populate temporary project tree
        (root / "README.md").write_text("# Welcome to LocalRAG\nFull local offline RAG.\n\n## Quickstart\nRun localrag index.")
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("""def start_server():
    print('Server listening')

def stop_server():
    print('Server stopped')
""")
        (root / "notes.txt").write_text("Just some notes on development.\nImportant reminder: use local models.")

        pipeline = IngestionPipeline(root)

        events = []
        def on_progress(meta, current, total):
            events.append((meta.relative_path, current, total))

        documents, stats = pipeline.process_directory(progress_callback=on_progress)

        assert stats.total_files_scanned == 3
        assert stats.total_files_parsed == 3
        assert stats.total_chunks_produced >= 3
        assert stats.total_characters > 0
        assert stats.total_estimated_tokens > 0
        assert stats.failed_files == 0

        assert len(events) == 3

        # Test streaming iter_chunks
        streamed_chunks = list(pipeline.iter_chunks())
        assert len(streamed_chunks) == stats.total_chunks_produced
