"""Tests for the unified IngestionPipeline."""

import tempfile
from pathlib import Path
from localrag.core.ingestion import IngestionPipeline
from localrag.storage.sqlite_store import SQLiteStore


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


def test_ingestion_pipeline_index_to_store_incremental():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        db_path = root / ".localrag" / "index.db"

        # Create files
        (root / "README.md").write_text("# Initial Doc\nExplaining architecture.")
        (root / "temp.txt").write_text("Temporary file to be deleted.")

        pipeline = IngestionPipeline(root)

        with SQLiteStore(db_path) as store:
            # First pass: Initial index
            stats1 = pipeline.index_to_store(store, incremental=True)
            assert stats1.files_added_or_modified == 2
            assert stats1.files_unchanged == 0
            assert stats1.files_deleted == 0
            assert stats1.chunks_indexed >= 2

            # FTS5 search check
            res = store.search_bm25("architecture")
            assert len(res) >= 1

            # Second pass: No modifications
            stats2 = pipeline.index_to_store(store, incremental=True)
            assert stats2.files_added_or_modified == 0
            assert stats2.files_unchanged == 2
            assert stats2.files_deleted == 0

            # Third pass: Modify README.md and delete temp.txt
            (root / "README.md").write_text("# Updated Doc\nNow talking about microservices.")
            (root / "temp.txt").unlink()

            stats3 = pipeline.index_to_store(store, incremental=True)
            assert stats3.files_added_or_modified == 1
            assert stats3.files_deleted == 1
            assert stats3.files_unchanged == 0

            # Verify deleted file chunks were purged
            assert len(store.search_bm25("Temporary")) == 0
            # Verify updated content was indexed
            assert len(store.search_bm25("microservices")) >= 1
