"""Tests for SQLiteStore, FTS5 BM25 search, vector similarity, and change detection."""

import tempfile
from pathlib import Path

from localrag.core.models import Chunk, ChunkMetadata, FileMetadata, FileType
from localrag.storage.sqlite_store import SQLiteStore
from localrag.storage.vector_ops import (
    batch_cosine_similarities,
    cosine_similarity,
    deserialize_vector,
    normalize_vector,
    serialize_vector,
)


def test_vector_ops():
    # Serialization & Deserialization
    vec = [0.1, 0.5, -0.3, 0.8]
    blob = serialize_vector(vec)
    assert isinstance(blob, bytes)
    recovered = deserialize_vector(blob)
    assert len(recovered) == 4
    for a, b in zip(vec, recovered):
        assert abs(a - b) < 1e-6

    # Normalization
    norm_vec = normalize_vector(vec)
    assert abs(sum(x * x for x in norm_vec) - 1.0) < 1e-5

    # Cosine similarity
    sim_identical = cosine_similarity(vec, vec)
    assert abs(sim_identical - 1.0) < 1e-5

    opposite = [-x for x in vec]
    sim_opp = cosine_similarity(vec, opposite)
    assert abs(sim_opp - (-1.0)) < 1e-5

    # Batch cosine similarities
    batch = batch_cosine_similarities(vec, [vec, opposite, [0.0, 1.0, 0.0, 0.0]])
    assert len(batch) == 3
    assert abs(batch[0] - 1.0) < 1e-5
    assert abs(batch[1] - (-1.0)) < 1e-5


def test_sqlite_store_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / ".localrag" / "index.db"
        with SQLiteStore(db_path) as store:
            stats = store.get_stats()
            assert stats.total_files == 0
            assert stats.total_chunks == 0
            assert db_path.exists()


def test_files_upsert_and_change_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "index.db"
        with SQLiteStore(db_path) as store:
            file_a = FileMetadata(
                file_path=str(Path(tmpdir) / "README.md"),
                relative_path="README.md",
                file_type=FileType.MARKDOWN,
                file_size_bytes=100,
                sha256="hash_a1",
                line_count=10,
                last_modified=1000.0,
            )
            file_b = FileMetadata(
                file_path=str(Path(tmpdir) / "main.py"),
                relative_path="main.py",
                file_type=FileType.CODE,
                file_size_bytes=200,
                sha256="hash_b1",
                line_count=20,
                last_modified=1000.0,
                language="python",
            )
            store.upsert_file(file_a)
            store.upsert_file(file_b)

            all_files = store.get_all_files()
            assert len(all_files) == 2

            # Now test change detection:
            # - README.md hash changed to hash_a2 (modified)
            # - main.py deleted (not in incoming)
            # - utils.py newly created (new)
            incoming = [
                FileMetadata(
                    file_path=str(Path(tmpdir) / "README.md"),
                    relative_path="README.md",
                    file_type=FileType.MARKDOWN,
                    file_size_bytes=110,
                    sha256="hash_a2",  # changed
                    line_count=11,
                    last_modified=1050.0,
                ),
                FileMetadata(
                    file_path=str(Path(tmpdir) / "utils.py"),
                    relative_path="utils.py",
                    file_type=FileType.CODE,
                    file_size_bytes=50,
                    sha256="hash_c1",  # new
                    line_count=5,
                    last_modified=1050.0,
                    language="python",
                ),
            ]

            modified_or_new, deleted = store.get_changed_files(incoming)
            mod_paths = {f.relative_path for f in modified_or_new}
            assert "README.md" in mod_paths
            assert "utils.py" in mod_paths
            assert deleted == ["main.py"]


def test_fts5_bm25_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "index.db"
        with SQLiteStore(db_path) as store:
            file_meta = FileMetadata(
                file_path="/dummy/docs.md",
                relative_path="docs.md",
                file_type=FileType.MARKDOWN,
                sha256="h1",
            )
            store.upsert_file(file_meta)

            chunk1 = Chunk(
                text="Authentication using JWT tokens and OAuth2 protocol.",
                metadata=ChunkMetadata(
                    chunk_id="chk_auth",
                    file_path="/dummy/docs.md",
                    relative_path="docs.md",
                    file_type=FileType.MARKDOWN,
                    start_line=1,
                    end_line=5,
                    section_title="Authentication",
                ),
            )
            chunk2 = Chunk(
                text="Database migrations and PostgreSQL connection pool configuration.",
                metadata=ChunkMetadata(
                    chunk_id="chk_db",
                    file_path="/dummy/docs.md",
                    relative_path="docs.md",
                    file_type=FileType.MARKDOWN,
                    start_line=6,
                    end_line=10,
                    section_title="Database",
                ),
            )
            store.upsert_chunks([chunk1, chunk2])

            # Keyword search for 'OAuth2'
            results_auth = store.search_bm25("OAuth2", limit=5)
            assert len(results_auth) == 1
            assert results_auth[0].chunk.id == "chk_auth"
            assert results_auth[0].score > 0

            # Keyword search for 'PostgreSQL'
            results_db = store.search_bm25("PostgreSQL connection", limit=5)
            assert len(results_db) == 1
            assert results_db[0].chunk.id == "chk_db"

            # Query with no matches
            results_none = store.search_bm25("Kubernetes deployment", limit=5)
            assert len(results_none) == 0


def test_vector_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "index.db"
        with SQLiteStore(db_path) as store:
            file_meta = FileMetadata(
                file_path="/dummy/code.py",
                relative_path="code.py",
                file_type=FileType.CODE,
                sha256="h2",
            )
            store.upsert_file(file_meta)

            # Chunk A: vector pointing along [1, 0, 0]
            chunk_a = Chunk(
                text="User login function.",
                metadata=ChunkMetadata(
                    chunk_id="chk_a",
                    file_path="/dummy/code.py",
                    relative_path="code.py",
                    file_type=FileType.CODE,
                ),
                embedding=[1.0, 0.0, 0.0],
            )
            # Chunk B: vector pointing along [0, 1, 0]
            chunk_b = Chunk(
                text="Payment processing function.",
                metadata=ChunkMetadata(
                    chunk_id="chk_b",
                    file_path="/dummy/code.py",
                    relative_path="code.py",
                    file_type=FileType.CODE,
                ),
                embedding=[0.0, 1.0, 0.0],
            )
            store.upsert_chunks([chunk_a, chunk_b])

            # Query vector close to Chunk A ([0.9, 0.1, 0.0])
            results = store.search_vector([0.9, 0.1, 0.0], limit=2)
            assert len(results) == 2
            assert results[0].chunk.id == "chk_a"
            assert results[0].score > 0.9
            assert results[1].chunk.id == "chk_b"
            assert results[1].score < 0.2


def test_cascade_deletion():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "index.db"
        with SQLiteStore(db_path) as store:
            file_meta = FileMetadata(
                file_path="/dummy/temp.py",
                relative_path="temp.py",
                file_type=FileType.CODE,
                sha256="h3",
            )
            store.upsert_file(file_meta)

            chunk = Chunk(
                text="Temporary logic to be deleted.",
                metadata=ChunkMetadata(
                    chunk_id="chk_temp",
                    file_path="/dummy/temp.py",
                    relative_path="temp.py",
                    file_type=FileType.CODE,
                ),
            )
            store.upsert_chunks([chunk])

            assert store.get_chunk("chk_temp") is not None
            assert len(store.search_bm25("Temporary logic")) == 1

            # Delete the parent file
            store.delete_file("temp.py")

            # Verify chunk and FTS were cascade deleted
            assert store.get_chunk("chk_temp") is None
            assert len(store.search_bm25("Temporary logic")) == 0
