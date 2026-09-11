"""SQLite-based zero-infrastructure hybrid storage engine for LocalRAG-Kit."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from pydantic import BaseModel, Field

from localrag.core.models import Chunk, ChunkMetadata, FileMetadata, FileType
from localrag.storage.schema import CREATE_TABLES_SQL, CREATE_TRIGGERS_SQL
from localrag.storage.vector_ops import (
    batch_cosine_similarities,
    deserialize_vector,
    normalize_vector,
    serialize_vector,
)


class SearchResult(BaseModel):
    """Result of a search query with score and provenance."""
    chunk: Chunk
    score: float = Field(description="Relevance score (higher is better)")
    match_type: str = Field(description="Search mechanism: 'bm25', 'vector', or 'hybrid'")


class StoreStats(BaseModel):
    """Aggregate storage metrics."""
    total_files: int = 0
    total_chunks: int = 0
    total_vectorized_chunks: int = 0
    database_size_bytes: int = 0
    db_path: str = ""


class SQLiteStore:
    """Manages files, chunks, FTS5 full-text indexing, and dense vector embeddings."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_connection()

    def _init_connection(self):
        """Configure SQLite connection with high-performance concurrency pragmas."""
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            # Enable WAL mode for high-throughput concurrent reads and writes
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA synchronous = NORMAL;")
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute("PRAGMA temp_store = MEMORY;")
            self._conn.execute("PRAGMA mmap_size = 268435456;")  # 256MB memory mapping
        self.init_db()

    def init_db(self):
        """Create tables, indexes, and synchronization triggers if not already present."""
        with self._conn:
            self._conn.executescript(CREATE_TABLES_SQL)
            self._conn.executescript(CREATE_TRIGGERS_SQL)

    def close(self):
        """Close the underlying SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -------------------------------------------------------------------------
    # Physical File Management & Incremental Change Detection
    # -------------------------------------------------------------------------

    def upsert_file(self, meta: FileMetadata):
        """Insert or update a physical file record."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO files (
                    relative_path, file_path, file_type, file_size_bytes,
                    sha256, line_count, last_modified, language, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(relative_path) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_type = excluded.file_type,
                    file_size_bytes = excluded.file_size_bytes,
                    sha256 = excluded.sha256,
                    line_count = excluded.line_count,
                    last_modified = excluded.last_modified,
                    language = excluded.language,
                    indexed_at = excluded.indexed_at;
                """,
                (
                    meta.relative_path,
                    meta.file_path,
                    meta.file_type.value,
                    meta.file_size_bytes,
                    meta.sha256,
                    meta.line_count,
                    meta.last_modified,
                    meta.language,
                    time.time(),
                ),
            )

    def delete_file(self, relative_path: str):
        """Delete a file and cascade delete all its associated chunks and FTS entries."""
        with self._conn:
            self._conn.execute("DELETE FROM files WHERE relative_path = ?", (relative_path,))

    def get_file(self, relative_path: str) -> Optional[FileMetadata]:
        """Fetch metadata for a stored file."""
        cur = self._conn.execute(
            "SELECT * FROM files WHERE relative_path = ?",
            (relative_path,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return FileMetadata(
            file_path=row["file_path"],
            relative_path=row["relative_path"],
            file_type=FileType(row["file_type"]),
            file_size_bytes=row["file_size_bytes"],
            sha256=row["sha256"],
            line_count=row["line_count"],
            last_modified=row["last_modified"],
            language=row["language"],
        )

    def get_all_files(self) -> List[FileMetadata]:
        """Retrieve all currently indexed physical files."""
        cur = self._conn.execute("SELECT * FROM files ORDER BY relative_path ASC")
        results = []
        for row in cur.fetchall():
            results.append(
                FileMetadata(
                    file_path=row["file_path"],
                    relative_path=row["relative_path"],
                    file_type=FileType(row["file_type"]),
                    file_size_bytes=row["file_size_bytes"],
                    sha256=row["sha256"],
                    line_count=row["line_count"],
                    last_modified=row["last_modified"],
                    language=row["language"],
                )
            )
        return results

    def get_changed_files(
        self,
        discovered_files: List[FileMetadata],
    ) -> Tuple[List[FileMetadata], List[str]]:
        """
        Compare discovered filesystem files against indexed database records.
        Returns:
            - modified_or_new: Files that need parsing/indexing (new or SHA-256 changed)
            - deleted_paths: Relative paths of files that no longer exist on disk
        """
        cur = self._conn.execute("SELECT relative_path, sha256 FROM files")
        existing_map = {row["relative_path"]: row["sha256"] for row in cur.fetchall()}

        modified_or_new: List[FileMetadata] = []
        discovered_paths: Set[str] = set()

        for f in discovered_files:
            discovered_paths.add(f.relative_path)
            stored_hash = existing_map.get(f.relative_path)
            if stored_hash is None or stored_hash != f.sha256:
                modified_or_new.append(f)

        deleted_paths = [path for path in existing_map if path not in discovered_paths]
        return modified_or_new, deleted_paths

    # -------------------------------------------------------------------------
    # Chunks & Dense Vector Management
    # -------------------------------------------------------------------------

    def upsert_chunks(self, chunks: List[Chunk]):
        """Batch insert or replace chunks along with their vector embeddings."""
        if not chunks:
            return

        rows = []
        for c in chunks:
            emb_blob = serialize_vector(c.embedding) if c.embedding else None
            extra_json = json.dumps(c.metadata.extra) if c.metadata.extra else None
            rows.append((
                c.metadata.chunk_id,
                c.metadata.relative_path,
                c.metadata.file_path,
                c.metadata.file_type.value,
                c.metadata.start_line,
                c.metadata.end_line,
                c.metadata.char_count,
                c.metadata.estimated_tokens,
                c.metadata.section_title,
                c.text,
                emb_blob,
                extra_json,
            ))

        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, relative_path, file_path, file_type,
                    start_line, end_line, char_count, estimated_tokens,
                    section_title, text, embedding, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    relative_path = excluded.relative_path,
                    file_path = excluded.file_path,
                    file_type = excluded.file_type,
                    start_line = excluded.start_line,
                    end_line = excluded.end_line,
                    char_count = excluded.char_count,
                    estimated_tokens = excluded.estimated_tokens,
                    section_title = excluded.section_title,
                    text = excluded.text,
                    embedding = excluded.embedding,
                    extra_json = excluded.extra_json;
                """,
                rows,
            )

    def delete_chunks_for_file(self, relative_path: str):
        """Remove all chunks associated with a specific relative file path."""
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE relative_path = ?", (relative_path,))

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Fetch a single chunk by ID."""
        cur = self._conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_chunk(row)

    def get_all_chunks(self) -> List[Chunk]:
        """Fetch all chunks stored in the database."""
        cur = self._conn.execute("SELECT * FROM chunks")
        return [self._row_to_chunk(row) for row in cur.fetchall()]

    # -------------------------------------------------------------------------
    # Search Engines (FTS5 BM25 + Dense Vector Cosine Similarity)
    # -------------------------------------------------------------------------

    def search_bm25(self, query: str, limit: int = 20) -> List[SearchResult]:
        """
        Execute BM25 keyword search using SQLite FTS5.
        Returns top matching chunks sorted by BM25 relevance.
        """
        cleaned_query = self._sanitize_fts_query(query)
        if not cleaned_query:
            return []

        # In SQLite FTS5, bm25() returns negative numbers (more negative = better match).
        # We transform it to a positive score: 1.0 / (1.0 + abs(rank))
        sql = """
            SELECT c.*, bm25(chunks_fts) as rank
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?;
        """
        try:
            cur = self._conn.execute(sql, (cleaned_query, limit))
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # If query syntax error, fallback to simple token search
            fallback_query = " OR ".join(f'"{tok}"' for tok in query.split() if tok.isalnum())
            if not fallback_query:
                return []
            cur = self._conn.execute(sql, (fallback_query, limit))
            rows = cur.fetchall()

        results = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            raw_rank = row["rank"]
            # Convert negative BM25 score to normalized positive relevance [0.0, 1.0]
            score = 1.0 / (1.0 + abs(raw_rank))
            results.append(SearchResult(chunk=chunk, score=score, match_type="bm25"))

        return results

    def search_vector(
        self,
        query_embedding: Sequence[float],
        limit: int = 20,
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """
        Execute dense vector cosine similarity search across all stored chunk embeddings.
        Returns top matching chunks sorted by cosine similarity.
        """
        if not query_embedding:
            return []

        cur = self._conn.execute(
            "SELECT * FROM chunks WHERE embedding IS NOT NULL"
        )
        rows = cur.fetchall()
        if not rows:
            return []

        candidate_chunks: List[Chunk] = []
        candidate_vectors: List[List[float]] = []

        for row in rows:
            chunk = self._row_to_chunk(row)
            if chunk.embedding:
                candidate_chunks.append(chunk)
                candidate_vectors.append(chunk.embedding)

        if not candidate_vectors:
            return []

        similarities = batch_cosine_similarities(query_embedding, candidate_vectors)

        # Pair chunks with scores and filter
        scored_pairs = [
            (chunk, score)
            for chunk, score in zip(candidate_chunks, similarities)
            if score >= min_similarity
        ]
        scored_pairs.sort(key=lambda p: p[1], reverse=True)

        results = [
            SearchResult(chunk=chunk, score=float(score), match_type="vector")
            for chunk, score in scored_pairs[:limit]
        ]
        return results

    # -------------------------------------------------------------------------
    # Diagnostics & Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> StoreStats:
        """Compute database statistics."""
        file_count = self._conn.execute("SELECT count(*) FROM files").fetchone()[0]
        chunk_count = self._conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        vec_count = self._conn.execute(
            "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()[0]

        size_bytes = 0
        if self.db_path.exists():
            size_bytes = self.db_path.stat().st_size
            # Include WAL file if present
            wal_path = self.db_path.with_name(self.db_path.name + "-wal")
            if wal_path.exists():
                size_bytes += wal_path.stat().st_size

        return StoreStats(
            total_files=file_count,
            total_chunks=chunk_count,
            total_vectorized_chunks=vec_count,
            database_size_bytes=size_bytes,
            db_path=str(self.db_path),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        """Hydrate SQLite row into a Chunk Pydantic model."""
        emb = deserialize_vector(row["embedding"]) if row["embedding"] else None
        extra = json.loads(row["extra_json"]) if row["extra_json"] else {}

        meta = ChunkMetadata(
            chunk_id=row["chunk_id"],
            file_path=row["file_path"],
            relative_path=row["relative_path"],
            file_type=FileType(row["file_type"]),
            start_line=row["start_line"],
            end_line=row["end_line"],
            char_count=row["char_count"],
            estimated_tokens=row["estimated_tokens"],
            section_title=row["section_title"],
            extra=extra,
        )
        return Chunk(text=row["text"], metadata=meta, embedding=emb)

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Clean raw input string into safe FTS5 MATCH query syntax."""
        tokens = []
        for word in query.strip().split():
            clean = "".join(ch for ch in word if ch.isalnum() or ch in ("_", "-"))
            if clean:
                tokens.append(f'"{clean}"')
        return " OR ".join(tokens)
