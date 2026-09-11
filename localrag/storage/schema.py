"""Database schema definitions and table creation for LocalRAG SQLite hybrid store."""

CREATE_TABLES_SQL = """
-- Track indexed physical files for incremental change detection
CREATE TABLE IF NOT EXISTS files (
    relative_path TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    line_count INTEGER NOT NULL,
    last_modified REAL NOT NULL,
    language TEXT,
    indexed_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_sha256 ON files(sha256);

-- Store chunk text, coordinates, provenance, and dense vector embeddings
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    estimated_tokens INTEGER NOT NULL,
    section_title TEXT,
    text TEXT NOT NULL,
    embedding BLOB,
    extra_json TEXT,
    FOREIGN KEY(relative_path) REFERENCES files(relative_path) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_relpath ON chunks(relative_path);

-- Full-Text Search (FTS5) index for BM25 keyword matching
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    relative_path UNINDEXED,
    section_title,
    text,
    tokenize='porter unicode61'
);
"""

# Triggers to keep FTS5 automatically synchronized with chunks table
CREATE_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(chunk_id, relative_path, section_title, text)
    VALUES (new.chunk_id, new.relative_path, coalesce(new.section_title, ''), new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.chunk_id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.chunk_id;
    INSERT INTO chunks_fts(chunk_id, relative_path, section_title, text)
    VALUES (new.chunk_id, new.relative_path, coalesce(new.section_title, ''), new.text);
END;
"""
