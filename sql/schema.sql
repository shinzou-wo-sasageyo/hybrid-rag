CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk. Dense half = embedding (HNSW), sparse half = fts (GIN).
-- doc_hash lets the sync job detect which documents changed; content_hash
-- lets it detect which individual chunks changed within a document.
CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,          -- e.g. "c_8841_0"
    doc_id        TEXT NOT NULL,
    position      INT  NOT NULL,             -- chunk order within the document
    content       TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}',
    doc_hash      TEXT NOT NULL,             -- hash of the whole source document
    content_hash  TEXT NOT NULL,             -- hash of this chunk's content
    embedding     VECTOR(384),               -- keep in sync with EMBEDDING_DIM
    fts           TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx   ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_embed_hnsw   ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_fts_gin      ON chunks USING gin (fts);
