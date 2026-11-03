"""Postgres/pgvector access: connection, writes, and the lookups the sync job
needs. Reads for retrieval live in rag_lab.retrieval."""
import json
import psycopg
from pgvector.psycopg import register_vector
from ..config import settings


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, autocommit=False)
    register_vector(conn)
    return conn


def get_existing(conn: psycopg.Connection, doc_id: str) -> dict[int, tuple[str, str]]:
    """position -> (chunk_id, content_hash) for what's currently stored."""
    rows = conn.execute(
        "SELECT position, id, content_hash FROM chunks WHERE doc_id = %s", (doc_id,)
    ).fetchall()
    return {pos: (cid, chash) for pos, cid, chash in rows}


def stored_doc_hash(conn: psycopg.Connection, doc_id: str) -> str | None:
    row = conn.execute(
        "SELECT doc_hash FROM chunks WHERE doc_id = %s LIMIT 1", (doc_id,)
    ).fetchone()
    return row[0] if row else None


def upsert_chunk(conn, chunk, embedding) -> None:
    conn.execute(
        """
        INSERT INTO chunks (id, doc_id, position, content, metadata, doc_hash, content_hash, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            doc_hash = EXCLUDED.doc_hash,
            content_hash = EXCLUDED.content_hash,
            embedding = EXCLUDED.embedding
        """,
        (chunk.id, chunk.doc_id, chunk.position, chunk.content,
         json.dumps(chunk.metadata), chunk.doc_hash, chunk.content_hash, embedding),
    )


def delete_chunk(conn, chunk_id: str) -> None:
    conn.execute("DELETE FROM chunks WHERE id = %s", (chunk_id,))


def hydrate(conn, ids: list[str]) -> list[dict]:
    """Fetch full chunk rows for a list of ids, preserving the given order."""
    if not ids:
        return []
    rows = conn.execute(
        "SELECT id, doc_id, content, metadata FROM chunks WHERE id = ANY(%s)", (ids,)
    ).fetchall()
    by_id = {r[0]: {"id": r[0], "doc_id": r[1], "content": r[2], "metadata": r[3]} for r in rows}
    return [by_id[i] for i in ids if i in by_id]
