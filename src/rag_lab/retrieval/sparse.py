"""Sparse retrieval: lexical match via Postgres full-text search.

Note: ts_rank is not textbook BM25 (no length normalization). Honest and good
enough for a learning project; the swap to a real BM25 extension (ParadeDB
pg_search) is a documented next step, not a hidden flaw."""


def sparse_search(conn, query: str, k: int = 50) -> list[str]:
    rows = conn.execute(
        """
        SELECT id
        FROM chunks
        WHERE fts @@ plainto_tsquery('english', %s)
        ORDER BY ts_rank(fts, plainto_tsquery('english', %s)) DESC
        LIMIT %s
        """,
        (query, query, k),
    ).fetchall()
    return [r[0] for r in rows]
