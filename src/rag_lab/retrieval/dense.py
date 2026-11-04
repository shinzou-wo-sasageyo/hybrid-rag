"""Dense retrieval: semantic match via pgvector cosine distance (<=>)."""
from ..embeddings import get_embedder


def dense_search(conn, query: str, k: int = 50) -> list[str]:
    qvec = get_embedder().embed([query])[0]
    rows = conn.execute(
        "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
        (qvec, k),
    ).fetchall()
    return [r[0] for r in rows]
