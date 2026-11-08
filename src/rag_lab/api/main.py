"""Thin serving layer: POST /query runs the online pipeline and returns the
fused top chunks (plus an answer when an Anthropic key is configured)."""
from fastapi import FastAPI
from pydantic import BaseModel

from ..ingest import store
from ..retrieval.fusion import hybrid_search

app = FastAPI(title="hybrid-rag-lab")


class Query(BaseModel):
    question: str
    k: int = 5


@app.post("/query")
def query(q: Query):
    conn = store.connect()
    try:
        ids = hybrid_search(conn, q.question, k=50)[: q.k]
        chunks = store.hydrate(conn, ids)
    finally:
        conn.close()
    return {"question": q.question, "chunks": chunks}
