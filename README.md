# hybrid-rag-lab

Hybrid retrieval-augmented generation, built from scratch to understand what
production RAG actually requires — and to **measure every design decision**
instead of guessing. Half working system, half lab notebook.

Most RAG repos are "chat with your PDF" wrappers around a framework. This one is
the opposite: the retrieval logic is hand-written, every retrieval choice is
backed by an eval number, and the parts real demos skip — incremental sync,
orphaned-chunk correctness, evaluation as a regression gate — are the point.

> Learning project, not a production system. The value is the reasoning and the
> measurements, documented in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## The result that matters

Dense vs. sparse vs. hybrid retrieval over **27 Python PEPs**, scored on a
hand-labeled set of 30 queries (document-level recall@k / MRR) by my own harness
(`make eval`). Run it and paste the table here:

| config | recall@5 | precision@5 | MRR |
|--------|----------|-------------|-----|
| dense  | _run it_ | _run it_    | _run it_ |
| sparse | _run it_ | _run it_    | _run it_ |
| hybrid | _run it_ | _run it_    | _run it_ |

The expected story: dense wins on paraphrased questions ("how do I create a
virtual environment?"), sparse wins on exact-term ones ("PEP 484", "walrus
operator", "yield from"), and **hybrid beats both** because they fail on
different queries. That's the whole argument for hybrid retrieval, shown rather
than asserted. PEPs are a good corpus for this precisely because they mix prose
with hard identifiers that reward lexical search.

## Architecture

```
          ingest (offline)                 query (online)
  docs ─► parse ─► chunk ─► embed ─► ┌── pgvector ──┐ ◄─ dense (HNSW, cosine)
                                     │   chunks     │ ◄─ sparse (GIN, full-text)
  sync ─► reconcile (diff + delete) ─┘              └─► RRF fusion ─► [rerank] ─► LLM
```

- **Dense + sparse + RRF** — two retrievers in parallel, fused by rank, no
  fragile score normalization.
- **Incremental sync** — re-index only what changed; delete orphaned chunks so
  the store never serves text that no longer exists in the source.
- **Eval-driven** — retrieval measured in isolation (recall@k, MRR) before any
  LLM is involved.

## Quickstart

```bash
cp .env.example .env
pip install -e .            # local embedding model, no API key needed
make up                     # postgres + pgvector via docker
make fetch-peps             # download the Python PEPs corpus (needs network)
make ingest                 # index data/corpus/
make eval                   # print the dense/sparse/hybrid comparison table
make serve                  # POST /query
```

Try the sync correctness demo: edit any file in `data/corpus/`, re-run
`make ingest`, and watch the reconcile report inserted/updated/deleted counts.

## Layout

```
src/rag_lab/
  embeddings.py        pluggable embedder (local by default; voyage/openai swap-in)
  ingest/              parse → chunk → store (pgvector schema)
  retrieval/           dense.py, sparse.py, fusion.py (hand-written RRF)
  sync/reconcile.py    incremental indexing: diff, upsert, delete-orphans, atomic
  eval/                metrics.py + harness.py (the comparison table)
  api/main.py          FastAPI /query endpoint
sql/schema.sql         chunks table, HNSW + GIN indexes
docs/DECISIONS.md      why each choice was made, and what I learned
```

## Roadmap

- **v1 (here):** ingest + dense + sparse + RRF + eval harness + sync.
- **v2:** cross-encoder reranker; show the before/after MRR lift in the table.
- **v3:** LangGraph orchestration with a conditional rerank + query-rewrite loop;
  real BM25 via a Postgres extension; a larger labeled corpus.

## What I learned

See [`docs/DECISIONS.md`](docs/DECISIONS.md) — the short version: the retrieval
loop is the easy 20%; data sync, access control, guardrails, and evaluation are
where the real engineering (and the real bugs) live.
