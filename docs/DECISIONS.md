# Design decisions & learnings

A record of the choices behind this project and the reasoning for each — the
part an interviewer actually cares about. Each entry: the decision, why, the
tradeoff accepted, and what I took away.

## 1. Hybrid retrieval, not pure vector search

**Decision.** Run dense (vector) and sparse (keyword) retrieval together and fuse.
**Why.** Dense embeddings capture meaning but smooth away exact tokens — error
codes, port numbers, IDs — and can return topically-close-but-wrong passages.
Keyword search nails exact terms but misses paraphrase. They fail on *different*
queries, so combining them raises recall without threshold tuning.
**Tradeoff.** Two retrievers and a fusion step instead of one query.
**Learned.** In 2026, pure vector search is treated as an architectural mistake;
hybrid + rerank is the assumed baseline. The eval table makes the case concrete:
sparse wins exact-term queries, dense wins paraphrases, hybrid wins overall.

## 2. RRF for fusion, not weighted score addition

**Decision.** Reciprocal Rank Fusion with k=60.
**Why.** Cosine distance and ts_rank live on incompatible, non-linear scales, so
normalizing and summing them is fragile and corpus-dependent. RRF uses only rank
position, which sidesteps normalization entirely.
**Tradeoff.** Throws away score magnitude (how *strong* a match was).
**Learned.** The "dumb" rank-based method is more robust than the clever
score-based one — and far easier to defend in an interview.

## 3. One Postgres store (pgvector), not a separate vector DB

**Decision.** Dense (`embedding` + HNSW) and sparse (`fts` + GIN) in one table.
**Why.** Both halves of hybrid retrieval live in one row, one transaction, one
backup story. Less operational surface for a learning project, and pgvector
scales to seven figures of chunks with HNSW.
**Tradeoff.** Postgres full-text search isn't true BM25 (no length
normalization). Documented as a known limitation, with ParadeDB `pg_search` as
the upgrade path — not hidden.

## 4. Local embedding model by default

**Decision.** `BAAI/bge-small-en-v1.5` (384-dim, CPU) behind a pluggable interface.
**Why.** The whole repo runs with no API key and no cost, so anyone can clone and
`make eval`. The interface makes Voyage/OpenAI a swap, not a rewrite.
**Learned.** Choosing an embedding model is a domain-fit + cost + dimensionality
tradeoff you should *measure on your own corpus*, not read off a leaderboard.

## 5. Incremental sync with orphan deletion (the correctness bug most demos have)

**Decision.** Re-indexing a changed document is a reconcile — diff desired chunks
against stored ones, then insert / update / delete, inside one transaction.
**Why.** The delete is load-bearing. If an edited document drops from 10 chunks to
7 and you don't delete the 3 orphans, they still retrieve — the system answers
from text that no longer exists in the source, and **no error is raised**.
**Tradeoff.** More logic than "delete all + re-insert"; chosen for lower embedding
cost (unchanged chunks are skipped) and to make the reasoning explicit.
**Learned.** Keeping a search index faithful to a moving source is ordinary
distributed-systems discipline — change detection, idempotency, atomic swaps —
applied to RAG. It's the box that separates a demo from a system.

## 6. Evaluate retrieval before generation

**Decision.** Measure recall@k, precision@k, MRR with no LLM in the loop; store
runs in DuckDB; make the comparison table the project's headline.
**Why.** If the right chunk is never retrieved, no prompt engineering helps — so
retrieval quality is the first thing to isolate and the first thing to gate in CI.
**Learned.** The gap between recall and MRR is a diagnostic: high recall + low MRR
means the right chunk is retrieved but ranked low — the signal to add a reranker.
Low recall means fix retrieval first; a reranker can't reorder what wasn't found.

## 7. Naive chunking first

**Decision.** Fixed-size with overlap; tune later.
**Why.** You can't justify semantic or structural chunking until the eval harness
shows fixed-size is the bottleneck. Measure, then complicate.
**Tradeoff.** Leaves recall on the table initially — by design, so the
improvement is visible as a number.

## 8. Where the real work lives (glue vs. custom code)

The core retrieve → fuse → generate loop is framework glue — a day of work that
looks the same at every company. The engineering that takes real time is the code
*around* it: document parsing, incremental sync, access control / multi-tenancy,
guardrails and abstain-on-no-match, prompt assembly, and the serving layer
(caching, tracing, retries, cost). A "deterministic" pipeline tends to have *more*
hand-written control-flow code, not less — every decision an agent would make at
runtime has to be pre-decided and hard-coded as a threshold or branch.

## 9. Built on a graph seam for a reason

The pipeline is modeled as discrete stages (dense, sparse, fuse, rerank, generate)
rather than one function. Today it's deterministic. The conditional/looping edges
are the seam where it becomes *agentic* — a node that judges the fused results
insufficient, rewrites the query, and retries. Adaptive RAG (route simple queries
down the cheap deterministic path, hard ones through the agentic loop) is the
production endpoint this is built toward.
