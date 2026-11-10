"""Run dense / sparse / hybrid over a labeled eval set and compare them.

Evaluated at the DOCUMENT level: a retrieved chunk counts as a hit if its parent
document is in the labeled relevant set. That makes labeling tractable for a doc
corpus — you tag "this question is answered by PEP 8", not an exact chunk offset.

The table this prints is the project's headline result. Runs are logged to DuckDB
so configs can be compared across changes (a CI regression gate, later)."""
import json
from pathlib import Path

import duckdb
from tabulate import tabulate

from ..ingest import store
from ..retrieval.dense import dense_search
from ..retrieval.sparse import sparse_search
from ..retrieval.fusion import hybrid_search
from . import metrics

CONFIGS = {"dense": dense_search, "sparse": sparse_search, "hybrid": hybrid_search}
K = 5


def load_eval_set(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def to_docs(chunk_ids: list[str]) -> list[str]:
    """Collapse a ranked list of chunk ids to a ranked list of unique doc ids."""
    seen, out = set(), []
    for cid in chunk_ids:
        doc = cid.rsplit(":", 1)[0]
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def run(eval_path: str = "eval/eval_set.jsonl", db_path: str = "eval_results.duckdb") -> None:
    examples = load_eval_set(eval_path)
    conn = store.connect()
    db = duckdb.connect(db_path)
    db.execute(
        "CREATE TABLE IF NOT EXISTS runs("
        "config TEXT, question TEXT, recall DOUBLE, precision DOUBLE, mrr DOUBLE, ts TIMESTAMP)"
    )

    summary = []
    for name, search in CONFIGS.items():
        r = p = m = 0.0
        for ex in examples:
            docs = to_docs(search(conn, ex["question"], k=50))
            relevant = set(ex["relevant_docs"])
            rr = metrics.recall_at_k(docs, relevant, K)
            pp = metrics.precision_at_k(docs, relevant, K)
            mm = metrics.mrr(docs, relevant)
            r, p, m = r + rr, p + pp, m + mm
            db.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, now())",
                       [name, ex["question"], rr, pp, mm])
        n = len(examples)
        summary.append([name, round(r / n, 3), round(p / n, 3), round(m / n, 3)])

    conn.close()
    print(f"\nRetrieval quality over {len(examples)} queries (k={K})\n")
    print(tabulate(summary, headers=["config", f"recall@{K}", f"precision@{K}", "MRR"], tablefmt="github"))
    print("\nPaste this table into the README.")


if __name__ == "__main__":
    run()
