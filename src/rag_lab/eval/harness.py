"""Run dense / sparse / hybrid over a labeled eval set and compare them.

This is the centerpiece of the whole project: the table it prints is the
evidence that hybrid beats either retriever alone. Results are stored in DuckDB
so runs can be compared across changes (a regression gate in CI, later)."""
import json
from pathlib import Path

import duckdb
from tabulate import tabulate

from ..ingest import store
from ..retrieval.dense import dense_search
from ..retrieval.sparse import sparse_search
from ..retrieval.fusion import hybrid_search
from . import metrics

CONFIGS = {
    "dense": dense_search,
    "sparse": sparse_search,
    "hybrid": hybrid_search,
}
K = 5


def load_eval_set(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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
            retrieved = search(conn, ex["question"], k=50)
            relevant = set(ex["relevant_ids"])
            r += metrics.recall_at_k(retrieved, relevant, K)
            p += metrics.precision_at_k(retrieved, relevant, K)
            m += metrics.mrr(retrieved, relevant)
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, now())",
                [name, ex["question"],
                 metrics.recall_at_k(retrieved, relevant, K),
                 metrics.precision_at_k(retrieved, relevant, K),
                 metrics.mrr(retrieved, relevant)],
            )
        n = len(examples)
        summary.append([name, round(r / n, 3), round(p / n, 3), round(m / n, 3)])

    conn.close()
    print(f"\nRetrieval quality over {len(examples)} queries (k={K})\n")
    print(tabulate(summary, headers=["config", f"recall@{K}", f"precision@{K}", "MRR"], tablefmt="github"))
    print("\nPaste this table into the README.")


if __name__ == "__main__":
    run()
