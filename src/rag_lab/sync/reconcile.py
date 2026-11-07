"""Incremental indexing — the part most RAG demos skip.

Re-indexing a changed document is not "run ingest again". It's a reconcile:
compute the desired chunk set, diff it against what's stored, then
INSERT new positions, UPDATE changed ones, and DELETE orphaned ones — all in a
single transaction so a concurrent query never sees a half-updated document.

The DELETE is the one that matters. Skip it and an edited document leaves stale
chunks behind that still retrieve — the system answers from text that no longer
exists in the source, with no error raised. That's a silent correctness bug.
"""
from ..config import settings
from ..embeddings import get_embedder
from ..ingest.parse import Document, load_corpus
from ..ingest.chunk import chunk_document
from ..ingest import store


def reconcile_document(conn, doc: Document) -> dict[str, int]:
    """Make the store match `doc`. Returns counts of inserted/updated/deleted."""
    desired = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
    existing = store.get_existing(conn, doc.doc_id)  # position -> (id, content_hash)
    embedder = get_embedder()

    inserted = updated = deleted = 0
    desired_positions = {c.position for c in desired}

    try:
        for chunk in desired:
            prev = existing.get(chunk.position)
            if prev is None:
                store.upsert_chunk(conn, chunk, embedder.embed([chunk.content])[0])
                inserted += 1
            elif prev[1] != chunk.content_hash:        # content changed -> re-embed
                store.upsert_chunk(conn, chunk, embedder.embed([chunk.content])[0])
                updated += 1
            # else: unchanged -> skip the embedding cost entirely

        for position, (chunk_id, _) in existing.items():  # orphans -> delete
            if position not in desired_positions:
                store.delete_chunk(conn, chunk_id)
                deleted += 1

        conn.commit()   # atomic: readers see all-old or all-new, never a blend
    except Exception:
        conn.rollback()
        raise

    return {"inserted": inserted, "updated": updated, "deleted": deleted}


def sync_corpus(conn, corpus_dir: str) -> None:
    """Index every document, skipping ones whose content hash is unchanged."""
    for doc in load_corpus(corpus_dir):
        if store.stored_doc_hash(conn, doc.doc_id) == doc.doc_hash:
            print(f"  = {doc.doc_id} (unchanged)")
            continue
        counts = reconcile_document(conn, doc)
        print(f"  ~ {doc.doc_id}: {counts}")


def main() -> None:
    import sys
    corpus = sys.argv[1] if len(sys.argv) > 1 else "sample_corpus"
    conn = store.connect()
    print(f"Syncing corpus: {corpus}")
    sync_corpus(conn, corpus)
    conn.close()


if __name__ == "__main__":
    main()
