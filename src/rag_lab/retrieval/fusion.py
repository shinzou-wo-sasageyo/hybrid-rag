"""Reciprocal Rank Fusion.

Dense and sparse scores live on incompatible, non-linear scales, so adding them
is meaningless. RRF sidesteps that entirely: it throws away the raw scores and
uses only each document's *rank* in each list. A constant k (60 is the standard
default) dampens the pull of the very top positions so one list can't dominate.

    score(d) = sum over lists of  1 / (k + rank_of_d_in_list)   # rank starts at 1
"""


def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_search(conn, query: str, k: int = 50, rrf_k: int = 60) -> list[str]:
    from .dense import dense_search
    from .sparse import sparse_search

    dense = dense_search(conn, query, k)
    sparse = sparse_search(conn, query, k)
    fused = rrf([dense, sparse], k=rrf_k)
    return [doc_id for doc_id, _ in fused]
