"""Embedding backends behind one interface, so the rest of the system never
knows (or cares) which model produced a vector. Default is a local model so
the whole project runs with no API key and no cost."""
from functools import lru_cache
from .config import settings


class LocalEmbedder:
    """BAAI/bge-small-en-v1.5 — 384 dims, runs on CPU, free."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


# Swap-in points. These are where a production system would reach for a hosted
# model (Voyage is Anthropic's recommended pairing for Claude; OpenAI's
# text-embedding-3-small is the cheap default). Left as TODOs on purpose:
# implementing one is a good way to feel the "choose an embedding model" tradeoff.
class VoyageEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError("Add the voyageai client here (set EMBEDDING_DIM to match).")


@lru_cache(maxsize=1)
def get_embedder():
    backend = settings.embedding_backend
    if backend == "local":
        return LocalEmbedder()
    if backend == "voyage":
        return VoyageEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")
