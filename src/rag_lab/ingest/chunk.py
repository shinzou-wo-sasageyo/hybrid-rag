"""Step 2 of indexing: split a document into retrievable chunks.

Naive fixed-size-with-overlap on purpose. The eval harness is how you'd justify
moving to semantic/structural chunking later — measure first, then complicate."""
import hashlib
from dataclasses import dataclass
from .parse import Document


@dataclass
class Chunk:
    id: str
    doc_id: str
    position: int
    content: str
    doc_hash: str
    content_hash: str
    metadata: dict


def _split(text: str, size: int, overlap: int) -> list[str]:
    if size <= overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pos, piece in enumerate(_split(doc.text, size, overlap)):
        chunks.append(
            Chunk(
                id=f"{doc.doc_id}:{pos}",
                doc_id=doc.doc_id,
                position=pos,
                content=piece,
                doc_hash=doc.doc_hash,
                content_hash=hashlib.sha256(piece.encode()).hexdigest()[:16],
                metadata=doc.metadata,
            )
        )
    return chunks
