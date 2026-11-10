"""Step 1 of indexing: turn a raw source file into clean text + metadata.

Kept deliberately small — real corpora need PDF/HTML/Confluence parsers here,
and that parsing is where most production RAG code (and most RAG bugs) live."""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def doc_hash(self) -> str:
        return _hash(self.text)


def load_corpus(corpus_dir: str) -> list[Document]:
    """Load every .txt / .md file under corpus_dir as one Document."""
    docs: list[Document] = []
    for path in sorted(Path(corpus_dir).rglob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".rst"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append(
            Document(
                doc_id=str(path.stem),
                text=text,
                metadata={"source": str(path)},
            )
        )
    return docs
