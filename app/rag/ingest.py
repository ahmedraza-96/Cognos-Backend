"""Document ingestion: load -> split -> embed -> upsert into Chroma.

`ingest_document` takes already-extracted text so it's easy to unit-test;
`load_file` handles reading text/markdown/PDF from disk.
"""
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.rag.vectorstore import get_vectorstore

_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def split_text(text: str) -> list[str]:
    return [c for c in _splitter.split_text(text) if c.strip()]


def load_file(path: str) -> str:
    """Extract plain text from a .txt/.md/.pdf file."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        reader = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return p.read_text(encoding="utf-8", errors="ignore")


def ingest_document(
    *,
    user_id: str,
    document_id: str,
    filename: str,
    text: str,
    embeddings: Embeddings | None = None,
    persist_directory: str | None = None,
) -> int:
    """Chunk `text`, embed, and store in the user's collection. Returns chunk count."""
    chunks = split_text(text)
    if not chunks:
        return 0

    vs = get_vectorstore(
        user_id, embeddings=embeddings, persist_directory=persist_directory
    )
    ids = [f"{document_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"document_id": document_id, "filename": filename, "chunk": i}
        for i in range(len(chunks))
    ]
    vs.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)
