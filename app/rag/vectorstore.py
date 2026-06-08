"""Chroma vector store access, namespaced per user.

This module owns Chroma. Each user gets their own collection so retrieval can
never cross user boundaries. Consumers call get_vectorstore / delete_document
rather than constructing Chroma directly.
"""
import re

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.rag.embeddings import get_embeddings


def collection_name(user_id: str) -> str:
    """Chroma-safe, per-user collection name."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
    return f"user_{safe}"


def get_vectorstore(
    user_id: str,
    *,
    embeddings: Embeddings | None = None,
    persist_directory: str | None = None,
) -> Chroma:
    return Chroma(
        collection_name=collection_name(user_id),
        embedding_function=embeddings or get_embeddings(),
        persist_directory=persist_directory or settings.chroma_dir,
    )


def delete_document(
    user_id: str,
    document_id: str,
    *,
    embeddings: Embeddings | None = None,
    persist_directory: str | None = None,
) -> int:
    """Remove all chunks for a document from the user's collection."""
    vs = get_vectorstore(
        user_id, embeddings=embeddings, persist_directory=persist_directory
    )
    existing = vs.get(where={"document_id": document_id})
    ids = existing.get("ids", [])
    if ids:
        vs.delete(ids=ids)
    return len(ids)
