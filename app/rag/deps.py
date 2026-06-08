"""FastAPI providers for RAG resources.

Indirection points so endpoints don't hard-code OpenAI / filesystem paths and
tests can override them with a fake embedding and temp directories.
"""
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.rag.embeddings import get_embeddings


def embeddings_provider() -> Embeddings:
    return get_embeddings()


def chroma_dir_provider() -> str:
    return settings.chroma_dir


def upload_dir_provider() -> str:
    return settings.upload_dir
