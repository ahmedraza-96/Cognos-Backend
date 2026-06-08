"""OpenAI embeddings factory.

Isolated here so the model/provider is a single swap point and so tests can
inject a fake embedding instead of calling OpenAI.
"""
from langchain_openai import OpenAIEmbeddings

from app.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
    )
