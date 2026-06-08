"""ChatOpenAI factory.

Single place the chat model is configured, so model/temperature/streaming are
swap points. `streaming=True` is required for token-level SSE.
"""
from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm(streaming: bool = True) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
        streaming=streaming,
    )
