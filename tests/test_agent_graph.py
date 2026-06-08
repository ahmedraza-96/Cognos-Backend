"""Test that the agent graph wires together into a runnable (no network calls)."""
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.config import settings

EMB = DeterministicFakeEmbedding(size=64)


def test_build_agent_returns_streaming_runnable(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "tavily_api_key", "")

    from app.agent.graph import build_agent

    agent = build_agent("u1", embeddings=EMB, persist_directory=str(tmp_path))
    # Compiled LangGraph agents expose astream_events for SSE streaming.
    assert hasattr(agent, "astream_events")
    assert hasattr(agent, "ainvoke")
