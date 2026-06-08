"""Tests for the agent tools: calculator, http allowlist, retrieval, registry."""
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.agent.tools import get_tools
from app.agent.tools.calculator import safe_eval
from app.agent.tools.http_tool import is_host_allowed
from app.agent.tools.retrieval import make_retrieval_tool
from app.config import settings
from app.rag import ingest

EMB = DeterministicFakeEmbedding(size=64)


# --- calculator ---
def test_safe_eval_basic_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14


def test_safe_eval_power_and_parentheses():
    assert safe_eval("(1 + 2) ** 3") == 27


def test_safe_eval_rejects_names_and_calls():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hi')")


def test_safe_eval_rejects_division_by_zero():
    with pytest.raises(ValueError):
        safe_eval("1 / 0")


# --- http tool SSRF allowlist ---
def test_http_allows_listed_host():
    assert is_host_allowed("https://api.github.com/repos/x", ["api.github.com"]) is True


def test_http_blocks_unlisted_host():
    assert is_host_allowed("https://evil.example.com", ["api.github.com"]) is False


def test_http_blocks_loopback_even_if_listed():
    assert is_host_allowed("http://localhost:8000/admin", ["localhost"]) is False
    assert is_host_allowed("http://127.0.0.1/", ["127.0.0.1"]) is False


def test_http_blocks_cloud_metadata_ip():
    assert is_host_allowed("http://169.254.169.254/latest/meta-data", ["*"]) is False


# --- retrieval tool ---
def test_retrieval_tool_returns_user_chunks(tmp_path):
    ingest.ingest_document(
        user_id="u1",
        document_id="d1",
        filename="f.txt",
        text="alpha bravo retrieval content here",
        embeddings=EMB,
        persist_directory=str(tmp_path),
    )
    tool = make_retrieval_tool("u1", embeddings=EMB, persist_directory=str(tmp_path))
    out = tool.invoke({"query": "alpha"})
    assert "f.txt" in out


def test_retrieval_tool_reports_when_empty(tmp_path):
    tool = make_retrieval_tool("nobody", embeddings=EMB, persist_directory=str(tmp_path))
    out = tool.invoke({"query": "anything"})
    assert "No relevant" in out


# --- registry ---
def test_get_tools_excludes_web_search_without_key(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "tavily_api_key", "")
    tools = get_tools("u1", embeddings=EMB, persist_directory=str(tmp_path))
    names = {t.name for t in tools}
    assert {"search_documents", "calculator", "current_datetime", "http_request"} <= names
    assert not any("tavily" in n for n in names)


def test_get_tools_includes_web_search_with_key(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "tavily_api_key", "fake-key")
    tools = get_tools("u1", embeddings=EMB, persist_directory=str(tmp_path))
    assert any("tavily" in t.name for t in tools)
