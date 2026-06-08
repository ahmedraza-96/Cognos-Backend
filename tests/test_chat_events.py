"""Tests for the pure SSE event-mapping helpers in chat.service."""
from langchain_core.messages import AIMessageChunk

from app.chat.service import format_sse, parse_agent_event


def test_format_sse_wraps_json_payload():
    assert format_sse({"type": "token", "content": "hi"}) == (
        'data: {"type": "token", "content": "hi"}\n\n'
    )


def test_parse_token_event():
    ev = {
        "event": "on_chat_model_stream",
        "name": "ChatOpenAI",
        "data": {"chunk": AIMessageChunk(content="Hello")},
    }
    assert parse_agent_event(ev) == {"type": "token", "content": "Hello"}


def test_parse_empty_token_is_ignored():
    ev = {
        "event": "on_chat_model_stream",
        "data": {"chunk": AIMessageChunk(content="")},
    }
    assert parse_agent_event(ev) is None


def test_parse_tool_start_event():
    ev = {"event": "on_tool_start", "name": "search_documents", "data": {}}
    assert parse_agent_event(ev) == {
        "type": "tool",
        "name": "search_documents",
        "status": "start",
    }


def test_parse_tool_end_event():
    ev = {"event": "on_tool_end", "name": "calculator", "data": {}}
    assert parse_agent_event(ev) == {
        "type": "tool",
        "name": "calculator",
        "status": "end",
    }


def test_parse_unrelated_event_returns_none():
    assert parse_agent_event({"event": "on_chain_start", "name": "x", "data": {}}) is None
