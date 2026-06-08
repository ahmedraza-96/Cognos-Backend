"""Integration tests for conversation CRUD and the streaming chat endpoint.

The streaming test injects a fake agent so the full stream→persist path is
exercised without calling OpenAI.
"""
from langchain_core.messages import AIMessageChunk

from app.chat.deps import get_agent_builder, get_checkpointer
from app.main import app


def _auth(client, email="c@h.com"):
    token = client.post(
        "/auth/signup", json={"email": email, "password": "password123"}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# --- conversation CRUD ---
def test_create_and_list_conversation(client):
    h = _auth(client)
    created = client.post("/conversations", headers=h)
    assert created.status_code == 201
    convo_id = created.json()["id"]

    listed = client.get("/conversations", headers=h)
    assert listed.status_code == 200
    assert [c["id"] for c in listed.json()] == [convo_id]


def test_conversations_are_per_user(client):
    ha = _auth(client, "a@h.com")
    hb = _auth(client, "b@h.com")
    client.post("/conversations", headers=ha)
    assert client.get("/conversations", headers=hb).json() == []


def test_delete_conversation(client):
    h = _auth(client)
    convo_id = client.post("/conversations", headers=h).json()["id"]
    assert client.delete(f"/conversations/{convo_id}", headers=h).status_code == 204
    assert client.get("/conversations", headers=h).json() == []


def test_get_messages_requires_ownership(client):
    ha = _auth(client, "a2@h.com")
    hb = _auth(client, "b2@h.com")
    convo_id = client.post("/conversations", headers=ha).json()["id"]
    assert client.get(f"/conversations/{convo_id}/messages", headers=hb).status_code == 404


# --- streaming chat ---
class _FakeAgent:
    async def astream_events(self, inputs, config=None, version=None):
        yield {"event": "on_tool_start", "name": "search_documents", "data": {}}
        yield {"event": "on_tool_end", "name": "search_documents", "data": {}}
        for tok in ["Hello", " world"]:
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content=tok)},
            }


def _use_fake_agent():
    app.dependency_overrides[get_agent_builder] = lambda: (
        lambda user_id, **kw: _FakeAgent()
    )
    app.dependency_overrides[get_checkpointer] = lambda: None


def test_chat_stream_emits_tool_and_token_frames(client):
    _use_fake_agent()
    h = _auth(client)
    r = client.post("/chat/stream", json={"content": "hi there"}, headers=h)
    assert r.status_code == 200
    body = r.text
    assert '"type": "tool"' in body
    assert '"type": "token"' in body
    assert '"type": "done"' in body
    assert "Hello" in body


def test_chat_stream_persists_user_and_assistant_messages(client):
    _use_fake_agent()
    h = _auth(client)
    r = client.post("/chat/stream", json={"content": "remember this"}, headers=h)
    convo_id = client.get("/conversations", headers=h).json()[0]["id"]
    msgs = client.get(f"/conversations/{convo_id}/messages", headers=h).json()
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant"]
    assert msgs[0]["content"] == "remember this"
    assert msgs[1]["content"] == "Hello world"
