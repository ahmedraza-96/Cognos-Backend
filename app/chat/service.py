"""Chat orchestration: drive the agent's event stream and map it to SSE frames.

`parse_agent_event` / `format_sse` are pure and unit-tested. `chat_event_stream`
is the async generator the endpoint streams: it relays tokens and tool events,
accumulates the assistant's reply, and persists it when the turn finishes.
"""
import json
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Message


def format_sse(payload: dict) -> str:
    """Serialize a payload as a single Server-Sent Events `data:` frame."""
    return f"data: {json.dumps(payload)}\n\n"


def parse_agent_event(event: dict) -> dict | None:
    """Map a LangGraph astream_events event to an SSE payload, or None to skip."""
    kind = event.get("event")

    if kind == "on_chat_model_stream":
        chunk = event.get("data", {}).get("chunk")
        text = getattr(chunk, "content", "") if chunk is not None else ""
        if text:
            return {"type": "token", "content": text}
        return None

    if kind == "on_tool_start":
        return {"type": "tool", "name": event.get("name", ""), "status": "start"}

    if kind == "on_tool_end":
        return {"type": "tool", "name": event.get("name", ""), "status": "end"}

    return None


async def chat_event_stream(
    *,
    agent,
    db: AsyncSession,
    conversation: Conversation,
    content: str,
) -> AsyncIterator[str]:
    """Stream one assistant turn as SSE frames and persist the assistant message."""
    assistant_parts: list[str] = []
    config = {"configurable": {"thread_id": conversation.id}}

    try:
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=content)]}, config=config, version="v2"
        ):
            payload = parse_agent_event(event)
            if payload is None:
                continue
            if payload["type"] == "token":
                assistant_parts.append(payload["content"])
            yield format_sse(payload)
    except Exception as exc:  # noqa: BLE001
        yield format_sse({"type": "error", "message": str(exc)})
        return

    final_text = "".join(assistant_parts)
    db.add(
        Message(conversation_id=conversation.id, role="assistant", content=final_text)
    )
    await db.commit()

    yield format_sse({"type": "done", "conversation_id": conversation.id})
