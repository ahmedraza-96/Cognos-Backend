"""Assemble the LangGraph agent.

`build_agent` wires the chat model, the user's tools, and an optional
checkpointer into LangGraph's prebuilt ReAct agent. The checkpointer (passed in
by the chat service) gives the agent per-conversation memory.
"""
from langchain_core.embeddings import Embeddings
from langgraph.prebuilt import create_react_agent

from app.agent.llm import get_llm
from app.agent.tools import get_tools

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools. "
    "Use the search_documents tool to answer questions about the user's uploaded "
    "documents, and cite the source filename when you do. Use the calculator for "
    "arithmetic, current_datetime for the date/time, web search for current events, "
    "and http_request only for explicitly allowlisted APIs. "
    "If a tool isn't needed, just answer directly."
)


def build_agent(
    user_id: str,
    *,
    checkpointer=None,
    embeddings: Embeddings | None = None,
    persist_directory: str | None = None,
):
    llm = get_llm()
    tools = get_tools(
        user_id, embeddings=embeddings, persist_directory=persist_directory
    )
    return create_react_agent(
        llm,
        tools,
        state_modifier=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
