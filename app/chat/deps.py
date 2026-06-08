"""Providers for the chat endpoint: the agent builder and the checkpointer.

Indirection so tests can inject a fake agent and skip the real checkpointer.
The checkpointer is created once at startup and stored on app.state (see
main.lifespan); here we just read it.
"""
from fastapi import Request

from app.agent.graph import build_agent


def get_agent_builder():
    return build_agent


def get_checkpointer(request: Request):
    return getattr(request.app.state, "checkpointer", None)
