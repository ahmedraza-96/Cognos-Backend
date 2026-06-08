"""Tool registry: assembles the agent's tools for a given user.

The retrieval tool is per-user; calculator/datetime/http are shared; web search
is included only when a Tavily key is configured.
"""
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool

from app.agent.tools.calculator import calculator, current_datetime
from app.agent.tools.http_tool import http_request
from app.agent.tools.retrieval import make_retrieval_tool
from app.config import settings


def get_tools(
    user_id: str,
    *,
    embeddings: Embeddings | None = None,
    persist_directory: str | None = None,
) -> list[BaseTool]:
    tools: list[BaseTool] = [
        make_retrieval_tool(
            user_id, embeddings=embeddings, persist_directory=persist_directory
        ),
        calculator,
        current_datetime,
        http_request,
    ]
    if settings.tavily_api_key:
        from app.agent.tools.web_search import make_web_search_tool

        tools.append(make_web_search_tool())
    return tools
