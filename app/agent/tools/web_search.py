"""Tavily web-search tool factory.

Only constructed when a Tavily API key is configured (see get_tools).
"""
from langchain_core.tools import BaseTool

from app.config import settings


def make_web_search_tool() -> BaseTool:
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

    wrapper = TavilySearchAPIWrapper(tavily_api_key=settings.tavily_api_key)
    return TavilySearchResults(max_results=3, api_wrapper=wrapper)
