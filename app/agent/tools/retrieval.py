"""RAG retrieval tool, bound to a single user's document collection."""
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool

from app.rag.vectorstore import get_vectorstore


def make_retrieval_tool(
    user_id: str,
    *,
    embeddings: Embeddings | None = None,
    persist_directory: str | None = None,
    k: int = 4,
) -> BaseTool:
    """Build a `search_documents` tool scoped to one user's Chroma collection."""

    @tool
    def search_documents(query: str) -> str:
        """Search the user's uploaded documents for passages relevant to the query."""
        vs = get_vectorstore(
            user_id, embeddings=embeddings, persist_directory=persist_directory
        )
        docs = vs.similarity_search(query, k=k)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(
            f"[{d.metadata.get('filename', '?')}] {d.page_content}" for d in docs
        )

    return search_documents
