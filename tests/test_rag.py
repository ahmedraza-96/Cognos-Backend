"""Tests for the RAG pipeline: chunking, ingestion, per-user retrieval.

Uses a deterministic offline fake embedding so the suite needs no OpenAI key.
These assert plumbing (chunking, per-user collections, metadata, isolation),
not embedding semantics.
"""
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.rag import ingest, vectorstore

EMB = DeterministicFakeEmbedding(size=64)


def test_split_text_produces_multiple_chunks():
    text = "sentence. " * 500  # long enough to split
    chunks = ingest.split_text(text)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and c for c in chunks)


def test_ingest_then_retrieve_from_user_collection(tmp_path):
    n = ingest.ingest_document(
        user_id="u1",
        document_id="d1",
        filename="facts.txt",
        text="The capital of France is Paris. Bananas are yellow fruit.",
        embeddings=EMB,
        persist_directory=str(tmp_path),
    )
    assert n >= 1

    vs = vectorstore.get_vectorstore("u1", embeddings=EMB, persist_directory=str(tmp_path))
    results = vs.similarity_search("anything", k=5)
    assert results, "expected retrieval from u1's collection"
    assert all(r.metadata.get("document_id") == "d1" for r in results)
    assert all(r.metadata.get("filename") == "facts.txt" for r in results)


def test_documents_are_isolated_per_user(tmp_path):
    ingest.ingest_document(
        user_id="u1",
        document_id="d1",
        filename="secret.txt",
        text="alpha bravo charlie secret content",
        embeddings=EMB,
        persist_directory=str(tmp_path),
    )
    other = vectorstore.get_vectorstore("u2", embeddings=EMB, persist_directory=str(tmp_path))
    assert other.similarity_search("alpha", k=5) == []


def test_delete_document_removes_its_chunks(tmp_path):
    ingest.ingest_document(
        user_id="u1",
        document_id="d1",
        filename="a.txt",
        text="content to be deleted later",
        embeddings=EMB,
        persist_directory=str(tmp_path),
    )
    vectorstore.delete_document("u1", "d1", embeddings=EMB, persist_directory=str(tmp_path))
    vs = vectorstore.get_vectorstore("u1", embeddings=EMB, persist_directory=str(tmp_path))
    assert vs.similarity_search("content", k=5) == []
