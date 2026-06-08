"""Document endpoints: upload (ingest into RAG), list, delete.

Uploads are validated (extension + size), saved to disk, chunked, embedded,
and stored in the user's Chroma collection. The Document row tracks the file;
its chunks live in the vector store, linked by document id.
"""
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from langchain_core.embeddings import Embeddings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Document, User
from app.rag import ingest, vectorstore
from app.rag.deps import (
    chroma_dir_provider,
    embeddings_provider,
    upload_dir_provider,
)
from app.schemas import DocumentResponse

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    embeddings: Embeddings = Depends(embeddings_provider),
    chroma_dir: str = Depends(chroma_dir_provider),
    upload_dir: str = Depends(upload_dir_provider),
) -> Document:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File exceeds {settings.max_upload_mb} MB limit")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    doc = Document(user_id=current_user.id, filename=file.filename, chunk_count=0)
    db.add(doc)
    await db.flush()  # populate doc.id

    os.makedirs(upload_dir, exist_ok=True)
    dest = Path(upload_dir) / f"{doc.id}{ext}"
    dest.write_bytes(content)

    text = ingest.load_file(str(dest))
    doc.chunk_count = ingest.ingest_document(
        user_id=current_user.id,
        document_id=doc.id,
        filename=file.filename,
        text=text,
        embeddings=embeddings,
        persist_directory=chroma_dir,
    )

    await db.commit()
    await db.refresh(doc)
    return doc


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[Document]:
    rows = await db.scalars(
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
    )
    return list(rows)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    embeddings: Embeddings = Depends(embeddings_provider),
    chroma_dir: str = Depends(chroma_dir_provider),
    upload_dir: str = Depends(upload_dir_provider),
) -> None:
    doc = await db.get(Document, document_id)
    if doc is None or doc.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found")

    vectorstore.delete_document(
        current_user.id, document_id, embeddings=embeddings, persist_directory=chroma_dir
    )
    for f in Path(upload_dir).glob(f"{document_id}.*"):
        f.unlink(missing_ok=True)

    await db.delete(doc)
    await db.commit()
