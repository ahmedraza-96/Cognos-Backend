"""Chat + conversation endpoints.

`/chat/stream` runs the agent and streams SSE frames; the conversation routes
provide history for the UI. The user message is persisted before streaming; the
assistant message is persisted by the stream generator when the turn completes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.embeddings import Embeddings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_current_user
from app.chat.deps import get_agent_builder, get_checkpointer
from app.chat.service import chat_event_stream
from app.database import get_db
from app.models import Conversation, Message, User
from app.rag.deps import chroma_dir_provider, embeddings_provider
from app.schemas import ChatRequest, ConversationResponse, MessageResponse

router = APIRouter(tags=["chat"])


async def _get_owned_conversation(
    db: AsyncSession, user: User, conversation_id: str
) -> Conversation:
    convo = await db.get(Conversation, conversation_id)
    if convo is None or convo.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return convo


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Conversation:
    convo = Conversation(user_id=current_user.id)
    db.add(convo)
    await db.commit()
    await db.refresh(convo)
    return convo


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[Conversation]:
    rows = await db.scalars(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(rows)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def get_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[Message]:
    convo = await _get_owned_conversation(db, current_user, conversation_id)
    rows = await db.scalars(
        select(Message)
        .where(Message.conversation_id == convo.id)
        .order_by(Message.created_at)
    )
    return list(rows)


@router.delete(
    "/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    convo = await _get_owned_conversation(db, current_user, conversation_id)
    await db.delete(convo)
    await db.commit()


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    agent_builder=Depends(get_agent_builder),
    checkpointer=Depends(get_checkpointer),
    embeddings: Embeddings = Depends(embeddings_provider),
    chroma_dir: str = Depends(chroma_dir_provider),
) -> StreamingResponse:
    if payload.conversation_id:
        convo = await _get_owned_conversation(db, current_user, payload.conversation_id)
    else:
        convo = Conversation(user_id=current_user.id, title=payload.content[:50])
        db.add(convo)
        await db.flush()

    db.add(Message(conversation_id=convo.id, role="user", content=payload.content))
    await db.commit()
    await db.refresh(convo)

    agent = agent_builder(
        current_user.id,
        checkpointer=checkpointer,
        embeddings=embeddings,
        persist_directory=chroma_dir,
    )
    return StreamingResponse(
        chat_event_stream(agent=agent, db=db, conversation=convo, content=payload.content),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Conversation-Id": convo.id,
        },
    )
