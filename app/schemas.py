"""Pydantic request/response DTOs shared across routers."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# --- Auth ---
class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Documents ---
class DocumentResponse(BaseModel):
    id: str
    filename: str
    chunk_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChunkResponse(BaseModel):
    # Built from a Chroma dict, not an ORM row, so no from_attributes.
    id: str
    chunk: int
    content: str


# --- Conversations / chat ---
class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatRequest(BaseModel):
    # If conversation_id is omitted, the server creates a new conversation.
    conversation_id: str | None = None
    content: str = Field(min_length=1)
