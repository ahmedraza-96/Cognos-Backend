# Cognos — Backend

> Chat with a tool-using AI agent over your own documents — streaming, with memory.

FastAPI service powering a tool-using AI agent with RAG, built on **LangChain +
LangGraph**. Pairs with the Next.js frontend (separate repo).

## Features

- 🔐 **Auth** — email/password signup & login (JWT, bcrypt)
- 💬 **Streaming chat** — token-by-token responses over SSE, with live tool events
- 🧠 **LangGraph agent** — prebuilt ReAct loop with per-conversation memory (Postgres checkpointer, in-memory fallback)
- 🧰 **Tools** — RAG document search, Tavily web search, calculator, allowlisted HTTP tool
- 📄 **RAG** — upload `.txt` / `.md` / `.pdf`, chunked + embedded into a per-user Chroma collection
- 🗂️ **History** — conversations and messages persisted in Postgres

## Module layout

```
app/
  main.py            FastAPI app, CORS, lifespan (DB + checkpointer)
  config.py          Settings (env)
  database.py        Async SQLAlchemy engine/session
  models.py          User, Conversation, Message, Document
  schemas.py         Pydantic DTOs
  auth/              security (JWT/bcrypt), deps (get_current_user), router
  agent/             llm, graph (create_react_agent), tools/
  rag/               embeddings, vectorstore (per-user Chroma), ingest
  chat/              service (SSE event mapping), router (stream + conversations)
  documents/         upload / list / delete
tests/               pytest suite (SQLite + fakes; no OpenAI/Postgres needed)
```

## Prerequisites

- Python 3.10+
- Docker Desktop (for Postgres)
- An OpenAI API key (required); a Tavily API key (optional, enables web search)

## Setup

```bash
# 1. Start Postgres
docker compose up -d

# 2. Create the virtualenv + install deps
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 3. Configure secrets
copy .env.example .env            # then edit .env:
#   - OPENAI_API_KEY
#   - JWT_SECRET   (python -c "import secrets; print(secrets.token_hex(32))")
#   - TAVILY_API_KEY (optional)

# 4. Run
uvicorn app.main:app --reload     # http://localhost:8000  (interactive docs at /docs)
```

## Tests

```bash
.venv\Scripts\python -m pytest    # 46 tests — uses SQLite + fake embeddings/agent
```

## Configuration reference

See `.env.example`. Notable vars:

| Var | Purpose |
|-----|---------|
| `OPENAI_API_KEY` | Chat + embeddings (required) |
| `TAVILY_API_KEY` | Enables the web-search tool (optional) |
| `DATABASE_URL` | Async SQLAlchemy URL (Postgres) |
| `CHECKPOINT_DB_URL` | Sync psycopg URL for the LangGraph checkpointer |
| `HTTP_TOOL_ALLOWED_HOSTS` | Comma-separated host allowlist for the HTTP tool (SSRF guard) |
| `MAX_UPLOAD_MB` | Upload size limit |
| `CORS_ORIGINS` | Allowed frontend origins 

## Security notes

- Passwords are bcrypt-hashed; API keys live only in this backend's `.env` (gitignored).
- Each user has an isolated Chroma collection — no cross-user document access.
- The HTTP tool refuses non-allowlisted hosts and blocks loopback/private/link-local addresses.
- For production: front with HTTPS, and have the frontend store the JWT in an httpOnly cookie rather than `localStorage`.
