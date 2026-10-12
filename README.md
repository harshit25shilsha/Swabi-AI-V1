# Swabi AI — Phase 0 (AI Foundation)

Standalone Python/FastAPI service providing the reusable AI foundation for the
Swabi AI Travel Assistant. This is **Phase 0** — infrastructure only, no
business features.

## Architecture

```
Swabi NestJS  = Business source of truth
Python FastAPI = AI service
Groq / LLM     = Reasoning and generation
LangGraph      = Orchestration
RAG            = Stable trusted knowledge (interface only in Phase 0)
Swabi APIs     = Live business data
```

The AI service **never** accesses Swabi PostgreSQL directly. All live data
flows through the NestJS backend via `SwabiClient`.

## Authentication

The AI service supports two outbound authentication modes when calling the
Swabi NestJS backend.

### 1. Caller-forwarded user JWT (preferred)

The frontend sends the logged-in user's JWT to the AI service:

```
Authorization: Bearer <user-jwt>
```

The AI service treats the token as **opaque**:

- It is never decoded, validated, or logged.
- It is never persisted (not in Redis, not in state, not on disk).
- It is forwarded unchanged to NestJS on every outbound call made during
  the same request.

NestJS remains the only authority for authentication and authorization.

### 2. Service API key (fallback)

For internal / background calls (health warmups, future batch jobs) the AI
service uses the service key:

```
Authorization: Bearer <SWABI_API_KEY>
```

Configure via `SWABI_API_KEY` in `.env`.

### Precedence

| Situation | Header sent to NestJS |
|---|---|
| User JWT present | `Authorization: Bearer <user-jwt>` |
| No user JWT, service key present | `Authorization: Bearer <service-key>` |
| Neither | No `Authorization` header |

### Security guarantees

- Bearer scheme only. `Basic`, cookie, and query-param tokens are ignored.
- Token is never logged.
- Token is never decoded in Python.
- Token is not stored across requests.
- 401 from NestJS is surfaced unchanged; the AI service does not refresh tokens.

### Where it lives in code

- `app/core/auth_context.py` — request-scoped contextvar holding the token.
- `app/main.py` — `auth_context_middleware` extracts the token.
- `app/swabi/client.py` — `_headers()` applies the precedence rule.


## Requirements

- Python 3.11+
- Redis 7+ (local, or remote URL)
- A Groq API key

## Local setup (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
cp .env.example .env
# edit .env with your GROQ_API_KEY, SWABI_API_KEY, REDIS_URL
```

### Run Redis locally

Install via your OS package manager or Homebrew, then:

```bash
redis-server
```

Verify:

```bash
redis-cli ping   # → PONG
```

### Run the AI service

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` — liveness
- `GET /ready` — readiness (checks Redis)
- `GET /docs` — OpenAPI UI

### Run tests

```bash
pytest
```

### Lint / format

```bash
ruff check .
ruff format .
```

## Configuration

All configuration is centralized in `app/core/config.py` and read from
environment variables (or `.env`). Never commit `.env`.

## Phase 0 scope

Included:

- FastAPI application, `/health`, `/ready`
- LLM gateway (Groq) with retries, timeouts, structured output
- Swabi HTTP client (no direct DB)
- Tool base pattern
- LangGraph minimal orchestration
- RAG interface + mock retriever
- Request ID, structured logging, safe logging helpers
- Redis connection manager
- Evaluation runner + sample dataset
- Unit / integration tests

Explicitly NOT included:

- Enquiry understanding, bid analyzer, semantic matching
- Full RAG (embeddings, vector DB, ingestion)
- Autonomous agents, payments, wallet, escrow operations
- Docker / Kubernetes / cloud infra