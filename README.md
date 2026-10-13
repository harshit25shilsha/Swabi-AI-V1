# Swabi AI

Standalone Python/FastAPI service providing AI capabilities to the Swabi
marketplace. Two capabilities are currently live:

- **AI Foundation** (Phase 0) — shared infrastructure for every capability.
- **Chat Safety** — marketplace chat moderation (contact-sharing, off-platform
  transactions, disguised numbers, scattered numbers).

The service is called **by** the Swabi NestJS backend and, for user-facing
routes, by the frontend through NestJS. It **never** accesses the Swabi
PostgreSQL database.

---

## Architecture

```
Swabi NestJS   = Business source of truth (auth, permissions, transactions)
Python FastAPI = AI service
Groq / Gemini  = Reasoning and generation
LangGraph      = Orchestration (foundation only today)
RAG            = Stable trusted knowledge (interface only today)
Swabi APIs     = Live business data (via SwabiClient, not direct DB)
```

### Request flows

**Chat Safety (service-to-service):**

```
NestJS ──(service token + message)──▶ /api/v1/validate-message
                                         │
                            deterministic detectors
                                         │
                            LLM classifier (Groq → Gemini)
                                         │
                            policy layer → ALLOW / BLOCK
                                         │
       ◀─────────────────────────────────┘
```

**User-facing routes (future, e.g. enquiry understanding):**

```
Frontend ──(user JWT)──▶ AI Service ──(same user JWT)──▶ NestJS APIs
```

For a full architecture reference, see
[`docs/architecture.md`](docs/architecture.md).

---

## Routes

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | none | Liveness. Status, uptime, versions. |
| `GET` | `/ready` | none | Readiness. Checks Redis. |
| `GET` | `/metrics` | none | Prometheus text. Counters + circuit state. |
| `POST` | `/api/v1/validate-message` | service token | Chat Safety classification. |

`/health`, `/ready`, and `/metrics` are unauthenticated by design — bind
behind a reverse proxy that restricts inbound access.

---

## Chat Safety

### Endpoint

```
POST /api/v1/validate-message
Authorization: Bearer <GUARDRAIL_API_TOKEN>
Content-Type: application/json

{
  "message": "Let's continue on WhatsApp.",
  "message_id": "msg_abc123",        // optional
  "conversation_id": "conv_456",     // optional
  "sender_id": "user_789",           // optional
  "sender_role": "user"              // optional: "user" | "vendor"
}
```

Only `message` is required. Length: 1–4000 chars.

### Response

```json
{
  "allowed": false,
  "action": "BLOCK",
  "category": "OFF_PLATFORM_COMMUNICATION",
  "confidence": 0.96,
  "source": "llm",
  "request_id": "gr_52b80e854eaa49ac",
  "prompt_version": "marketplace_<hash>",
  "provider": "groq"
}
```

| Field | Meaning |
|---|---|
| `allowed` | Whether the marketplace may deliver the message |
| `action` | `ALLOW` or `BLOCK` |
| `category` | Moderation category, or `null` for allow |
| `confidence` | `0.0`–`1.0` |
| `source` | `deterministic`, `llm`, `review`, or `fail_open` |
| `request_id` | Correlates with the `X-Request-ID` response header |
| `prompt_version` | Version hash of the active system prompt |
| `provider` | `groq`, `gemini`, or `""` for deterministic decisions |

The public response contract is locked by
`tests/contract/test_chat_safety_contract.py`. Do not add or remove fields
without coordinating with the NestJS team.

> **`source="review"`** is emitted when a soft detector (currently the
> scattered-number rule) sees something suspicious but the LLM returns
> ALLOW. The response surfaces this as `allowed=false, action=BLOCK` with
> `source="review"` and lower confidence (0.65), so the NestJS caller can
> distinguish it from a hard block if it wants a different UX. The public
> field set is unchanged.

### Categories

`PHONE_NUMBER`, `EMAIL`, `SOCIAL_CONTACT`, `EXTERNAL_URL`,
`PAYMENT_INFORMATION`, `OFF_PLATFORM_COMMUNICATION`,
`OFF_PLATFORM_TRANSACTION`, `OTHER_PROHIBITED_CONTENT`.

### Detection layers

1. **Hard deterministic** — phone, email, URL, UPI, disguised number runs,
   arithmetic encoding, high digit density. A match returns BLOCK
   immediately; the LLM is skipped.
2. **Soft deterministic** — scattered-number detection. Returns
   BLOCK / REVIEW / ALLOW based on aggregate chain analysis. A REVIEW
   signal escalates even when the LLM says ALLOW.
3. **LLM classifier** — Groq with Gemini fallback. Handles contextual intent,
   obfuscated contact sharing, and mixed-language messages (English, Hindi,
   Hinglish, Devanagari).
4. **Policy layer** — maps evidence to a final decision.

#### Scattered-number detection

Catches phone numbers hidden across natural language — for example:

> *ek room, do bed ke saath, teen raat ke liye, chaar log hain, paach bje
> phuchenge, das bje khayenge, gyarah bje niklenge, bara bje lunch
> karenge, pandrah minute late honge.*

Nine numeric tokens embedded in ordinary sentences. Each number looks
innocent in isolation; the aggregate pattern is the leak. The detector
analyzes chain length, distinct-value count, and gap between numbers,
returning:

- `BLOCK` — 8+ distinct values in a chain
- `REVIEW` — 6–7 distinct values
- `ALLOW` — fewer, or clearly a quantity list with repeats

The public response maps REVIEW to `allowed=false, action=BLOCK,
source="review"` so it is delivered as a block. NestJS can distinguish
soft from hard blocks if needed.

### Failure policy

When all LLM providers are unavailable (or all provider circuits are open),
the service **fails open**, unless a soft detector already flagged REVIEW:

- No REVIEW signal: `allowed=true, action=ALLOW, confidence=0.0,
  source="fail_open"`
- REVIEW signal present: `allowed=false, action=BLOCK,
  source="review"`

> **Production release gate.** The fail-open policy for the no-signal case
> is a deliberate product decision carried over from the original guardrail
> service. It must be approved in writing by the Swabi product/backend owner
> before production traffic. See
> [`docs/guardrail_integration_plan.md`](docs/guardrail_integration_plan.md) §7.2.

### Circuit breaker

Per-provider. Opens after `LLM_CIRCUIT_FAILURE_THRESHOLD` consecutive
failures. Half-open probe after `LLM_CIRCUIT_OPEN_SECONDS`. State is
exposed via `/metrics`.

### Connection pooling

`GroqProvider.generate_sync()` uses a module-level shared `httpx.Client`
with `max_keepalive_connections=20`, `max_connections=50`, and
`keepalive_expiry=30s`. The client is created lazily on first use and
closed on FastAPI shutdown (and again via `atexit` as a safety net). This
eliminates per-request TCP + TLS handshakes on the Chat Safety path.

---

## Authentication

There are **two distinct authentication flows**. They must not be conflated.

### 1. Inbound service-to-service (Chat Safety)

The NestJS backend calls `/api/v1/validate-message` with a shared service
token:

```
Authorization: Bearer <GUARDRAIL_API_TOKEN>
```

This token is **not** a user JWT. It carries no user identity, role, or
permissions. Comparison uses `secrets.compare_digest`.

### 2. Outbound user JWT forwarding (future user-facing routes)

User-facing AI routes accept the logged-in user's JWT and forward it
unchanged to NestJS when calling `SwabiClient`. NestJS authorizes against
the user's real identity.

```
Frontend ──(user JWT)──▶ AI Service ──(same user JWT)──▶ NestJS APIs
```

The AI service treats the token as **opaque**:

- Never decoded, validated, or logged.
- Never persisted (not in Redis, not in state, not on disk).
- Forwarded unchanged on every outbound call within the same request.

NestJS remains the only authority for authentication and authorization.

### 3. The boundary — do not forward user JWTs to the guardrail

`auth_context_middleware` only captures the user JWT for routes under
`USER_JWT_ROUTE_PREFIXES`. Service-to-service routes — including
`/api/v1/validate-message` — are excluded and explicitly clear the context.

This prevents the guardrail's service token from being captured as a user
JWT and forwarded to NestJS as if it were one. See
`tests/integration/test_auth_boundary.py`. Do not add a route to the
allow-list unless it genuinely acts on behalf of a logged-in user.

### Precedence (outbound)

| Situation | Header sent to NestJS |
|---|---|
| User JWT present | `Authorization: Bearer <user-jwt>` |
| No user JWT, service key present | `Authorization: Bearer <SWABI_API_KEY>` |
| Neither | No `Authorization` header |

### Where it lives in code

- `app/core/auth_context.py` — request-scoped contextvar.
- `app/main.py` — `auth_context_middleware`, `USER_JWT_ROUTE_PREFIXES`.
- `app/swabi/client.py` — `_headers()` applies the precedence rule.
- `app/capabilities/chat_safety/dependencies.py` — `verify_token` for the
  guardrail route.

---

## Requirements

- Python 3.11+
- Redis 7+ (local, or remote URL)
- A Groq API key (required)
- A Gemini API key (optional — enables the fallback provider)

## Local setup (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
cp .env.example .env
# edit .env with at minimum:
#   GROQ_API_KEY=<your key>
#   GUARDRAIL_API_TOKEN=<generate a long random string>
#   REDIS_URL=redis://localhost:6379/0
```

Generate the guardrail token:

```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

### Run Redis locally

```bash
redis-server
redis-cli ping   # → PONG
```

### Run the AI service

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` — liveness
- `GET /ready` — readiness (checks Redis)
- `GET /metrics` — Prometheus metrics
- `POST /api/v1/validate-message` — Chat Safety
- `GET /docs` — OpenAPI UI

### Smoke test

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/api/v1/validate-message \
  -H "Authorization: Bearer $GUARDRAIL_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Call me at 9876543210"}'
```

---

## Configuration

All configuration lives in `app/core/config.py::Settings`. Field names are
lower-case Pydantic names; upper-case properties exist as read-only
aliases for legacy guardrail code compatibility.

**Required with no default:**

- `GROQ_API_KEY`
- `GUARDRAIL_API_TOKEN`

**Everything else has a development-safe default.**

Key settings:

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq model |
| `GROQ_CHAT_URL` | `https://api.groq.com/openai/v1/chat/completions` | Groq endpoint |
| `LLM_TIMEOUT` | `60` | Seconds |
| `LLM_MAX_RETRIES` | `2` | |
| `LLM_MAX_TOKENS` | `512` | Moderation JSON is small, but reasoning models spend 80–150 tokens on internal reasoning before emitting JSON. 128 caused `json_validate_failed`; 512 leaves headroom. |
| `LLM_TEMPERATURE` | `0.0` | |
| `LLM_CONSENSUS_ATTEMPTS` | `1` | Single call per message by default. Raising it requires evaluation evidence. |
| `LLM_FALLBACK_ENABLED` | `true` | Enables Gemini fallback when a key is present |
| `GEMINI_API_KEY` | `""` | Empty disables the fallback |
| `GEMINI_MODEL` | `gemini-2.0-flash` | |
| `GEMINI_TIMEOUT_SECONDS` | `15` | Wall-clock cap for the fallback |
| `LLM_CIRCUIT_FAILURE_THRESHOLD` | `5` | |
| `LLM_CIRCUIT_OPEN_SECONDS` | `60` | |
| `LOG_FORMAT` | `json` | `json` or `text` |
| `REDIS_URL` | `redis://localhost:6379/0` | Empty disables Redis |

Never commit `.env`. The template is `.env.example`.

See [`docs/architecture.md`](docs/architecture.md) §8 for the full list.

---

## Testing

```bash
# Default run: unit + integration with mocks. No live providers.
pytest

# Live-provider tests (opt-in; requires real GROQ_API_KEY in .env).
pytest -m integration -v

# Lint and format.
ruff check .
ruff format --check .
```

Test layout:

- `tests/unit/` — pure logic, mocked dependencies.
- `tests/integration/` — FastAPI `TestClient`, mocked external calls.
- `tests/contract/` — public API shape guards.
- Live-provider tests are marked `@pytest.mark.integration` and excluded from
  the default run via `addopts = "-m 'not integration'"`.

---

## Project layout

```
app/
├── main.py                     FastAPI app: routes, middleware, handlers
├── core/                       Config, logging, exceptions, Redis, auth context
├── llm/                        Shared LLM gateway (Groq via httpx, retries)
├── capabilities/
│   └── chat_safety/            Chat Safety (detectors, policy, prompt, adapter)
├── observability/              Counters, circuit breaker, request middleware
├── api/routes/                 health, metrics, chat_safety
├── swabi/                      Async client for NestJS APIs
├── rag/                        Retriever interface (Phase 2)
├── graph/                      LangGraph foundation
└── evaluation/                 Dataset loader + runner

tests/{unit,integration,contract}/

docs/
├── architecture.md
├── guardrail_integration_plan.md
├── gdv1_test_verification.md
└── phase_guardrail_integration_report.md
```

---

## Scope

### Included

- FastAPI application with `/health`, `/ready`, `/metrics`,
  `/api/v1/validate-message`.
- Shared LLM gateway with retries, timeouts, structured output.
- Swabi HTTP client (no direct DB access).
- Tool base pattern; minimal LangGraph foundation; RAG interface.
- Request ID correlation, structured logging, safe logging helpers.
- Redis connection manager.
- Chat Safety: hard and soft deterministic detectors, LLM classifier,
  policy layer, REVIEW escalation.
- Scattered-number detection (aggregate chain analysis with REVIEW
  escalation).
- Shared sync `httpx.Client` with connection pooling for Groq calls.
- Per-provider circuit breakers and fail-open on outage.
- Evaluation runner + sample dataset.
- Unit / integration / contract tests.

### Explicitly NOT included

- Enquiry understanding, bid analyzer, semantic matching.
- Full RAG (embeddings, vector DB, ingestion).
- Autonomous agents, payments, wallet, escrow operations.
- Docker / Kubernetes / cloud infrastructure.
- Direct database access from Python.
- A `Guardrail` ABC or plugin registry (deferred until a second capability
  needs it).
- Async connection pooling (only the sync path pools today).

---

## Production release gates

The codebase is ready for review. Before production traffic, the following
must be closed (tracked in
[`docs/guardrail_integration_plan.md`](docs/guardrail_integration_plan.md) §7.2):

- [ ] Fail-open policy approved in writing by the Swabi product/backend owner.
- [ ] Token rotation procedure documented and rehearsed once on staging.
- [ ] Deployment worker count decided and documented (metrics are per-process).
- [ ] Rate-limit expectation confirmed with the NestJS team.
- [ ] Manager sign-off on the failure policy and the auth boundary.

---

## Further reading

- [`docs/architecture.md`](docs/architecture.md) — reference for future
  contributors and new capabilities.
- [`docs/guardrail_integration_plan.md`](docs/guardrail_integration_plan.md) —
  the integration plan v2.
- [`docs/gdv1_test_verification.md`](docs/gdv1_test_verification.md) — test-suite
  audit performed before the guardrail move.
- [`docs/phase_guardrail_integration_report.md`](docs/phase_guardrail_integration_report.md) —
  final report for the Chat Safety integration.