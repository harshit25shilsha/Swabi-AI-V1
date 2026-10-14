# Swabi AI

Standalone Python/FastAPI service providing AI capabilities to the Swabi
marketplace. Three capabilities are currently live:

- **AI Foundation** (Phase 0) — shared infrastructure for every capability.
- **Chat Safety** — marketplace chat moderation (contact-sharing, off-platform
  transactions, disguised numbers, scattered numbers).
- **Enquiry Understanding** (Phase 1) — free-text travel request → structured
  enquiry fields + clarifying questions.

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

**Service-to-service (Chat Safety, Enquiry Understanding):**

```
NestJS ──(service token + payload)──▶ /api/v1/...
                                         │
                            capability-specific logic
                                         │
                            LLM call through shared gateway
                                         │
       ◀─────────────────────────────────┘
```

**User-facing routes (future):**

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
| `POST` | `/api/v1/enquiry/understand` | service token | Enquiry Understanding extraction. |

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
`tests/contract/test_chat_safety_contract.py`.

> **`source="review"`** is emitted when a soft detector (currently the
> scattered-number rule) sees something suspicious but the LLM returns
> ALLOW. The response surfaces this as `allowed=false, action=BLOCK` with
> `source="review"` and lower confidence (0.65).

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

### Failure policy

When all LLM providers are unavailable (or all provider circuits are open),
the service **fails open**, unless a soft detector already flagged REVIEW:

- No REVIEW signal: `allowed=true, action=ALLOW, confidence=0.0,
  source="fail_open"`
- REVIEW signal present: `allowed=false, action=BLOCK, source="review"`

> **Production release gate.** The fail-open policy for the no-signal case
> is a deliberate product decision. It must be approved in writing by the
> Swabi product/backend owner before production traffic.

### Circuit breaker

Per-provider. Opens after `LLM_CIRCUIT_FAILURE_THRESHOLD` consecutive
failures. Half-open probe after `LLM_CIRCUIT_OPEN_SECONDS`. State is
exposed via `/metrics`.

### Connection pooling

`GroqProvider.generate_sync()` uses a module-level shared `httpx.Client`
with `max_keepalive_connections=20`, `max_connections=50`, and
`keepalive_expiry=30s`. The client is created lazily on first use and
closed on FastAPI shutdown (and again via `atexit` as a safety net).

---

## Enquiry Understanding

### Endpoint

```
POST /api/v1/enquiry/understand
Authorization: Bearer <GUARDRAIL_API_TOKEN>
Content-Type: application/json

{
  "message": "I want a 5-day family trip to Manali in December for 2 adults and two children aged 7 and 12, around 60k INR.",
  "user_id": "u_123",           // optional, reserved for personalization
  "currency_hint": "INR"        // optional, from customer profile
}
```

Only `message` is required. Length: 1–4000 chars.

### Response

```json
{
  "extraction": {
    "title": "5-day family trip to Manali",
    "description": "5-day family trip to Manali in December for a family of 4",
    "destination": "Manali",
    "travel_date": null,
    "travel_date_end": null,
    "date_flexibility": null,
    "duration_days": 5,
    "duration_nights": null,
    "travelers": 4,
    "adults": 2,
    "children": [{"age": 7}, {"age": 12}],
    "budget": {"amount": 60000.0, "currency": "INR", "is_per_person": false},
    "budget_range_min": null,
    "budget_range_max": null,
    "requirements": {
      "accommodation": "5-Star Hotel",
      "meals": ["All Meals"],
      "transport": null,
      "activities": ["Skiing", "Local Sightseeing"],
      "tour_guide": null,
      "special_requests": "senior citizen friendly"
    },
    "rooms": null,
    "property_type": null,
    "trip_type": "family",
    "flags": [],
    "tags": ["family"]
  },
  "missing_fields": ["transport", "tour_guide", "travel_date"],
  "clarifying_questions": [
    {"field": "transport", "question": "What transport would you prefer?"},
    {"field": "tour_guide", "question": "Would you like a tour guide?"},
    {"field": "travel_date", "question": "What date would you like to start?"}
  ],
  "completeness": 0.8,
  "provider": "groq",
  "prompt_version": "enquiry_37956fc3",
  "request_id": "gr_..."
}
```

The public response contract is locked by
`tests/contract/test_enquiry_contract.py`.

### Extraction fields

Every field is optional. Missing fields are `null` (scalars) or `[]`
(lists). No fabricated defaults.

The extraction schema is a **superset** of three sources:

- the live Enquiry form (13 fields)
- the `CreateEnquiryDto` (NestJS Swagger contract)
- the Swabi BRD `CUST-ENQ-*` fields (for future UI additions)

Fields the UI does not currently render (`rooms`, `property_type`,
`trip_type`, `flags`, `adults`, `children[]`) are populated when the
user mentions them and are dormant otherwise.

### Date resolution rules

- **Month-only** expressions (`"in December"`) leave `travel_date` null.
  The frontend's date picker collects the actual date.
- **Explicit dates** (`"15/12/26"`) are returned as `dd/mm/yy`.
- **Ordinal weekdays** (`"first Saturday of December"`, `"next Monday"`)
  are resolved using the current date as reference.
- `travel_date_end` is extracted when the user gives a range, or derived
  from `travel_date + duration_days - 1`.

### Canonical activities

The ten known activities are matched by name:

```
Shikara Ride, Gulmarg Gondola, Houseboat Stay, Skiing, Trekking,
Paragliding, Local Sightseeing, River Rafting, Camel Safari, Scuba Diving
```

Custom activities are returned as free strings (e.g. "Fly fishing").

### Enum values

- **`accommodation`** (single string): `3-Star Hotel`, `4-Star Hotel`,
  `5-Star Hotel`, `Luxury Resort`, `Houseboat`, `Homestay`, `Villa`
- **`transport`** (single string): `Private Cab (Sedan)`, `Private SUV`,
  `Tempo Traveller (12-Seater)`, `Luxury Sedan`, `Shared Coach`
- **`meals`** (list): `Breakfast`, `Lunch`, `Dinner`, `All Meals`
- **`tour_guide`** (single string): `English Speaking Guide`,
  `Local Language Guide`, `Photography Guide`, `Adventure Guide`, `Optional`
- **`property_type`** (single string, BRD field): `villa`, `apartment`,
  `hotel`, `homestay`, `resort`, `guesthouse`, `other`
- **`trip_type`** (single string, BRD field): `myself`, `family`, `others`
- **`flags`** (list, BRD field): `work`, `pets`, `couple_friendly`,
  `entire_apartment`

### Missing-field detection

Deterministic, in the order the frontend validates. Required fields first
(destination, when, travelers, accommodation, transport, meals, tour_guide,
activities, special_instructions, description), then askable fields
(budget, duration, travel_date). `when` is satisfied by any of
`travel_date`, `duration_days`, or `duration_nights`.

Clarifying questions are generated for missing fields, capped at
`MAX_CLARIFYING_QUESTIONS = 4`.

### Scope

Read-only. Does **not**:

- create an enquiry
- call vendors or bids
- touch bookings, payments, or escrow
- write anything to the database

It is a pure text → JSON transform.

---

## Authentication

There are **two distinct authentication flows**. They must not be conflated.

### 1. Inbound service-to-service

Both `/api/v1/validate-message` and `/api/v1/enquiry/understand` are
called by NestJS with a shared service token:

```
Authorization: Bearer <GUARDRAIL_API_TOKEN>
```

This token is **not** a user JWT. It carries no user identity, role, or
permissions. Comparison uses `secrets.compare_digest`.

The shared dependency lives in `app/core/service_auth.py` as
`verify_service_token`. It is re-exported by Chat Safety's
`dependencies.py` for backward compatibility.

### 2. Outbound user JWT forwarding (future user-facing routes)

User-facing AI routes accept the logged-in user's JWT and forward it
unchanged to NestJS when calling `SwabiClient`. NestJS authorizes
against the user's real identity.

The AI service treats the token as **opaque**:

- Never decoded, validated, or logged.
- Never persisted (not in Redis, not in state, not on disk).
- Forwarded unchanged on every outbound call within the same request.

### 3. The boundary — do not forward user JWTs to service-to-service routes

`auth_context_middleware` only captures the user JWT for routes under
`USER_JWT_ROUTE_PREFIXES`. `/api/v1/validate-message` and
`/api/v1/enquiry/understand` are excluded and explicitly clear the
context.

This prevents a service token from being captured as a user JWT and
forwarded to NestJS as if it were one. See
`tests/integration/test_auth_boundary.py`. Do not add a route to the
allow-list unless it genuinely acts on behalf of a logged-in user.

### Precedence (outbound)

| Situation | Header sent to NestJS |
|---|---|
| User JWT present | `Authorization: Bearer <user-jwt>` |
| No user JWT, service key present | `Authorization: Bearer <SWABI_API_KEY>` |
| Neither | No `Authorization` header |

### Where it lives in code

- `app/core/service_auth.py` — inbound service-token check.
- `app/core/auth_context.py` — request-scoped user-JWT contextvar.
- `app/main.py` — `auth_context_middleware`, `USER_JWT_ROUTE_PREFIXES`.
- `app/swabi/client.py` — `_headers()` applies the outbound precedence.

---

## Requirements

- Python 3.11+
- Redis 7+ (local, or remote URL)
- A Groq API key (required)
- A Gemini API key (optional — enables the Chat Safety fallback)

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

Generate the service token:

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
- `POST /api/v1/enquiry/understand` — Enquiry Understanding
- `GET /docs` — OpenAPI UI

### Smoke test

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/api/v1/validate-message \
  -H "Authorization: Bearer $GUARDRAIL_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Call me at 9876543210"}'

curl -X POST http://localhost:8000/api/v1/enquiry/understand \
  -H "Authorization: Bearer $GUARDRAIL_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"5-day family trip to Manali in December for 2 adults and two children aged 7 and 12."}'
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
| `LLM_CONSENSUS_ATTEMPTS` | `1` | Single call per message by default. |
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
- Live-provider tests are marked `@pytest.mark.integration` and excluded
  from the default run via `addopts = "-m 'not integration'"`.

---

## Project layout

```
app/
├── main.py                     FastAPI app: routes, middleware, handlers
├── core/                       Config, logging, exceptions, Redis,
│                               auth context, service auth
├── llm/                        Shared LLM gateway (Groq via httpx, retries)
├── capabilities/
│   ├── chat_safety/            Chat Safety
│   └── enquiry/                Enquiry Understanding
├── observability/              Counters, circuit breaker, request middleware
├── api/routes/                 health, metrics, chat_safety, enquiry
├── swabi/                      Async client for NestJS APIs
├── rag/                        Retriever interface (Phase 2)
├── graph/                      LangGraph foundation
└── evaluation/                 Dataset loader + runner

tests/{unit,integration,contract}/

docs/
├── architecture.md
├── guardrail_integration_plan.md
├── gdv1_test_verification.md
├── phase_guardrail_integration_report.md
└── phase_enquiry_report.md
```

---

## Scope

### Included

- FastAPI application with `/health`, `/ready`, `/metrics`.
- `POST /api/v1/validate-message` — Chat Safety.
- `POST /api/v1/enquiry/understand` — Enquiry Understanding.
- Shared LLM gateway with retries, timeouts, structured output.
- Swabi HTTP client (no direct DB access).
- Shared service-to-service authentication.
- Request ID correlation, structured logging, safe logging helpers.
- Redis connection manager.
- Per-provider circuit breakers and fail-open on outage (Chat Safety).
- Connection pooling for Groq calls.
- Unit / integration / contract tests.

### Explicitly NOT included

- Bid analyzer, semantic matching.
- Full RAG (embeddings, vector DB, ingestion).
- Autonomous agents, payments, wallet, escrow operations.
- Docker / Kubernetes / cloud infrastructure.
- Direct database access from Python.
- Async connection pooling (only the sync path pools today).
- Enquiry creation on the AI service (the frontend posts it directly).

---

## Production release gates

The Chat Safety integration is ready for review. Before production
traffic, the following must be closed:

- [ ] Fail-open policy approved in writing by the Swabi product/backend owner.
- [ ] Token rotation procedure documented and rehearsed once on staging.
- [ ] Deployment worker count decided and documented.
- [ ] Rate-limit expectation confirmed with the NestJS team.
- [ ] Manager sign-off on the failure policy and the auth boundary.

Enquiry Understanding is a read-only extraction and does not raise the
same failure-policy questions.

---

## Further reading

- [`docs/architecture.md`](docs/architecture.md) — reference for future
  contributors and new capabilities.
- [`docs/guardrail_integration_plan.md`](docs/guardrail_integration_plan.md) —
  the Chat Safety integration plan.
- [`docs/gdv1_test_verification.md`](docs/gdv1_test_verification.md) —
  test-suite audit for the guardrail move.
- [`docs/phase_guardrail_integration_report.md`](docs/phase_guardrail_integration_report.md) —
  final report for the Chat Safety integration.
- [`docs/phase_enquiry_report.md`](docs/phase_enquiry_report.md) —
  final report for the Enquiry Understanding phase.