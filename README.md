# Swabi AI

Standalone Python/FastAPI service providing AI capabilities to the Swabi
marketplace. Four capabilities are currently implemented:

- **AI Foundation** (Phase 0) — shared infrastructure for every capability.
- **Chat Safety** — marketplace chat moderation (contact-sharing, off-platform
  transactions, disguised numbers, scattered numbers).
- **Enquiry Understanding** (Phase 1) — free-text travel request → structured
  enquiry fields + clarifying questions.
- **AI Support / RAG** (Phase 2 implementation) — grounded support answers
  from approved knowledge and live Swabi APIs, with audience-aware retrieval,
  policy freshness checks, and hybrid BM25 + dense search.

The service is called **by** the Swabi NestJS backend and, for user-facing
routes, by the frontend through NestJS. AI Support accepts the user's bearer
JWT and forwards it to authorized NestJS endpoints when needed. The service
**never** accesses the Swabi PostgreSQL database.

---

## Architecture

```
Swabi NestJS   = Business source of truth (auth, permissions, transactions)
Python FastAPI = AI service
Groq / Gemini  = Reasoning and generation
LangGraph      = Orchestration (foundation only today)
RAG            = Approved knowledge (dense + BM25, fused with RRF)
Swabi APIs     = Live business data (called by AI Support, never direct DB)
```

### Request flows

**Gateway mode (target production flow for every AI capability):**

```text
Frontend -- user JWT --> NestJS validates identity and scopes
                             |
                             +-- signed principal assertion --> private AI service
                                                                    |
                                             KB answer or delegated live NestJS API read
```

NestJS validates before both KB-only and live Support queries. For a live
request, the AI service forwards the original opaque user JWT to the
allowlisted NestJS API; NestJS authorizes the requested resource.

**Legacy migration flows (temporary only):**

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

**User-facing AI Support:**

```
Frontend ──(user JWT via NestJS)──▶ AI Service /api/v1/support/ask
                                          ├── approved knowledge in Qdrant
                                          └── authorized live calls ──▶ NestJS APIs
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
| `POST` | `/api/v1/validate-message` | gateway assertion + `ai:chat-safety` | Chat Safety classification. |
| `POST` | `/api/v1/enquiry/understand` | gateway assertion + `ai:enquiry:understand` | Enquiry Understanding extraction. |
| `POST` | `/api/v1/support/ask` | gateway assertion + `ai:support:ask` | RAG support answer; uses approved knowledge and/or delegated live data. |

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

## AI Support (Phase 2)

### Endpoint and authentication

```
POST /api/v1/support/ask
Authorization: Bearer <user JWT>
Content-Type: application/json

{
  "question": "How can I book a package through an enquiry?",
  "audience": "customer"
}
```

For multi-turn follow-ups, the caller may include up to 10 recent turns:

```json
{
  "question": "Only show the latest two",
  "audience": "customer",
  "conversation_history": [
    {"role": "user", "content": "Show my payment history"},
    {"role": "assistant", "content": "I found your payment history..."}
  ]
}
```

The integration must bind the audience and JWT to the authenticated session
and enforce role scope server-side; a client-supplied audience is not an
authorization grant. Conversation history helps resolve references and
constraints but never grants access or changes role scope.

The route requires a user bearer token. Private live-data calls forward that
token to NestJS, where authorization remains authoritative. Public catalogue
intents (packages, package details, activities, and popular destinations) do
not require a JWT at the Swabi API call. The AI service never queries the
Swabi database directly.

### Answer sources

- **Approved knowledge:** Stable policies and process guidance are retrieved
  from Qdrant, filtered by audience and knowledge area. Policy chunks must be
  approved, effective, and not expired.
- **Hybrid intent and answers:** Deterministic guards enforce audience scope
  and preserve exact-ID fast paths. The LLM resolves list, catalogue, and
  contextual follow-up requests; authorized facts are fetched from NestJS and
  combined with approved RAG evidence when needed. The answer LLM receives
  only allowlisted live fields and generates the user-facing response.
- **Conversation context and limits:** Clients may send up to 10 recent
  `conversation_history` turns (`user` or `assistant`, at most 2,000
  characters each). Explicit list limits are applied before answer generation;
  over-limit generated lists fall back to the exact trimmed live results.
  Broad payment-history requests summarize up to 20 available records and
  offer a follow-up; other unspecified lists default to five records.
  Live-only requests skip Qdrant and knowledge-version checks.
- **Latency:** Obvious unfiltered catalogue-list requests use a deterministic
  fast path; destination/context-dependent requests still use semantic
  classification. Live API attempts use a 5-second timeout and at most one
  retry.
- **No unsupported claims:** The current package endpoint filters only by
  destination. It does not expose season, travel-date availability, budget,
  or group-size filters; the answer says these constraints could not be
  checked and does not claim matching availability.
- **Grounding:** RAG answers cite retrieved chunks. Live-only answers have no
  knowledge citations and identify their source through `live_data_sources`.
  Unsupported or out-of-scope questions receive a safe fallback with
  audience-specific app guidance.

### Phase 2 implementation and current status

Implemented: dense + BM25 retrieval fused with RRF; optional local
cross-encoder reranking; YAML policy lifecycle metadata and Qdrant eligibility
filters; `CURRENT` versus Qdrant alias authority checks; live API routing;
grounding/retrieval evaluation tools; and structured-output truncation
handling.

The RAG customer-support scope is integrated for the current release. The
platform-purpose knowledge update was published and indexed as version
`48fa84f6d107`; the service authority check matched `CURRENT` to the Qdrant
alias, retrieval returned the new overview chunk, and the platform-overview
question was smoke-tested successfully. The current implementation is covered
by the full test suite (769 passed, 44 deselected at the latest recorded run).
Treat these test results as a code baseline, not a substitute for deployment-
specific provider and NestJS contract checks.

The latest 15-case retrieval report measured hybrid Recall@5 **0.973**, MRR
**0.950**, nDCG **0.937**, and mean latency **2.14 s**. In the same run,
reranking scored lower (Recall@5 **0.940**, MRR **0.922**, nDCG **0.911**) and
was slower (**3.64 s**); therefore serving reranking remains disabled.

The current support/RAG feature phase is complete for the agreed scope. The
latest grounding report predates later structured-output, citation, and
support-routing changes: judge coverage was **46.7%** (8 judge errors),
citation precision **70.0%**, and citation recall **71.6%**. These figures are
historical and must not be presented as validation of the latest implementation.
For wider production rollout, rerun grounding and review per-case findings,
validate NestJS endpoints/auth/decryption in the target environment, and run
deployment/configuration/latency checks. The current evaluation reports are
`retrieval-reranked-phase2.json` and
`grounding-report-phase2.json`; detailed interpretation and commands are in
[`docs/hybrid_retrieval.md`](docs/hybrid_retrieval.md).

---

## Authentication

### Current target: NestJS gateway authentication

All three user-facing AI routes have one target security boundary: NestJS
validates the user JWT, derives user role/audience and allowed AI capabilities,
then calls this private AI service. NestJS supplies the original Bearer token
plus a signed, short-lived principal assertion. The AI service checks the
assertion before it starts Chat Safety, Enquiry, KB retrieval, or a live-data
request.

The AI service keeps the user JWT opaque: it never decodes, logs, or persists
it. When a question needs personal live data (for example, a coins balance),
the service forwards the original JWT to the appropriate NestJS API; NestJS
then authorizes that resource read again.

```ini
# Production target, after NestJS implements the signed-header contract
AI_AUTH_MODE=gateway
AI_GATEWAY_AUTH_SECRET=<same-long-random-secret-in-NestJS-and-AI>
```

The gateway assertion contains a verified user ID, `customer` or `vendor`
audience, allowed scopes, and a short expiry. Required scopes are
`ai:chat-safety`, `ai:enquiry:understand`, and `ai:support:ask`. For Support,
the client-provided `audience` cannot differ from this verified audience.

See [the NestJS-to-AI gateway contract](docs/auth_gateway_contract.md) for
the exact headers, HMAC signing process, and deployment protections. The AI
service must not be publicly reachable; frontend requests go through NestJS.

### Legacy migration mode (temporary)

Until NestJS sends the signed assertion, use:

```ini
AI_AUTH_MODE=legacy_service_token
GUARDRAIL_API_TOKEN=<existing-service-token>
AI_GATEWAY_AUTH_SECRET=<new-shared-gateway-secret>
```

In legacy mode, Chat Safety and Enquiry use `GUARDRAIL_API_TOKEN`. Support
accepts a non-empty Bearer token and forwards it for live reads, but its
KB-only path is not truly JWT-verified. Do not retain this mode for production
after NestJS gateway integration is available.

### Legacy behavior retained during migration

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

### 2. Outbound user JWT forwarding (AI Support)

`/api/v1/support/ask` requires a bearer JWT. Its support service passes the
token explicitly to live-data fetchers, which forward it unchanged to
authorized NestJS endpoints. NestJS authorizes against the user's real
identity. Public catalogue calls may omit the token.

The AI service treats the token as **opaque**:

- Never decoded, validated, or logged.
- Never persisted (not in Redis, not in state, not on disk).
- Forwarded unchanged on every outbound call within the same request.

### 3. The boundary — do not forward user JWTs to service-to-service routes

`auth_context_middleware` only captures the user JWT for routes under
`USER_JWT_ROUTE_PREFIXES`. `/api/v1/validate-message` and
`/api/v1/enquiry/understand` are excluded and explicitly clear the
context. The support route currently reads and validates its bearer token
in the route handler and passes it explicitly to the support service; it
does not rely on the request auth context.

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

Gateway mode also requires `AI_GATEWAY_AUTH_SECRET`, shared only between the
NestJS and AI-service environments. `AI_AUTH_MODE` defaults to `gateway`;
set it to `legacy_service_token` only during the NestJS migration described
above.

**Everything else has a development-safe default.**

Key settings:

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `openai/gpt-oss-120b` | Groq model |
| `GROQ_CHAT_URL` | `https://api.groq.com/openai/v1/chat/completions` | Groq endpoint |
| `LLM_TIMEOUT` | `60` | Seconds |
| `LLM_MAX_RETRIES` | `2` | |
| `LLM_MAX_TOKENS` | `512` | Moderation JSON is small, but reasoning models spend 80–150 tokens on internal reasoning before emitting JSON. 128 caused `json_validate_failed`; 512 leaves headroom. |
| `GROQ_MAX_CONCURRENT_REQUESTS` | `1` | Maximum in-flight Groq requests per process |
| `GROQ_TPM_LIMIT` | unset | Optional TPM override; otherwise learned from Groq response headers or rate-limit errors |
| `GROQ_TPM_TARGET_UTILIZATION` | `0.75` | Fraction of the detected/configured TPM limit to reserve |
| `GROQ_TPM_WINDOW_SECONDS` | `60` | Rolling token-budget window |
| `LLM_TEMPERATURE` | `0.0` | |
| `LLM_CONSENSUS_ATTEMPTS` | `1` | Single call per message by default. |
| `LLM_FALLBACK_ENABLED` | `true` | Enables Gemini fallback when a key is present |
| `GEMINI_API_KEY` | `""` | Empty disables the fallback |
| `GEMINI_MODEL` | `gemini-2.0-flash` | |
| `GEMINI_TIMEOUT_SECONDS` | `15` | Wall-clock cap for the fallback |
| `LLM_CIRCUIT_FAILURE_THRESHOLD` | `5` | |
| `LLM_CIRCUIT_OPEN_SECONDS` | `60` | |
| `SUPPORT_MAX_CONTEXT_CHUNKS` | `5` | Maximum retrieved chunks passed to RAG answer generation. |
| `SUPPORT_MAX_CHUNK_CHARS` | `1200` | Per-chunk context truncation limit. |
| `SUPPORT_ANSWER_MAX_TOKENS` | `1536` | Completion budget for structured support answers; truncated structured output retries with a larger budget. |
| `RAG_RERANKER_ENABLED` | `false` | Cross-encoder reranking is optional and currently disabled based on the latest benchmark. |
| `RAG_RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Local reranker model used when enabled. |
| `LOG_FORMAT` | `json` | `json` or `text` |
| `REDIS_URL` | `redis://localhost:6379/0` | Empty disables Redis |

Groq TPM pacing does not assume an 8,000-token quota. It learns the model's
reported limit and leaves the configured utilization headroom. Redis shares
reservations between workers; without Redis, pacing is process-local. Set
`GROQ_TPM_LIMIT` only when you need to override the provider-reported quota.

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

# Compare dense, BM25, and hybrid retrieval against the labeled query set.
python scripts/evaluate_retrieval.py --k 5

# Compare hybrid RRF with the local FastEmbed/ONNX cross-encoder. Model
# weights are downloaded on first use.
python scripts/evaluate_retrieval.py --k 5 --include-reranked

# Test different hybrid fusion settings.
python scripts/evaluate_retrieval.py --k 5 --rrf-rank-constant 20 --candidate-multiplier 2

# Generate support answers, score evidence/citations, and optionally use the
# configured LLM provider to judge semantic support.
python scripts/evaluate_grounding.py
python scripts/evaluate_grounding.py --judge --output grounding-report.json
```

Retrieval evaluation requires the current knowledge index to be published
and its Qdrant alias to be authoritative. The initial labeled set lives at
[`app/evaluation/datasets/retrieval_v1.json`](app/evaluation/datasets/retrieval_v1.json);
review and expand its relevance labels as the knowledge base evolves.
Each query may label multiple relevant chunks when the answer spans
several passages. The evaluator reports Precision@k, Recall@k, MRR@k,
nDCG@k, and mean retrieval latency.
BM25 indexes are cached by published index version, audience, area, and UTC
date; each new filter scope incurs one initial corpus load, then reuses the
in-memory index. Retrieval pins dense and BM25 searches to the verified
published version. Use `--rrf-rank-constant` and `--candidate-multiplier`
to evaluate settings without changing production defaults (60 and 4). Add
`--include-reranked` to evaluate the same fused candidate pool with the local
cross-encoder and report quality/latency deltas against hybrid. Reranking
remains disabled in serving by default until its benchmark impact and
operational cost are reviewed; enable it with `RAG_RERANKER_ENABLED=true`.

Grounding evaluation generates answers through the support service and
compares retrieved/cited chunk IDs with the labeled evidence IDs in the
retrieval dataset. `--judge` additionally sends the question, answer, and
retrieved chunks to the configured LLM provider for semantic answer/citation
support scores; use only with approved evaluation data and expect provider
cost and latency. Judge coverage and per-case errors are reported; a failed
judge case does not prevent later cases from being evaluated. Deterministic
metrics are preserved for all cases.

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
│   ├── enquiry/                Enquiry Understanding
│   └── support/                RAG support, intent routing, live data
├── observability/              Counters, circuit breaker, request middleware
├── api/routes/                 health, metrics, chat_safety, enquiry, support
├── swabi/                      NestJS transport helpers and response crypto
├── rag/                        Ingestion, policy lifecycle, Qdrant, hybrid retrieval/reranking
├── graph/                      LangGraph foundation
└── evaluation/                 Datasets, retrieval + grounding evaluation

tests/{unit,integration,contract}/
scripts/                         Ingestion and evaluation CLIs

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
- `POST /api/v1/support/ask` — authenticated RAG support with authorized
  live NestJS API reads.
- Shared LLM gateway with retries, timeouts, structured output.
- Swabi HTTP client (no direct DB access).
- Shared service-to-service authentication.
- Request ID correlation, structured logging, safe logging helpers.
- Redis connection manager.
- RAG ingestion and hybrid retrieval over the published knowledge base
  (BM25 keyword search + dense vector search with Reciprocal Rank Fusion).
- AI Support with audience-aware answers, YAML policy lifecycle metadata,
  Qdrant policy filters, local `CURRENT`/alias authority checks, and live API
  integration.
- Retrieval and grounding evaluation tools; optional cross-encoder reranking
  (disabled by default based on the latest 15-case benchmark).
- Per-provider circuit breakers and fail-open on outage (Chat Safety).
- Connection pooling for Groq calls.
- Unit / integration / contract tests.

### Explicitly NOT included

- Bid analyzer, semantic matching.
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

The RAG customer-support feature phase is complete for the current agreed
scope. The knowledge update for the Swabi overview was published, indexed,
and smoke-tested against the active development index. This is distinct from
production rollout acceptance; deployment-specific checks still include:

- [ ] Rerun grounding against the latest answer-generation and citation code;
  review semantic judge coverage and per-case findings. The checked-in report
  predates those changes.
- [ ] Verify real NestJS endpoint response shapes, JWT forwarding, decryption,
  empty results, and error handling for customer, vendor, and public intents.
- [ ] Exercise stale `CURRENT`/Qdrant-alias mismatch rejection and confirm
  recovery after a policy update in the target deployment.
- [ ] Run deployment-specific latency, configuration/secrets, logging, and
  service-behavior checks.

The last recorded full test run passed 769 tests with 44 deselected. It is a
code regression baseline, not a substitute for the external-service checks
above. For a knowledge release, run `python -m scripts.rag_ingest --release`,
then `python -m scripts.rag_index --version <version>` with the version
reported by ingestion. Confirm `data/rag_index/CURRENT` matches the Qdrant
alias before serving RAG requests; the service fails closed on a mismatch.

The current package API accepts only `page`, `limit`, and `destination`.
Season, travel-date availability, budget, and traveler-count matching remain
unavailable until NestJS exposes those filters and corresponding data.

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
- [`docs/hybrid_retrieval.md`](docs/hybrid_retrieval.md) — how hybrid RAG
  retrieval works, how it is evaluated, and how to run tuning experiments.
