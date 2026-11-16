# NestJS gateway to Swabi AI authentication contract

## Request flow

The browser sends its normal user JWT only to NestJS. NestJS validates the
JWT, authorizes the requested AI capability, then proxies the request over the
private NestJS-to-AI connection. The AI service does not decode, log, or store
the user JWT.

NestJS must forward the original user JWT in `Authorization: Bearer <token>`.
The AI service forwards that unchanged token only for user-specific live API
reads. NestJS remains authoritative when those live endpoints authorize the
delegated call.

For every protected AI request NestJS also sends:

| Header | Value |
|---|---|
| `X-Swabi-Principal` | Base64url-encoded UTF-8 JSON, without padding |
| `X-Swabi-Principal-Signature` | Lowercase hex HMAC-SHA-256 of the exact principal-header value, using `AI_GATEWAY_AUTH_SECRET` |

The principal JSON has exactly the authorization data that AI needs:

```json
{
  "sub": "authenticated-user-id",
  "audience": "customer",
  "scopes": ["ai:support:ask"],
  "exp": 1790000000
}
```

`audience` is `customer` or `vendor`; it is derived by NestJS from verified
identity/roles, never from the AI request body. `exp` is a Unix timestamp and
must be short lived. Required capability scopes are:

- `ai:chat-safety` for `POST /api/v1/validate-message`
- `ai:enquiry:understand` for `POST /api/v1/enquiry/understand`
- `ai:support:ask` for `POST /api/v1/support/ask`

## Security requirements

- The AI service must be private; clients must not reach it directly.
- Terminate public traffic at NestJS and strip both `X-Swabi-Principal*`
  headers from untrusted requests.
- Use mTLS (or equivalent workload authentication) in addition to the HMAC.
- Rotate `AI_GATEWAY_AUTH_SECRET` in NestJS and AI together. Do not reuse it
  as a user, API, or database secret.
- Do not substitute `SWABI_API_KEY` or `GUARDRAIL_API_TOKEN` for a user JWT on
  user-specific live calls.
- `AI_AUTH_MODE=legacy_service_token` is a temporary migration mode only;
  production target state is `AI_AUTH_MODE=gateway`.

## Expected failures

Missing/malformed Bearer token, missing assertion, invalid signature, expired
assertion, or malformed principal returns `401` with `WWW-Authenticate:
Bearer`. A valid principal lacking the route scope returns `403`. For Support,
a body `audience` that differs from the verified principal returns `403`.
