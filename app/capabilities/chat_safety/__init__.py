"""Chat Safety capability.

Detects off-platform communication attempts in marketplace chat:
phone numbers, emails, URLs, payment details, and requests to move the
conversation or transaction outside Swabi.

Public surface (for step 10 wiring):
    service.moderate(message) -> ModerationResponse
    dependencies.verify_token  (FastAPI dependency)
    schemas.ModerationRequest, ModerationResponse
    prompts.SYSTEM_PROMPT, PROMPT_VERSION
"""
