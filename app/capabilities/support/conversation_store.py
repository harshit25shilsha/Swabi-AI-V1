"""Redis-backed conversation history for Customer Support."""

from __future__ import annotations

import hashlib

from app.capabilities.support.schemas import ConversationTurn
from app.core.config import settings
from app.core.logging import get_logger
from app.core.redis import redis_manager

log = get_logger(__name__)


class SupportConversationStore:
    """Stores bounded support conversation history in Redis.

    Conversation ownership is enforced through a key scoped to the trusted
    authenticated user ID and audience. The raw user ID is never placed
    in the Redis key.
    """

    KEY_PREFIX = "swabi:support:conversation:v1"
    MAX_MESSAGE_LENGTH = 2000

    def __init__(
        self,
        *,
        ttl_seconds: int | None = None,
        max_turns: int | None = None,
    ) -> None:
        self.ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else settings.support_conversation_ttl_seconds
        )
        self.max_turns = (
            max_turns if max_turns is not None else settings.support_conversation_max_turns
        )

    @classmethod
    def _make_key(
        cls,
        *,
        user_id: str,
        audience: str,
        conversation_id: str,
    ) -> str:
        """Build a user- and audience-scoped conversation key."""
        user_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()

        return f"{cls.KEY_PREFIX}:{user_hash}:{audience}:{conversation_id}"

    async def load_history(
        self,
        *,
        user_id: str,
        audience: str,
        conversation_id: str,
    ) -> list[ConversationTurn]:
        """Load recent messages and refresh conversation expiry.

        Returns an empty list if Redis is unavailable or history cannot
        be loaded.
        """
        client = redis_manager.client

        if client is None:
            return []

        key = self._make_key(
            user_id=user_id,
            audience=audience,
            conversation_id=conversation_id,
        )

        try:
            # Read only the configured number of recent exchanges.
            raw_messages = await client.lrange(
                key,
                -self.max_turns * 2,
                -1,
            )

            if not raw_messages:
                return []

            history: list[ConversationTurn] = []

            for raw_message in raw_messages:
                try:
                    history.append(ConversationTurn.model_validate_json(raw_message))
                except (ValueError, TypeError):
                    log.warning(
                        "Invalid support conversation message skipped",
                        extra={"event": "support_conversation_invalid_message"},
                    )

            # Refresh inactivity expiry for an existing conversation.
            await client.expire(
                key,
                self.ttl_seconds,
            )

            return history

        except Exception:
            log.warning(
                "Support conversation history read failed",
                extra={"event": "support_conversation_read_failed"},
                exc_info=True,
            )
            return []

    async def append_exchange(
        self,
        *,
        user_id: str,
        audience: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> bool:
        """Append a completed user/assistant exchange atomically.

        Both messages are bounded to the ConversationTurn schema limit.

        Returns:
            True: Both messages were successfully persisted.
            False: Redis is unavailable or the write failed.
        """
        client = redis_manager.client

        if client is None:
            return False

        key = self._make_key(
            user_id=user_id,
            audience=audience,
            conversation_id=conversation_id,
        )

        try:
            # Keep both messages within the schema's content limit.
            user_message = user_message[: self.MAX_MESSAGE_LENGTH]
            assistant_message = assistant_message[: self.MAX_MESSAGE_LENGTH]

            user_turn = ConversationTurn(
                role="user",
                content=user_message,
            )

            assistant_turn = ConversationTurn(
                role="assistant",
                content=assistant_message,
            )

            # One Redis transaction appends the exchange, trims old
            # messages, and refreshes the conversation TTL.
            async with client.pipeline(transaction=True) as pipeline:
                pipeline.rpush(
                    key,
                    user_turn.model_dump_json(),
                    assistant_turn.model_dump_json(),
                )

                pipeline.ltrim(
                    key,
                    -self.max_turns * 2,
                    -1,
                )

                pipeline.expire(
                    key,
                    self.ttl_seconds,
                )

                await pipeline.execute()

            return True

        except Exception:
            log.warning(
                "Support conversation history write failed",
                extra={"event": "support_conversation_write_failed"},
                exc_info=True,
            )
            return False


support_conversation_store = SupportConversationStore()
