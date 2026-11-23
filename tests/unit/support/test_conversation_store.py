from unittest.mock import patch

import pytest

from app.capabilities.support.conversation_store import (
    SupportConversationStore,
)


def test_conversation_key_is_scoped_by_user():
    key_a = SupportConversationStore._make_key(
        user_id="user-1",
        audience="customer",
        conversation_id="conversation-1",
    )
    key_b = SupportConversationStore._make_key(
        user_id="user-2",
        audience="customer",
        conversation_id="conversation-1",
    )

    assert key_a != key_b


def test_conversation_key_is_scoped_by_audience():
    customer_key = SupportConversationStore._make_key(
        user_id="user-1",
        audience="customer",
        conversation_id="conversation-1",
    )
    vendor_key = SupportConversationStore._make_key(
        user_id="user-1",
        audience="vendor",
        conversation_id="conversation-1",
    )

    assert customer_key != vendor_key


def test_raw_user_id_is_not_in_redis_key():
    key = SupportConversationStore._make_key(
        user_id="private-user-id",
        audience="customer",
        conversation_id="conversation-1",
    )

    assert "private-user-id" not in key


@pytest.mark.asyncio
async def test_load_history_returns_empty_when_redis_unavailable():
    store = SupportConversationStore()

    with patch("app.capabilities.support.conversation_store.redis_manager") as redis:
        redis.client = None

        history = await store.load_history(
            user_id="user-1",
            audience="customer",
            conversation_id="conversation-1",
        )

    assert history == []


@pytest.mark.asyncio
async def test_append_returns_false_when_redis_unavailable():
    store = SupportConversationStore()

    with patch("app.capabilities.support.conversation_store.redis_manager") as redis:
        redis.client = None

        saved = await store.append_exchange(
            user_id="user-1",
            audience="customer",
            conversation_id="conversation-1",
            user_message="Show my payments.",
            assistant_message="Here are your payments.",
        )

    assert saved is False
