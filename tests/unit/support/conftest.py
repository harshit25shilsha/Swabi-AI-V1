"""Service tests use an offline intent plan; planner behavior has its own tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def use_offline_service_intent_resolver(monkeypatch):
    from app.capabilities.support import service
    from app.capabilities.support.greeting import GreetingClassification
    from app.capabilities.support.intent_resolver import ResolvedIntent
    from app.capabilities.support.live_data_check import (
        is_general_process_question,
        is_live_data_question,
    )
    from app.capabilities.support.live_data_classifier import classify

    class OfflineIntentResolver:
        def resolve(self, question, audience, conversation_history=None):
            classified = classify(question, audience)
            use_rag = is_general_process_question(question)
            if use_rag and not is_live_data_question(question):
                classified.intent = type(classified.intent).NONE
            return ResolvedIntent(
                intent=classified.intent,
                entities=classified.entities,
                source="deterministic",
                reason=classified.reason,
                use_rag=use_rag,
                knowledge_query=question if use_rag else None,
            )

    class OfflineGreetingClassifier:
        def classify(self, _message):
            return GreetingClassification(greeting_only=False, confidence=1.0)

    monkeypatch.setattr(service, "_get_resolver", OfflineIntentResolver)
    monkeypatch.setattr(
        service,
        "_get_greeting_classifier",
        OfflineGreetingClassifier,
    )
