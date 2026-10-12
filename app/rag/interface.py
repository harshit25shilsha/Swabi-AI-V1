"""Retriever protocol + a mock implementation for tests."""

from __future__ import annotations

from typing import Protocol

from app.rag.schemas import DocumentMetadata, RetrievalResult, RetrievedDocument


class Retriever(Protocol):
    async def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]: ...


class MockRetriever:
    """Deterministic stub used in tests and local development."""

    def __init__(self, documents: list[RetrievedDocument] | None = None) -> None:
        self._documents = documents or [
            RetrievedDocument(
                content="Swabi is a travel bidding marketplace.",
                score=0.9,
                metadata=DocumentMetadata(source="mock", title="About Swabi"),
            )
        ]

    async def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        return self._documents[:top_k]

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        docs = await self.search(query, top_k=top_k)
        return RetrievalResult(query=query, documents=docs)
