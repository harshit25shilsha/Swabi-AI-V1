import pytest

from app.rag.interface import MockRetriever


@pytest.mark.asyncio
async def test_mock_retriever_returns_documents():
    retriever = MockRetriever()
    docs = await retriever.search("anything", top_k=3)
    assert len(docs) >= 1
    assert docs[0].content


@pytest.mark.asyncio
async def test_mock_retriever_respects_top_k():
    retriever = MockRetriever()
    docs = await retriever.search("q", top_k=0)
    assert docs == []
