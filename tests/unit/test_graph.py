import pytest

from app.graph.base import base_graph


@pytest.mark.asyncio
async def test_base_graph_runs():
    result = await base_graph.ainvoke(
        {
            "request_id": "r1",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {},
            "errors": [],
        }
    )
    assert result["final_response"] == "echo: hi"
    assert result["errors"] == []
