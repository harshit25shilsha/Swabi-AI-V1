"""Minimal LangGraph foundation used as a building block for later phases."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.core.logging import get_logger
from app.graph.state import GraphState

log = get_logger(__name__)


def _input_node(state: GraphState) -> GraphState:
    """Trivial input node — verifies orchestration wiring."""
    log.debug("graph.input_node request_id=%s", state.get("request_id"))
    return {"metadata": {**state.get("metadata", {}), "input_processed": True}}


def _validation_node(state: GraphState) -> GraphState:
    errors = list(state.get("errors") or [])
    if not state.get("messages"):
        errors.append("empty_messages")
    return {"errors": errors}


def _processing_node(state: GraphState) -> GraphState:
    """Placeholder processing step. Later phases replace this."""
    messages = state.get("messages") or []
    last = messages[-1]["content"] if messages else ""
    return {"final_response": f"echo: {last}"}


def _response_node(state: GraphState) -> GraphState:
    return {"metadata": {**state.get("metadata", {}), "response_ready": True}}


def build_base_graph():
    """Build and compile the minimal Phase 0 graph."""
    graph = StateGraph(GraphState)
    graph.add_node("input", _input_node)
    graph.add_node("validation", _validation_node)
    graph.add_node("processing", _processing_node)
    graph.add_node("response", _response_node)

    graph.add_edge(START, "input")
    graph.add_edge("input", "validation")
    graph.add_edge("validation", "processing")
    graph.add_edge("processing", "response")
    graph.add_edge("response", END)

    return graph.compile()


base_graph = build_base_graph()
