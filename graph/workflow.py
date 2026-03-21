"""
graph/workflow.py
Defines and compiles the LangGraph StateGraph for the video workflow pipeline.

Flow:
    START → transcribe → extract_frames → requirements →
    research → alignment → (confident?) → synthesis → END
                               ↑_____________________|
                               (not confident & iterations < MAX)
"""

import os
import logging
from langgraph.graph import StateGraph, START, END

from graph.state import WorkflowState, MAX_ALIGNMENT_ITERATIONS
from graph.nodes import (
    transcribe_node,
    extract_frames_node,
    requirements_node,
    research_node,
    alignment_node,
    synthesis_node,
)

logger = logging.getLogger(__name__)

_MAX_ITER = int(os.getenv("MAX_ALIGNMENT_ITERATIONS", MAX_ALIGNMENT_ITERATIONS))


# ── Conditional routing ────────────────────────────────────────────────────────

def should_loop_or_synthesize(state: WorkflowState) -> str:
    """
    After alignment: route to 'research' if not confident and under iteration limit.
    Otherwise route to 'synthesis'.
    """
    is_confident    = state.get("alignment_confident", False)
    iteration_count = state.get("iteration_count", 0)

    if is_confident:
        logger.info("Alignment confident — proceeding to synthesis")
        return "synthesis"

    if iteration_count >= _MAX_ITER:
        logger.warning(
            "Max alignment iterations (%d) reached — forcing synthesis", _MAX_ITER
        )
        return "synthesis"

    logger.info(
        "Alignment not confident (iteration %d/%d) — looping back to research",
        iteration_count,
        _MAX_ITER,
    )
    return "research"


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph StateGraph."""
    builder = StateGraph(WorkflowState)

    # Add nodes
    builder.add_node("transcribe",     transcribe_node)
    builder.add_node("extract_frames", extract_frames_node)
    builder.add_node("requirements",   requirements_node)
    builder.add_node("research",       research_node)
    builder.add_node("alignment",      alignment_node)
    builder.add_node("synthesis",      synthesis_node)

    # Linear edges
    builder.add_edge(START,            "transcribe")
    builder.add_edge("transcribe",     "extract_frames")
    builder.add_edge("extract_frames", "requirements")
    builder.add_edge("requirements",   "research")
    builder.add_edge("research",       "alignment")

    # Conditional edge: alignment → synthesis OR research
    builder.add_conditional_edges(
        "alignment",
        should_loop_or_synthesize,
        {
            "synthesis": "synthesis",
            "research":  "research",
        },
    )

    builder.add_edge("synthesis", END)

    graph = builder.compile()
    logger.info("LangGraph compiled successfully")
    return graph


# ── Module-level compiled graph (import this in app.py) ───────────────────────
workflow = build_graph()
