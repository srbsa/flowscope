"""
tests/test_workflow.py
Tests for graph/workflow.py — graph construction and conditional routing.
"""

import pytest

from graph.workflow import should_loop_or_synthesize, build_graph


class TestConditionalRouting:
    def test_confident_routes_to_synthesis(self):
        state = {"alignment_confident": True, "iteration_count": 1}
        assert should_loop_or_synthesize(state) == "synthesis"

    def test_not_confident_routes_to_research(self):
        state = {"alignment_confident": False, "iteration_count": 1}
        assert should_loop_or_synthesize(state) == "research"

    def test_max_iterations_forces_synthesis(self):
        state = {"alignment_confident": False, "iteration_count": 99}
        assert should_loop_or_synthesize(state) == "synthesis"

    def test_defaults_when_keys_missing(self):
        # alignment_confident defaults to False, iteration_count to 0
        state = {}
        assert should_loop_or_synthesize(state) == "research"


class TestGraphBuild:
    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None
