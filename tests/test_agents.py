"""
tests/test_agents.py
Tests for the LLM-backed agents — requirements, alignment, synthesis, research.
All LLM calls are mocked; we validate wiring, state writes, and parsing logic.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestRequirementsAgent:
    @patch("agents.requirements_agent.chat")
    def test_returns_requirements_text(self, mock_chat, tmp_path):
        mock_chat.return_value = "## Workflow Overview\nSome requirements output"
        run_dir = str(tmp_path / "run")

        from agents.requirements_agent import run_requirements_agent
        result = run_requirements_agent(
            transcript="User shows their CRM process",
            frame_chunk_summaries=["**Frames 1–25:** User navigates CRM dashboard and opens a contact form."],
            provider="lm_studio",
            run_dir=run_dir,
        )

        assert "Workflow Overview" in result
        mock_chat.assert_called_once()

    @patch("agents.requirements_agent.chat")
    def test_writes_state_file(self, mock_chat, tmp_path):
        mock_chat.return_value = "output"
        run_dir = str(tmp_path / "run")

        from agents.requirements_agent import run_requirements_agent
        run_requirements_agent(
            transcript="test", frame_chunk_summaries=[],
            provider="lm_studio", run_dir=run_dir,
        )

        from utils.state_manager import get_status
        assert get_status("requirements", run_dir) == "complete"


class TestAlignmentAgent:
    @patch("agents.alignment_agent.chat")
    def test_confident_verdict_parsed(self, mock_chat, tmp_path):
        mock_chat.return_value = (
            "## Alignment Verdict\nVERDICT: confident\n"
            "## Confidence Score\nScore: 9/10\n## Summary\nAll good."
        )
        run_dir = str(tmp_path / "run")

        from agents.alignment_agent import run_alignment_agent
        verdict_text, is_confident, notes = run_alignment_agent(
            requirements="req", research="res",
            provider="lm_studio", run_dir=run_dir,
        )

        assert is_confident is True
        assert notes == ""
        assert "confident" in verdict_text.lower()

    @patch("agents.alignment_agent.chat")
    def test_not_confident_extracts_feedback(self, mock_chat, tmp_path):
        mock_chat.return_value = (
            "## Alignment Verdict\nVERDICT: not_confident\n"
            "## Confidence Score\nScore: 4/10\n"
            "## Feedback for Researcher\n- Need more SaaS options\n- Check pricing\n"
            "## Summary\nNot enough."
        )
        run_dir = str(tmp_path / "run")

        from agents.alignment_agent import run_alignment_agent
        _, is_confident, notes = run_alignment_agent(
            requirements="req", research="res",
            provider="lm_studio", run_dir=run_dir,
        )

        assert is_confident is False
        assert "SaaS" in notes
        assert "pricing" in notes


class TestSynthesisAgent:
    @patch("agents.synthesis_agent.chat")
    def test_returns_synthesis_text(self, mock_chat, tmp_path):
        mock_chat.return_value = "# Workflow Improvement Report\n\n## Executive Summary\nGreat."
        run_dir = str(tmp_path / "run")

        from agents.synthesis_agent import run_synthesis_agent
        result = run_synthesis_agent(
            requirements="req", research="res",
            alignment_verdict="VERDICT: confident",
            transcript="video transcript",
            provider="lm_studio", run_dir=run_dir,
        )

        assert "Workflow Improvement Report" in result


class TestResearchAgent:
    @patch("agents.research_agent._agentic_search_loop")
    def test_returns_research_and_sources(self, mock_loop, tmp_path):
        mock_loop.return_value = ("## Research Summary\nFindings here", ["https://example.com"])
        run_dir = str(tmp_path / "run")

        from agents.research_agent import run_research_agent
        text, sources = run_research_agent(
            requirements="req",
            provider="lm_studio",
            run_dir=run_dir,
        )

        assert "Research Summary" in text
        assert len(sources) == 1

    @patch("agents.research_agent._agentic_search_loop")
    def test_includes_alignment_notes_on_iteration(self, mock_loop, tmp_path):
        mock_loop.return_value = ("research", [])
        run_dir = str(tmp_path / "run")

        from agents.research_agent import run_research_agent
        run_research_agent(
            requirements="req",
            alignment_notes="Need more SaaS options",
            iteration=1,
            provider="lm_studio",
            run_dir=run_dir,
        )

        # Verify the user message sent to the loop included alignment notes
        call_args = mock_loop.call_args
        messages = call_args[0][0]
        assert "SaaS" in messages[0]["content"]
