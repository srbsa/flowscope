"""
tests/test_nodes.py
Tests for graph/nodes.py — node functions with mocked agents.
"""

import pytest
from unittest.mock import patch


class TestDescribeFrames:
    @patch("utils.llm_client.get_client")
    @patch("utils.llm_client.get_model", return_value="test-model")
    def test_chains_previous_description(self, mock_model, mock_client, tmp_path):
        # Create fake frame files
        frame1 = tmp_path / "f1.jpg"
        frame2 = tmp_path / "f2.jpg"
        frame1.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG-like
        frame2.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        # Mock the OpenAI client
        mock_obj = mock_client.return_value
        resp1 = type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "Dashboard with charts"})()})]})()
        resp2 = type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "Form opened on top of dashboard"})()})]})()
        mock_obj.chat.completions.create.side_effect = [resp1, resp2]

        from graph.nodes import _describe_frames
        descriptions = _describe_frames([str(frame1), str(frame2)], "lm_studio")

        assert len(descriptions) == 2
        assert descriptions[0] == "Dashboard with charts"
        assert descriptions[1] == "Form opened on top of dashboard"

        # Check second call included previous description in prompt
        calls = mock_obj.chat.completions.create.call_args_list
        second_call_msgs = calls[1].kwargs["messages"]
        user_content = second_call_msgs[-1]["content"]
        # The text prompt should reference previous frame
        found_chain = any(
            isinstance(part, dict) and "Previous frame" in part.get("text", "")
            for part in user_content
        ) if isinstance(user_content, list) else "Previous frame" in str(user_content)
        assert found_chain

    @patch("utils.llm_client.get_client")
    @patch("utils.llm_client.get_model", return_value="test-model")
    def test_empty_paths_returns_empty(self, mock_model, mock_client):
        from graph.nodes import _describe_frames
        assert _describe_frames([], "lm_studio") == []


class TestTranscribeNode:
    @patch("agents.transcriber.transcribe_video", return_value="Hello world transcript")
    def test_returns_transcript(self, mock_transcribe):
        from graph.nodes import transcribe_node
        result = transcribe_node({
            "video_path": "/tmp/test.mp4",
            "provider": "lm_studio",
            "run_dir": "/tmp/run",
        })
        assert result["transcript"] == "Hello world transcript"
        assert result["current_step"] == "frame_extractor"


class TestSynthesisNode:
    @patch("agents.synthesis_agent.run_synthesis_agent", return_value="Final report")
    def test_returns_synthesis(self, mock_synth):
        from graph.nodes import synthesis_node
        result = synthesis_node({
            "requirements": "req",
            "research": "res",
            "alignment_verdict": "confident",
            "transcript": "text",
            "iteration_count": 1,
            "provider": "lm_studio",
            "run_dir": "/tmp/run",
        })
        assert result["synthesis"] == "Final report"
        assert result["current_step"] == "complete"
