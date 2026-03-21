"""
tests/test_state.py
Tests for graph/state.py — WorkflowState, initial_state, constants.
"""

import os
import shutil
import pytest
from pathlib import Path

from graph.state import (
    initial_state,
    WorkflowState,
    ALL_AGENTS,
    PROVIDER_LM_STUDIO,
    PROVIDER_OPENAI,
    STATE_OUTPUTS_DIR,
    AGENT_TRANSCRIBER,
)


class TestInitialState:
    def test_creates_state_with_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEFAULT_PROVIDER", "lm_studio")
        state = initial_state("/tmp/video.mp4")

        assert state["video_path"] == "/tmp/video.mp4"
        assert state["provider"] == PROVIDER_LM_STUDIO
        assert state["run_id"].startswith("run_")
        assert state["run_dir"].startswith(STATE_OUTPUTS_DIR)
        assert state["transcript"] == ""
        assert state["frame_paths"] == []
        assert state["iteration_count"] == 0
        assert state["current_step"] == AGENT_TRANSCRIBER
        assert state["error"] is None

        # Clean up the created run dir
        if os.path.exists(state["run_dir"]):
            shutil.rmtree(state["run_dir"])

    def test_explicit_provider(self, tmp_path):
        state = initial_state("/tmp/v.mp4", provider=PROVIDER_OPENAI)
        assert state["provider"] == PROVIDER_OPENAI

        if os.path.exists(state["run_dir"]):
            shutil.rmtree(state["run_dir"])

    def test_run_dir_created(self):
        state = initial_state("/tmp/v.mp4")
        assert os.path.isdir(state["run_dir"])
        assert os.path.isdir(os.path.join(state["run_dir"], "frames"))

        shutil.rmtree(state["run_dir"])

    def test_run_id_format(self):
        state = initial_state("/tmp/v.mp4")
        assert state["run_id"].startswith("run_")
        # run_YYYYMMDD_HHMMSS → 19 chars
        assert len(state["run_id"]) == 19

        shutil.rmtree(state["run_dir"])


class TestConstants:
    def test_all_agents_has_six(self):
        assert len(ALL_AGENTS) == 6

    def test_providers_are_strings(self):
        assert PROVIDER_LM_STUDIO == "lm_studio"
        assert PROVIDER_OPENAI == "openai"
