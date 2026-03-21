"""
tests/test_state_manager.py
Tests for utils/state_manager.py — read/write .sh state files with run_dir.
"""

import os
import pytest
from pathlib import Path

from utils.state_manager import (
    write_agent_state,
    read_agent_state,
    get_status,
    get_output,
    clear_agent_state,
    clear_all_states,
    all_statuses,
)


@pytest.fixture
def run_dir(tmp_path):
    """Create a temporary run directory."""
    d = tmp_path / "run_test"
    d.mkdir()
    return str(d)


class TestWriteAndRead:
    def test_write_then_read(self, run_dir):
        write_agent_state("test_agent", "complete", output_full="hello world",
                          output_summary="hello", run_dir=run_dir)
        state = read_agent_state("test_agent", run_dir)

        assert state["AGENT"] == "test_agent"
        assert state["STATUS"] == "complete"
        assert state["OUTPUT_FULL"] == "hello world"
        assert state["OUTPUT_SUMMARY"] == "hello"
        assert state["ITERATION"] == "0"

    def test_single_quotes_escaped(self, run_dir):
        write_agent_state("q_agent", "complete",
                          output_full="it's a test with 'quotes'",
                          run_dir=run_dir)
        state = read_agent_state("q_agent", run_dir)
        assert state["OUTPUT_FULL"] == "it's a test with 'quotes'"

    def test_multiline_summary_collapsed(self, run_dir):
        write_agent_state("ml_agent", "complete",
                          output_summary="line1\nline2\nline3",
                          run_dir=run_dir)
        state = read_agent_state("ml_agent", run_dir)
        assert "\n" not in state["OUTPUT_SUMMARY"]

    def test_sh_file_is_created(self, run_dir):
        write_agent_state("file_agent", "running", run_dir=run_dir)
        assert (Path(run_dir) / "file_agent" / "state.sh").exists()

    def test_read_nonexistent_returns_empty(self, run_dir):
        assert read_agent_state("missing", run_dir) == {}


class TestHelpers:
    def test_get_status(self, run_dir):
        write_agent_state("s_agent", "running", run_dir=run_dir)
        assert get_status("s_agent", run_dir) == "running"

    def test_get_status_default(self, run_dir):
        assert get_status("nope", run_dir) == "waiting"

    def test_get_output(self, run_dir):
        write_agent_state("o_agent", "complete", output_full="data here", run_dir=run_dir)
        assert get_output("o_agent", run_dir) == "data here"

    def test_get_output_missing(self, run_dir):
        assert get_output("nope", run_dir) == ""


class TestClear:
    def test_clear_agent_state(self, run_dir):
        write_agent_state("del_agent", "complete", run_dir=run_dir)
        clear_agent_state("del_agent", run_dir)
        assert not (Path(run_dir) / "del_agent" / "state.sh").exists()

    def test_clear_all_states(self, run_dir):
        write_agent_state("a", "complete", run_dir=run_dir)
        write_agent_state("b", "complete", run_dir=run_dir)
        clear_all_states(run_dir)
        assert list(Path(run_dir).glob("**/state.sh")) == []

    def test_clear_preserves_frames_dir(self, run_dir):
        frames = Path(run_dir) / "frames"
        frames.mkdir(exist_ok=True)
        (frames / "frame.jpg").touch()
        write_agent_state("x", "complete", run_dir=run_dir)
        clear_all_states(run_dir)
        assert frames.exists()
        assert (frames / "frame.jpg").exists()


class TestAllStatuses:
    def test_returns_dict_for_all_agents(self, run_dir):
        statuses = all_statuses(run_dir)
        from graph.state import ALL_AGENTS
        assert set(statuses.keys()) == set(ALL_AGENTS)
        # All should be 'waiting' since nothing written
        assert all(v == "waiting" for v in statuses.values())
