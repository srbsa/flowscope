"""
tests/conftest.py
Shared fixtures for the test suite.
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    """Set safe defaults for env vars used by the codebase."""
    monkeypatch.setenv("DEFAULT_PROVIDER", "lm_studio")
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LM_STUDIO_MODEL", "qwen/qwen3.5-9b")
    monkeypatch.setenv("LM_STUDIO_VISION_MODEL", "qwen/qwen3.5-9b")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "gpt-4o")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("WHISPER_MODEL", "base")
    monkeypatch.setenv("VIDEO_MAX_WIDTH", "480")
    monkeypatch.setenv("FRAME_DIFF_THRESHOLD", "30")
    monkeypatch.setenv("MOUSE_REGION_SIZE", "100")
    monkeypatch.setenv("MAX_ALIGNMENT_ITERATIONS", "3")
