"""
tests/test_frame_extractor.py
Tests for agents/frame_extractor.py — downscale logic and keyframe extraction.
"""

import numpy as np
import pytest

from agents.frame_extractor import _maybe_downscale, _is_mouse_only_change


class TestMaybeDownscale:
    def test_no_downscale_when_max_width_zero(self, monkeypatch):
        monkeypatch.setattr("agents.frame_extractor._MAX_WIDTH", 0)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = _maybe_downscale(frame)
        assert result.shape == (1080, 1920, 3)

    def test_no_downscale_when_within_limit(self, monkeypatch):
        monkeypatch.setattr("agents.frame_extractor._MAX_WIDTH", 480)
        frame = np.zeros((320, 400, 3), dtype=np.uint8)
        result = _maybe_downscale(frame)
        assert result.shape == (320, 400, 3)

    def test_downscale_preserves_aspect_ratio(self, monkeypatch):
        monkeypatch.setattr("agents.frame_extractor._MAX_WIDTH", 480)
        # 1920x1080 → 480x270
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = _maybe_downscale(frame)
        assert result.shape[1] == 480  # width
        assert result.shape[0] == 270  # height (1080 * 480/1920)

    def test_downscale_exact_boundary(self, monkeypatch):
        monkeypatch.setattr("agents.frame_extractor._MAX_WIDTH", 480)
        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        result = _maybe_downscale(frame)
        assert result.shape == (480, 480, 3)  # exactly at limit, no resize


class TestMouseOnlyChange:
    def test_all_zeros_is_mouse_only(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert _is_mouse_only_change(mask) is True

    def test_small_region_is_mouse(self):
        mask = np.zeros((500, 500), dtype=np.uint8)
        mask[10:50, 10:50] = 255  # 40x40 region
        assert _is_mouse_only_change(mask) is True

    def test_large_region_is_not_mouse(self):
        mask = np.zeros((500, 500), dtype=np.uint8)
        mask[10:300, 10:300] = 255  # 290x290 region
        assert _is_mouse_only_change(mask) is False
