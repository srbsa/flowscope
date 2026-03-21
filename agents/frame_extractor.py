"""
agents/frame_extractor.py
Extracts semantically unique frames from a video using OpenCV frame differencing.

Strategy:
- Optionally downscale frames to VIDEO_MAX_WIDTH before processing
- Compare each frame against the previous kept frame
- If mean absolute pixel diff > FRAME_DIFF_THRESHOLD → unique frame
- MOUSE HEURISTIC: if the detected change region fits within a
  MOUSE_REGION_SIZE × MOUSE_REGION_SIZE bounding box, skip it (cursor movement)
- Saves keyframes as JPEG to <run_dir>/frames/
"""

import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from utils.state_manager import write_agent_state
from graph.state import (
    AGENT_FRAME_EXTRACTOR,
    FRAME_DIFF_THRESHOLD,
    MOUSE_REGION_SIZE,
    VIDEO_MAX_WIDTH,
    STATUS_RUNNING,
    STATUS_COMPLETE,
    STATUS_FAILED,
)

logger = logging.getLogger(__name__)

_THRESHOLD   = int(os.getenv("FRAME_DIFF_THRESHOLD", FRAME_DIFF_THRESHOLD))
_MOUSE_SIZE  = int(os.getenv("MOUSE_REGION_SIZE",    MOUSE_REGION_SIZE))
_MAX_WIDTH   = int(os.getenv("VIDEO_MAX_WIDTH",      VIDEO_MAX_WIDTH))
_SAMPLE_RATE = 2   # Process every Nth frame to reduce computation


def _is_mouse_only_change(diff_mask: np.ndarray) -> bool:
    """Return True if the changed region fits within a MOUSE_REGION_SIZE bounding box."""
    coords = cv2.findNonZero(diff_mask)
    if coords is None:
        return True
    x, y, w, h = cv2.boundingRect(coords)
    return w <= _MOUSE_SIZE and h <= _MOUSE_SIZE


def _maybe_downscale(frame: np.ndarray) -> np.ndarray:
    """Downscale a frame so its width ≤ VIDEO_MAX_WIDTH, preserving aspect ratio."""
    if _MAX_WIDTH <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= _MAX_WIDTH:
        return frame
    scale = _MAX_WIDTH / w
    new_w = _MAX_WIDTH
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_keyframes(
    video_path: str,
    run_dir: str | None = None,
) -> List[str]:
    """
    Extract unique keyframes from a video, ignoring mouse-only movement.

    Args:
        video_path: Path to the video file.
        run_dir:    Per-run output directory (frames saved to <run_dir>/frames/).

    Returns:
        List of absolute paths to saved JPEG keyframes.
    """
    write_agent_state(
        AGENT_FRAME_EXTRACTOR, STATUS_RUNNING,
        output_summary="Scanning frames for scene changes…",
        run_dir=run_dir,
    )

    frames_dir = Path(run_dir, "frames") if run_dir else Path("state_outputs/frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        prev_gray: np.ndarray | None = None
        keyframe_paths: List[str] = []
        frame_idx = 0

        logger.info(
            "Extracting keyframes from %s  (%.1f fps, %d frames, max_width=%s)",
            video_path, fps, total_frames, _MAX_WIDTH or "original",
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if frame_idx % _SAMPLE_RATE != 0:
                continue

            frame = _maybe_downscale(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                path = _save_frame(frame, frames_dir, frame_idx)
                keyframe_paths.append(path)
                prev_gray = gray
                continue

            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(diff.mean())

            if mean_diff < _THRESHOLD:
                continue

            _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

            if _is_mouse_only_change(mask):
                logger.debug("Frame %d: mouse-only change (mean diff=%.1f), skipping", frame_idx, mean_diff)
                continue

            path = _save_frame(frame, frames_dir, frame_idx)
            keyframe_paths.append(path)
            prev_gray = gray
            logger.debug("Keyframe saved: %s  (mean diff=%.1f)", path, mean_diff)

        cap.release()

        summary = f"{len(keyframe_paths)} unique frames extracted from {total_frames} total"
        write_agent_state(
            AGENT_FRAME_EXTRACTOR, STATUS_COMPLETE,
            output_full="\n".join(keyframe_paths),
            output_summary=summary,
            run_dir=run_dir,
        )

        logger.info("Frame extraction complete: %d keyframes", len(keyframe_paths))
        return keyframe_paths

    except Exception as exc:
        error_msg = f"Frame extraction failed: {exc}"
        logger.exception(error_msg)
        write_agent_state(
            AGENT_FRAME_EXTRACTOR, STATUS_FAILED,
            output_summary=error_msg, run_dir=run_dir,
        )
        raise RuntimeError(error_msg) from exc


def _save_frame(frame: np.ndarray, output_dir: Path, frame_idx: int) -> str:
    """Save a BGR frame as JPEG and return its absolute path."""
    filename = f"frame_{frame_idx:06d}.jpg"
    path = output_dir / filename
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(path.resolve())
