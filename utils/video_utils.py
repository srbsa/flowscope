"""
utils/video_utils.py
Helpers for inspecting video files before processing.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


def get_video_metadata(video_path: str) -> dict:
    """
    Return basic metadata for a video file.

    Returns:
        {
            "duration_seconds": float,
            "fps": float,
            "total_frames": int,
            "width": int,
            "height": int,
            "codec": str,
            "file_size_mb": float,
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int   = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec        = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
    duration     = total_frames / fps if fps > 0 else 0.0
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

    cap.release()

    return {
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "codec": codec.strip(),
        "file_size_mb": round(file_size_mb, 2),
    }


def validate_video(video_path: str) -> Tuple[bool, str]:
    """
    Check that the file exists and is a readable video.

    Returns:
        (is_valid: bool, message: str)
    """
    path = Path(video_path)
    if not path.exists():
        return False, f"File not found: {video_path}"
    if not path.is_file():
        return False, f"Not a file: {video_path}"
    if path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return False, f"Unsupported extension: {path.suffix}"

    cap = cv2.VideoCapture(video_path)
    opened = cap.isOpened()
    cap.release()

    if not opened:
        return False, "OpenCV could not open the file (corrupted or unsupported codec)"
    return True, "OK"


def extract_audio_track(video_path: str, output_path: str) -> str:
    """
    Extract the audio track from a video to a WAV file for Whisper.

    Uses ffmpeg under the hood. Returns the output path.
    """
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM WAV
        "-ar", "16000",           # 16 kHz (Whisper's native rate)
        "-ac", "1",               # mono
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    logger.info("Extracted audio → %s", output_path)
    return output_path
