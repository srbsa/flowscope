"""
agents/transcriber.py
Transcribes a video file using local Whisper or OpenAI Whisper API.
Writes transcript text to <run_dir>/transcriber.sh
"""

import logging
import os
import tempfile
from pathlib import Path

from utils.state_manager import write_agent_state
from utils.video_utils import extract_audio_track
from graph.state import (
    AGENT_TRANSCRIBER,
    PROVIDER_OPENAI,
    STATUS_RUNNING,
    STATUS_COMPLETE,
    STATUS_FAILED,
)

logger = logging.getLogger(__name__)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# Lazy-loaded local model (loaded once, reused)
_model = None


def _get_local_model():
    global _model
    if _model is None:
        import whisper
        logger.info("Loading local Whisper model: %s", WHISPER_MODEL)
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def _transcribe_local(wav_path: str) -> str:
    """Transcribe using local Whisper model."""
    model = _get_local_model()
    result = model.transcribe(wav_path, verbose=False)
    return result["text"].strip()


def _transcribe_openai(wav_path: str) -> str:
    """Transcribe using OpenAI Whisper API."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(wav_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return result.text.strip()


def transcribe_video(
    video_path: str,
    provider: str = "",
    run_dir: str | None = None,
) -> str:
    """
    Transcribe audio from a video file.

    Uses OpenAI Whisper API when provider is 'openai', local Whisper otherwise.

    Args:
        video_path: Absolute path to the video file.
        provider:   'lm_studio' or 'openai'.
        run_dir:    Per-run output directory.

    Returns:
        Full transcript as a string.
    """
    write_agent_state(
        AGENT_TRANSCRIBER, STATUS_RUNNING,
        output_summary="Extracting audio…", run_dir=run_dir,
    )

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        extract_audio_track(video_path, wav_path)

        api_key = os.getenv("OPENAI_API_KEY", "")
        use_openai = (
            provider == PROVIDER_OPENAI
            and bool(api_key)
            and not api_key.startswith("your_")
        )
        if use_openai:
            logger.info("Transcribing via OpenAI Whisper API")
            transcript = _transcribe_openai(wav_path)
        else:
            if provider == PROVIDER_OPENAI:
                logger.warning(
                    "OpenAI API key not configured — falling back to local Whisper (%s)",
                    WHISPER_MODEL,
                )
            else:
                logger.info("Transcribing via local Whisper (%s)", WHISPER_MODEL)
            transcript = _transcribe_local(wav_path)

        Path(wav_path).unlink(missing_ok=True)

        summary = transcript[:120].replace("\n", " ") + ("…" if len(transcript) > 120 else "")
        write_agent_state(
            AGENT_TRANSCRIBER, STATUS_COMPLETE,
            output_full=transcript, output_summary=summary,
            run_dir=run_dir,
        )

        logger.info("Transcription complete — %d characters", len(transcript))
        return transcript

    except Exception as exc:
        error_msg = f"Transcription failed: {exc}"
        logger.exception(error_msg)
        write_agent_state(
            AGENT_TRANSCRIBER, STATUS_FAILED,
            output_summary=error_msg, run_dir=run_dir,
        )
        raise RuntimeError(error_msg) from exc
