"""
graph/state.py
LangGraph state definition and shared constants for the video workflow pipeline.
"""

import os
from datetime import datetime, timezone
from typing import TypedDict, List, Optional

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_ALIGNMENT_ITERATIONS = 3
FRAME_DIFF_THRESHOLD = 30          # Mean pixel diff to consider frame "unique"
MOUSE_REGION_SIZE = 100            # px² region to ignore as mouse movement
VIDEO_MAX_WIDTH = int(os.getenv("VIDEO_MAX_WIDTH", "480"))  # 0 = no downscale
MAX_FRAMES_DESCRIBED = int(os.getenv("MAX_FRAMES_DESCRIBED", "100"))  # max frames sent to vision LLM
MAX_ALIGNMENT_SEARCH_ROUNDS = int(os.getenv("MAX_ALIGNMENT_SEARCH_ROUNDS", "3"))
MAX_SYNTHESIS_SEARCH_ROUNDS = int(os.getenv("MAX_SYNTHESIS_SEARCH_ROUNDS", "3"))
SYNTHESIS_SEARCH_ENABLED = os.getenv("SYNTHESIS_SEARCH_ENABLED", "false").lower() == "true"
STATE_OUTPUTS_DIR = "state_outputs"

# Provider constants
PROVIDER_LM_STUDIO = "lm_studio"
PROVIDER_OPENAI = "openai"

# Agent names (also used as .sh file basenames)
AGENT_TRANSCRIBER    = "transcriber"
AGENT_FRAME_EXTRACTOR = "frame_extractor"
AGENT_REQUIREMENTS   = "requirements"
AGENT_RESEARCH       = "research"
AGENT_ALIGNMENT      = "alignment"
AGENT_SYNTHESIS      = "synthesis"

ALL_AGENTS = [
    AGENT_TRANSCRIBER,
    AGENT_FRAME_EXTRACTOR,
    AGENT_REQUIREMENTS,
    AGENT_RESEARCH,
    AGENT_ALIGNMENT,
    AGENT_SYNTHESIS,
]

# Status values written to .sh files
STATUS_WAITING  = "waiting"
STATUS_RUNNING  = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED   = "failed"


# ── State ──────────────────────────────────────────────────────────────────────

class WorkflowState(TypedDict):
    """Single source of truth flowing through all LangGraph nodes."""

    # ── Input ──────────────────────────────────────────────────────────────────
    video_path: str                         # Absolute path to uploaded video

    # ── Run context ────────────────────────────────────────────────────────────
    run_id: str                             # Unique run identifier (run_YYYYMMDD_HHMMSS)
    run_dir: str                            # Absolute path to run output directory
    provider: str                           # 'lm_studio' or 'openai'

    # ── Step 1: Transcription ──────────────────────────────────────────────────
    transcript: str                         # Full Whisper transcript

    # ── Step 2: Frame extraction ───────────────────────────────────────────────
    frame_paths: List[str]                  # Paths to saved keyframe JPEGs
    frame_descriptions: List[str]           # Brief LLM description of each frame
    frame_chunk_summaries: List[str]        # Narrative summaries per 25-frame chunk

    # ── Step 3: Requirements ───────────────────────────────────────────────────
    requirements: str                       # Streamlined workflow requirements doc

    # ── Step 4: Research (loops) ───────────────────────────────────────────────
    research: str                           # Researcher's recommendations
    research_sources: List[str]             # URLs / citations used
    research_search_queries: List[str]      # All web search queries executed across iterations

    # ── Step 5: Alignment ─────────────────────────────────────────────────────
    alignment_verdict: str                  # "confident" | "not_confident"
    alignment_confident: bool               # Parsed bool for routing
    alignment_notes: str                    # Cumulative notes back to researcher

    # ── Step 6: Synthesis ─────────────────────────────────────────────────────
    synthesis: str                          # Final structured output

    # ── Control ───────────────────────────────────────────────────────────────
    iteration_count: int                    # Alignment loop iteration counter
    current_step: str                       # Which agent is currently running
    error: Optional[str]                    # Last error message if any


def initial_state(video_path: str, provider: str = "") -> WorkflowState:
    """Return a fresh WorkflowState for a new run."""
    if not provider:
        provider = os.getenv("DEFAULT_PROVIDER", PROVIDER_LM_STUDIO)

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(STATE_OUTPUTS_DIR, run_id)
    os.makedirs(os.path.join(run_dir, "frames"), exist_ok=True)

    return WorkflowState(
        video_path=video_path,
        run_id=run_id,
        run_dir=run_dir,
        provider=provider,
        transcript="",
        frame_paths=[],
        frame_descriptions=[],
        frame_chunk_summaries=[],
        requirements="",
        research="",
        research_sources=[],
        research_search_queries=[],
        alignment_verdict="",
        alignment_confident=False,
        alignment_notes="",
        synthesis="",
        iteration_count=0,
        current_step=AGENT_TRANSCRIBER,
        error=None,
    )
