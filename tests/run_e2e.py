#!/usr/bin/env python3
"""
tests/run_e2e.py
End-to-end CLI runner — runs the full Video Workflow Agent pipeline against
a video file without needing the Streamlit UI.

Usage:
    python tests/run_e2e.py                                         # defaults to test video
    python tests/run_e2e.py path/to/video.mp4
    python tests/run_e2e.py path/to/video.mp4 --provider openai
    python tests/run_e2e.py path/to/video.mp4 --provider lm_studio

Outputs:
    state_outputs/<run_id>/  — all .sh state files + .md outputs
    state_outputs/<run_id>/frames/  — extracted keyframe JPEGs
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Path bootstrap ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_VIDEO = ROOT / "tests" / "video" / "airtableRoadmap.mp4"


def _print_separator(title: str = "") -> None:
    line = "─" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Video Workflow Agent — end-to-end pipeline runner",
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help="Path to video file (default: tests/video/airtableRoadmap.mp4)",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("DEFAULT_PROVIDER", "lm_studio"),
        choices=["lm_studio", "openai"],
        help="LLM provider (default: $DEFAULT_PROVIDER or lm_studio)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show DEBUG-level logs",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return 1

    _print_separator("Video Workflow Agent — E2E Run")
    print(f"  Video:    {video_path}")
    print(f"  Provider: {args.provider}")
    _print_separator()

    from graph.state import initial_state, ALL_AGENTS
    from graph.workflow import workflow
    from utils.state_manager import all_statuses

    state = initial_state(str(video_path), provider=args.provider)
    run_dir = Path(state["run_dir"]).resolve()

    print(f"  Run ID:   {state['run_id']}")
    print(f"  Out dir:  {run_dir}")
    _print_separator()

    completed_nodes: list[str] = []
    errors: list[str] = []

    try:
        for event in workflow.stream(state):
            for node_name, node_output in event.items():
                if node_name.startswith("__"):
                    continue
                completed_nodes.append(node_name)
                step = node_output.get("current_step", "")
                print(f"  ✅  {node_name:20s}  →  {step}")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        errors.append(str(exc))

    # ── Summary ─────────────────────────────────────────────────────────────────
    _print_separator("Run Summary")
    statuses = all_statuses(str(run_dir))
    for agent in ALL_AGENTS:
        status = statuses.get(agent, "waiting")
        icon = "✅" if status == "complete" else ("❌" if status == "failed" else "⬜")
        print(f"  {icon}  {agent:20s}  {status}")

    # ── Output files ────────────────────────────────────────────────────────────
    _print_separator("Output Files")
    for f in sorted(run_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.name:30s}  ({size:,} bytes)")

    frames_dir = run_dir / "frames"
    if frames_dir.exists():
        frame_count = len(list(frames_dir.glob("*.jpg")))
        print(f"  frames/                         ({frame_count} keyframes)")

    # ── Final report ─────────────────────────────────────────────────────────────
    synthesis_md = run_dir / "synthesis.md"
    if synthesis_md.exists():
        _print_separator("Final Report (synthesis.md)")
        print(synthesis_md.read_text(encoding="utf-8"))
    else:
        _print_separator()
        print("  synthesis.md not found — pipeline may have failed before synthesis.")

    if errors:
        _print_separator("Errors")
        for err in errors:
            print(f"  ❌ {err}")
        return 1

    print(f"\n  All outputs in: {run_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
