"""
utils/state_manager.py
Reads and writes agent state files using a nested folder structure:

    state_outputs/
      run_YYYYMMDD_HHMMSS/
        frames/                  ← keyframe JPEGs (written by frame_extractor)
        transcriber/
          state.sh               ← machine-readable shell-sourceable state
        requirements/
          state.sh
          output.md              ← human-readable markdown (for MD_AGENTS)
        research/
          state.sh
          output.md
        alignment/
          state.sh
          output.md
        synthesis/
          state.sh
          output.md

state.sh format:
    export AGENT='requirements'
    export STATUS='complete'
    export TIMESTAMP='2025-01-01T00:00:00Z'
    export ITERATION='0'
    export OUTPUT_SUMMARY='One-line summary'
    export OUTPUT_FULL='Multi-line full output with escaped single quotes'

All functions accept a ``run_dir`` parameter (the run-level directory).
When omitted the default ``state_outputs/`` path is used.
"""

import os
import re
import logging
from datetime import datetime, timezone
from pathlib import Path

from graph.state import STATE_OUTPUTS_DIR

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path(STATE_OUTPUTS_DIR)


def _resolve_dir(run_dir: str | None) -> Path:
    """Return a Path for the run directory, falling back to the default."""
    return Path(run_dir) if run_dir else _DEFAULT_DIR


def _agent_dir(agent: str, run_dir: str | None = None) -> Path:
    """Return the per-agent subdirectory: <run_dir>/<agent>/"""
    return _resolve_dir(run_dir) / agent


def _ensure_dir(agent: str, run_dir: str | None = None) -> None:
    """Create the run dir, shared frames dir, and per-agent subdir."""
    d = _resolve_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "frames").mkdir(parents=True, exist_ok=True)
    _agent_dir(agent, run_dir).mkdir(parents=True, exist_ok=True)


def _sh_path(agent: str, run_dir: str | None = None) -> Path:
    return _agent_dir(agent, run_dir) / "state.sh"


def _md_path(agent: str, run_dir: str | None = None) -> Path:
    return _agent_dir(agent, run_dir) / "output.md"


# Agents whose full output is worth rendering as standalone markdown
_MD_AGENTS = {"frame_extractor", "requirements", "research", "alignment", "synthesis"}


# Agents that loop — iteration number is meaningful in their .md heading
_ITER_AGENTS = {"research", "alignment"}


def write_agent_state(
    agent: str,
    status: str,
    output_full: str = "",
    output_summary: str = "",
    iteration: int = 0,
    run_dir: str | None = None,
) -> None:
    """
    Write agent output to <run_dir>/<agent>.sh as a shell-sourceable file.
    When status is 'complete' and output_full is provided, also writes
    <run_dir>/<agent>.md for human-readable inspection.

    Args:
        agent:          Agent name (e.g. "requirements")
        status:         "waiting" | "running" | "complete" | "failed"
        output_full:    Complete agent output text
        output_summary: One-line human-readable summary
        iteration:      Alignment loop iteration number
        run_dir:        Per-run output directory (None → default state_outputs/)
    """
    _ensure_dir(agent, run_dir)

    timestamp = datetime.now(timezone.utc).isoformat()

    safe_full    = output_full.replace("'", "'\\''")
    safe_summary = output_summary.replace("'", "'\\''").replace("\n", " ")

    content = (
        f"export AGENT='{agent}'\n"
        f"export STATUS='{status}'\n"
        f"export TIMESTAMP='{timestamp}'\n"
        f"export ITERATION='{iteration}'\n"
        f"export OUTPUT_SUMMARY='{safe_summary}'\n"
        f"export OUTPUT_FULL='{safe_full}'\n"
    )

    path = _sh_path(agent, run_dir)
    path.write_text(content, encoding="utf-8")
    logger.debug("Wrote state for agent '%s' → %s  [%s]", agent, path.parent, status)

    # Also write a human-readable .md file for agents with structured text output
    if status == "complete" and output_full and agent in _MD_AGENTS:
        title = agent.replace("_", " ").title()
        iter_note = f"  *(loop {iteration + 1})*" if iteration > 0 and agent in _ITER_AGENTS else ""
        md_content = (
            f"# {title}{iter_note}\n\n"
            f"> Generated: {timestamp}\n\n"
            f"---\n\n"
            f"{output_full}\n"
        )
        _md_path(agent, run_dir).write_text(md_content, encoding="utf-8")
        logger.debug("Wrote markdown for agent '%s'", agent)


def _parse_sh_value(text: str, start: int) -> tuple[str, int]:
    """Parse a shell single-quoted value starting immediately after the opening `'`.

    Handles the ``'\''`` escape sequence used by write_agent_state to embed
    literal single-quotes in the value.  Works correctly for multiline values.

    Returns:
        (decoded_value, position_after_closing_quote)
    """
    parts: list[str] = []
    i = start
    n = len(text)
    while i < n:
        if text[i] == "'":
            if text[i : i + 4] == "'\\''":
                # Escaped single quote: replace '\'' → '
                parts.append("'")
                i += 4
            else:
                # Unescaped ' → closing quote
                return "".join(parts), i + 1
        else:
            parts.append(text[i])
            i += 1
    # Unterminated value — return what we have
    return "".join(parts), n


def read_agent_state(agent: str, run_dir: str | None = None) -> dict:
    """
    Read and parse a .sh state file for an agent.

    Returns a dict with keys: AGENT, STATUS, TIMESTAMP, ITERATION,
    OUTPUT_SUMMARY, OUTPUT_FULL.  Returns empty dict if file not found.

    Uses a proper single-quote parser so that multiline OUTPUT_FULL
    values are decoded correctly.
    """
    path = _sh_path(agent, run_dir)
    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8")
    result: dict = {}

    for m in re.finditer(r"export\s+(\w+)='", raw):
        key = m.group(1)
        val, _ = _parse_sh_value(raw, m.end())
        result[key] = val

    return result


def get_status(agent: str, run_dir: str | None = None) -> str:
    """Return the STATUS field for an agent, or 'waiting' if not yet written."""
    state = read_agent_state(agent, run_dir)
    return state.get("STATUS", "waiting")


def get_output(agent: str, run_dir: str | None = None) -> str:
    """Return the full output for an agent, or empty string."""
    state = read_agent_state(agent, run_dir)
    return state.get("OUTPUT_FULL", "")


def clear_agent_state(agent: str, run_dir: str | None = None) -> None:
    """Delete the per-agent subdirectory (resets that step for re-run)."""
    import shutil
    adir = _agent_dir(agent, run_dir)
    if adir.exists():
        shutil.rmtree(adir)
    logger.info("Cleared state for agent '%s'", agent)


def clear_all_states(run_dir: str | None = None) -> None:
    """Delete all agent subdirectories (full pipeline reset, preserves frames and run dir)."""
    import shutil
    d = _resolve_dir(run_dir)
    for sub in d.iterdir():
        if sub.is_dir() and sub.name != "frames":
            shutil.rmtree(sub)
    logger.info("Cleared all agent state dirs in %s", d)


def all_statuses(run_dir: str | None = None) -> dict[str, str]:
    """Return a dict of {agent_name: status} for all known agents."""
    from graph.state import ALL_AGENTS
    return {agent: get_status(agent, run_dir) for agent in ALL_AGENTS}
