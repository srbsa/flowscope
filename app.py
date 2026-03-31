"""
app.py
Streamlit UI for the Video Workflow Agent pipeline.

Layout:
- Sidebar: provider selector, previous-run picker, pipeline status with elapsed times
- Main: video upload + step-by-step collapsible agent outputs
- Shows alignment loop iteration counter
- Re-run from any step by clearing downstream state files
- Download button for the final synthesis report
"""

import os
import sys
import logging
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from graph.state import (
    ALL_AGENTS,
    AGENT_TRANSCRIBER,
    AGENT_FRAME_EXTRACTOR,
    AGENT_REQUIREMENTS,
    AGENT_RESEARCH,
    AGENT_ALIGNMENT,
    AGENT_SYNTHESIS,
    PROVIDER_LM_STUDIO,
    PROVIDER_OPENAI,
    STATE_OUTPUTS_DIR,
    STATUS_WAITING,
    STATUS_RUNNING,
    STATUS_COMPLETE,
    STATUS_FAILED,
    WorkflowState,
    initial_state,
)
from utils.state_manager import (
    all_statuses,
    read_agent_state,
    clear_all_states,
    clear_agent_state,
    get_output,
)
from utils.video_utils import get_video_metadata, validate_video

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Workflow Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
STATUS_COLORS = {
    STATUS_WAITING:  ("⬜", "#888888"),
    STATUS_RUNNING:  ("🔵", "#1E90FF"),
    STATUS_COMPLETE: ("✅", "#28a745"),
    STATUS_FAILED:   ("❌", "#dc3545"),
}

AGENT_LABELS = {
    AGENT_TRANSCRIBER:     "1. Transcriber",
    AGENT_FRAME_EXTRACTOR: "2. Frame Extractor",
    AGENT_REQUIREMENTS:    "3. Requirements",
    AGENT_RESEARCH:        "4. Researcher",
    AGENT_ALIGNMENT:       "5. Alignment PM",
    AGENT_SYNTHESIS:       "6. Synthesis",
}

AGENT_DESCRIPTIONS = {
    AGENT_TRANSCRIBER:     "Transcribes video audio using Whisper",
    AGENT_FRAME_EXTRACTOR: "Extracts unique keyframes and describes them with vision AI",
    AGENT_REQUIREMENTS:    "Distils workflow requirements from transcript + visuals",
    AGENT_RESEARCH:        "Researches tools and best practices via web search",
    AGENT_ALIGNMENT:       "Checks if research aligns with requirements",
    AGENT_SYNTHESIS:       "Produces the final implementation report",
}

PROVIDER_DISPLAY = {
    PROVIDER_LM_STUDIO: "LM Studio (local)",
    PROVIDER_OPENAI:    "OpenAI",
}

_DOWNSTREAM: dict[str, list[str]] = {
    AGENT_TRANSCRIBER:     [AGENT_FRAME_EXTRACTOR, AGENT_REQUIREMENTS, AGENT_RESEARCH, AGENT_ALIGNMENT, AGENT_SYNTHESIS],
    AGENT_FRAME_EXTRACTOR: [AGENT_REQUIREMENTS, AGENT_RESEARCH, AGENT_ALIGNMENT, AGENT_SYNTHESIS],
    AGENT_REQUIREMENTS:    [AGENT_RESEARCH, AGENT_ALIGNMENT, AGENT_SYNTHESIS],
    AGENT_RESEARCH:        [AGENT_ALIGNMENT, AGENT_SYNTHESIS],
    AGENT_ALIGNMENT:       [AGENT_SYNTHESIS],
    AGENT_SYNTHESIS:       [],
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_run_dir() -> str | None:
    return st.session_state.get("run_dir")


def _parse_timestamp(ts: str) -> datetime | None:
    """Parse an ISO timestamp string, returning None on failure."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _elapsed_between(start_ts: str, end_ts: str) -> str:
    """Return a human-friendly elapsed time string between two ISO timestamps."""
    start = _parse_timestamp(start_ts)
    end = _parse_timestamp(end_ts)
    if not start or not end:
        return ""
    delta = end - start
    total_secs = int(delta.total_seconds())
    if total_secs < 0:
        return ""
    if total_secs < 60:
        return f"{total_secs}s"
    mins, secs = divmod(total_secs, 60)
    return f"{mins}m {secs}s"


def _total_pipeline_elapsed(run_dir: str | None) -> str:
    """Compute total pipeline elapsed time from first to last agent timestamp."""
    if not run_dir:
        return ""
    timestamps: list[datetime] = []
    for agent in ALL_AGENTS:
        state = read_agent_state(agent, run_dir)
        ts = _parse_timestamp(state.get("TIMESTAMP", ""))
        if ts:
            timestamps.append(ts)
    if len(timestamps) < 2:
        return ""
    delta = max(timestamps) - min(timestamps)
    total_secs = int(delta.total_seconds())
    if total_secs < 60:
        return f"{total_secs}s"
    mins, secs = divmod(total_secs, 60)
    return f"{mins}m {secs}s"


def _list_previous_runs() -> list[str]:
    """Return run directory names sorted newest-first."""
    outputs = Path(STATE_OUTPUTS_DIR)
    if not outputs.exists():
        return []
    return sorted(
        [d.name for d in outputs.iterdir() if d.is_dir() and d.name.startswith("run_")],
        reverse=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(statuses: dict[str, str]) -> None:
    run_dir = _get_run_dir()
    any_running = any(s == STATUS_RUNNING for s in statuses.values())

    with st.sidebar:
        st.markdown("## 🎬 Pipeline Status")

        # Provider selector — disabled while pipeline is running
        default_prov = os.getenv("DEFAULT_PROVIDER", PROVIDER_LM_STUDIO)
        provider_keys = list(PROVIDER_DISPLAY.keys())
        default_idx = provider_keys.index(default_prov) if default_prov in provider_keys else 0
        selected = st.selectbox(
            "LLM Provider",
            options=provider_keys,
            format_func=lambda k: PROVIDER_DISPLAY[k],
            index=default_idx,
            key="provider_select",
            disabled=any_running,
        )
        st.session_state["provider"] = selected

        # Previous runs selector
        prev_runs = _list_previous_runs()
        if prev_runs and not any_running:
            current_run_name = Path(run_dir).name if run_dir else None
            options = ["(current)"] + prev_runs
            default_idx = 0
            if current_run_name and current_run_name in prev_runs:
                default_idx = prev_runs.index(current_run_name) + 1

            chosen = st.selectbox(
                "Previous Runs",
                options=options,
                index=default_idx,
                key="run_picker",
            )
            if chosen != "(current)" and chosen != current_run_name:
                new_dir = str(Path(STATE_OUTPUTS_DIR) / chosen)
                st.session_state["run_dir"] = new_dir
                st.session_state["run_id"] = chosen
                st.rerun()

        if run_dir:
            st.caption(f"Run: `{Path(run_dir).name}`")

        st.divider()

        # Collect timestamps for elapsed-time computation between consecutive steps
        agent_timestamps: dict[str, str] = {}
        for agent in ALL_AGENTS:
            state_data = read_agent_state(agent, run_dir)
            agent_timestamps[agent] = state_data.get("TIMESTAMP", "")

        for i, agent in enumerate(ALL_AGENTS):
            status = statuses.get(agent, STATUS_WAITING)
            icon, color = STATUS_COLORS.get(status, ("⬜", "#888"))
            label = AGENT_LABELS.get(agent, agent)
            state_data = read_agent_state(agent, run_dir)
            summary = state_data.get("OUTPUT_SUMMARY", "")
            iteration = state_data.get("ITERATION", "0")

            # Compute elapsed time for completed steps
            elapsed = ""
            if status == STATUS_COMPLETE and i > 0:
                prev_agent = ALL_AGENTS[i - 1]
                elapsed = _elapsed_between(
                    agent_timestamps.get(prev_agent, ""),
                    agent_timestamps[agent],
                )

            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:2px'>"
                f"<span style='font-size:18px'>{icon}</span>"
                f"<span style='color:{color};font-weight:600'>{label}</span>"
                + (f"<span style='color:#888;font-size:12px;margin-left:auto'>"
                   f"{elapsed}</span>" if elapsed else "")
                + "</div>",
                unsafe_allow_html=True,
            )
            if summary:
                st.caption(summary[:100])
            if agent in (AGENT_RESEARCH, AGENT_ALIGNMENT) and iteration and iteration != "0":
                st.caption(f"🔁 Iteration: {iteration}")

            if status == STATUS_COMPLETE and not any_running:
                if st.button("↩ Re-run from here", key=f"rerun_{agent}"):
                    for downstream in _DOWNSTREAM.get(agent, []):
                        clear_agent_state(downstream, run_dir)
                    clear_agent_state(agent, run_dir)
                    st.rerun()

        st.divider()

        # Total pipeline time
        total_elapsed = _total_pipeline_elapsed(run_dir)
        if total_elapsed:
            st.markdown(f"**Total pipeline time:** {total_elapsed}")

        if st.button("🗑 Clear all & restart", use_container_width=True, disabled=any_running):
            clear_all_states(run_dir)
            for k in ("video_path", "run_dir", "run_id"):
                st.session_state.pop(k, None)
            st.rerun()


# ── Main content ───────────────────────────────────────────────────────────────

def render_agent_output(agent: str, label: str, statuses: dict[str, str]) -> None:
    run_dir = _get_run_dir()
    state_data = read_agent_state(agent, run_dir)
    status = state_data.get("STATUS", STATUS_WAITING)
    icon, _ = STATUS_COLORS.get(status, ("⬜", "#888"))
    desc = AGENT_DESCRIPTIONS.get(agent, "")

    # Auto-expand running or complete steps; collapse waiting ones
    expand = status in (STATUS_RUNNING, STATUS_COMPLETE)

    with st.expander(f"{icon} {label}", expanded=expand):
        if desc:
            st.caption(desc)

        if status == STATUS_WAITING:
            st.info("Waiting to run…")
        elif status == STATUS_RUNNING:
            progress_msg = state_data.get("OUTPUT_SUMMARY", "")
            if progress_msg:
                st.info(f"⏳ {progress_msg}")
            else:
                st.info("Running…")
        elif status == STATUS_FAILED:
            st.error(state_data.get("OUTPUT_SUMMARY", "Failed"))
        elif status == STATUS_COMPLETE:
            if agent == AGENT_TRANSCRIBER:
                _render_transcript_output(state_data)
            elif agent == AGENT_FRAME_EXTRACTOR:
                _render_frame_extractor_output(run_dir, state_data)
            elif agent == AGENT_SYNTHESIS:
                _render_synthesis_output(run_dir, state_data)
            else:
                output = state_data.get("OUTPUT_FULL", "")
                st.markdown(output)

            ts = state_data.get("TIMESTAMP", "")
            if ts:
                st.caption(f"Completed at {ts}")


def _render_transcript_output(state_data: dict) -> None:
    """Render transcript with word count and copy-friendly display."""
    transcript = state_data.get("OUTPUT_FULL", "")
    if not transcript:
        st.info("No transcript available.")
        return

    words = len(transcript.split())
    chars = len(transcript)
    st.markdown(f"**{words:,} words** · {chars:,} characters")
    st.text_area(
        "Full Transcript",
        value=transcript,
        height=200,
        label_visibility="collapsed",
        disabled=True,
    )


def _render_frame_extractor_output(run_dir: str | None, state_data: dict) -> None:
    """Render the frame extractor output: chunk summaries + thumbnail gallery."""
    output = state_data.get("OUTPUT_FULL", "")

    # Split output into chunk summaries and frame-by-frame sections
    if "## Frame-by-Frame Descriptions" in output:
        chunks_section, frames_section = output.split("## Frame-by-Frame Descriptions", 1)
    else:
        chunks_section = output
        frames_section = ""

    # Always show chunk summaries
    if chunks_section.strip():
        st.markdown(chunks_section)

    # Thumbnail gallery scanned directly from the frames/ directory
    frames_dir = Path(run_dir) / "frames" if run_dir else None
    if frames_dir and frames_dir.exists():
        frame_files = sorted(frames_dir.glob("*.jpg"))
        if frame_files:
            st.markdown(f"**{len(frame_files)} keyframes saved**")
            # Paginated gallery
            page_size = 12
            total_pages = max(1, (len(frame_files) + page_size - 1) // page_size)
            page = st.number_input(
                "Frame page",
                min_value=1, max_value=total_pages, value=1,
                label_visibility="collapsed",
                key="frame_page",
            ) if total_pages > 1 else 1
            start = (page - 1) * page_size
            page_frames = frame_files[start:start + page_size]

            cols = st.columns(4)
            for i, fp in enumerate(page_frames):
                with cols[i % 4]:
                    st.image(str(fp), caption=f"Frame {start + i + 1}", use_container_width=True)
            if total_pages > 1:
                st.caption(f"Page {page} of {total_pages} · {len(frame_files)} total frames")
        else:
            st.info("No keyframes extracted.")

    # Frame-by-frame descriptions in a collapsible sub-section
    if frames_section.strip():
        with st.expander("📝 Frame-by-Frame Descriptions", expanded=False):
            st.markdown(frames_section)


def _render_synthesis_output(run_dir: str | None, state_data: dict) -> None:
    """Render synthesis output with a download button for the report."""
    output = state_data.get("OUTPUT_FULL", "")
    if not output:
        st.info("No synthesis output available.")
        return

    st.markdown(output)

    # Download button
    st.download_button(
        label="📥 Download Report",
        data=output,
        file_name="workflow-optimisation-report.md",
        mime="text/markdown",
        use_container_width=True,
    )


def _run_pipeline_thread(video_path: str, provider: str, run_dir: str, run_id: str) -> None:
    """Run the LangGraph pipeline in a background thread."""
    try:
        from graph.workflow import workflow
        state = WorkflowState(
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
            current_step="transcriber",
            error=None,
        )
        for event in workflow.stream(state):  # type: ignore[union-attr]
            logger.info("Graph event: %s", list(event.keys()))
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        from utils.state_manager import write_agent_state
        write_agent_state("pipeline_error", STATUS_FAILED, output_summary=str(exc), run_dir=run_dir)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🎬 Video Workflow Agent")
    st.caption(
        "Upload a screen recording or talking-head video. "
        "The pipeline transcribes it, extracts keyframes, distills requirements, "
        "researches recommendations, checks alignment, and synthesises a report."
    )

    run_dir = _get_run_dir()
    statuses = all_statuses(run_dir)
    render_sidebar(statuses)

    # ── Pipeline error banner ─────────────────────────────────────────────────
    error_state = read_agent_state("pipeline_error", run_dir)
    if error_state.get("STATUS") == STATUS_FAILED:
        st.error(f"⚠️ Pipeline error: {error_state.get('OUTPUT_SUMMARY', 'Unknown error')}")

    # ── Video upload section ──────────────────────────────────────────────────
    st.subheader("Upload Video")

    uploaded = st.file_uploader(
        "Drop your video here",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        label_visibility="collapsed",
    )

    if uploaded:
        if "video_path" not in st.session_state:
            suffix = Path(uploaded.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()
            st.session_state["video_path"] = tmp.name

        video_path = st.session_state["video_path"]

        is_valid, msg = validate_video(video_path)
        if not is_valid:
            st.error(f"Invalid video: {msg}")
            return

        meta = get_video_metadata(video_path)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duration", f"{meta['duration_seconds']:.0f}s")
        col2.metric("Resolution", f"{meta['width']}×{meta['height']}")
        col3.metric("FPS", f"{meta['fps']:.0f}")
        col4.metric("Size", f"{meta['file_size_mb']:.1f} MB")

        with st.expander("🎥 Video Preview", expanded=False):
            st.video(video_path)

        # ── Run pipeline button ───────────────────────────────────────────────
        all_done = all(s == STATUS_COMPLETE for s in statuses.values())
        any_running = any(s == STATUS_RUNNING for s in statuses.values())
        any_started = any(s != STATUS_WAITING for s in statuses.values())

        provider = st.session_state.get("provider", PROVIDER_LM_STUDIO)

        col_run, col_status = st.columns([2, 3])

        with col_run:
            if not any_started:
                if st.button("▶ Run Pipeline", type="primary", use_container_width=True):
                    # Create a fresh run dir
                    state = initial_state(video_path, provider=provider)
                    st.session_state["run_dir"] = state["run_dir"]
                    st.session_state["run_id"] = state["run_id"]
                    run_dir = state["run_dir"]

                    thread = threading.Thread(
                        target=_run_pipeline_thread,
                        args=(video_path, provider, run_dir, state["run_id"]),
                        daemon=True,
                    )
                    thread.start()
                    st.session_state["pipeline_running"] = True
                    st.rerun()

            elif any_running:
                st.button("⏳ Running…", disabled=True, use_container_width=True)

            elif all_done:
                st.success("✅ Pipeline complete!")

            else:
                if st.button("▶ Continue Pipeline", type="primary", use_container_width=True):
                    if not run_dir:
                        state = initial_state(video_path, provider=provider)
                        st.session_state["run_dir"] = state["run_dir"]
                        st.session_state["run_id"] = state["run_id"]
                        run_dir = state["run_dir"]

                    run_id = st.session_state.get("run_id", Path(run_dir).name)
                    thread = threading.Thread(
                        target=_run_pipeline_thread,
                        args=(video_path, provider, run_dir, run_id),
                        daemon=True,
                    )
                    thread.start()
                    st.rerun()

        with col_status:
            completed = sum(1 for s in statuses.values() if s == STATUS_COMPLETE)
            st.progress(completed / len(ALL_AGENTS), text=f"{completed}/{len(ALL_AGENTS)} steps complete")

        # ── Alignment loop counter ────────────────────────────────────────────
        alignment_state = read_agent_state(AGENT_ALIGNMENT, run_dir)
        if alignment_state.get("ITERATION") and int(alignment_state["ITERATION"]) > 0:
            st.info(
                f"🔁 Alignment loop iteration: "
                f"{alignment_state['ITERATION']} / {os.getenv('MAX_ALIGNMENT_ITERATIONS', '3')}"
            )

        # ── Agent output sections ─────────────────────────────────────────────
        st.divider()
        st.subheader("Pipeline Outputs")

        for agent in ALL_AGENTS:
            render_agent_output(agent, AGENT_LABELS[agent], statuses)

        # Auto-refresh while running
        if any_running or st.session_state.get("pipeline_running"):
            time.sleep(2)
            st.rerun()

    else:
        st.info("👆 Upload a video to get started.")
        st.markdown(
            """
            **What this pipeline does:**
            1. 🎙 **Transcribes** the video with Whisper
            2. 🖼 **Extracts keyframes** (unique scene changes, no mouse jitter)
            3. 📋 **Distils requirements** from transcript + visual context
            4. 🔍 **Researches** tools and best practices (with web search)
            5. ✅ **Alignment PM** checks recommendations vs requirements
            6. 📄 **Synthesises** a final implementation report

            The alignment loop repeats up to 3× until the PM is confident.
            """
        )


if __name__ == "__main__":
    main()
