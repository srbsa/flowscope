"""
app.py
Streamlit UI for the Video Workflow Agent pipeline.

Layout:
- Sidebar: provider selector + pipeline status badges (one per agent, colour-coded)
- Main: video upload + step-by-step collapsible agent outputs
- Shows alignment loop iteration counter
- Re-run from any step by clearing downstream state files
"""

import os
import sys
import logging
import tempfile
import threading
import time
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


def _get_run_dir() -> str | None:
    return st.session_state.get("run_dir")


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(statuses: dict[str, str]) -> None:
    run_dir = _get_run_dir()

    with st.sidebar:
        st.markdown("## 🎬 Pipeline Status")

        # Provider selector
        default_prov = os.getenv("DEFAULT_PROVIDER", PROVIDER_LM_STUDIO)
        provider_keys = list(PROVIDER_DISPLAY.keys())
        default_idx = provider_keys.index(default_prov) if default_prov in provider_keys else 0
        selected = st.selectbox(
            "LLM Provider",
            options=provider_keys,
            format_func=lambda k: PROVIDER_DISPLAY[k],
            index=default_idx,
            key="provider_select",
        )
        st.session_state["provider"] = selected

        if run_dir:
            st.caption(f"Run: `{Path(run_dir).name}`")

        st.divider()

        for agent in ALL_AGENTS:
            status = statuses.get(agent, STATUS_WAITING)
            icon, color = STATUS_COLORS.get(status, ("⬜", "#888"))
            label = AGENT_LABELS.get(agent, agent)
            state_data = read_agent_state(agent, run_dir)
            summary = state_data.get("OUTPUT_SUMMARY", "")
            iteration = state_data.get("ITERATION", "0")

            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
                f"<span style='font-size:18px'>{icon}</span>"
                f"<span style='color:{color};font-weight:600'>{label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if summary:
                st.caption(summary[:80])
            if agent in (AGENT_RESEARCH, AGENT_ALIGNMENT) and iteration and iteration != "0":
                st.caption(f"Iteration: {iteration}")

            if status == STATUS_COMPLETE:
                if st.button(f"↩ Re-run from here", key=f"rerun_{agent}"):
                    for downstream in _DOWNSTREAM.get(agent, []):
                        clear_agent_state(downstream, run_dir)
                    clear_agent_state(agent, run_dir)
                    st.rerun()

        st.divider()
        if st.button("🗑 Clear all & restart", use_container_width=True):
            clear_all_states(run_dir)
            for k in ("video_path", "run_dir", "run_id"):
                st.session_state.pop(k, None)
            st.rerun()


# ── Main content ───────────────────────────────────────────────────────────────

def render_agent_output(agent: str, label: str) -> None:
    run_dir = _get_run_dir()
    state_data = read_agent_state(agent, run_dir)
    status = state_data.get("STATUS", STATUS_WAITING)
    icon, _ = STATUS_COLORS.get(status, ("⬜", "#888"))

    with st.expander(f"{icon} {label}", expanded=(status == STATUS_COMPLETE)):
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
            if agent == AGENT_FRAME_EXTRACTOR:
                _render_frame_extractor_output(run_dir, state_data)
            else:
                output = state_data.get("OUTPUT_FULL", "")
                st.markdown(output)

            ts = state_data.get("TIMESTAMP", "")
            if ts:
                st.caption(f"Completed at {ts}")


def _render_frame_extractor_output(run_dir: str | None, state_data: dict) -> None:
    """Render the frame extractor output: thumbnail gallery + chunk summaries."""
    # Show chunk summaries (markdown written to OUTPUT_FULL)
    output = state_data.get("OUTPUT_FULL", "")
    if output:
        st.markdown(output)

    # Thumbnail gallery scanned directly from the frames/ directory
    frames_dir = Path(run_dir) / "frames" if run_dir else None
    if frames_dir and frames_dir.exists():
        frame_files = sorted(frames_dir.glob("*.jpg"))
        if frame_files:
            st.markdown(f"**{len(frame_files)} keyframes saved**")
            cols = st.columns(4)
            for i, fp in enumerate(frame_files[:20]):
                with cols[i % 4]:
                    st.image(str(fp), caption=f"Frame {i + 1}", use_container_width=True)
            if len(frame_files) > 20:
                st.caption(f"… and {len(frame_files) - 20} more frames (showing first 20)")
        else:
            st.info("No keyframes extracted.")


def _render_frame_gallery(output: str) -> None:
    frame_paths = [p.strip() for p in output.splitlines() if p.strip()]
    if not frame_paths:
        st.info("No keyframes extracted.")
        return

    st.write(f"**{len(frame_paths)} unique frames extracted**")
    cols = st.columns(4)
    for i, path in enumerate(frame_paths[:20]):
        if Path(path).exists():
            with cols[i % 4]:
                st.image(path, caption=f"Frame {i+1}", use_container_width=True)

    if len(frame_paths) > 20:
        st.caption(f"… and {len(frame_paths) - 20} more frames (showing first 20)")


def _run_pipeline_thread(video_path: str, provider: str, run_dir: str, run_id: str) -> None:
    """Run the LangGraph pipeline in a background thread."""
    try:
        from graph.workflow import workflow
        # Re-use the already-created run directory rather than creating a new one
        state = WorkflowState(
            video_path=video_path,
            run_id=run_id,
            run_dir=run_dir,
            provider=provider,
            transcript="",
            frame_paths=[],
            frame_descriptions=[],
            requirements="",
            research="",
            research_sources=[],
            alignment_verdict="",
            alignment_confident=False,
            alignment_notes="",
            synthesis="",
            iteration_count=0,
            current_step="transcriber",
            error=None,
        )
        for event in workflow.stream(state):
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
                st.success("Pipeline complete!")
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

        render_agent_output(AGENT_TRANSCRIBER,     AGENT_LABELS[AGENT_TRANSCRIBER])
        render_agent_output(AGENT_FRAME_EXTRACTOR, AGENT_LABELS[AGENT_FRAME_EXTRACTOR])
        render_agent_output(AGENT_REQUIREMENTS,    AGENT_LABELS[AGENT_REQUIREMENTS])
        render_agent_output(AGENT_RESEARCH,        AGENT_LABELS[AGENT_RESEARCH])
        render_agent_output(AGENT_ALIGNMENT,       AGENT_LABELS[AGENT_ALIGNMENT])
        render_agent_output(AGENT_SYNTHESIS,       AGENT_LABELS[AGENT_SYNTHESIS])

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
