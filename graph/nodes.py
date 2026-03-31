"""
graph/nodes.py
LangGraph node functions. Each node wraps an agent and updates WorkflowState.
Node signature: (state: WorkflowState) -> dict  (return only changed keys)
"""

import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from graph.state import (
    WorkflowState,
    AGENT_TRANSCRIBER,
    AGENT_FRAME_EXTRACTOR,
    AGENT_REQUIREMENTS,
    AGENT_RESEARCH,
    AGENT_ALIGNMENT,
    AGENT_SYNTHESIS,
    MAX_FRAMES_DESCRIBED,
    FRAME_DESCRIPTION_WORKERS,
    FRAME_CHUNK_SIZE,
    STATUS_RUNNING,
    STATUS_COMPLETE,
)
from utils.state_manager import write_agent_state

logger = logging.getLogger(__name__)

# ── Helper: describe frames via vision with chained context ───────────────────

FRAME_VISION_SYSTEM = """\
You are a workflow analyst examining screen recording keyframes sequentially.
For each frame, identify: the software/tool being used, what workflow action is
being performed, and what data or content is visible. Be specific about product
names if recognisable. One concise sentence per frame."""

FRAME_VISION_FIRST = (
    "What software/tool is shown? What workflow action is being performed? "
    "What data or content is visible on screen? Answer in one specific sentence."
)

FRAME_VISION_CHAIN = (
    "Previous frame: {prev}\n\n"
    "What changed? What new action, screen, or content is now visible? "
    "If the tool/product is identifiable, name it. One sentence."
)

# ── Helper: chunk-summarise frame descriptions ──────────────────────────────────

CHUNK_SUMMARY_SYSTEM = """\
You are a workflow analyst. You have been given a sequence of consecutive screen
recording frame descriptions from a business workflow walkthrough.
Produce a nuanced, flowing narrative summary (3–5 sentences) of what the user
was doing during this segment. Focus on: the workflow actions performed,
tools or interfaces used, any friction or inefficiency visible, and how this
segment connects to the broader workflow. Be specific and observational."""

CHUNK_SUMMARY_USER = """\
Frame segment {start}–{end} of {total} total frames:
{descriptions}

Summarise what was happening during this workflow segment. Describe the actions,
tools, and any friction or inefficiency visible."""


def _describe_segment(
    segment: List[tuple[int, str]],
    provider: str,
    total_frames: int,
    run_dir: str | None,
) -> List[tuple[int, str]]:
    """
    Describe a contiguous segment of frames sequentially with chained context.
    Each segment starts its own chain (first frame uses FRAME_VISION_FIRST).
    Returns list of (original_index, description) tuples.
    Progress updates are written to the state file every 5 frames.
    """
    from utils.llm_client import describe_image

    results: list[tuple[int, str]] = []
    prev_desc: str = ""

    for position, (orig_idx, path) in enumerate(segment):
        try:
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            # Chain within segment; first frame of each segment is treated as fresh
            if position == 0 or not prev_desc:
                text_prompt = FRAME_VISION_FIRST
            else:
                text_prompt = FRAME_VISION_CHAIN.format(prev=prev_desc)

            desc = describe_image(
                provider=provider,
                image_b64=img_b64,
                text_prompt=text_prompt,
                system=FRAME_VISION_SYSTEM,
                max_tokens=400,
                thinking=False,  # thinking disabled: 1-sentence task consumes entire
                                 # budget during <think> phase, leaving empty content
            )
            # llm_client already strips <think> blocks; remove any residual
            desc = desc.strip()

            if not desc:
                logger.warning(
                    "Empty description for frame %d (%s) — vision model returned no content",
                    orig_idx + 1, path.split("/")[-1],
                )
                desc = f"(no description — frame {orig_idx + 1})"

            prev_desc = desc

        except Exception as exc:
            logger.warning("Frame description failed for %s: %s", path, exc)
            desc = f"(frame: {path.split('/')[-1]})"
            prev_desc = desc

        results.append((orig_idx, desc))

        # Emit progress every 5 frames so the UI stays updated
        if run_dir and (len(results) % 5 == 0 or len(results) == len(segment)):
            completed = orig_idx + 1
            write_agent_state(
                AGENT_FRAME_EXTRACTOR, STATUS_RUNNING,
                output_summary=f"Describing frame {completed}/{total_frames}…",
                run_dir=run_dir,
            )

    return results


def _describe_frames(
    frame_paths: List[str],
    provider: str,
    run_dir: str | None = None,
) -> List[str]:
    """
    Describe each keyframe using vision with chained context.

    When FRAME_DESCRIPTION_WORKERS > 1 the frame list is split into equal
    segments processed in parallel, each with its own internal chain.
    Workers > 1 trades cross-segment chain continuity for speed — useful
    when using the OpenAI provider. Set to 1 (default) for LM Studio where
    requests are serialised on a single GPU anyway (sequential = full chain).

    Returns a list of one-line descriptions (one per frame).
    """
    if not frame_paths:
        return []

    paths_to_describe = frame_paths[:MAX_FRAMES_DESCRIBED]
    total = len(paths_to_describe)
    n_workers = max(1, FRAME_DESCRIPTION_WORKERS)

    write_agent_state(
        AGENT_FRAME_EXTRACTOR, STATUS_RUNNING,
        output_summary=f"Describing frame 0/{total}…",
        run_dir=run_dir,
    )

    # Split into n_workers segments (contiguous, for chain context within each)
    segment_size = max(1, (total + n_workers - 1) // n_workers)
    indexed = list(enumerate(paths_to_describe))
    segments = [indexed[i:i + segment_size] for i in range(0, total, segment_size)]

    all_results: list[tuple[int, str]] = []

    if n_workers == 1:
        # Sequential: full chain across all frames
        all_results = _describe_segment(segments[0], provider, total, run_dir)
    else:
        # Parallel segments
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_describe_segment, seg, provider, total, run_dir): seg
                for seg in segments
            }
            for future in as_completed(futures):
                all_results.extend(future.result())

    # Restore original order
    all_results.sort(key=lambda x: x[0])
    descriptions = [desc for _, desc in all_results]

    # Placeholder labels for any frames beyond MAX_FRAMES_DESCRIBED
    if len(frame_paths) > MAX_FRAMES_DESCRIBED:
        descriptions += [
            f"(frame {MAX_FRAMES_DESCRIBED + i + 1} — not described)"
            for i in range(len(frame_paths) - MAX_FRAMES_DESCRIBED)
        ]

    return descriptions


def _summarise_chunks(descriptions: List[str], provider: str) -> List[str]:
    """
    Group frame descriptions into chunks of FRAME_CHUNK_SIZE and summarise
    each group with a focused LLM call, producing a nuanced narrative per chunk.
    Returns a list of paragraph summaries, one per chunk.
    Falls back to a joined description list on error.
    """
    from utils.llm_client import chat

    if not descriptions:
        return []

    summaries: list[str] = []
    total = len(descriptions)

    for start_idx in range(0, total, FRAME_CHUNK_SIZE):
        chunk = descriptions[start_idx: start_idx + FRAME_CHUNK_SIZE]
        start_label = start_idx + 1
        end_label = min(start_idx + FRAME_CHUNK_SIZE, total)
        numbered = "\n".join(f"{start_label + j}. {d}" for j, d in enumerate(chunk))

        try:
            summary = chat(
                provider=provider,
                messages=[{
                    "role": "user",
                    "content": CHUNK_SUMMARY_USER.format(
                        start=start_label,
                        end=end_label,
                        total=total,
                        descriptions=numbered,
                    ),
                }],
                system=CHUNK_SUMMARY_SYSTEM,
                max_tokens=4000,
            )
            summaries.append(f"**Frames {start_label}–{end_label}:** {summary.strip()}")
        except Exception as exc:
            logger.warning("Chunk summary failed for frames %d–%d: %s", start_label, end_label, exc)
            summaries.append(f"**Frames {start_label}–{end_label}:** " + "; ".join(chunk))

    return summaries


# ── Node 1: Transcribe ─────────────────────────────────────────────────────────

def transcribe_node(state: WorkflowState) -> dict:
    """Transcribe video audio using Whisper."""
    from agents.transcriber import transcribe_video
    logger.info("NODE: transcribe")

    transcript = transcribe_video(
        state["video_path"],
        provider=state.get("provider", ""),
        run_dir=state.get("run_dir"),
    )
    return {
        "transcript": transcript,
        "current_step": AGENT_FRAME_EXTRACTOR,
    }


# ── Node 2: Extract Frames ─────────────────────────────────────────────────────

def extract_frames_node(state: WorkflowState) -> dict:
    """Extract unique keyframes from video, describe them with vision, and chunk-summarise."""
    from agents.frame_extractor import extract_keyframes
    logger.info("NODE: extract_frames")

    frame_paths = extract_keyframes(
        state["video_path"],
        run_dir=state.get("run_dir"),
    )
    run_dir = state.get("run_dir")
    frame_descriptions = _describe_frames(frame_paths, state.get("provider", ""), run_dir=run_dir)
    frame_chunk_summaries = _summarise_chunks(frame_descriptions, state.get("provider", ""))

    # Persist both individual descriptions and chunk summaries to frame_extractor/output.md
    chunk_section = (
        "\n\n".join(frame_chunk_summaries)
        if frame_chunk_summaries else "_No frames extracted._"
    )
    desc_section = (
        "\n".join(f"{i + 1}. {d}" for i, d in enumerate(frame_descriptions))
        if frame_descriptions else "_No frame descriptions available._"
    )
    output_full = (
        "## Chunk Summaries (every 25 frames)\n\n"
        f"{chunk_section}\n\n"
        "---\n\n"
        "## Frame-by-Frame Descriptions\n\n"
        f"{desc_section}"
    )
    summary_line = (
        f"{len(frame_paths)} keyframes extracted, "
        f"{len(frame_chunk_summaries)} chunk "
        f"{'summary' if len(frame_chunk_summaries) == 1 else 'summaries'} generated"
    )
    write_agent_state(
        AGENT_FRAME_EXTRACTOR, STATUS_COMPLETE,
        output_full=output_full,
        output_summary=summary_line,
        run_dir=state.get("run_dir"),
    )

    return {
        "frame_paths": frame_paths,
        "frame_descriptions": frame_descriptions,
        "frame_chunk_summaries": frame_chunk_summaries,
        "current_step": AGENT_REQUIREMENTS,
    }


# ── Node 3: Requirements ───────────────────────────────────────────────────────

def requirements_node(state: WorkflowState) -> dict:
    """Distill transcript + frame chunk summaries into structured requirements."""
    from agents.requirements_agent import run_requirements_agent
    logger.info("NODE: requirements")

    requirements = run_requirements_agent(
        transcript=state["transcript"],
        frame_chunk_summaries=state.get("frame_chunk_summaries", []),
        provider=state.get("provider", ""),
        run_dir=state.get("run_dir"),
    )
    return {
        "requirements": requirements,
        "current_step": AGENT_RESEARCH,
    }


# ── Node 4: Research ───────────────────────────────────────────────────────────

def research_node(state: WorkflowState) -> dict:
    """Research recommendations (loops if alignment feedback present)."""
    from agents.research_agent import run_research_agent
    logger.info("NODE: research  (iteration=%d)", state["iteration_count"])

    research, sources, queries = run_research_agent(
        requirements=state["requirements"],
        alignment_notes=state.get("alignment_notes", ""),
        iteration=state["iteration_count"],
        provider=state.get("provider", ""),
        run_dir=state.get("run_dir"),
    )
    # Accumulate queries across loop iterations
    prev_queries = state.get("research_search_queries", [])
    return {
        "research": research,
        "research_sources": sources,
        "research_search_queries": prev_queries + queries,
        "current_step": AGENT_ALIGNMENT,
    }


# ── Node 5: Alignment ──────────────────────────────────────────────────────────

def alignment_node(state: WorkflowState) -> dict:
    """PM alignment check — routes to synthesis or back to research."""
    from agents.alignment_agent import run_alignment_agent
    logger.info("NODE: alignment  (iteration=%d)", state["iteration_count"])

    verdict_text, is_confident, notes = run_alignment_agent(
        requirements=state["requirements"],
        research=state["research"],
        iteration=state["iteration_count"],
        provider=state.get("provider", ""),
        run_dir=state.get("run_dir"),
    )

    prev_notes = state.get("alignment_notes", "")
    new_notes = (
        f"{prev_notes}\n\n--- Iteration {state['iteration_count'] + 1} feedback ---\n{notes}"
        if notes else prev_notes
    ).strip()

    return {
        "alignment_verdict": verdict_text,
        "alignment_confident": is_confident,
        "alignment_notes": new_notes,
        "iteration_count": state["iteration_count"] + 1,
        "current_step": AGENT_SYNTHESIS if is_confident else AGENT_RESEARCH,
    }


# ── Node 6: Synthesis ──────────────────────────────────────────────────────────

def synthesis_node(state: WorkflowState) -> dict:
    """Final synthesis — produce the polished recommendation report."""
    from agents.synthesis_agent import run_synthesis_agent
    logger.info("NODE: synthesis")

    synthesis = run_synthesis_agent(
        requirements=state["requirements"],
        research=state["research"],
        alignment_verdict=state["alignment_verdict"],
        transcript=state["transcript"],
        iteration=state["iteration_count"],
        provider=state.get("provider", ""),
        run_dir=state.get("run_dir"),
    )
    return {
        "synthesis": synthesis,
        "current_step": "complete",
    }
