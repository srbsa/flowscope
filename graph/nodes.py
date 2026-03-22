"""
graph/nodes.py
LangGraph node functions. Each node wraps an agent and updates WorkflowState.
Node signature: (state: WorkflowState) -> dict  (return only changed keys)
"""

import base64
import logging
import re
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
FRAME_CHUNK_SIZE = 25

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


def _describe_frames(frame_paths: List[str], provider: str) -> List[str]:
    """
    Describe each keyframe using vision, chaining previous descriptions
    so the model understands sequential context.
    Returns a list of one-line descriptions (one per frame).
    Falls back to filename-based labels on error.
    """
    from utils.llm_client import get_client, get_model

    if not frame_paths:
        return []

    client = get_client(provider)
    model = get_model(provider, vision=True)
    descriptions: list[str] = []

    for i, path in enumerate(frame_paths[:MAX_FRAMES_DESCRIBED]):
        try:
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            # Build the text prompt (chain previous frame context)
            if i == 0 or not descriptions:
                text_prompt = FRAME_VISION_FIRST
            else:
                text_prompt = FRAME_VISION_CHAIN.format(prev=descriptions[-1])

            response = client.chat.completions.create(
                model=model,
                max_tokens=150,
                messages=[
                    {"role": "system", "content": FRAME_VISION_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}",
                                },
                            },
                            {"type": "text", "text": text_prompt},
                        ],
                    },
                ],
            )
            desc = (response.choices[0].message.content or "").strip()
            # Strip thinking-model tags
            desc = re.sub(r"<think>.*?</think>", "", desc, flags=re.DOTALL).strip()
            descriptions.append(desc)
        except Exception as exc:
            logger.warning("Frame description failed for %s: %s", path, exc)
            descriptions.append(f"(frame: {path.split('/')[-1]})")

    if len(frame_paths) > MAX_FRAMES_DESCRIBED:
        descriptions += [f"(frame {i+MAX_FRAMES_DESCRIBED+1})" for i in range(len(frame_paths) - MAX_FRAMES_DESCRIBED)]

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
                max_tokens=400,
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
    frame_descriptions = _describe_frames(frame_paths, state.get("provider", ""))
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

    research, sources = run_research_agent(
        requirements=state["requirements"],
        alignment_notes=state.get("alignment_notes", ""),
        iteration=state["iteration_count"],
        provider=state.get("provider", ""),
        run_dir=state.get("run_dir"),
    )
    return {
        "research": research,
        "research_sources": sources,
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
