"""
agents/requirements_agent.py
Distills the raw transcript and frame descriptions into a clean, structured
workflow requirements document using the configured LLM provider.
"""

import logging
from typing import List

from utils.llm_client import chat
from utils.state_manager import write_agent_state
from graph.state import (
    AGENT_REQUIREMENTS,
    STATUS_RUNNING,
    STATUS_COMPLETE,
    STATUS_FAILED,
)

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior workflow optimisation consultant. A business owner has recorded
a video walkthrough of one of their operational workflows. Your job is to deeply
analyse that workflow and produce a structured analysis that will drive creative,
high-impact recommendations.

You are NOT writing a feature spec for the tool shown in the video. You ARE
identifying the business goals, workflow bottlenecks, and optimisation opportunities.

Output format (use these exact headers):

## Business Context
Infer: what type of business is this? What role does this workflow serve?
What business outcomes depend on this workflow being done well?

## Workflow Map
For each step observed in the video, produce a numbered list:
- **Step N: [action]** | Actor: ... | Tool/System: ... | Estimated effort: ...
  - What happens: ...
  - Friction observed: ... (or "none")
  - Dependency: which previous step feeds this one?

## Bottleneck Analysis
For each friction point or bottleneck identified, score it:
| Bottleneck | Severity (1-5) | Frequency | Impact on Business Goal | Root Cause |
And explain WHY it matters to the business (not just that it exists).

## Automation Opportunity Map
For each workflow step, assess:
| Step | Automation Potential (High/Med/Low) | Current State | Possible Approach |
Where "Possible Approach" can include: AI agent, no-code automation (n8n/Zapier/Make),
tool integration, custom internal tool, process redesign, or "keep manual".

## Business Goals & Success Metrics
What is this workflow ultimately trying to achieve? What metrics would indicate
the workflow is performing well? (e.g., cycle time, accuracy, throughput, cost)

## Solution Strategy Framework
Propose 3-5 distinct optimisation strategies worth exploring. For each:
- **Strategy name** (e.g., "Automate via n8n", "Replace with [tool]", "Build custom agent")
- Why it might work for THIS specific business/workflow
- Key research questions to answer before committing
- Risk level (Low/Med/High)

Do NOT just list features of the tool shown in the video. Think like a consultant
who needs to 10x this workflow's efficiency.
"""

USER_PROMPT_TEMPLATE = """\
## Transcript of Workflow Video
{transcript}

## Visual Frame Summaries (chunked every 25 frames)
{frame_summaries}

Analyse this workflow deeply. Focus on business goals, bottlenecks, and
optimisation opportunities — not on describing the tool's features.
"""


def run_requirements_agent(
    transcript: str,
    frame_chunk_summaries: List[str],
    provider: str = "",
    run_dir: str | None = None,
) -> str:
    """
    Streamline the raw transcript + frame chunk summaries into a requirements doc.

    Args:
        transcript:           Full Whisper transcript text.
        frame_chunk_summaries: Narrative summaries per 25-frame chunk from frame extraction.
        provider:             LLM provider ('lm_studio' or 'openai').
        run_dir:              Per-run output directory.

    Returns:
        Structured requirements document as a string.
    """
    write_agent_state(
        AGENT_REQUIREMENTS, STATUS_RUNNING,
        output_summary="Distilling requirements…", run_dir=run_dir,
    )

    frame_summaries_text = (
        "\n\n".join(frame_chunk_summaries)
        if frame_chunk_summaries else "No visual context available."
    )

    user_msg = USER_PROMPT_TEMPLATE.format(
        transcript=transcript or "(no transcript)",
        frame_summaries=frame_summaries_text,
    )

    try:
        logger.info("Running requirements agent (provider=%s)", provider)
        requirements = chat(
            provider=provider,
            messages=[{"role": "user", "content": user_msg}],
            system=SYSTEM_PROMPT,
            max_tokens=10000,
        )

        summary_line = requirements.split("\n")[0][:120]
        write_agent_state(
            AGENT_REQUIREMENTS, STATUS_COMPLETE,
            output_full=requirements, output_summary=summary_line,
            run_dir=run_dir,
        )
        logger.info("Requirements agent complete (%d chars)", len(requirements))
        return requirements

    except Exception as exc:
        error_msg = f"Requirements agent failed: {exc}"
        logger.exception(error_msg)
        write_agent_state(
            AGENT_REQUIREMENTS, STATUS_FAILED,
            output_summary=error_msg, run_dir=run_dir,
        )
        raise RuntimeError(error_msg) from exc
