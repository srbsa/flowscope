"""
agents/synthesis_agent.py
Final synthesis agent — produces a polished, executive-ready recommendation
document by combining all prior pipeline outputs into a unified deliverable.
"""

import logging

from utils.llm_client import chat
from utils.state_manager import write_agent_state
from graph.state import (
    AGENT_SYNTHESIS,
    STATUS_RUNNING,
    STATUS_COMPLETE,
    STATUS_FAILED,
)

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a principal workflow optimisation consultant writing a final
recommendations report for a business owner who showed you a video of their
operational workflow. They want concrete, creative, actionable ways to make
this workflow dramatically more efficient.

Your report MUST present MULTIPLE SOLUTION PATHS — the business owner should be
able to choose based on their budget, timeline, and risk appetite.

Rules:
- Be specific: name actual tools, give cost estimates, outline exact steps
- Be creative: don't just find a similar SaaS tool — consider automation (n8n,
  Zapier, Make), AI agents, custom builds, process redesign
- Be honest: acknowledge trade-offs and risks for each path
- Be concise: no filler, no hedging, every sentence must add value
- Write for a non-technical business owner, but include enough detail for
  their technical team to act immediately

Output structure:

# Workflow Optimisation Report

## Executive Summary
5 sentences max: what you observed, the core problem, the #1 recommendation,
expected impact, and recommended next step.

## Workflow as Observed
Narrative description of the workflow from the video. Focus on what the person
is TRYING TO ACCOMPLISH (business goal), not tool features.

## Key Bottlenecks & Opportunities
For each major bottleneck, one paragraph: what it is, why it matters, how much
it costs the business (time/money/quality).

## Solution Paths

### Path A: Optimise Current Setup
What can be done with the existing tool/process. Quick wins, configuration
changes, integrations. Estimated cost and timeline.

### Path B: Automate with No-Code (n8n / Zapier / Make)
Specific automation workflows that eliminate manual steps. Describe triggers,
actions, and connections. Estimated cost and timeline.

### Path C: AI-Powered Solution
Where AI agents, custom GPT actions, or LLM-powered automation could handle
parts of this workflow. What would the agent do? What's the build effort?

### Path D: Replace / Rebuild
If the current tool is fundamentally limiting, what's the alternative?
SaaS replacement or custom internal tool. Cost, timeline, migration risk.

(Include only paths that are genuinely viable — skip any that don't apply.)

## Decision Matrix
| Criteria | Path A | Path B | Path C | Path D |
| Cost | ... | ... | ... | ... |
| Time to Value | ... | ... | ... | ... |
| Impact (1-5) | ... | ... | ... | ... |
| Risk (1-5) | ... | ... | ... | ... |
| Maintenance | ... | ... | ... | ... |

## Recommended Path & Quick Wins
Your #1 recommendation with rationale. Then list 3-5 things they can do THIS WEEK.

## Implementation Roadmap
| Phase | Timeline | Actions | Expected Outcome |
Phases should be: "This Week", "This Month", "This Quarter"

## What We'd Need From You
2-3 questions for the business owner that would refine recommendations further.
"""

USER_PROMPT_TEMPLATE = """\
## Workflow Analysis
{requirements}

## Research & Recommendations
{research}

## Alignment Assessment
{alignment_verdict}

## Original Transcript (first 1500 chars)
{transcript_preview}

Synthesize into the final multi-path recommendation report. Be specific, creative, and actionable.
"""


def run_synthesis_agent(
    requirements: str,
    research: str,
    alignment_verdict: str,
    transcript: str,
    iteration: int = 0,
    provider: str = "",
    run_dir: str | None = None,
) -> str:
    """
    Generate the final synthesis report.

    Args:
        requirements:     Structured requirements doc.
        research:         Researcher's approved recommendations.
        alignment_verdict: Full alignment review text.
        transcript:       Original video transcript.
        iteration:        Final loop iteration count (for state file).
        provider:         LLM provider.
        run_dir:          Per-run output directory.

    Returns:
        Final synthesis report as a string.
    """
    write_agent_state(
        AGENT_SYNTHESIS, STATUS_RUNNING,
        output_summary="Synthesizing final report…",
        iteration=iteration, run_dir=run_dir,
    )

    user_msg = USER_PROMPT_TEMPLATE.format(
        requirements=requirements,
        research=research,
        alignment_verdict=alignment_verdict,
        transcript_preview=transcript[:1500] + ("…" if len(transcript) > 1500 else ""),
    )

    try:
        synthesis = chat(
            provider=provider,
            messages=[{"role": "user", "content": user_msg}],
            system=SYSTEM_PROMPT,
            max_tokens=10000,
            thinking=False,
        )

        summary_line = synthesis.split("\n")[2][:120] if "\n\n" in synthesis else synthesis[:120]
        write_agent_state(
            AGENT_SYNTHESIS, STATUS_COMPLETE,
            output_full=synthesis, output_summary=summary_line,
            iteration=iteration, run_dir=run_dir,
        )
        logger.info("Synthesis complete (%d chars)", len(synthesis))
        return synthesis

    except Exception as exc:
        error_msg = f"Synthesis agent failed: {exc}"
        logger.exception(error_msg)
        write_agent_state(
            AGENT_SYNTHESIS, STATUS_FAILED,
            output_summary=error_msg, iteration=iteration, run_dir=run_dir,
        )
        raise RuntimeError(error_msg) from exc
