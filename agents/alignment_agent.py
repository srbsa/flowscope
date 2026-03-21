"""
agents/alignment_agent.py
Alignment PM agent — acts as a senior product manager who critically evaluates
whether the researcher's recommendations genuinely address the workflow requirements.

Outputs a structured verdict: "confident" or "not_confident", plus detailed notes
that feed back to the researcher on the next loop iteration.
"""

import logging
import re
from typing import Tuple

from utils.llm_client import chat
from utils.state_manager import write_agent_state
from graph.state import (
    AGENT_ALIGNMENT,
    STATUS_RUNNING,
    STATUS_COMPLETE,
    STATUS_FAILED,
)

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior product strategist reviewing whether a researcher's work is
genuinely useful for a business owner who wants to optimise their workflow.

You must evaluate BEYOND simple requirement matching. Ask yourself:

1. **Creative breadth**: Were multiple solution paradigms explored (optimise existing,
   replace, automate via n8n/Zapier, build AI agent, custom tool)? Or did the
   researcher just find similar SaaS tools?
2. **Business context**: Do recommendations connect to the actual business goals
   and bottlenecks — or are they generic product comparisons?
3. **Actionability**: Could a business owner or their team act on these
   recommendations THIS WEEK? Are there specific tool names, costs, steps?
4. **Specificity**: Are recommendations tailored to THIS workflow, or could they
   apply to any business?
5. **Quick wins identified**: Are there immediate low-effort improvements?
6. **ROI reasoning**: Is there any cost/benefit thinking, not just feature matching?

Respond in this exact format:

## Alignment Verdict
VERDICT: confident   ← or: VERDICT: not_confident

## Confidence Score
Score: X/10  (where 8+ = confident)

## Creative Breadth Assessment
Were multiple solution paradigms explored? Which were missing?

## Business Alignment
Do recommendations connect to actual business goals and bottlenecks?

## Actionability Check
Could a team act on these recommendations immediately?

## Gaps & Concerns
- ...

## Feedback for Researcher (only if not_confident)
Specific, actionable instructions for the next research iteration:
- ...

## Summary
One paragraph explanation of the verdict.
"""

USER_PROMPT_TEMPLATE = """\
## Workflow Analysis (bottlenecks, opportunities, strategies)
{requirements}

## Researcher's Recommendations
{research}

Evaluate alignment and provide your verdict.
"""

_VERDICT_RE = re.compile(r"VERDICT:\s*(confident|not_confident)", re.IGNORECASE)


def run_alignment_agent(
    requirements: str,
    research: str,
    iteration: int = 0,
    provider: str = "",
    run_dir: str | None = None,
) -> Tuple[str, bool, str]:
    """
    Evaluate alignment between requirements and research recommendations.

    Args:
        requirements: Structured requirements doc.
        research:     Researcher's recommendations.
        iteration:    Current loop iteration count.
        provider:     LLM provider.
        run_dir:      Per-run output directory.

    Returns:
        (verdict_text, is_confident, notes_for_researcher)
    """
    write_agent_state(
        AGENT_ALIGNMENT, STATUS_RUNNING,
        output_summary=f"Evaluating alignment… (iteration {iteration + 1})",
        iteration=iteration, run_dir=run_dir,
    )

    user_msg = USER_PROMPT_TEMPLATE.format(
        requirements=requirements,
        research=research,
    )

    try:
        verdict_text = chat(
            provider=provider,
            messages=[{"role": "user", "content": user_msg}],
            system=SYSTEM_PROMPT,
            max_tokens=8000,
        )

        match = _VERDICT_RE.search(verdict_text)
        is_confident = bool(match and match.group(1).lower() == "confident")

        notes = _extract_feedback_section(verdict_text) if not is_confident else ""

        summary = f"[Iter {iteration+1}] {'✓ CONFIDENT' if is_confident else '✗ NOT CONFIDENT'}"
        write_agent_state(
            AGENT_ALIGNMENT, STATUS_COMPLETE,
            output_full=verdict_text, output_summary=summary,
            iteration=iteration, run_dir=run_dir,
        )

        logger.info("Alignment verdict: %s  (iteration %d)", "confident" if is_confident else "not_confident", iteration)
        return verdict_text, is_confident, notes

    except Exception as exc:
        error_msg = f"Alignment agent failed: {exc}"
        logger.exception(error_msg)
        write_agent_state(
            AGENT_ALIGNMENT, STATUS_FAILED,
            output_summary=error_msg, iteration=iteration, run_dir=run_dir,
        )
        raise RuntimeError(error_msg) from exc


def _extract_feedback_section(verdict_text: str) -> str:
    """Extract the '## Feedback for Researcher' section from the verdict text."""
    lines = verdict_text.splitlines()
    capturing = False
    feedback_lines: list[str] = []

    for line in lines:
        if line.strip().startswith("## Feedback for Researcher"):
            capturing = True
            continue
        if capturing:
            if line.startswith("## ") and feedback_lines:
                break
            feedback_lines.append(line)

    return "\n".join(feedback_lines).strip()
