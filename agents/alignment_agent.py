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
from utils.search import run_search
from utils.state_manager import write_agent_state
from graph.state import (
    AGENT_ALIGNMENT,
    STATUS_RUNNING,
    STATUS_COMPLETE,
    STATUS_FAILED,
    MAX_ALIGNMENT_SEARCH_ROUNDS,
)

logger = logging.getLogger(__name__)

_MAX_SPOT_CHECK_ROUNDS = MAX_ALIGNMENT_SEARCH_ROUNDS
_SPOT_CHECK_SCORE_THRESHOLD = 7  # Spot-check triggered when score < this

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior product strategist reviewing whether a researcher's work is
genuinely useful for a business owner who wants to optimise their workflow.

You must evaluate BEYOND simple requirement matching. Ask yourself:

1. **Creative breadth**: Were multiple solution approaches explored where they
   make sense for this specific workflow? Did the researcher avoid tunnel-visioning
   on a single type of solution (e.g. only SaaS comparisons)? If certain approaches
   were omitted, is the reasoning sound for this workflow?
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
Which solution approaches were explored? Were the right ones chosen for this workflow?
If any were skipped, was the omission justified?

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
_SCORE_RE = re.compile(r"Score:\s*(\d+)/10", re.IGNORECASE)

# ── Spot-check prompts ──────────────────────────────────────────────────────────

SPOT_CHECK_QUERY_PROMPT = """\
Based on the gaps and concerns identified below, generate up to {max_rounds} highly
specific web search queries to fact-check the most important verifiable claims
(tool pricing, feature availability, integration support, real-world viability).

Rules:
- Only query SPECIFIC, VERIFIABLE facts — not subjective assessments
- Each query must be 5–10 words and highly targeted
- Output ONLY the queries, one per line, prefixed with "QUERY: "
- Fewer is better: only generate a query if it would meaningfully resolve a concern

## Gaps & Concerns
{gaps}
"""


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
        score = _parse_confidence_score(verdict_text)

        notes = _extract_feedback_section(verdict_text) if not is_confident else ""

        # Spot-check: verify factual claims when score is low
        if score < _SPOT_CHECK_SCORE_THRESHOLD and _MAX_SPOT_CHECK_ROUNDS > 0:
            spot_evidence = _run_spot_check(
                verdict_text=verdict_text,
                provider=provider,
            )
            if spot_evidence:
                notes = f"{notes}\n\n{spot_evidence}".strip()
                verdict_text = f"{verdict_text}\n\n{spot_evidence}"
                logger.info("Spot-check evidence appended to alignment notes")

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


def _extract_gaps_section(verdict_text: str) -> str:
    """Extract the '## Gaps & Concerns' section from the verdict text."""
    lines = verdict_text.splitlines()
    capturing = False
    gap_lines: list[str] = []

    for line in lines:
        if line.strip().startswith("## Gaps & Concerns"):
            capturing = True
            continue
        if capturing:
            if line.startswith("## ") and gap_lines:
                break
            gap_lines.append(line)

    return "\n".join(gap_lines).strip()


def _parse_confidence_score(verdict_text: str) -> int:
    """Parse the numeric confidence score from the verdict text. Returns 10 if not found."""
    match = _SCORE_RE.search(verdict_text)
    return int(match.group(1)) if match else 10


def _run_spot_check(verdict_text: str, provider: str) -> str:
    """
    Generate targeted search queries from the identified gaps and run them.
    Returns a formatted '## Spot-check Evidence' section string, or empty string on failure.
    """
    gaps = _extract_gaps_section(verdict_text)
    if not gaps or gaps == "-":
        return ""

    try:
        query_response = chat(
            provider=provider,
            messages=[{"role": "user", "content": SPOT_CHECK_QUERY_PROMPT.format(
                max_rounds=_MAX_SPOT_CHECK_ROUNDS,
                gaps=gaps,
            )}],
            max_tokens=200,
        )

        queries = [
            line.replace("QUERY: ", "", 1).strip()
            for line in query_response.splitlines()
            if line.strip().startswith("QUERY: ")
        ][:_MAX_SPOT_CHECK_ROUNDS]

        if not queries:
            return ""

        evidence_parts: list[str] = []
        for query in queries:
            try:
                text, sources = run_search(query, max_results=3)
                source_list = "  ".join(f"- {s}" for s in sources[:3])
                evidence_parts.append(
                    f"**Query:** {query}\n"
                    f"**Findings:** {text[:500]}\n"
                    f"**Sources:**\n{source_list}"
                )
            except Exception as exc:
                logger.warning("Spot-check search failed for '%s': %s", query, exc)

        if not evidence_parts:
            return ""

        return (
            "## Spot-check Evidence\n"
            "_Fact-checked against web sources to ground the feedback above._\n\n"
            + "\n\n---\n\n".join(evidence_parts)
        )

    except Exception as exc:
        logger.warning("Spot-check query generation failed: %s", exc)
        return ""
