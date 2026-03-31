"""
agents/synthesis_agent.py
Final synthesis agent — produces a polished, executive-ready recommendation
document by combining all prior pipeline outputs into a unified deliverable.
"""

import logging
import re

from utils.llm_client import chat
from utils.search import run_search
from utils.state_manager import write_agent_state
from graph.state import (
    AGENT_SYNTHESIS,
    STATUS_RUNNING,
    STATUS_COMPLETE,
    STATUS_FAILED,
    SYNTHESIS_SEARCH_ENABLED,
    MAX_SYNTHESIS_SEARCH_ROUNDS,
)

logger = logging.getLogger(__name__)

_SYNTHESIS_SEARCH_ENABLED = SYNTHESIS_SEARCH_ENABLED
_MAX_SYNTHESIS_SEARCH_ROUNDS = MAX_SYNTHESIS_SEARCH_ROUNDS

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
Present the most relevant and genuinely viable solution paths for THIS specific
workflow. Name each path descriptively (e.g. "Automate with n8n", "Migrate to
Linear", "Build a custom intake agent") — not generically as Path A/B/C/D.
Include ONLY paths where you have specific, actionable recommendations. For each:

### [Descriptive Path Name]
What it involves: specific tools, steps, or approaches.
Estimated cost and timeline. Key trade-offs and risks.

## Decision Matrix
Compare paths across the dimensions most relevant to this workflow. Adapt columns
to what matters (typically cost, time-to-value, impact, risk, maintenance burden).

| Criteria | [Path 1 Name] | [Path 2 Name] | [Path N Name] |
| ... | ... | ... | ... |

## Recommended Path & Quick Wins
Your #1 recommendation with rationale. Then list 3-5 things they can do THIS WEEK.

## Implementation Roadmap
| Phase | Timeline | Actions | Expected Outcome |
Phases should be: "This Week", "This Month", "This Quarter"

## What We'd Need From You
2-3 questions for the business owner that would refine recommendations further.
"""

EVIDENCE_QUERY_PROMPT = """\
You are reviewing a workflow optimisation report that recommends specific solution paths.
Generate up to {max_rounds} targeted web search queries to find real-world evidence
(case studies, verified pricing, integration guides) that strengthens the top recommended paths.

Rules:
- Focus on the HIGHEST-IMPACT paths in the report
- Each query must be 5–10 words, highly specific
- Output ONLY the queries, one per line, prefixed with "QUERY: "
- Prioritise case studies and pricing verification over generic product info

## Report Summary (solution paths section)
{paths_excerpt}
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

        # Optional: enrich decision matrix with web-sourced evidence
        if _SYNTHESIS_SEARCH_ENABLED and _MAX_SYNTHESIS_SEARCH_ROUNDS > 0:
            synthesis = _enrich_with_evidence(synthesis, provider)

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


def _enrich_with_evidence(synthesis: str, provider: str) -> str:
    """
    Optionally append a 'Supporting Evidence' section to the final report by
    running a small number of targeted searches for case studies and pricing.
    Only called when SYNTHESIS_SEARCH_ENABLED=true. Falls back silently on error.
    """
    # Extract the solution paths section as context for query generation
    paths_excerpt = _extract_section(synthesis, "## Solution Paths", max_chars=1200)
    if not paths_excerpt:
        return synthesis

    try:
        query_response = chat(
            provider=provider,
            messages=[{"role": "user", "content": EVIDENCE_QUERY_PROMPT.format(
                max_rounds=_MAX_SYNTHESIS_SEARCH_ROUNDS,
                paths_excerpt=paths_excerpt,
            )}],
            max_tokens=200,
        )

        queries = [
            line.replace("QUERY: ", "", 1).strip()
            for line in query_response.splitlines()
            if line.strip().startswith("QUERY: ")
        ][:_MAX_SYNTHESIS_SEARCH_ROUNDS]

        if not queries:
            return synthesis

        evidence_parts: list[str] = []
        for query in queries:
            try:
                text, sources = run_search(query, max_results=3)
                source_list = "\n".join(f"- {s}" for s in sources[:3])
                evidence_parts.append(
                    f"**{query}**\n{text[:500]}\n\n*Sources:* {source_list}"
                )
            except Exception as exc:
                logger.warning("Evidence search failed for '%s': %s", query, exc)

        if not evidence_parts:
            return synthesis

        appendix = (
            "\n\n---\n\n"
            "## Supporting Evidence\n"
            "_Post-research evidence gathered to substantiate the recommended paths._\n\n"
            + "\n\n".join(evidence_parts)
        )
        return synthesis + appendix

    except Exception as exc:
        logger.warning("Evidence enrichment failed: %s", exc)
        return synthesis


def _extract_section(text: str, header: str, max_chars: int = 1200) -> str:
    """Extract content from a markdown section header up to max_chars."""
    idx = text.find(header)
    if idx == -1:
        return ""
    excerpt = text[idx: idx + max_chars]
    # Trim at next top-level header to avoid spill
    next_header = re.search(r"\n## ", excerpt[len(header):])
    if next_header:
        return excerpt[: len(header) + next_header.start()]
    return excerpt
