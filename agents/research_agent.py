"""
agents/research_agent.py
Researcher agent that uses Tavily web search to find tools, frameworks, and
best practices aligned with the workflow requirements.

On alignment loop iterations, it receives notes from the Alignment PM and
sharpens its recommendations accordingly.
"""

import json
import logging
import os
import re
from typing import List

from utils.llm_client import get_client, get_model
from utils.search import run_search
from utils.state_manager import write_agent_state
from graph.state import (
    AGENT_RESEARCH,
    STATUS_RUNNING,
    STATUS_COMPLETE,
    STATUS_FAILED,
)

logger = logging.getLogger(__name__)

# Lazily initialised Tavily client lives in utils/search.py

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior solutions architect and workflow automation expert. You have
been given a detailed workflow analysis identifying bottlenecks, automation
opportunities, and multiple solution strategies worth exploring.

Your job is to research ACROSS MULTIPLE SOLUTION APPROACHES — not just find a
tool that matches features. For each bottleneck and opportunity, explore the
approaches that are genuinely relevant to THIS specific workflow.

Common solution types to consider (use only those that fit — skip or adapt as needed):
1. **Optimise Current Tool** — settings, plugins, integrations that fix the problem
2. **Replace with Alternative SaaS** — better-fit tools for this specific workflow
3. **No-Code Automation** — n8n, Zapier, Make, Power Automate workflows that
   connect existing tools and eliminate manual steps
4. **AI Agent / Custom Automation** — LangChain agents, custom GPT actions,
   AI-powered bots that can handle parts of the workflow autonomously
5. **Custom Internal Tool** — when nothing off-the-shelf fits, a lightweight
   custom build (estimate scope: days, weeks, or months)

You are NOT required to cover all five. Include only approaches that are genuinely
valuable for this specific workflow. If a better framing exists for certain
bottlenecks — e.g. "Process Redesign", "Vendor Consolidation", "Training & SOPs" —
use it. Depth and relevance beats exhaustive paradigm coverage.

Use the web_search function aggressively. Aim for 6–10 targeted searches. Be creative
with queries — look for case studies, automation recipes, n8n templates, AI agent
examples, not just product pages.

Output format:

## Executive Research Summary
3-5 sentences: what you found, key insight, strongest recommendation.

## Solution Landscape
For each identified bottleneck/opportunity from the workflow analysis, provide:

### [Bottleneck/Opportunity Name]
For each viable solution approach, include a row. Omit approaches that genuinely
don't apply — a brief note on why is helpful where the omission might surprise.

| Approach | Solution | Fit Score (1-5) | Effort | Time-to-Value | Cost Estimate | Source |
| [Approach Name] | ... | ... | ... | ... | ... | ... |

Add trade-off notes after the table where they add insight.

## Quick Wins (implement in < 1 week)
Numbered list of immediate improvements with specific steps.

## Strategic Recommendations
The top 3 highest-impact recommendations with:
- What to do (specific tool/approach name)
- Why (business case, ROI estimate where possible)
- How (implementation steps, 3-5 bullets)
- Risk/dependency

## n8n / Automation Recipes
If applicable, describe specific automation workflows:
- Trigger → Action → Action pattern
- Which tools/APIs to connect
- Expected time savings

## Open Questions for Stakeholder
What decisions or context from the business owner would change recommendations?

## Pre-Finalisation Self-Check
Before returning your output, score yourself on each of these criteria (the reviewing
strategist will check all six). If any score < 7/10, run one more targeted web_search
to fill the gap before finalising:
1. **Approach breadth** — multiple relevant solution types explored (not just SaaS comparisons)
2. **Business grounding** — every recommendation connects to a specific bottleneck or goal
3. **Immediate actionability** — a team could start THIS WEEK (specific tool names, exact costs, steps)
4. **Workflow specificity** — recommendations are tailored to THIS workflow, not generic advice
5. **Quick wins** — at least 2–3 improvements achievable in < 1 week are explicitly listed
6. **ROI reasoning** — cost/benefit or time-saving estimates are present, not just feature lists
"""

USER_PROMPT_TEMPLATE = """\
## Workflow Analysis
{requirements}

{alignment_notes_section}

Research across ALL solution paradigms (optimise, replace, automate, AI agent, custom build).
Use web_search for EACH major bottleneck. Be creative — look for n8n templates,
AI agent examples, automation case studies, not just product feature pages.
"""

ALIGNMENT_NOTES_SECTION = """\
## Alignment Feedback (from previous iteration — MUST address these)
{notes}
"""

# ── OpenAI-format tool spec for web_search ─────────────────────────────────────

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current tools, frameworks, best practices, and case studies.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (5–10 words, specific and targeted)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                },
            },
            "required": ["query"],
        },
    },
}


def _run_search(query: str, max_results: int = 5) -> str:
    """Execute a search and return only the formatted text (for tool-use loop compatibility)."""
    text, _ = run_search(query, max_results)
    return text


MAX_TOOL_ROUNDS = int(os.environ.get("MAX_RESEARCH_TOOL_ROUNDS", "6"))


def _agentic_search_loop(
    messages: list,
    provider: str,
) -> tuple[str, list[str], list[str]]:
    """
    Run the tool-use loop until the LLM stops calling tools or hits MAX_TOOL_ROUNDS.
    Returns (final_text, all_source_urls, all_search_queries).
    """
    client = get_client(provider)
    model = get_model(provider)
    sources: list[str] = []
    queries: list[str] = []
    final_text = ""
    rounds = 0

    while rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            max_tokens=16000,
            temperature=0.7,
            tools=[WEB_SEARCH_TOOL],
        )

        choice = response.choices[0]

        # Thinking-model fallback: use reasoning_content when content is empty
        msg_content = choice.message.content or getattr(choice.message, "reasoning_content", None) or ""
        # Strip thinking-model tags
        msg_content = re.sub(r"<think>.*?</think>", "", msg_content, flags=re.DOTALL).strip()
        if msg_content:
            final_text = msg_content

        if not choice.message.tool_calls:
            break

        # Append assistant message with tool calls
        messages.append(choice.message)

        for tc in choice.message.tool_calls:
            if tc.function.name == "web_search":
                args = json.loads(tc.function.arguments)
                query = args.get("query", "")
                queries.append(query)
                search_result = _run_search(
                    query=query,
                    max_results=args.get("max_results", 5),
                )
                for line in search_result.splitlines():
                    if line.startswith("URL: "):
                        sources.append(line[5:].strip())

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": search_result,
                })

    # If we hit the round limit and still don't have a final text, force a
    # final completion WITHOUT tools so the model produces its synthesis.
    if not final_text and messages:
        logger.info("Max tool rounds reached (%d) — forcing final synthesis", MAX_TOOL_ROUNDS)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages + [
                {"role": "user", "content": "You have completed your research. Now synthesize all findings into the final output format. Do NOT call any more tools."}
            ],
            max_tokens=16000,
            temperature=0.7,
        )
        msg_content = response.choices[0].message.content or getattr(response.choices[0].message, "reasoning_content", None) or ""
        msg_content = re.sub(r"<think>.*?</think>", "", msg_content, flags=re.DOTALL).strip()
        if msg_content:
            final_text = msg_content

    return final_text, sources, queries


def run_research_agent(
    requirements: str,
    alignment_notes: str = "",
    iteration: int = 0,
    provider: str = "",
    run_dir: str | None = None,
) -> tuple[str, list[str]]:
    """
    Run the researcher agent with optional alignment feedback.

    Args:
        requirements:    Structured requirements doc from requirements agent.
        alignment_notes: Feedback from previous alignment check (may be empty).
        iteration:       Current loop iteration count.
        provider:        LLM provider.
        run_dir:         Per-run output directory.

    Returns:
        (research_text, sources_list, search_queries)
    """
    write_agent_state(
        AGENT_RESEARCH, STATUS_RUNNING,
        output_summary=f"Researching… (iteration {iteration + 1})",
        iteration=iteration, run_dir=run_dir,
    )

    notes_section = (
        ALIGNMENT_NOTES_SECTION.format(notes=alignment_notes)
        if alignment_notes.strip()
        else ""
    )

    user_msg = USER_PROMPT_TEMPLATE.format(
        requirements=requirements,
        alignment_notes_section=notes_section,
    )

    messages = [{"role": "user", "content": user_msg}]

    try:
        research, sources, queries = _agentic_search_loop(messages, provider)

        summary = f"[Iteration {iteration+1}] " + (research.split("\n")[0][:100] if research else "Research complete")

        write_agent_state(
            AGENT_RESEARCH, STATUS_COMPLETE,
            output_full=research, output_summary=summary,
            iteration=iteration, run_dir=run_dir,
        )
        logger.info("Research agent complete (%d chars, %d sources, %d searches)", len(research), len(sources), len(queries))
        return research, sources, queries

    except Exception as exc:
        error_msg = f"Research agent failed: {exc}"
        logger.exception(error_msg)
        write_agent_state(
            AGENT_RESEARCH, STATUS_FAILED,
            output_summary=error_msg, iteration=iteration, run_dir=run_dir,
        )
        raise RuntimeError(error_msg) from exc
