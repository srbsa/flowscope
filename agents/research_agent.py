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

from tavily import TavilyClient

from utils.llm_client import get_client, get_model
from utils.state_manager import write_agent_state
from graph.state import (
    AGENT_RESEARCH,
    STATUS_RUNNING,
    STATUS_COMPLETE,
    STATUS_FAILED,
)

logger = logging.getLogger(__name__)

# Lazily initialised — so tests and imports don't fail when no key is set
_tavily: TavilyClient | None = None


def _get_tavily() -> TavilyClient:
    """Return a TavilyClient, raising clearly if the API key is absent."""
    global _tavily
    if _tavily is None:
        key = os.environ.get("TAVILY_API_KEY", "")
        if not key:
            raise ValueError(
                "TAVILY_API_KEY is not set. Set it in your .env file. "
                "Get a free key at https://tavily.com"
            )
        _tavily = TavilyClient(api_key=key)
    return _tavily

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior solutions architect and workflow automation expert. You have
been given a detailed workflow analysis identifying bottlenecks, automation
opportunities, and multiple solution strategies worth exploring.

Your job is to research ACROSS MULTIPLE SOLUTION PARADIGMS — not just find a
tool that matches features. For each bottleneck and opportunity, explore:

1. **Optimise Current Tool** — settings, plugins, integrations that fix the problem
2. **Replace with Alternative SaaS** — better-fit tools for this specific workflow
3. **No-Code Automation** — n8n, Zapier, Make, Power Automate workflows that
   connect existing tools and eliminate manual steps
4. **AI Agent / Custom Automation** — LangChain agents, custom GPT actions,
   AI-powered bots that can handle parts of the workflow autonomously
5. **Custom Internal Tool** — when nothing off-the-shelf fits, a lightweight
   custom build (estimate scope: days, weeks, or months)

Use the web_search function aggressively. Aim for 6–10 targeted searches. Be creative
with queries — look for case studies, automation recipes, n8n templates, AI agent
examples, not just product pages.

Output format:

## Executive Research Summary
3-5 sentences: what you found, key insight, strongest recommendation.

## Solution Landscape
For each identified bottleneck/opportunity from the workflow analysis, provide:

### [Bottleneck/Opportunity Name]
| Approach | Solution | Fit Score (1-5) | Effort | Time-to-Value | Cost Estimate | Source |
| Optimise Current | ... | ... | ... | ... | ... | ... |
| Alternative SaaS | ... | ... | ... | ... | ... | ... |
| No-Code Automation | ... | ... | ... | ... | ... | ... |
| AI Agent | ... | ... | ... | ... | ... | ... |
| Custom Build | ... | ... | ... | ... | ... | ... |

Include the BEST option per approach. Leave blank if no viable option exists.

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
    """Execute a Tavily search and return formatted results."""
    logger.info("Searching: %s", query)
    results = _get_tavily().search(query=query, max_results=max_results, search_depth="advanced")
    formatted = []
    for r in results.get("results", []):
        formatted.append(
            f"**{r.get('title', 'Untitled')}**\n"
            f"URL: {r.get('url', '')}\n"
            f"{r.get('content', '')[:400]}\n"
        )
    return "\n---\n".join(formatted) or "No results found."


MAX_TOOL_ROUNDS = int(os.environ.get("MAX_RESEARCH_TOOL_ROUNDS", "6"))


def _agentic_search_loop(
    messages: list,
    provider: str,
) -> tuple[str, list[str]]:
    """
    Run the tool-use loop until the LLM stops calling tools or hits MAX_TOOL_ROUNDS.
    Returns (final_text, all_source_urls).
    """
    client = get_client(provider)
    model = get_model(provider)
    sources: list[str] = []
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
                search_result = _run_search(
                    query=args.get("query", ""),
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

    return final_text, sources


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
        (research_text, sources_list)
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
        research, sources = _agentic_search_loop(messages, provider)

        summary = f"[Iteration {iteration+1}] " + (research.split("\n")[0][:100] if research else "Research complete")

        write_agent_state(
            AGENT_RESEARCH, STATUS_COMPLETE,
            output_full=research, output_summary=summary,
            iteration=iteration, run_dir=run_dir,
        )
        logger.info("Research agent complete (%d chars, %d sources)", len(research), len(sources))
        return research, sources

    except Exception as exc:
        error_msg = f"Research agent failed: {exc}"
        logger.exception(error_msg)
        write_agent_state(
            AGENT_RESEARCH, STATUS_FAILED,
            output_summary=error_msg, iteration=iteration, run_dir=run_dir,
        )
        raise RuntimeError(error_msg) from exc
