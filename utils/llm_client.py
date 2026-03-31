"""
utils/llm_client.py
Unified LLM client factory for LM Studio (local) and OpenAI providers.

Both providers use the OpenAI SDK — LM Studio exposes an OpenAI-compatible API.
This is the single module that owns provider configuration.

OpenAI provider uses the Responses API (client.responses.create) — the primary
API for gpt-5+ models. LM Studio uses the Chat Completions API since local
servers do not yet implement the Responses endpoint.
"""

import logging
import os
import re

from openai import OpenAI

logger = logging.getLogger(__name__)


# ── Thinking-model helpers ─────────────────────────────────────────────────────

# Patterns that indicate meta-reasoning rather than actual answer content.
_META_PATTERNS = re.compile(
    r"^(The user wants|Wait,|Actually,|Looking at|Thinking Process|Let me |Hmm,|"
    r"I need to |So,? (?:the|I)|OK,? |However,? looking)",
    re.IGNORECASE,
)


def _extract_conclusion(reasoning: str) -> str:
    """Best-effort extraction of the actual answer from raw reasoning_content.

    When a thinking-model exhausts max_tokens during reasoning it never writes
    ``content``, so the fallback returns ``reasoning_content``.  The tail end
    of that text is most likely to contain the usable answer.  This function:

    1. Splits the text by double-newline into paragraphs.
    2. Walks backwards, keeping paragraphs that look like real content.
    3. Returns the last substantial block (>120 chars) that does not start
       with common meta-reasoning patterns.

    Falls back to the original text (stripped) if nothing better is found.
    """
    paragraphs = [p.strip() for p in reasoning.split("\n\n") if p.strip()]
    if not paragraphs:
        return reasoning.strip()

    # Walk from end: once we find a substantial paragraph that looks like
    # real content (not meta), collect it and any contiguous real content
    # preceding it.
    conclusion_parts: list[str] = []
    for para in reversed(paragraphs):
        if _META_PATTERNS.match(para):
            # Stop once we hit meta-reasoning going backwards
            if conclusion_parts:
                break
            continue
        conclusion_parts.append(para)
        # If we've collected enough content, stop
        if sum(len(p) for p in conclusion_parts) > 400:
            break

    if conclusion_parts:
        conclusion_parts.reverse()
        result = "\n\n".join(conclusion_parts)
        if len(result) > 120:
            return result

    # If extraction failed, return the last 40% of the text as a heuristic
    cutpoint = max(0, len(reasoning) - int(len(reasoning) * 0.4))
    tail = reasoning[cutpoint:].strip()
    return tail if tail else reasoning.strip()

# ── Provider constants ─────────────────────────────────────────────────────────

PROVIDER_LM_STUDIO = "lm_studio"
PROVIDER_OPENAI = "openai"

# ── Defaults (overridden by .env) ──────────────────────────────────────────────

_LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
_LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3.5-9b")
_LM_STUDIO_VISION_MODEL = os.getenv("LM_STUDIO_VISION_MODEL", "qwen/qwen3.5-9b")

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
_OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", PROVIDER_LM_STUDIO)


def get_client(provider: str) -> OpenAI:
    """Return an OpenAI-compatible client for the given provider."""
    if provider == PROVIDER_LM_STUDIO:
        return OpenAI(
            base_url=_LM_STUDIO_BASE_URL,
            api_key="lm-studio",  # LM Studio ignores the key but SDK requires one
            timeout=600.0,  # 10 min timeout — generous for thinking models on local HW
        )
    elif provider == PROVIDER_OPENAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set — required for OpenAI provider")
        return OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use '{PROVIDER_LM_STUDIO}' or '{PROVIDER_OPENAI}'.")


def get_model(provider: str, vision: bool = False) -> str:
    """Return the model name for the given provider + mode."""
    if provider == PROVIDER_LM_STUDIO:
        return _LM_STUDIO_VISION_MODEL if vision else _LM_STUDIO_MODEL
    elif provider == PROVIDER_OPENAI:
        return _OPENAI_VISION_MODEL if vision else _OPENAI_MODEL
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


def chat(
    provider: str,
    messages: list[dict],
    system: str = "",
    max_tokens: int = 8096,
    temperature: float = 0.7,
    tools: list[dict] | None = None,
    vision: bool = False,
    thinking: bool = True,
) -> str:
    """
    Send a completion and return the assistant's text content.

    For the OpenAI provider, uses the Responses API (client.responses.create)
    which is the primary API for gpt-5+ models. For LM Studio, uses the Chat
    Completions API since local servers do not implement the Responses endpoint.

    Handles thinking-mode models (e.g. qwen3, QwQ — LM Studio only) where the
    model may put all its output in ``reasoning_content`` and return an empty
    ``content`` field.

    Args:
        provider:    'lm_studio' or 'openai'
        messages:    List of message dicts (role/content)
        system:      System prompt
        max_tokens:  Max response tokens
        temperature: Sampling temperature
        tools:       Optional function-calling tools (OpenAI format)
        vision:      If True, use the vision model variant
        thinking:    If False, disable thinking mode for Qwen3/LM Studio models

    Returns:
        Assistant message text content.
    """
    client = get_client(provider)
    model = get_model(provider, vision=vision)

    logger.info(
        "LLM call → provider=%s  model=%s  msgs=%d  max_tokens=%d",
        provider, model, len(messages), max_tokens,
    )

    if provider == PROVIDER_OPENAI:
        # ── Responses API (gpt-5+) ─────────────────────────────────────────────
        kwargs: dict = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["instructions"] = system
        if tools:
            kwargs["tools"] = tools
        response = client.responses.create(**kwargs)
        return response.output_text or ""

    else:
        # ── Chat Completions API (LM Studio) ───────────────────────────────────
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if not thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        response = client.chat.completions.create(**kwargs)

        msg = response.choices[0].message
        content = msg.content or ""

        # Thinking-model fallback: qwen3/QwQ puts the reasoning in reasoning_content
        # and may leave content empty if max_tokens was exhausted during thinking.
        if not content:
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if reasoning:
                logger.debug("Fell back to reasoning_content (%d chars)", len(reasoning))
                content = _extract_conclusion(reasoning)

        # Strip <think>...</think> blocks emitted by thinking-mode models (Qwen3, QwQ)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content


def describe_image(
    provider: str,
    image_b64: str,
    text_prompt: str,
    system: str = "",
    max_tokens: int = 400,
    thinking: bool = True,
) -> str:
    """
    Send a vision request with an inline base64-encoded image.

    For the OpenAI provider, uses the Responses API with ``input_image`` /
    ``input_text`` content types. For LM Studio, uses the Chat Completions API
    with the ``image_url`` content type.

    Args:
        provider:    'lm_studio' or 'openai'
        image_b64:   Base64-encoded JPEG image
        text_prompt: Text prompt to accompany the image
        system:      Optional system prompt
        max_tokens:  Max response tokens
        thinking:    If False, disable extended thinking (LM Studio / Qwen3 only).
                     Recommended False for simple per-frame descriptions to save
                     context tokens and avoid empty responses on small budgets.

    Returns:
        Model's text description of the image.
    """
    client = get_client(provider)
    model = get_model(provider, vision=True)

    logger.info("Vision call → provider=%s  model=%s", provider, model)

    if provider == PROVIDER_OPENAI:
        # ── Responses API vision (input_image + input_text) ───────────────────
        kwargs: dict = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}",
                        },
                        {"type": "input_text", "text": text_prompt},
                    ],
                }
            ],
            "max_output_tokens": max_tokens,
        }
        if system:
            kwargs["instructions"] = system
        response = client.responses.create(**kwargs)
        return response.output_text or ""

    else:
        # ── Chat Completions API vision (image_url) ───────────────────────────
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": text_prompt},
            ],
        })
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if not thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        content = (msg.content or "").strip()

        # Thinking-model fallback: reasoning_content may hold the answer when
        # max_tokens was exhausted during the <think> phase.
        if not content:
            reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
            if reasoning:
                logger.debug("describe_image: fell back to reasoning_content (%d chars)", len(reasoning))
                content = _extract_conclusion(reasoning)

        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
