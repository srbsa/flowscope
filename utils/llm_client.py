"""
utils/llm_client.py
Unified LLM client factory for LM Studio (local) and OpenAI providers.

Both providers use the OpenAI SDK — LM Studio exposes an OpenAI-compatible API.
This is the single module that owns provider configuration.
"""

import logging
import os
import re

from openai import OpenAI

logger = logging.getLogger(__name__)

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
    Send a chat completion and return the assistant's text content.

    Handles thinking-mode models (e.g. qwen3, QwQ) where the model may put all
    its output in ``reasoning_content`` and return an empty ``content`` field.
    In that case, ``reasoning_content`` is returned as fallback so callers
    always receive non-empty text.

    Args:
        provider:    'lm_studio' or 'openai'
        messages:    List of message dicts (role/content)
        system:      System prompt (prepended as system message)
        max_tokens:  Max response tokens (default 8096 — generous for thinking models)
        temperature: Sampling temperature
        tools:       Optional function-calling tools (OpenAI format)
        vision:      If True, use the vision model variant
        thinking:    If False, disable thinking mode for Qwen3 models (saves tokens)

    Returns:
        Assistant message text content.
    """
    client = get_client(provider)
    model = get_model(provider, vision=vision)

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    kwargs: dict = {
        "model": model,
        "messages": full_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
    if not thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    logger.info("LLM call → provider=%s  model=%s  msgs=%d  max_tokens=%d  thinking=%s", provider, model, len(full_messages), max_tokens, thinking)
    response = client.chat.completions.create(**kwargs)

    msg = response.choices[0].message
    content = msg.content or ""

    # Thinking-model fallback: qwen3/QwQ puts the reasoning in reasoning_content
    # and may leave content empty if max_tokens was exhausted during thinking.
    # Use reasoning_content as-is so structured fields (VERDICT:, Score:) are
    # still present for downstream regex parsing.
    if not content:
        content = getattr(msg, "reasoning_content", None) or ""
        if content:
            logger.debug("Fell back to reasoning_content (%d chars)", len(content))

    # Strip <think>...</think> blocks emitted by thinking-mode models (Qwen3, QwQ)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    return content
