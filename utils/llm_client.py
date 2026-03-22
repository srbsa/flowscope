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

# ── Provider constants ─────────────────────────────────────────────────────────

PROVIDER_LM_STUDIO = "lm_studio"
PROVIDER_OPENAI = "openai"

# ── Defaults (overridden by .env) ──────────────────────────────────────────────

_LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
_LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3.5-9b")
_LM_STUDIO_VISION_MODEL = os.getenv("LM_STUDIO_VISION_MODEL", "qwen/qwen3.5-9b")

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
_OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.4")

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
            content = getattr(msg, "reasoning_content", None) or ""
            if content:
                logger.debug("Fell back to reasoning_content (%d chars)", len(content))

        # Strip <think>...</think> blocks emitted by thinking-mode models (Qwen3, QwQ)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content


def describe_image(
    provider: str,
    image_b64: str,
    text_prompt: str,
    system: str = "",
    max_tokens: int = 150,
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
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        content = (response.choices[0].message.content or "").strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
