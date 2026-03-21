"""
tests/test_llm_client.py
Tests for utils/llm_client.py — provider factory, model selection, chat wrapper.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from utils.llm_client import (
    get_client,
    get_model,
    chat,
    PROVIDER_LM_STUDIO,
    PROVIDER_OPENAI,
)


class TestGetClient:
    def test_lm_studio_returns_openai_with_local_base_url(self):
        client = get_client(PROVIDER_LM_STUDIO)
        assert client is not None
        # The client should have the base_url set to the LM Studio endpoint
        assert "1234" in str(client.base_url)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"})
    def test_openai_returns_client_with_api_key(self):
        client = get_client(PROVIDER_OPENAI)
        assert client is not None

    def test_openai_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove OPENAI_API_KEY if present
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
                get_client(PROVIDER_OPENAI)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_client("azure")


class TestGetModel:
    def test_lm_studio_text_model(self):
        model = get_model(PROVIDER_LM_STUDIO, vision=False)
        assert isinstance(model, str)
        assert len(model) > 0

    def test_lm_studio_vision_model(self):
        model = get_model(PROVIDER_LM_STUDIO, vision=True)
        assert isinstance(model, str)
        assert len(model) > 0

    def test_openai_text_model(self):
        model = get_model(PROVIDER_OPENAI, vision=False)
        assert model  # Should be gpt-4o or whatever env says

    def test_openai_vision_model(self):
        model = get_model(PROVIDER_OPENAI, vision=True)
        assert model

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_model("bedrock")


class TestChat:
    @patch("utils.llm_client.get_client")
    def test_chat_sends_system_and_user_messages(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from LLM"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = chat(
            provider=PROVIDER_LM_STUDIO,
            messages=[{"role": "user", "content": "test"}],
            system="You are helpful.",
        )

        assert result == "Hello from LLM"
        call_kwargs = mock_client.chat.completions.create.call_args
        msgs = call_kwargs.kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."
        assert msgs[1]["role"] == "user"

    @patch("utils.llm_client.get_client")
    def test_chat_without_system_prompt(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        chat(provider=PROVIDER_LM_STUDIO, messages=[{"role": "user", "content": "hi"}])

        call_kwargs = mock_client.chat.completions.create.call_args
        msgs = call_kwargs.kwargs["messages"]
        # No system message when system=""
        assert msgs[0]["role"] == "user"

    @patch("utils.llm_client.get_client")
    def test_chat_passes_tools_when_provided(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        tools = [{"type": "function", "function": {"name": "search"}}]
        chat(
            provider=PROVIDER_LM_STUDIO,
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["tools"] == tools
