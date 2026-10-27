
from unittest.mock import Mock, patch

import pytest

from memora.llm_client import GroqLLMClient, LLMResponse


@pytest.fixture
def llm_client():
    with patch("memora.llm_client.Groq") as mock_groq:
        client = GroqLLMClient()
        yield client, mock_groq.return_value


def test_empty_prompt_is_rejected(llm_client):
    client, mock_api = llm_client

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        client.generate("   ")

    mock_api.chat.completions.create.assert_not_called()


def test_invalid_token_limit_is_rejected(llm_client):
    client, mock_api = llm_client

    with pytest.raises(ValueError, match="positive integer"):
        client.generate("Hello", max_completion_tokens=0)

    mock_api.chat.completions.create.assert_not_called()


def test_generate_returns_response_and_usage(llm_client):
    client, mock_api = llm_client

    mock_response = Mock()
    mock_response.model = "openai/gpt-oss-20b"
    mock_response.choices = [
        Mock(message=Mock(content="Hello from the mock"))
    ]
    mock_response.usage = Mock(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    mock_api.chat.completions.create.return_value = mock_response

    result = client.generate("Say hello")

    assert isinstance(result, LLMResponse)
    assert result.text == "Hello from the mock"
    assert result.model == "openai/gpt-oss-20b"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15
    assert result.latency_ms >= 0

    mock_api.chat.completions.create.assert_called_once()


def test_generate_handles_missing_usage(llm_client):
    client, mock_api = llm_client

    mock_response = Mock()
    mock_response.model = "openai/gpt-oss-20b"
    mock_response.choices = [
        Mock(message=Mock(content="Response without usage"))
    ]
    mock_response.usage = None

    mock_api.chat.completions.create.return_value = mock_response

    result = client.generate("Say hello")

    assert result.text == "Response without usage"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_tokens is None
