"""Tests for the Claude LLM client via Azure Foundry."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from teams_attendant.agent.llm import (
    AnthropicClient,
    ClaudeClient,
    LLMAuthError,
    LLMRateLimitError,
    LLMResponse,
    Message,
    create_llm_client,
)
from teams_attendant.agent.openai_llm import OpenAIClient
from teams_attendant.config import AppConfig, AzureFoundryConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> AzureFoundryConfig:
    return AzureFoundryConfig(
        endpoint="https://test.azure.com",
        api_key="test-key-123",
        model_deployment="claude-sonnet",
    )


@pytest.fixture
def client(config: AzureFoundryConfig) -> ClaudeClient:
    return ClaudeClient(config)


def _api_response(
    text: str = "Hello!",
    input_tokens: int = 10,
    output_tokens: int = 5,
    model: str = "claude-sonnet",
    stop_reason: str = "end_turn",
) -> dict:
    """Build a realistic Anthropic Messages API response dict."""
    return {
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _httpx_response(
    status_code: int = 200,
    json_body: dict | None = None,
) -> httpx.Response:
    """Create an httpx.Response for mocking."""
    resp = httpx.Response(
        status_code=status_code,
        json=json_body or _api_response(),
        request=httpx.Request("POST", "https://test.azure.com/models/claude-sonnet/messages"),
    )
    return resp


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestAlias:
    def test_claude_client_alias(self) -> None:
        assert ClaudeClient is AnthropicClient


class TestFactory:
    def test_create_llm_client_anthropic(self) -> None:
        config = AppConfig(
            llm_provider="anthropic",
            azure={"foundry": {"endpoint": "https://e.com", "api_key": "k"}},
        )
        client = create_llm_client(config)
        assert isinstance(client, AnthropicClient)

    def test_create_llm_client_openai(self) -> None:
        config = AppConfig(
            llm_provider="openai",
            openai={"api_key": "sk-test"},
        )
        client = create_llm_client(config)
        assert isinstance(client, OpenAIClient)


class TestClaudeClientInit:
    def test_stores_config(self, config: AzureFoundryConfig) -> None:
        client = ClaudeClient(config)
        assert client._config is config
        assert client._model == "claude-sonnet"
        assert client._base_url == "https://test.azure.com"

    def test_trailing_slash_stripped(self) -> None:
        cfg = AzureFoundryConfig(
            endpoint="https://example.com/", api_key="k", model_deployment="m",
        )
        client = ClaudeClient(cfg)
        assert client._base_url == "https://example.com"

    def test_headers_contain_auth(self, client: ClaudeClient) -> None:
        headers = client._build_headers()
        assert headers["api-key"] == "test-key-123"
        assert headers["Authorization"] == "Bearer test-key-123"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


class TestBuildRequestBody:
    def test_basic_body(self, client: ClaudeClient) -> None:
        msgs = [Message(role="user", content="Hi")]
        body = client._build_request_body(msgs, system="", max_tokens=512, temperature=0.5)
        assert body["model"] == "claude-sonnet"
        assert body["max_tokens"] == 512
        assert body["temperature"] == 0.5
        assert body["messages"] == [{"role": "user", "content": "Hi"}]
        assert "system" not in body
        assert "stream" not in body

    def test_system_prompt(self, client: ClaudeClient) -> None:
        msgs = [Message(role="user", content="Hi")]
        body = client._build_request_body(msgs, system="Be helpful", max_tokens=100, temperature=0)
        assert body["system"] == "Be helpful"

    def test_stream_flag(self, client: ClaudeClient) -> None:
        msgs = [Message(role="user", content="Hi")]
        body = client._build_request_body(
            msgs, system="", max_tokens=100, temperature=0, stream=True,
        )
        assert body["stream"] is True


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_parse_standard(self) -> None:
        raw = _api_response(text="world", input_tokens=20, output_tokens=10, model="m")
        resp = ClaudeClient._parse_response(raw)
        assert isinstance(resp, LLMResponse)
        assert resp.content == "world"
        assert resp.input_tokens == 20
        assert resp.output_tokens == 10
        assert resp.model == "m"

    def test_multiple_content_blocks(self) -> None:
        raw = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "model": "m",
            "stop_reason": "end_turn",
        }
        resp = ClaudeClient._parse_response(raw)
        assert resp.content == "Hello world"

    def test_missing_usage(self) -> None:
        raw = {"content": [{"type": "text", "text": "ok"}], "model": "m"}
        resp = ClaudeClient._parse_response(raw)
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------


class TestComplete:
    async def test_sends_correct_request(self, client: ClaudeClient) -> None:
        mock_resp = _httpx_response(json_body=_api_response(text="Reply"))
        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.complete(
                [Message(role="user", content="Hello")],
                system="sys",
                max_tokens=256,
                temperature=0.3,
            )

            client._client.post.assert_awaited_once()
            call_kwargs = client._client.post.call_args
            assert "claude-sonnet/messages" in call_kwargs.args[0]
            body = call_kwargs.kwargs["json"]
            assert body["max_tokens"] == 256
            assert body["system"] == "sys"
            assert body["temperature"] == 0.3

        assert result.content == "Reply"

    async def test_parses_response(self, client: ClaudeClient) -> None:
        mock_resp = _httpx_response(
            json_body=_api_response(text="Parsed", input_tokens=15, output_tokens=7),
        )
        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.complete([Message(role="user", content="x")])

        assert result.content == "Parsed"
        assert result.input_tokens == 15
        assert result.output_tokens == 7


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


class TestStream:
    async def test_yields_text_chunks(self, client: ClaudeClient) -> None:
        sse_lines = (
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n\n'
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}\n\n'
            "data: [DONE]\n\n"
        )

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200

        async def _aiter_text():
            yield sse_lines

        mock_response.aiter_text = _aiter_text
        mock_response.aclose = AsyncMock()

        with (
            patch.object(
                client._client, "build_request", return_value=httpx.Request("POST", "https://x"),
            ),
            patch.object(
                client._client, "send", new_callable=AsyncMock, return_value=mock_response,
            ),
        ):
            chunks: list[str] = []
            async for chunk in client.stream([Message(role="user", content="Hi")]):
                chunks.append(chunk)

        assert chunks == ["Hello", " world"]


# ---------------------------------------------------------------------------
# complete_with_vision()
# ---------------------------------------------------------------------------


class TestVision:
    async def test_includes_image_blocks(self, client: ClaudeClient) -> None:
        img_data = b"\x89PNG_fake_image_data"
        mock_resp = _httpx_response(json_body=_api_response(text="I see a cat"))
        captured_body: dict = {}

        async def _capture_post(url, *, headers, json):  # noqa: F811
            captured_body.update(json)
            return mock_resp

        with patch.object(client._client, "post", side_effect=_capture_post):
            result = await client.complete_with_vision(
                messages=[Message(role="user", content="What is this?")],
                images=[img_data],
                system="describe",
                max_tokens=512,
            )

        assert result.content == "I see a cat"
        # The user message should now have image + text blocks
        user_msg = captured_body["messages"][-1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "image"
        assert user_msg["content"][0]["source"]["type"] == "base64"
        assert user_msg["content"][-1]["type"] == "text"
        assert user_msg["content"][-1]["text"] == "What is this?"

    def test_build_vision_content(self, client: ClaudeClient) -> None:
        msgs = [Message(role="user", content="Describe")]
        images = [b"img1", b"img2"]
        result = client._build_vision_content(msgs, images)
        assert len(result) == 1
        content = result[0].content
        assert isinstance(content, list)
        # 2 image blocks + 1 text block
        assert len(content) == 3
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "image"
        assert content[2]["type"] == "text"

    def test_build_vision_no_user_message(self, client: ClaudeClient) -> None:
        msgs = [Message(role="assistant", content="ok")]
        result = client._build_vision_content(msgs, [b"img"])
        # No user message to attach images to; messages unchanged
        assert len(result) == 1
        assert result[0].content == "ok"


# ---------------------------------------------------------------------------
# Retry & error handling
# ---------------------------------------------------------------------------


class TestRetryOnRateLimit:
    async def test_retries_on_429(self, client: ClaudeClient) -> None:
        rate_resp = _httpx_response(status_code=429, json_body={})
        ok_resp = _httpx_response(json_body=_api_response(text="ok"))

        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return rate_resp
            return ok_resp

        with patch.object(client._client, "post", side_effect=_side_effect):
            with patch("teams_attendant.agent.llm.asyncio.sleep", new_callable=AsyncMock):
                result = await client.complete([Message(role="user", content="hi")])

        assert result.content == "ok"
        assert call_count == 3

    async def test_raises_after_max_retries(self, client: ClaudeClient) -> None:
        rate_resp = _httpx_response(status_code=429, json_body={})

        with patch.object(
            client._client, "post", new_callable=AsyncMock, return_value=rate_resp,
        ):
            with patch("teams_attendant.agent.llm.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(LLMRateLimitError, match="Rate limited"):
                    await client.complete([Message(role="user", content="hi")])


class TestAuthErrors:
    @pytest.mark.parametrize("status", [401, 403])
    async def test_auth_error_raises(self, client: ClaudeClient, status: int) -> None:
        auth_resp = _httpx_response(status_code=status, json_body={})
        with patch.object(
            client._client, "post", new_callable=AsyncMock, return_value=auth_resp,
        ):
            with pytest.raises(LLMAuthError, match="Authentication failed"):
                await client.complete([Message(role="user", content="hi")])


class TestNetworkErrors:
    async def test_retries_once_then_raises(self, client: ClaudeClient) -> None:
        exc = httpx.ConnectError("Connection refused")
        mock_post = AsyncMock(side_effect=exc)
        with patch.object(client._client, "post", mock_post):
            with patch("teams_attendant.agent.llm.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(httpx.ConnectError):
                    await client.complete([Message(role="user", content="hi")])
        # Should have been called twice (initial + 1 retry)
        assert mock_post.await_count == 2


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_closes_client(self, client: ClaudeClient) -> None:
        with patch.object(client._client, "aclose", new_callable=AsyncMock) as mock_close:
            await client.close()
            mock_close.assert_awaited_once()
