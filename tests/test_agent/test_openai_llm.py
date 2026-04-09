"""Tests for the OpenAI GPT LLM client."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from teams_attendant.agent.llm import LLMResponse, Message, create_llm_client
from teams_attendant.agent.openai_llm import OpenAIClient
from teams_attendant.config import AppConfig, OpenAIConfig
from teams_attendant.errors import LLMAuthError, LLMRateLimitError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> OpenAIConfig:
    return OpenAIConfig(
        api_key="test-key-123",
        endpoint="https://api.openai.com/v1",
        model="gpt-4o",
        organization="test-org",
    )


@pytest.fixture
def config_no_org() -> OpenAIConfig:
    return OpenAIConfig(
        api_key="test-key-123",
        endpoint="https://api.openai.com/v1",
        model="gpt-4o",
        organization="",
    )


@pytest.fixture
def client(config: OpenAIConfig) -> OpenAIClient:
    return OpenAIClient(config)


@pytest.fixture
def client_no_org(config_no_org: OpenAIConfig) -> OpenAIClient:
    return OpenAIClient(config_no_org)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api_response(
    text: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    model: str = "gpt-4o",
    finish_reason: str = "stop",
) -> dict:
    """Build a realistic OpenAI Chat Completions API response dict."""
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
        "model": model,
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _httpx_response(
    status_code: int = 200,
    json_body: dict | None = None,
) -> httpx.Response:
    """Create an httpx.Response for mocking."""
    return httpx.Response(
        status_code=status_code,
        json=json_body or _api_response(),
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )


# ---------------------------------------------------------------------------
# Config / Setup
# ---------------------------------------------------------------------------


class TestClientCreation:
    def test_client_creation(self, config: OpenAIConfig) -> None:
        c = OpenAIClient(config)
        assert c._config is config
        assert c._model == "gpt-4o"
        assert c._base_url == "https://api.openai.com/v1"

    def test_trailing_slash_stripped(self) -> None:
        cfg = OpenAIConfig(
            api_key="k", endpoint="https://api.openai.com/v1/", model="m",
        )
        c = OpenAIClient(cfg)
        assert c._base_url == "https://api.openai.com/v1"

    def test_completions_url(self, client: OpenAIClient) -> None:
        assert client._completions_url == "https://api.openai.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


class TestHeaders:
    def test_headers_contain_auth(self, client: OpenAIClient) -> None:
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer test-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_headers_include_organization(self, client: OpenAIClient) -> None:
        headers = client._build_headers()
        assert headers["OpenAI-Organization"] == "test-org"

    def test_headers_omit_organization_when_empty(self, client_no_org: OpenAIClient) -> None:
        headers = client_no_org._build_headers()
        assert "OpenAI-Organization" not in headers


# ---------------------------------------------------------------------------
# Request body building
# ---------------------------------------------------------------------------


class TestBuildRequestBody:
    def test_build_request_body_basic(self, client: OpenAIClient) -> None:
        msgs = [Message(role="user", content="Hi")]
        body = client._build_request_body(msgs, system="", max_tokens=512, temperature=0.5)
        assert body["model"] == "gpt-4o"
        assert body["max_tokens"] == 512
        assert body["temperature"] == 0.5
        assert body["messages"] == [{"role": "user", "content": "Hi"}]
        assert "stream" not in body

    def test_build_request_body_with_system(self, client: OpenAIClient) -> None:
        msgs = [Message(role="user", content="Hi")]
        body = client._build_request_body(msgs, system="Be helpful", max_tokens=100, temperature=0)
        assert body["messages"][0] == {"role": "system", "content": "Be helpful"}
        assert body["messages"][1] == {"role": "user", "content": "Hi"}

    def test_build_request_body_no_system_when_empty(self, client: OpenAIClient) -> None:
        msgs = [Message(role="user", content="Hi")]
        body = client._build_request_body(msgs, system="", max_tokens=100, temperature=0)
        assert body["messages"] == [{"role": "user", "content": "Hi"}]

    def test_build_request_body_stream(self, client: OpenAIClient) -> None:
        msgs = [Message(role="user", content="Hi")]
        body = client._build_request_body(
            msgs, system="", max_tokens=100, temperature=0, stream=True,
        )
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# Vision content
# ---------------------------------------------------------------------------


class TestVisionContent:
    def test_build_vision_content(self, client: OpenAIClient) -> None:
        msgs = [Message(role="user", content="Describe")]
        images = [b"img1", b"img2"]
        result = client._build_vision_content(msgs, images)
        assert len(result) == 1
        content = result[0].content
        assert isinstance(content, list)
        # 2 image blocks + 1 text block
        assert len(content) == 3
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "text"
        assert content[2]["text"] == "Describe"

    def test_build_vision_content_data_uri(self, client: OpenAIClient) -> None:
        raw = b"\x89PNG_fake"
        msgs = [Message(role="user", content="What?")]
        result = client._build_vision_content(msgs, [raw])
        img_block = result[0].content[0]
        expected_b64 = base64.b64encode(raw).decode("ascii")
        assert img_block["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"

    def test_build_vision_content_preserves_text(self, client: OpenAIClient) -> None:
        msgs = [
            Message(role="assistant", content="Sure"),
            Message(role="user", content="Look at this"),
        ]
        result = client._build_vision_content(msgs, [b"img"])
        # Assistant message untouched
        assert result[0].content == "Sure"
        # User message has image + text
        user_content = result[1].content
        assert isinstance(user_content, list)
        assert user_content[-1] == {"type": "text", "text": "Look at this"}

    def test_build_vision_no_user_message(self, client: OpenAIClient) -> None:
        msgs = [Message(role="assistant", content="ok")]
        result = client._build_vision_content(msgs, [b"img"])
        assert len(result) == 1
        assert result[0].content == "ok"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_parse_response(self) -> None:
        raw = _api_response(
            text="world", prompt_tokens=20, completion_tokens=10, model="gpt-4o",
        )
        resp = OpenAIClient._parse_response(raw)
        assert isinstance(resp, LLMResponse)
        assert resp.content == "world"
        assert resp.input_tokens == 20
        assert resp.output_tokens == 10
        assert resp.model == "gpt-4o"
        assert resp.stop_reason == "stop"

    def test_parse_response_empty_choices(self) -> None:
        raw = {"choices": [], "model": "gpt-4o", "usage": {"prompt_tokens": 5, "completion_tokens": 0}}
        resp = OpenAIClient._parse_response(raw)
        assert resp.content == ""
        assert resp.stop_reason == ""

    def test_parse_response_missing_usage(self) -> None:
        raw = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "model": "m"}
        resp = OpenAIClient._parse_response(raw)
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------


class TestComplete:
    async def test_complete_success(self, client: OpenAIClient) -> None:
        mock_resp = _httpx_response(json_body=_api_response(text="Reply", prompt_tokens=15))
        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.complete(
                [Message(role="user", content="Hello")],
                system="sys",
                max_tokens=256,
                temperature=0.3,
            )
            client._client.post.assert_awaited_once()
            call_kwargs = client._client.post.call_args
            assert "chat/completions" in call_kwargs.args[0]
            body = call_kwargs.kwargs["json"]
            assert body["max_tokens"] == 256
            assert body["messages"][0] == {"role": "system", "content": "sys"}
            assert body["temperature"] == 0.3

        assert result.content == "Reply"
        assert result.input_tokens == 15

    async def test_complete_auth_error(self, client: OpenAIClient) -> None:
        auth_resp = _httpx_response(status_code=401, json_body={})
        with patch.object(
            client._client, "post", new_callable=AsyncMock, return_value=auth_resp,
        ):
            with pytest.raises(LLMAuthError, match="Authentication failed"):
                await client.complete([Message(role="user", content="hi")])

    async def test_complete_rate_limit(self, client: OpenAIClient) -> None:
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
            with patch("teams_attendant.agent.openai_llm.asyncio.sleep", new_callable=AsyncMock):
                result = await client.complete([Message(role="user", content="hi")])

        assert result.content == "ok"
        assert call_count == 3

    async def test_complete_rate_limit_exhausted(self, client: OpenAIClient) -> None:
        rate_resp = _httpx_response(status_code=429, json_body={})
        with patch.object(
            client._client, "post", new_callable=AsyncMock, return_value=rate_resp,
        ):
            with patch("teams_attendant.agent.openai_llm.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(LLMRateLimitError, match="Rate limited"):
                    await client.complete([Message(role="user", content="hi")])


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


class TestStream:
    async def test_stream_yields_text(self, client: OpenAIClient) -> None:
        sse_lines = (
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
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

    async def test_stream_handles_done(self, client: OpenAIClient) -> None:
        sse_lines = (
            'data: {"choices":[{"delta":{"content":"A"}}]}\n\n'
            "data: [DONE]\n\n"
            'data: {"choices":[{"delta":{"content":"B"}}]}\n\n'
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

        # "B" after [DONE] should not appear
        assert chunks == ["A"]


# ---------------------------------------------------------------------------
# complete_with_vision()
# ---------------------------------------------------------------------------


class TestVision:
    async def test_includes_image_blocks(self, client: OpenAIClient) -> None:
        img_data = b"\x89PNG_fake_image_data"
        mock_resp = _httpx_response(json_body=_api_response(text="I see a cat"))
        captured_body: dict = {}

        async def _capture_post(url, *, headers, json):
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
        user_msg = captured_body["messages"][-1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "image_url"
        assert user_msg["content"][-1]["type"] == "text"
        assert user_msg["content"][-1]["text"] == "What is this?"


# ---------------------------------------------------------------------------
# Network errors
# ---------------------------------------------------------------------------


class TestNetworkErrors:
    async def test_retries_once_then_raises(self, client: OpenAIClient) -> None:
        exc = httpx.ConnectError("Connection refused")
        mock_post = AsyncMock(side_effect=exc)
        with patch.object(client._client, "post", mock_post):
            with patch("teams_attendant.agent.openai_llm.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(httpx.ConnectError):
                    await client.complete([Message(role="user", content="hi")])
        assert mock_post.await_count == 2


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_closes_client(self, client: OpenAIClient) -> None:
        with patch.object(client._client, "aclose", new_callable=AsyncMock) as mock_close:
            await client.close()
            mock_close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_llm_client_openai(self) -> None:
        cfg = AppConfig(
            llm_provider="openai",
            openai=OpenAIConfig(api_key="k", model="gpt-4o"),
        )
        c = create_llm_client(cfg)
        assert isinstance(c, OpenAIClient)

    def test_create_llm_client_anthropic(self) -> None:
        from teams_attendant.agent.llm import AnthropicClient

        cfg = AppConfig(llm_provider="anthropic")
        c = create_llm_client(cfg)
        assert isinstance(c, AnthropicClient)
