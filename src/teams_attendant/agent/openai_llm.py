"""OpenAI-compatible GPT client via Azure Foundry."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import TYPE_CHECKING, AsyncIterator

import httpx
import structlog

from teams_attendant.agent.llm import LLMResponse, Message
from teams_attendant.errors import LLMAuthError, LLMRateLimitError

if TYPE_CHECKING:
    from teams_attendant.config import AzureFoundryConfig

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Retry constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpenAIClient:
    """Client for OpenAI-compatible models deployed on Azure Foundry."""

    def __init__(self, config: AzureFoundryConfig) -> None:
        self._config = config
        self._base_url = config.endpoint.rstrip("/")
        self._model = config.model_deployment
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0, read=120.0, write=30.0, pool=30.0,
            ),
            limits=httpx.Limits(
                max_connections=20, max_keepalive_connections=10,
            ),
        )

    # -- public API ---------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send a completion request and return the full response."""
        body = self._build_request_body(
            messages, system, max_tokens, temperature,
        )
        raw = await self._post(body)
        response = self._parse_response(raw)
        log.info(
            "llm.complete",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=response.model,
        )
        return response

    async def stream(
        self,
        messages: list[Message],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream a completion response, yielding text chunks."""
        body = self._build_request_body(
            messages, system, max_tokens, temperature, stream=True,
        )
        async for chunk in self._post_stream(body):
            yield chunk

    async def complete_with_vision(
        self,
        messages: list[Message],
        images: list[bytes],
        system: str = "",
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send a completion with images for vision analysis."""
        vision_messages = self._build_vision_content(messages, images)
        body = self._build_request_body(
            vision_messages, system, max_tokens, temperature=0.7,
        )
        raw = await self._post(body)
        response = self._parse_response(raw)
        log.info(
            "llm.complete_with_vision",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            images=len(images),
        )
        return response

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    # -- request helpers ----------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        return {
            "api-key": self._config.api_key,
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

    @property
    def _completions_url(self) -> str:
        return f"{self._base_url}/models/{self._model}/chat/completions"

    def _build_request_body(
        self,
        messages: list[Message],
        system: str,
        max_tokens: int,
        temperature: float,
        *,
        stream: bool = False,
    ) -> dict:
        api_messages: list[dict] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        for m in messages:
            api_messages.append(self._serialise_message(m))
        body: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    @staticmethod
    def _serialise_message(msg: Message) -> dict:
        return {"role": msg.role, "content": msg.content}

    def _build_vision_content(
        self, messages: list[Message], images: list[bytes],
    ) -> list[Message]:
        """Prepend base64-encoded image blocks to the last user message."""
        if not messages:
            return messages

        result = list(messages)
        last_user_idx: int | None = None
        for i in reversed(range(len(result))):
            if result[i].role == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return result

        original = result[last_user_idx]
        if isinstance(original.content, str):
            text_blocks: list[dict] = [
                {"type": "text", "text": original.content},
            ]
        else:
            text_blocks = list(original.content)

        image_blocks: list[dict] = []
        for img in images:
            encoded = base64.b64encode(img).decode("ascii")
            log.debug("llm.vision_image", size_bytes=len(img))
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded}",
                    },
                },
            )

        result[last_user_idx] = Message(
            role="user",
            content=[*image_blocks, *text_blocks],
        )
        return result

    # -- response helpers ---------------------------------------------------

    @staticmethod
    def _parse_response(data: dict) -> LLMResponse:
        """Parse an OpenAI Chat Completions API response."""
        choices = data.get("choices", [])
        content = ""
        stop_reason = ""
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content", "") or ""
            stop_reason = choice.get("finish_reason", "")

        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=data.get("model", ""),
            stop_reason=stop_reason,
        )

    # -- HTTP layer with retries --------------------------------------------

    async def _post(self, body: dict) -> dict:
        """POST to the chat completions endpoint with retry logic."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                log.debug(
                    "llm.request",
                    attempt=attempt,
                    url=self._completions_url,
                )
                resp = await self._client.post(
                    self._completions_url,
                    headers=self._build_headers(),
                    json=body,
                )
                if resp.status_code in (401, 403):
                    raise LLMAuthError(
                        f"Authentication failed (HTTP {resp.status_code}). "
                        "Check your OpenAI API key and endpoint."
                    )
                if resp.status_code == 429:
                    backoff = _INITIAL_BACKOFF * (2**attempt)
                    log.warning(
                        "llm.rate_limited",
                        retry_after=backoff,
                        attempt=attempt,
                    )
                    await asyncio.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (LLMAuthError, LLMRateLimitError):
                raise
            except httpx.HTTPStatusError:
                raise
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt == 0:
                    log.warning(
                        "llm.network_error",
                        error=str(exc),
                        retrying=True,
                    )
                    await asyncio.sleep(_INITIAL_BACKOFF)
                    continue
                raise

        raise LLMRateLimitError(
            "Rate limited after maximum retries",
        ) from last_exc

    async def _post_stream(self, body: dict) -> AsyncIterator[str]:
        """POST with streaming; yield text deltas from SSE."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                log.debug(
                    "llm.stream_request",
                    attempt=attempt,
                    url=self._completions_url,
                )
                req = self._client.build_request(
                    "POST",
                    self._completions_url,
                    headers=self._build_headers(),
                    json=body,
                )
                resp = await self._client.send(req, stream=True)

                if resp.status_code in (401, 403):
                    await resp.aclose()
                    raise LLMAuthError(
                        f"Authentication failed (HTTP {resp.status_code}). "
                        "Check your OpenAI API key and endpoint."
                    )
                if resp.status_code == 429:
                    await resp.aclose()
                    backoff = _INITIAL_BACKOFF * (2**attempt)
                    log.warning(
                        "llm.stream_rate_limited",
                        retry_after=backoff,
                        attempt=attempt,
                    )
                    await asyncio.sleep(backoff)
                    continue
                if resp.status_code >= 400:
                    await resp.aclose()
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=req,
                        response=resp,
                    )

                async for text in self._parse_sse(resp):
                    yield text
                return

            except (LLMAuthError, LLMRateLimitError):
                raise
            except httpx.HTTPStatusError:
                raise
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt == 0:
                    log.warning(
                        "llm.stream_network_error",
                        error=str(exc),
                        retrying=True,
                    )
                    await asyncio.sleep(_INITIAL_BACKOFF)
                    continue
                raise

        raise LLMRateLimitError(
            "Rate limited after maximum retries",
        ) from last_exc

    @staticmethod
    async def _parse_sse(resp: httpx.Response) -> AsyncIterator[str]:
        """Parse SSE lines from a streaming response, yielding text."""
        buffer = ""
        async for raw_chunk in resp.aiter_text():
            buffer += raw_chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    return
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text
        await resp.aclose()
