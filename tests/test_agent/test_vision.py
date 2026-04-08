"""Tests for the vision analysis module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from teams_attendant.agent.llm import LLMResponse
from teams_attendant.agent.vision import VISION_SYSTEM_PROMPT, VisionAnalyzer
from teams_attendant.utils.events import Event, EventBus, ScreenCaptureEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.complete_with_vision.return_value = LLMResponse(
        content="A slide showing Q3 revenue growth.",
        input_tokens=100,
        output_tokens=20,
        model="claude-sonnet",
    )
    return llm


@pytest.fixture
def mock_context() -> MagicMock:
    return MagicMock()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def analyzer(mock_llm: AsyncMock, mock_context: MagicMock, event_bus: EventBus) -> VisionAnalyzer:
    return VisionAnalyzer(
        llm_client=mock_llm,
        context=mock_context,
        event_bus=event_bus,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_init_defaults(analyzer: VisionAnalyzer) -> None:
    assert analyzer._running is False
    assert analyzer._analysis_count == 0
    assert analyzer._analysis_prompt == VISION_SYSTEM_PROMPT


def test_init_custom_prompt(mock_llm: AsyncMock, mock_context: MagicMock, event_bus: EventBus) -> None:
    custom = "Custom prompt."
    va = VisionAnalyzer(mock_llm, mock_context, event_bus, analysis_prompt=custom)
    assert va._analysis_prompt == custom


# ---------------------------------------------------------------------------
# Start / Stop
# ---------------------------------------------------------------------------


async def test_start_subscribes(analyzer: VisionAnalyzer, event_bus: EventBus) -> None:
    await analyzer.start()
    assert analyzer._running is True
    assert analyzer._handle_screen_capture in event_bus._handlers["screen_capture"]


async def test_stop_unsubscribes(analyzer: VisionAnalyzer, event_bus: EventBus) -> None:
    await analyzer.start()
    await analyzer.stop()
    assert analyzer._running is False
    assert analyzer._handle_screen_capture not in event_bus._handlers["screen_capture"]


# ---------------------------------------------------------------------------
# analyze_image
# ---------------------------------------------------------------------------


async def test_analyze_image_calls_llm(analyzer: VisionAnalyzer, mock_llm: AsyncMock) -> None:
    image_data = b"\x89PNG fake image data"
    result = await analyzer.analyze_image(image_data)

    assert result == "A slide showing Q3 revenue growth."
    mock_llm.complete_with_vision.assert_called_once()
    call_kwargs = mock_llm.complete_with_vision.call_args
    assert call_kwargs.kwargs["images"] == [image_data]
    assert call_kwargs.kwargs["system"] == VISION_SYSTEM_PROMPT
    assert call_kwargs.kwargs["max_tokens"] == 500

    messages = call_kwargs.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0].role == "user"


async def test_analyze_image_returns_description(analyzer: VisionAnalyzer) -> None:
    result = await analyzer.analyze_image(b"img")
    assert result == "A slide showing Q3 revenue growth."


async def test_analyze_image_handles_llm_error(
    analyzer: VisionAnalyzer, mock_llm: AsyncMock
) -> None:
    mock_llm.complete_with_vision.side_effect = RuntimeError("API down")
    result = await analyzer.analyze_image(b"img")
    assert result == ""


async def test_analyze_image_empty_data(analyzer: VisionAnalyzer, mock_llm: AsyncMock) -> None:
    result = await analyzer.analyze_image(b"")
    assert result == ""
    mock_llm.complete_with_vision.assert_not_called()


# ---------------------------------------------------------------------------
# _handle_screen_capture
# ---------------------------------------------------------------------------


async def test_handle_screen_capture_adds_description(
    analyzer: VisionAnalyzer, mock_context: MagicMock
) -> None:
    event = ScreenCaptureEvent(image_data=b"png-bytes", description="")
    await analyzer._handle_screen_capture(event)

    mock_context.add_screen_description.assert_called_once_with("A slide showing Q3 revenue growth.")


async def test_handle_screen_capture_skips_empty_description(
    analyzer: VisionAnalyzer, mock_llm: AsyncMock, mock_context: MagicMock
) -> None:
    mock_llm.complete_with_vision.return_value = LLMResponse(content="", input_tokens=0, output_tokens=0)
    event = ScreenCaptureEvent(image_data=b"png-bytes")
    await analyzer._handle_screen_capture(event)

    mock_context.add_screen_description.assert_not_called()


async def test_handle_screen_capture_skips_no_image(
    analyzer: VisionAnalyzer, mock_llm: AsyncMock, mock_context: MagicMock
) -> None:
    event = Event(type="screen_capture", data={"image_data": b""})
    await analyzer._handle_screen_capture(event)

    mock_llm.complete_with_vision.assert_not_called()
    mock_context.add_screen_description.assert_not_called()


# ---------------------------------------------------------------------------
# analysis_count
# ---------------------------------------------------------------------------


async def test_analysis_count_increments(analyzer: VisionAnalyzer) -> None:
    assert analyzer._analysis_count == 0
    event = ScreenCaptureEvent(image_data=b"img1")
    await analyzer._handle_screen_capture(event)
    assert analyzer._analysis_count == 1

    event2 = ScreenCaptureEvent(image_data=b"img2")
    await analyzer._handle_screen_capture(event2)
    assert analyzer._analysis_count == 2


async def test_analysis_count_no_increment_on_empty(
    analyzer: VisionAnalyzer, mock_llm: AsyncMock
) -> None:
    mock_llm.complete_with_vision.return_value = LLMResponse(content="  ", input_tokens=0, output_tokens=0)
    event = ScreenCaptureEvent(image_data=b"img")
    await analyzer._handle_screen_capture(event)
    assert analyzer._analysis_count == 0
