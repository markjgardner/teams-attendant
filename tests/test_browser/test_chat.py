"""Tests for teams_attendant.browser.chat module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock


from teams_attendant.browser.chat import (
    ChatMessage,
    ChatObserver,
    _SEL_MESSAGE_AUTHOR,
    _SEL_MESSAGE_BODY,
)
from teams_attendant.utils.events import Event, EventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_element(*, inner_text: str = "") -> AsyncMock:
    """Build a mock Playwright ElementHandle."""
    el = AsyncMock()
    el.click = AsyncMock()
    el.inner_text = AsyncMock(return_value=inner_text)
    el.query_selector = AsyncMock(return_value=None)
    return el


def _make_message_element(author: str = "Alice", text: str = "hello") -> AsyncMock:
    """Build a mock message element with author and body sub-elements."""
    el = _make_element()

    author_el = _make_element(inner_text=author)
    body_el = _make_element(inner_text=text)

    async def _query_selector(selector: str) -> AsyncMock | None:
        if _SEL_MESSAGE_AUTHOR in selector or "author" in selector:
            return author_el
        if _SEL_MESSAGE_BODY in selector or "body" in selector:
            return body_el
        return None

    el.query_selector = AsyncMock(side_effect=_query_selector)
    return el


def _make_page() -> AsyncMock:
    """Build a mock Playwright Page with common stubs."""
    page = AsyncMock()
    type(page).is_closed = MagicMock(return_value=False)
    page.goto = AsyncMock()
    page.close = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.wait_for_selector = AsyncMock(return_value=_make_element())
    page.query_selector_all = AsyncMock(return_value=[])
    page.keyboard = AsyncMock()
    page.keyboard.type = AsyncMock()
    page.keyboard.press = AsyncMock()
    return page


def _make_event_bus() -> EventBus:
    """Build a real EventBus for integration tests."""
    return EventBus()


# ---------------------------------------------------------------------------
# ChatMessage dataclass
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_construction(self) -> None:
        msg = ChatMessage(
            id="abc", author="Bob", text="hi", timestamp=MagicMock(), is_own=False
        )
        assert msg.id == "abc"
        assert msg.author == "Bob"
        assert msg.text == "hi"
        assert msg.is_own is False

    def test_is_own_default(self) -> None:
        msg = ChatMessage(id="x", author="A", text="t", timestamp=MagicMock())
        assert msg.is_own is False


# ---------------------------------------------------------------------------
# _generate_message_id
# ---------------------------------------------------------------------------


class TestGenerateMessageId:
    def test_deterministic(self) -> None:
        page = _make_page()
        bus = _make_event_bus()
        obs = ChatObserver(page, bus)
        id1 = obs._generate_message_id("Alice", "hello", 0)
        id2 = obs._generate_message_id("Alice", "hello", 0)
        assert id1 == id2

    def test_different_for_different_inputs(self) -> None:
        page = _make_page()
        bus = _make_event_bus()
        obs = ChatObserver(page, bus)
        id1 = obs._generate_message_id("Alice", "hello", 0)
        id2 = obs._generate_message_id("Bob", "hello", 0)
        id3 = obs._generate_message_id("Alice", "hello", 1)
        assert id1 != id2
        assert id1 != id3


# ---------------------------------------------------------------------------
# _open_chat_panel
# ---------------------------------------------------------------------------


class TestOpenChatPanel:
    async def test_already_open(self) -> None:
        """Should not click chat button if panel is already visible."""
        page = _make_page()
        panel_el = _make_element()
        page.wait_for_selector = AsyncMock(return_value=panel_el)
        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        await obs._open_chat_panel()

        # wait_for_selector called for panel check; button should NOT be clicked
        page.wait_for_selector.assert_awaited_once()

    async def test_clicks_chat_button_when_closed(self) -> None:
        """Should click chat button and wait for panel when panel not visible."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        page = _make_page()
        chat_btn = _make_element()
        panel_el = _make_element()

        call_count = 0

        async def _selector_router(selector: str, **kwargs: object) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: checking if panel is open → not found
                raise PlaywrightTimeout("timeout")
            if call_count == 2:
                # Second call: finding chat button
                return chat_btn
            # Third call: waiting for panel after click
            return panel_el

        page.wait_for_selector = AsyncMock(side_effect=_selector_router)
        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        await obs._open_chat_panel()

        chat_btn.click.assert_awaited_once()

    async def test_error_when_button_not_found(self) -> None:
        """Should log error but not raise when chat button is not found."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        page = _make_page()
        page.wait_for_selector = AsyncMock(
            side_effect=PlaywrightTimeout("timeout")
        )
        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        # Should not raise
        await obs._open_chat_panel()


# ---------------------------------------------------------------------------
# _parse_messages_from_dom
# ---------------------------------------------------------------------------


class TestParseMessagesFromDom:
    async def test_extracts_messages(self) -> None:
        """Should parse author and text from message elements."""
        page = _make_page()
        msg1 = _make_message_element(author="Alice", text="hello")
        msg2 = _make_message_element(author="Bob", text="world")
        page.query_selector_all = AsyncMock(return_value=[msg1, msg2])

        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        messages = await obs._parse_messages_from_dom()

        assert len(messages) == 2
        assert messages[0].author == "Alice"
        assert messages[0].text == "hello"
        assert messages[1].author == "Bob"
        assert messages[1].text == "world"

    async def test_skips_empty_text(self) -> None:
        """Should skip messages with no text."""
        page = _make_page()
        msg = _make_message_element(author="Alice", text="")
        page.query_selector_all = AsyncMock(return_value=[msg])

        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        messages = await obs._parse_messages_from_dom()
        assert len(messages) == 0

    async def test_handles_dom_error(self) -> None:
        """Should return empty list on query_selector_all failure."""
        page = _make_page()
        page.query_selector_all = AsyncMock(side_effect=Exception("DOM error"))

        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        messages = await obs._parse_messages_from_dom()
        assert messages == []

    async def test_handles_single_message_error(self) -> None:
        """Should skip individual messages that fail to parse."""
        page = _make_page()
        bad_el = _make_element()
        bad_el.query_selector = AsyncMock(side_effect=Exception("parse error"))
        good_el = _make_message_element(author="Bob", text="ok")
        page.query_selector_all = AsyncMock(return_value=[bad_el, good_el])

        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        messages = await obs._parse_messages_from_dom()
        assert len(messages) == 1
        assert messages[0].author == "Bob"


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    async def test_start_opens_panel_and_snapshots(self) -> None:
        """start() should open chat panel, snapshot existing messages, start polling."""
        page = _make_page()
        msg = _make_message_element(author="Alice", text="old message")
        page.query_selector_all = AsyncMock(return_value=[msg])

        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        await obs.start()
        assert obs._running is True
        assert obs._task is not None
        # Existing message should be in seen set
        assert len(obs._seen_messages) == 1

        await obs.stop()

    async def test_start_idempotent(self) -> None:
        """start() called twice should not create a second task."""
        page = _make_page()
        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        await obs.start()
        task1 = obs._task
        await obs.start()
        assert obs._task is task1

        await obs.stop()

    async def test_stop_halts_polling(self) -> None:
        """stop() should set _running to False and cancel the task."""
        page = _make_page()
        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        await obs.start()
        assert obs._running is True

        await obs.stop()
        assert obs._running is False
        assert obs._task is None


# ---------------------------------------------------------------------------
# Polling – new messages trigger events
# ---------------------------------------------------------------------------


class TestPolling:
    async def test_new_messages_trigger_events(self) -> None:
        """New messages found during polling should be published to event bus."""
        page = _make_page()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus = _make_event_bus()
        bus.subscribe("chat_message", handler)
        await bus.start()

        obs = ChatObserver(page, bus)
        obs._poll_interval = 0.05

        # First poll returns empty (snapshot), second poll returns a new message
        call_count = 0

        async def _query_messages(selector: str) -> list[AsyncMock]:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return []
            return [_make_message_element(author="Alice", text="new msg")]

        page.query_selector_all = AsyncMock(side_effect=_query_messages)

        await obs.start()
        await asyncio.sleep(0.3)
        await obs.stop()
        await bus.stop()

        assert len(received) == 1
        assert received[0].data["author"] == "Alice"
        assert received[0].data["text"] == "new msg"

    async def test_seen_messages_not_reemitted(self) -> None:
        """Messages already in _seen_messages should not be published again."""
        page = _make_page()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus = _make_event_bus()
        bus.subscribe("chat_message", handler)
        await bus.start()

        msg = _make_message_element(author="Alice", text="hello")

        # Always return the same message
        page.query_selector_all = AsyncMock(return_value=[msg])

        obs = ChatObserver(page, bus)
        obs._poll_interval = 0.05

        # Snapshot picks up the existing message
        await obs.start()
        await asyncio.sleep(0.3)
        await obs.stop()
        await bus.stop()

        # Message was seen at snapshot time, so no events should fire
        assert len(received) == 0

    async def test_poll_handles_dom_error_gracefully(self) -> None:
        """Polling should continue even if DOM parsing raises."""
        page = _make_page()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus = _make_event_bus()
        bus.subscribe("chat_message", handler)
        await bus.start()

        call_count = 0

        async def _flaky_query(selector: str) -> list[AsyncMock]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []  # snapshot
            if call_count == 2:
                raise Exception("transient DOM error")
            return [_make_message_element(author="Bob", text="after error")]

        page.query_selector_all = AsyncMock(side_effect=_flaky_query)

        obs = ChatObserver(page, bus)
        obs._poll_interval = 0.05

        await obs.start()
        await asyncio.sleep(0.4)
        await obs.stop()
        await bus.stop()

        # Should have recovered and found the message after the error
        assert len(received) >= 1
        assert received[0].data["author"] == "Bob"


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------


class TestSendMessage:
    async def test_send_focuses_types_and_enters(self) -> None:
        """send_message should click input, type text, and press Enter."""
        page = _make_page()
        input_el = _make_element()

        page.wait_for_selector = AsyncMock(return_value=input_el)
        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        await obs.send_message("Hello world")

        input_el.click.assert_awaited()
        page.keyboard.type.assert_awaited_once_with("Hello world", delay=30)
        page.keyboard.press.assert_awaited_once_with("Enter")

    async def test_send_handles_missing_input(self) -> None:
        """send_message should not raise when input box is not found."""
        page = _make_page()
        page.wait_for_selector = AsyncMock(return_value=None)

        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        # Should not raise
        await obs.send_message("Hello")

    async def test_send_handles_timeout(self) -> None:
        """send_message should not raise on PlaywrightTimeout."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        page = _make_page()

        call_count = 0

        async def _selector_router(selector: str, **kwargs: object) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            # First call is _open_chat_panel panel check
            if call_count == 1:
                return _make_element()
            # Second call is finding the input
            raise PlaywrightTimeout("timeout")

        page.wait_for_selector = AsyncMock(side_effect=_selector_router)

        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        # Should not raise
        await obs.send_message("Hello")


# ---------------------------------------------------------------------------
# get_recent_messages
# ---------------------------------------------------------------------------


class TestGetRecentMessages:
    async def test_returns_last_n_messages(self) -> None:
        page = _make_page()
        elements = [
            _make_message_element(author=f"User{i}", text=f"msg{i}") for i in range(5)
        ]
        page.query_selector_all = AsyncMock(return_value=elements)

        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        messages = await obs.get_recent_messages(count=3)
        assert len(messages) == 3
        assert messages[0].author == "User2"
        assert messages[2].author == "User4"

    async def test_returns_all_when_count_exceeds(self) -> None:
        page = _make_page()
        elements = [_make_message_element(author="A", text="msg")]
        page.query_selector_all = AsyncMock(return_value=elements)

        bus = _make_event_bus()
        obs = ChatObserver(page, bus)

        messages = await obs.get_recent_messages(count=50)
        assert len(messages) == 1
