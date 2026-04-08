"""Tests for teams_attendant.browser.transcript module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from teams_attendant.browser.transcript import (
    CaptionEntry,
    TranscriptObserver,
    _SEL_CAPTIONS_CONTAINER,
    _SEL_CAPTION_SPEAKER,
    _SEL_CAPTION_TEXT,
    _SEL_MORE_ACTIONS,
    _SEL_TRANSCRIPT_BUTTON,
    _SEL_TRANSCRIPT_PANEL,
    _SEL_TURN_ON_CAPTIONS,
)
from teams_attendant.utils.events import Event, EventBus

# We need Playwright's TimeoutError for simulating selector failures.
from playwright.async_api import TimeoutError as PlaywrightTimeout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page() -> AsyncMock:
    """Build a mock Playwright Page."""
    page = AsyncMock()
    type(page).is_closed = MagicMock(return_value=False)
    page.wait_for_selector = AsyncMock(return_value=AsyncMock())
    page.query_selector = AsyncMock(return_value=None)
    return page


def _make_element(*, inner_text: str = "") -> AsyncMock:
    """Build a mock DOM element."""
    el = AsyncMock()
    el.click = AsyncMock()
    el.inner_text = AsyncMock(return_value=inner_text)
    el.query_selector = AsyncMock(return_value=None)
    el.query_selector_all = AsyncMock(return_value=[])
    return el


def _make_caption_element(speaker: str = "Alice", text: str = "hello") -> AsyncMock:
    """Build a mock caption entry element with speaker and text sub-elements."""
    speaker_el = _make_element(inner_text=speaker)
    text_el = _make_element(inner_text=text)

    el = _make_element(inner_text=f"{speaker}\n{text}")

    async def _qs(selector: str) -> AsyncMock | None:
        if selector == _SEL_CAPTION_SPEAKER:
            return speaker_el
        if selector == _SEL_CAPTION_TEXT:
            return text_el
        return None

    el.query_selector = AsyncMock(side_effect=_qs)
    return el


def _make_event_bus() -> EventBus:
    """Build a real EventBus."""
    return EventBus()


def _make_container(*entry_elements: AsyncMock) -> AsyncMock:
    """Build a mock captions/transcript container with child entries."""
    container = _make_element()
    container.query_selector_all = AsyncMock(return_value=list(entry_elements))
    return container


# ---------------------------------------------------------------------------
# CaptionEntry dataclass
# ---------------------------------------------------------------------------


class TestCaptionEntry:
    def test_construction(self) -> None:
        from datetime import datetime, timezone

        entry = CaptionEntry(
            id="abc", speaker="Alice", text="hello",
            timestamp=datetime.now(timezone.utc),
        )
        assert entry.id == "abc"
        assert entry.speaker == "Alice"
        assert entry.text == "hello"


# ---------------------------------------------------------------------------
# _generate_entry_id
# ---------------------------------------------------------------------------


class TestGenerateEntryId:
    def test_deterministic(self) -> None:
        obs = TranscriptObserver(_make_page(), _make_event_bus())
        id1 = obs._generate_entry_id("Alice", "hello", 0)
        id2 = obs._generate_entry_id("Alice", "hello", 0)
        assert id1 == id2

    def test_different_for_different_inputs(self) -> None:
        obs = TranscriptObserver(_make_page(), _make_event_bus())
        id1 = obs._generate_entry_id("Alice", "hello", 0)
        id2 = obs._generate_entry_id("Bob", "hello", 0)
        id3 = obs._generate_entry_id("Alice", "hello", 1)
        assert id1 != id2
        assert id1 != id3

    def test_seen_entries_prevents_duplicates(self) -> None:
        obs = TranscriptObserver(_make_page(), _make_event_bus())
        entry_id = obs._generate_entry_id("Alice", "hello", 0)
        obs._seen_entries.add(entry_id)
        assert entry_id in obs._seen_entries
        # A different entry is not in the set
        other_id = obs._generate_entry_id("Bob", "world", 1)
        assert other_id not in obs._seen_entries


# ---------------------------------------------------------------------------
# start – captions
# ---------------------------------------------------------------------------


class TestStartCaptions:
    async def test_start_enables_captions_returns_true(self) -> None:
        """Captions not initially visible; More → Turn on captions → container appears."""
        page = _make_page()
        more_btn = _make_element()
        captions_btn = _make_element()
        container_el = _make_element()

        container_call_count = 0

        async def _selector_router(selector: str, **kwargs: object) -> AsyncMock:
            nonlocal container_call_count
            if selector == _SEL_CAPTIONS_CONTAINER:
                container_call_count += 1
                if container_call_count == 1:
                    raise PlaywrightTimeout("not visible yet")
                return container_el
            if selector == _SEL_MORE_ACTIONS:
                return more_btn
            if selector == _SEL_TURN_ON_CAPTIONS:
                return captions_btn
            # Confirmation dialog – not present
            raise PlaywrightTimeout("not found")

        page.wait_for_selector = AsyncMock(side_effect=_selector_router)
        page.query_selector = AsyncMock(return_value=None)

        obs = TranscriptObserver(page, _make_event_bus())
        result = await obs.start()

        assert result is True
        assert obs.source == "captions"
        assert obs.is_available is True
        more_btn.click.assert_awaited()
        captions_btn.click.assert_awaited()

        await obs.stop()

    async def test_start_captions_already_on(self) -> None:
        """Captions container already visible → returns True immediately."""
        page = _make_page()
        container_el = _make_element()
        page.wait_for_selector = AsyncMock(return_value=container_el)
        page.query_selector = AsyncMock(return_value=None)

        obs = TranscriptObserver(page, _make_event_bus())
        result = await obs.start()

        assert result is True
        assert obs.source == "captions"

        await obs.stop()


# ---------------------------------------------------------------------------
# start – transcript fallback
# ---------------------------------------------------------------------------


class TestStartTranscriptFallback:
    async def test_start_falls_back_to_transcript_panel(self) -> None:
        """Captions fail; transcript panel opens as fallback."""
        page = _make_page()
        transcript_btn = _make_element()
        panel_el = _make_element()

        panel_call_count = 0

        async def _selector_router(selector: str, **kwargs: object) -> AsyncMock:
            nonlocal panel_call_count
            # All caption selectors fail
            if selector in (
                _SEL_CAPTIONS_CONTAINER, _SEL_MORE_ACTIONS,
                _SEL_TURN_ON_CAPTIONS,
            ):
                raise PlaywrightTimeout("not found")
            if selector == _SEL_TRANSCRIPT_PANEL:
                panel_call_count += 1
                if panel_call_count == 1:
                    raise PlaywrightTimeout("panel not open yet")
                return panel_el
            if selector == _SEL_TRANSCRIPT_BUTTON:
                return transcript_btn
            raise PlaywrightTimeout("not found")

        page.wait_for_selector = AsyncMock(side_effect=_selector_router)
        page.query_selector = AsyncMock(return_value=None)

        obs = TranscriptObserver(page, _make_event_bus())
        result = await obs.start()

        assert result is True
        assert obs.source == "transcript"
        assert obs.is_available is True
        transcript_btn.click.assert_awaited()

        await obs.stop()


# ---------------------------------------------------------------------------
# start – no source available
# ---------------------------------------------------------------------------


class TestStartNoSource:
    async def test_start_no_source_returns_false(self) -> None:
        """Both captions and transcript panel fail → returns False."""
        page = _make_page()
        page.wait_for_selector = AsyncMock(
            side_effect=PlaywrightTimeout("nothing found"),
        )

        obs = TranscriptObserver(page, _make_event_bus())
        result = await obs.start()

        assert result is False
        assert obs.is_available is False
        assert obs.source == ""


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestStop:
    async def test_stop_cancels_task(self) -> None:
        """stop() should cancel the polling task and reset state."""
        page = _make_page()
        page.wait_for_selector = AsyncMock(return_value=_make_element())
        page.query_selector = AsyncMock(return_value=None)

        obs = TranscriptObserver(page, _make_event_bus())

        await obs.start()
        assert obs._running is True
        assert obs._task is not None

        await obs.stop()
        assert obs._running is False
        assert obs._task is None


# ---------------------------------------------------------------------------
# _parse_entries_from_dom
# ---------------------------------------------------------------------------


class TestParseEntries:
    async def test_parse_caption_entries(self) -> None:
        """Should parse speaker and text from caption entry elements."""
        page = _make_page()
        entry1 = _make_caption_element(speaker="Alice", text="Hello world")
        entry2 = _make_caption_element(speaker="Bob", text="Hi there")
        container = _make_container(entry1, entry2)
        page.query_selector = AsyncMock(return_value=container)

        obs = TranscriptObserver(page, _make_event_bus())
        obs._source = "captions"

        entries = await obs._parse_entries_from_dom()

        assert len(entries) == 2
        assert entries[0].speaker == "Alice"
        assert entries[0].text == "Hello world"
        assert entries[1].speaker == "Bob"
        assert entries[1].text == "Hi there"

    async def test_parse_entries_skips_empty_text(self) -> None:
        """Entries with empty text should be skipped."""
        page = _make_page()
        empty = _make_caption_element(speaker="Alice", text="")
        good = _make_caption_element(speaker="Bob", text="hi")
        container = _make_container(empty, good)
        page.query_selector = AsyncMock(return_value=container)

        obs = TranscriptObserver(page, _make_event_bus())
        obs._source = "captions"

        entries = await obs._parse_entries_from_dom()

        assert len(entries) == 1
        assert entries[0].speaker == "Bob"

    async def test_parse_entries_fallback_text_extraction(self) -> None:
        """When no text sub-element exists, extract text from inner_text minus speaker."""
        page = _make_page()

        speaker_el = _make_element(inner_text="Alice")
        el = _make_element(inner_text="Alice\nHello from fallback")

        async def _qs(selector: str) -> AsyncMock | None:
            if selector == _SEL_CAPTION_SPEAKER:
                return speaker_el
            return None  # _SEL_CAPTION_TEXT → not found

        el.query_selector = AsyncMock(side_effect=_qs)

        container = _make_container(el)
        page.query_selector = AsyncMock(return_value=container)

        obs = TranscriptObserver(page, _make_event_bus())
        obs._source = "captions"

        entries = await obs._parse_entries_from_dom()

        assert len(entries) == 1
        assert entries[0].speaker == "Alice"
        assert "Hello from fallback" in entries[0].text

    async def test_parse_entries_no_source_returns_empty(self) -> None:
        """When source is empty string, returns empty list."""
        obs = TranscriptObserver(_make_page(), _make_event_bus())
        obs._source = ""

        entries = await obs._parse_entries_from_dom()
        assert entries == []


# ---------------------------------------------------------------------------
# Polling – new entries trigger events
# ---------------------------------------------------------------------------


class TestPolling:
    async def test_poll_publishes_new_entries(self) -> None:
        """New entries found during polling should be published to event bus."""
        page = _make_page()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus = _make_event_bus()
        bus.subscribe("transcript", handler)
        await bus.start()

        obs = TranscriptObserver(page, bus)
        obs._source = "captions"
        obs._captions_available = True
        obs._running = True
        obs._poll_interval = 0.05

        # Pre-populate with a "seen" entry ID
        seen_id = obs._generate_entry_id("Alice", "old msg", 0)
        obs._seen_entries.add(seen_id)

        old_entry = _make_caption_element(speaker="Alice", text="old msg")
        new_entry = _make_caption_element(speaker="Bob", text="new msg")
        container = _make_container(old_entry, new_entry)
        page.query_selector = AsyncMock(return_value=container)

        obs._task = asyncio.create_task(obs._poll_entries())
        await asyncio.sleep(0.25)

        obs._running = False
        obs._task.cancel()
        try:
            await obs._task
        except asyncio.CancelledError:
            pass
        await bus.stop()

        texts = [e.data["text"] for e in received]
        assert "new msg" in texts
        assert "old msg" not in texts

    async def test_seen_entries_not_reemitted(self) -> None:
        """Messages already in _seen_entries should not be published again."""
        page = _make_page()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus = _make_event_bus()
        bus.subscribe("transcript", handler)
        await bus.start()

        entry = _make_caption_element(speaker="Alice", text="hello")
        container = _make_container(entry)
        page.query_selector = AsyncMock(return_value=container)

        obs = TranscriptObserver(page, bus)
        obs._source = "captions"
        obs._captions_available = True
        obs._running = True
        obs._poll_interval = 0.05

        # Mark the entry as already seen
        seen_id = obs._generate_entry_id("Alice", "hello", 0)
        obs._seen_entries.add(seen_id)

        obs._task = asyncio.create_task(obs._poll_entries())
        await asyncio.sleep(0.25)

        obs._running = False
        obs._task.cancel()
        try:
            await obs._task
        except asyncio.CancelledError:
            pass
        await bus.stop()

        assert len(received) == 0


# ---------------------------------------------------------------------------
# Snapshot on start
# ---------------------------------------------------------------------------


class TestSnapshotOnStart:
    async def test_snapshot_existing_on_start(self) -> None:
        """Existing entries at start() time are snapshotted, not published."""
        page = _make_page()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus = _make_event_bus()
        bus.subscribe("transcript", handler)
        await bus.start()

        # Captions container already visible
        page.wait_for_selector = AsyncMock(return_value=_make_element())

        # _parse_entries_from_dom will find this existing entry
        existing = _make_caption_element(speaker="Alice", text="pre-existing")
        container = _make_container(existing)
        page.query_selector = AsyncMock(return_value=container)

        obs = TranscriptObserver(page, bus)
        obs._poll_interval = 0.05

        await obs.start()

        # Existing entry should be in _seen_entries
        assert len(obs._seen_entries) == 1

        # Let polling run a couple of cycles
        await asyncio.sleep(0.25)
        await obs.stop()
        await bus.stop()

        # No events — the only entry was snapshotted before polling began
        assert len(received) == 0
