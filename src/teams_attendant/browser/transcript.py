"""Meeting transcript/caption observation via DOM scraping."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from playwright.async_api import Page

    from teams_attendant.utils.events import EventBus

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Selectors – Teams Web UI captions / transcript
# ---------------------------------------------------------------------------

_SEL_MORE_ACTIONS = (
    "[data-tid='more-button'],"
    "button[aria-label*='More actions' i],"
    "button[aria-label*='More' i]:not([aria-label*='emoji' i])"
)

_SEL_TURN_ON_CAPTIONS = (
    "[data-tid='toggle-captions'],"
    "button[aria-label*='Turn on live captions' i],"
    "button:has-text('Turn on live captions'),"
    "button:has-text('Live captions'),"
    "[data-tid='captions-button']"
)

_SEL_CAPTIONS_CONFIRM = (
    "button:has-text('Got it'),"
    "button:has-text('OK'),"
    "button:has-text('Confirm'),"
    "[data-tid='captions-confirm']"
)

_SEL_CAPTIONS_CONTAINER = (
    "[data-tid='live-captions-container'],"
    "[data-tid='captions-container'],"
    "[class*='captions-container'],"
    "[class*='caption-container'],"
    "[aria-label*='captions' i][role='log'],"
    "[aria-label*='captions' i][role='region']"
)

_SEL_CAPTION_ENTRY = (
    "[data-tid='caption-text'],"
    "[data-tid='live-caption-entry'],"
    "[class*='caption-entry'],"
    "[class*='caption-line'],"
    "[class*='caption-item']"
)

_SEL_CAPTION_SPEAKER = (
    "[data-tid='caption-speaker'],"
    "[class*='caption-speaker'],"
    "[class*='speaker-name']"
)

_SEL_CAPTION_TEXT = (
    "[data-tid='caption-text-content'],"
    "[class*='caption-text'],"
    "[class*='caption-content']"
)

_SEL_TRANSCRIPT_BUTTON = (
    "[data-tid='transcript-button'],"
    "button[aria-label*='Transcript' i],"
    "button:has-text('Transcript')"
)

_SEL_TRANSCRIPT_PANEL = (
    "[data-tid='transcript-pane'],"
    "[data-tid='transcript-panel'],"
    "[class*='transcript-pane'],"
    "[aria-label*='Transcript' i][role='log']"
)

_SEL_TRANSCRIPT_ENTRY = (
    "[data-tid='transcript-entry'],"
    "[class*='transcript-entry'],"
    "[class*='transcript-item']"
)


@dataclass
class CaptionEntry:
    """A parsed caption/transcript entry."""

    id: str
    speaker: str
    text: str
    timestamp: datetime


class TranscriptObserver:
    """Observes Teams meeting captions/transcript via DOM scraping.

    Attempts to enable and read live captions.  Falls back to the
    transcript panel if available.  Publishes ``TranscriptEvent`` on the
    event bus — the same event type used by audio-based STT.
    """

    def __init__(self, page: Page, event_bus: EventBus) -> None:
        self._page = page
        self._event_bus = event_bus
        self._running: bool = False
        self._seen_entries: set[str] = set()
        self._poll_interval: float = 1.5
        self._task: asyncio.Task[None] | None = None
        self._captions_available: bool = False
        self._source: str = ""

    # -- public API ---------------------------------------------------------

    async def start(self) -> bool:
        """Start observing transcript/caption entries.

        Returns ``True`` when a source (captions or transcript panel) is
        available, ``False`` otherwise.
        """
        if self._running:
            return self._captions_available

        log.info("transcript_observer.starting")

        if await self._enable_live_captions():
            self._captions_available = True
        elif await self._open_transcript_panel():
            self._captions_available = True
        else:
            log.warning("transcript_observer.no_source")
            return False

        # Snapshot existing entries so we don't re-emit them
        existing = await self._parse_entries_from_dom()
        for entry in existing:
            self._seen_entries.add(entry.id)
        log.info("transcript_observer.snapshot", count=len(existing))

        self._running = True
        self._task = asyncio.create_task(self._poll_entries())
        return True

    async def stop(self) -> None:
        """Stop observing."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("transcript_observer.stopped")

    @property
    def is_available(self) -> bool:
        """Whether a caption/transcript source was found."""
        return self._captions_available

    @property
    def source(self) -> str:
        """Active source: ``"captions"``, ``"transcript"``, or ``""``."""
        return self._source

    # -- activation helpers -------------------------------------------------

    async def _enable_live_captions(self) -> bool:
        """Try to turn on live captions via the meeting toolbar."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        # Already visible?
        try:
            container = await self._page.wait_for_selector(
                _SEL_CAPTIONS_CONTAINER, timeout=1_000, state="visible",
            )
            if container:
                log.debug("transcript_observer.captions_already_on")
                self._source = "captions"
                return True
        except PlaywrightTimeout:
            pass

        # Open the "More actions" menu
        try:
            more_btn = await self._page.wait_for_selector(
                _SEL_MORE_ACTIONS, timeout=5_000, state="visible",
            )
            if more_btn:
                await more_btn.click()
        except PlaywrightTimeout:
            log.debug("transcript_observer.more_button_not_found")
            return False

        # Click "Turn on live captions"
        try:
            captions_btn = await self._page.wait_for_selector(
                _SEL_TURN_ON_CAPTIONS, timeout=5_000, state="visible",
            )
            if captions_btn:
                await captions_btn.click()
            else:
                return False
        except PlaywrightTimeout:
            log.debug("transcript_observer.captions_button_not_found")
            return False

        # Dismiss any confirmation dialog
        try:
            confirm_btn = await self._page.wait_for_selector(
                _SEL_CAPTIONS_CONFIRM, timeout=3_000, state="visible",
            )
            if confirm_btn:
                await confirm_btn.click()
        except PlaywrightTimeout:
            pass  # no confirmation needed

        # Wait for the captions container to appear
        try:
            await self._page.wait_for_selector(
                _SEL_CAPTIONS_CONTAINER, timeout=5_000, state="visible",
            )
            self._source = "captions"
            log.info("transcript_observer.captions_enabled")
            return True
        except PlaywrightTimeout:
            log.warning("transcript_observer.captions_container_timeout")
            return False

    async def _open_transcript_panel(self) -> bool:
        """Try to open the transcript panel as a fallback source."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        # Already visible?
        try:
            panel = await self._page.wait_for_selector(
                _SEL_TRANSCRIPT_PANEL, timeout=1_000, state="visible",
            )
            if panel:
                log.debug("transcript_observer.panel_already_open")
                self._source = "transcript"
                return True
        except PlaywrightTimeout:
            pass

        # Click the transcript button
        try:
            btn = await self._page.wait_for_selector(
                _SEL_TRANSCRIPT_BUTTON, timeout=5_000, state="visible",
            )
            if btn:
                await btn.click()
                await self._page.wait_for_selector(
                    _SEL_TRANSCRIPT_PANEL, timeout=5_000, state="visible",
                )
                self._source = "transcript"
                log.info("transcript_observer.panel_opened")
                return True
        except PlaywrightTimeout:
            log.debug("transcript_observer.panel_open_failed")

        return False

    # -- polling ------------------------------------------------------------

    async def _poll_entries(self) -> None:
        """Polling loop for new caption/transcript entries."""
        from teams_attendant.utils.events import TranscriptEvent

        while self._running:
            try:
                entries = await self._parse_entries_from_dom()
                for entry in entries:
                    if entry.id not in self._seen_entries:
                        self._seen_entries.add(entry.id)
                        event = TranscriptEvent(
                            text=entry.text,
                            speaker=entry.speaker,
                            is_final=True,
                            confidence=1.0,
                        )
                        await self._event_bus.publish(event)
                        log.info(
                            "transcript_observer.new_entry",
                            speaker=entry.speaker,
                            text=entry.text[:80],
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("transcript_observer.poll_error", error=str(exc))

            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                raise

    # -- DOM parsing --------------------------------------------------------

    async def _parse_entries_from_dom(self) -> list[CaptionEntry]:
        """Parse caption/transcript entries from the Teams DOM."""
        entries: list[CaptionEntry] = []

        if self._source == "captions":
            container_sel = _SEL_CAPTIONS_CONTAINER
            entry_sel = _SEL_CAPTION_ENTRY
        elif self._source == "transcript":
            container_sel = _SEL_TRANSCRIPT_PANEL
            entry_sel = _SEL_TRANSCRIPT_ENTRY
        else:
            return entries

        try:
            container = await self._page.query_selector(container_sel)
            if not container:
                return entries
            elements = await container.query_selector_all(entry_sel)
        except Exception as exc:
            log.warning(
                "transcript_observer.parse_container_error", error=str(exc),
            )
            return entries

        for idx, el in enumerate(elements):
            try:
                # Extract speaker name
                speaker = ""
                speaker_el = await el.query_selector(_SEL_CAPTION_SPEAKER)
                if speaker_el:
                    speaker = (await speaker_el.inner_text()).strip()

                # Extract caption/transcript text
                text = ""
                text_el = await el.query_selector(_SEL_CAPTION_TEXT)
                if text_el:
                    text = (await text_el.inner_text()).strip()
                else:
                    # Fallback: full inner text minus speaker
                    full = (await el.inner_text()).strip()
                    if speaker and full.startswith(speaker):
                        text = full[len(speaker):].strip()
                    else:
                        text = full

                if not text:
                    continue

                entry_id = self._generate_entry_id(speaker, text, idx)
                entries.append(
                    CaptionEntry(
                        id=entry_id,
                        speaker=speaker,
                        text=text,
                        timestamp=datetime.now(timezone.utc),
                    )
                )
            except Exception as exc:
                log.warning(
                    "transcript_observer.parse_entry_error",
                    index=idx,
                    error=str(exc),
                )
                continue

        return entries

    def _generate_entry_id(self, speaker: str, text: str, index: int) -> str:
        """Generate a unique-ish ID for a caption/transcript entry."""
        raw = f"{speaker}:{text}:{index}"
        return hashlib.md5(raw.encode()).hexdigest()  # noqa: S324
