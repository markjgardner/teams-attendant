"""Screen capture for vision mode."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from playwright.async_api import Page
    from teams_attendant.utils.events import EventBus

from teams_attendant.utils.events import ScreenCaptureEvent

log = structlog.get_logger()

# Selectors for shared content areas (ordered by specificity)
_SEL_SHARED_CONTENT = "[data-tid='shared-content']"
_SEL_VIDEO_STREAM = "[data-tid='video-stream']"
_SEL_MAIN_STAGE = "[data-tid='main-stage']"

# Selectors for detecting screen sharing activity
_SEL_SHARING_INDICATORS = (
    "[data-tid='shared-content'],"
    "[aria-label*='sharing' i],"
    "[data-tid='screen-sharing-indicator'],"
    "[data-tid='sharing-screen'],"
    "text='is presenting'"
)


class ScreenCaptureObserver:
    """Captures screen-shared content during Teams meetings."""

    def __init__(
        self,
        page: Page,
        event_bus: EventBus,
        capture_interval: float = 15.0,
        only_when_sharing: bool = True,
    ) -> None:
        self._page = page
        self._event_bus = event_bus
        self._capture_interval = capture_interval
        self._only_when_sharing = only_when_sharing
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_capture_hash: str = ""

    async def start(self) -> None:
        """Start periodic screen capture."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._capture_loop())
        log.info("screen_capture.started", interval=self._capture_interval)

    async def stop(self) -> None:
        """Stop screen capture."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("screen_capture.stopped")

    async def capture_now(self) -> bytes | None:
        """Take a screenshot of the meeting content immediately."""
        try:
            # Try specific shared-content selectors first
            for selector in (_SEL_SHARED_CONTENT, _SEL_VIDEO_STREAM, _SEL_MAIN_STAGE):
                try:
                    element = await self._page.wait_for_selector(
                        selector, timeout=1_000, state="visible"
                    )
                    if element:
                        image_data: bytes = await element.screenshot(type="png")
                        log.debug("screen_capture.element", selector=selector)
                        return image_data
                except Exception:
                    continue

            # Fall back to full page screenshot
            image_data = await self._page.screenshot(type="png", full_page=False)
            log.debug("screen_capture.full_page")
            return image_data
        except Exception:
            log.warning("screen_capture.failed", exc_info=True)
            return None

    async def _capture_loop(self) -> None:
        """Periodic capture loop."""
        while self._running:
            try:
                await asyncio.sleep(self._capture_interval)

                if self._only_when_sharing:
                    sharing = await self._is_screen_sharing_active()
                    if not sharing:
                        log.debug("screen_capture.skip_no_sharing")
                        continue

                image_data = await self.capture_now()
                if image_data is None:
                    continue

                if self._image_changed(image_data):
                    event = ScreenCaptureEvent(
                        image_data=image_data, description="screen capture"
                    )
                    await self._event_bus.publish(event)
                    log.debug("screen_capture.published")
                else:
                    log.debug("screen_capture.duplicate_skipped")
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("screen_capture.loop_error", exc_info=True)

    async def _is_screen_sharing_active(self) -> bool:
        """Check if someone is sharing their screen."""
        try:
            el = await self._page.wait_for_selector(
                _SEL_SHARING_INDICATORS, timeout=1_000, state="visible"
            )
            return el is not None
        except Exception:
            return False

    def _image_changed(self, image_data: bytes) -> bool:
        """Check if the captured image is significantly different from the last one."""
        current_hash = hashlib.md5(image_data).hexdigest()
        if current_hash != self._last_capture_hash:
            self._last_capture_hash = current_hash
            return True
        return False
