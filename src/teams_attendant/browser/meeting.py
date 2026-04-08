"""Meeting join/leave and UI interaction."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

import structlog
from playwright.async_api import BrowserContext, Page, TimeoutError as PlaywrightTimeout

from teams_attendant.errors import MeetingJoinError

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Selectors – Teams Web UI (ordered by stability)
# ---------------------------------------------------------------------------

# Pre-join / lobby
_SEL_USE_WEB_APP = (
    "[data-tid='joinOnWeb'],"
    "button:has-text('Continue on this browser'),"
    "button:has-text('Join on the web instead'),"
    "a:has-text('Continue on this browser'),"
    "button:has-text('Use Teams on the web')"
)
_SEL_PREJOIN_SCREEN = (
    "[data-tid='prejoin-screen'],"
    "[data-tid='pre-join-screen'],"
    "[data-tid='prejoin'],"
    "[data-tid='lobby-join-button'],"
    "button:has-text('Join now')"
)

# Camera / microphone toggles
_SEL_CAMERA_TOGGLE = (
    "[data-tid='toggle-video'],"
    "[data-tid='prejoin-video-toggle'],"
    "button[aria-label*='camera' i],"
    "button[aria-label*='video' i]"
)
_SEL_MIC_TOGGLE = (
    "[data-tid='toggle-mute'],"
    "[data-tid='prejoin-audio-toggle'],"
    "button[aria-label*='microphone' i],"
    "button[aria-label*='mute' i]"
)

# Join button
_SEL_JOIN_BUTTON = (
    "[data-tid='prejoin-join-button'],"
    "[data-tid='lobby-join-button'],"
    "button:has-text('Join now'),"
    "button[aria-label*='Join now' i]"
)

# In-meeting indicators
_SEL_IN_MEETING = (
    "[data-tid='call-composite'],"
    "[data-tid='calling-screen'],"
    "[data-tid='hangup-button'],"
    "button[aria-label*='Leave' i],"
    "[data-tid='leave-call-button']"
)

# Waiting room
_SEL_WAITING_ROOM = (
    "[data-tid='waiting-room'],"
    "text='Waiting for someone to let you in',"
    "text='waiting to be admitted',"
    "text='Someone in the meeting should let you in soon'"
)

# Leave / hangup button
_SEL_LEAVE_BUTTON = (
    "[data-tid='hangup-button'],"
    "[data-tid='leave-call-button'],"
    "button[aria-label*='Leave' i],"
    "button[aria-label*='Hang up' i],"
    "button:has-text('Leave')"
)

# Meeting ended
_SEL_MEETING_ENDED = (
    "text='The meeting has ended',"
    "text='Meeting has ended',"
    "text='You left the meeting',"
    "[data-tid='meeting-ended'],"
    "text='Return to home screen'"
)

# Meeting title
_SEL_MEETING_TITLE = (
    "[data-tid='call-title'],"
    "[data-tid='meeting-title'],"
    "[data-tid='call-composite'] [data-tid='title']"
)

# Participant count
_SEL_PARTICIPANT_COUNT = (
    "[data-tid='participant-count'],"
    "[data-tid='roster-button'],"
    "button[aria-label*='participant' i]"
)

# Recording indicator
_SEL_RECORDING = (
    "[data-tid='recording-indicator'],"
    "[data-tid='recording-started'],"
    "text='Recording has started'"
)

# Screen sharing indicator
_SEL_SCREEN_SHARING = (
    "[data-tid='screen-sharing-indicator'],"
    "[data-tid='sharing-screen'],"
    "text='is presenting'"
)


class MeetingState(Enum):
    """Lifecycle states of a meeting session."""

    NOT_STARTED = "not_started"
    WAITING_ROOM = "waiting_room"
    JOINED = "joined"
    ENDED = "ended"
    ERROR = "error"


@dataclass
class MeetingInfo:
    """Snapshot of meeting metadata."""

    title: str = ""
    state: MeetingState = MeetingState.NOT_STARTED
    participant_count: int = 0
    is_recording: bool = False
    is_screen_sharing: bool = False


class MeetingController:
    """Controls a Teams meeting session via browser automation."""

    def __init__(self, context: BrowserContext) -> None:
        self._context = context
        self._page: Page | None = None
        self._state = MeetingState.NOT_STARTED

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def join(
        self,
        meeting_url: str,
        timeout_seconds: float = 120,
    ) -> MeetingInfo:
        """Join a Teams meeting and return info once connected.

        Raises ``TimeoutError`` if the meeting cannot be joined within
        *timeout_seconds*.
        """
        timeout_ms = int(timeout_seconds * 1000)

        log.info("meeting.join.start", url=meeting_url)
        self._page = await self._context.new_page()

        try:
            # 1. Navigate to meeting URL
            await self._page.goto(meeting_url, wait_until="domcontentloaded", timeout=timeout_ms)
            log.info("meeting.join.navigated")

            # 2. Handle "Continue on this browser" prompt
            await self._click_if_visible(self._page, _SEL_USE_WEB_APP, timeout_ms=10_000)

            # 3. Wait for pre-join screen
            await self._wait_for_any(self._page, _SEL_PREJOIN_SCREEN, timeout_ms=timeout_ms)
            log.info("meeting.join.prejoin_loaded")

            # 4. Toggle camera OFF
            await self._ensure_camera_off(self._page)

            # 5. Toggle microphone ON
            await self._ensure_mic_on(self._page)

            # 6. Click "Join now"
            await self._click_first(self._page, _SEL_JOIN_BUTTON, timeout_ms=10_000)
            log.info("meeting.join.clicked_join")

            # 7. Wait for in-meeting UI or waiting room
            joined = await self._wait_for_join_or_waiting_room(self._page, timeout_ms=timeout_ms)

            if not joined:
                # In waiting room – wait for admission
                log.info("meeting.join.waiting_room")
                self._state = MeetingState.WAITING_ROOM
                await self._wait_for_admission(self._page, timeout_ms=timeout_ms)

            self._state = MeetingState.JOINED
            log.info("meeting.join.connected")

            info = await self.get_info()
            return info

        except PlaywrightTimeout as exc:
            self._state = MeetingState.ERROR
            log.error("meeting.join.timeout", error=str(exc))
            raise MeetingJoinError(
                f"Timed out joining the meeting after {timeout_seconds}s"
            ) from exc
        except Exception as exc:
            self._state = MeetingState.ERROR
            log.error("meeting.join.error", error=str(exc))
            raise MeetingJoinError(str(exc)) from exc

    async def leave(self) -> None:
        """Leave the meeting and close the page."""
        if self._page is None or self._page.is_closed():
            log.warning("meeting.leave.no_page")
            return

        log.info("meeting.leave.start")
        try:
            await self._click_first(self._page, _SEL_LEAVE_BUTTON, timeout_ms=5_000)
            log.info("meeting.leave.clicked")
            await self._page.wait_for_timeout(1_500)
        except (PlaywrightTimeout, Exception) as exc:
            log.warning("meeting.leave.button_failed", error=str(exc))
        finally:
            if not self._page.is_closed():
                await self._page.close()
            self._state = MeetingState.ENDED
            log.info("meeting.leave.done")

    async def get_state(self) -> MeetingState:
        """Detect the current meeting state from the DOM."""
        if self._page is None or self._page.is_closed():
            self._state = MeetingState.ENDED
            return self._state

        try:
            if await self._is_visible(self._page, _SEL_MEETING_ENDED):
                self._state = MeetingState.ENDED
            elif await self._is_visible(self._page, _SEL_WAITING_ROOM):
                self._state = MeetingState.WAITING_ROOM
            elif await self._is_visible(self._page, _SEL_IN_MEETING):
                self._state = MeetingState.JOINED
        except Exception:
            self._state = MeetingState.ERROR

        return self._state

    async def get_info(self) -> MeetingInfo:
        """Extract meeting metadata from the DOM."""
        state = await self.get_state()
        title = ""
        participant_count = 0
        is_recording = False
        is_screen_sharing = False

        if self._page and not self._page.is_closed():
            title = await self._extract_text(self._page, _SEL_MEETING_TITLE)
            participant_count = await self._extract_participant_count(self._page)
            is_recording = await self._is_visible(self._page, _SEL_RECORDING)
            is_screen_sharing = await self._is_visible(self._page, _SEL_SCREEN_SHARING)

        return MeetingInfo(
            title=title,
            state=state,
            participant_count=participant_count,
            is_recording=is_recording,
            is_screen_sharing=is_screen_sharing,
        )

    async def wait_for_meeting_end(self, poll_interval: float = 5.0) -> None:
        """Block until the meeting ends or the page closes."""
        log.info("meeting.wait_for_end.start", poll_interval=poll_interval)

        while True:
            if self._page is None or self._page.is_closed():
                log.info("meeting.wait_for_end.page_closed")
                self._state = MeetingState.ENDED
                break

            state = await self.get_state()
            if state in (MeetingState.ENDED, MeetingState.ERROR):
                log.info("meeting.wait_for_end.detected", state=state.value)
                break

            await asyncio.sleep(poll_interval)

    @property
    def page(self) -> Page | None:
        """The Playwright page used for the meeting, if any."""
        return self._page

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _click_if_visible(
        self,
        page: Page,
        selector: str,
        *,
        timeout_ms: int = 5_000,
    ) -> bool:
        """Click the first matching element if it appears within *timeout_ms*.

        Returns ``True`` if an element was clicked, ``False`` otherwise.
        """
        try:
            el = await page.wait_for_selector(selector, timeout=timeout_ms, state="visible")
            if el:
                await el.click()
                log.debug("meeting._click_if_visible.clicked", selector=selector)
                return True
        except PlaywrightTimeout:
            log.debug("meeting._click_if_visible.not_found", selector=selector)
        return False

    async def _click_first(
        self,
        page: Page,
        selector: str,
        *,
        timeout_ms: int = 10_000,
    ) -> None:
        """Wait for the first matching element and click it."""
        el = await page.wait_for_selector(selector, timeout=timeout_ms, state="visible")
        if el:
            await el.click()

    async def _wait_for_any(
        self,
        page: Page,
        selector: str,
        *,
        timeout_ms: int = 30_000,
    ) -> None:
        """Wait until at least one selector from a comma-separated list appears."""
        await page.wait_for_selector(selector, timeout=timeout_ms, state="visible")

    async def _is_visible(self, page: Page, selector: str) -> bool:
        """Return ``True`` if any element matching *selector* is visible."""
        try:
            el = await page.wait_for_selector(selector, timeout=1_000, state="visible")
            return el is not None
        except (PlaywrightTimeout, Exception):
            return False

    async def _extract_text(self, page: Page, selector: str) -> str:
        """Return the inner text of the first matching element, or ``""``."""
        try:
            el = await page.wait_for_selector(selector, timeout=2_000, state="visible")
            if el:
                return (await el.inner_text()).strip()
        except (PlaywrightTimeout, Exception):
            pass
        return ""

    async def _extract_participant_count(self, page: Page) -> int:
        """Try to parse a participant count from the roster button or badge."""
        text = await self._extract_text(page, _SEL_PARTICIPANT_COUNT)
        if not text:
            return 0
        # Extract first integer from the text (e.g. "Participants (5)" → 5)
        digits = "".join(ch for ch in text if ch.isdigit())
        return int(digits) if digits else 0

    async def _ensure_camera_off(self, page: Page) -> None:
        """Toggle camera off if it is currently on."""
        try:
            el = await page.wait_for_selector(_SEL_CAMERA_TOGGLE, timeout=5_000, state="visible")
            if el:
                aria = await el.get_attribute("aria-checked")
                aria_pressed = await el.get_attribute("aria-pressed")
                is_on = (aria or aria_pressed or "").lower() in ("true",)

                if is_on:
                    await el.click()
                    log.info("meeting.camera.toggled_off")
                else:
                    log.info("meeting.camera.already_off")
        except PlaywrightTimeout:
            log.warning("meeting.camera.toggle_not_found")

    async def _ensure_mic_on(self, page: Page) -> None:
        """Toggle microphone on if it is currently muted."""
        try:
            el = await page.wait_for_selector(_SEL_MIC_TOGGLE, timeout=5_000, state="visible")
            if el:
                aria = await el.get_attribute("aria-checked")
                aria_pressed = await el.get_attribute("aria-pressed")
                # For mic: aria-checked="true" or aria-pressed="true" means unmuted
                is_on = (aria or aria_pressed or "").lower() in ("true",)

                if not is_on:
                    await el.click()
                    log.info("meeting.mic.toggled_on")
                else:
                    log.info("meeting.mic.already_on")
        except PlaywrightTimeout:
            log.warning("meeting.mic.toggle_not_found")

    async def _wait_for_join_or_waiting_room(
        self,
        page: Page,
        *,
        timeout_ms: int = 60_000,
    ) -> bool:
        """Wait for either the in-meeting UI or the waiting room.

        Returns ``True`` if directly joined, ``False`` if in waiting room.
        """
        combined = f"{_SEL_IN_MEETING},{_SEL_WAITING_ROOM}"
        try:
            el = await page.wait_for_selector(combined, timeout=timeout_ms, state="visible")
            if el:
                # Check which one matched
                for sel in _SEL_WAITING_ROOM.split(","):
                    sel = sel.strip()
                    try:
                        waiting = await page.wait_for_selector(
                            sel, timeout=500, state="visible"
                        )
                        if waiting:
                            return False
                    except PlaywrightTimeout:
                        continue
                return True
        except PlaywrightTimeout:
            raise
        return True

    async def _wait_for_admission(
        self,
        page: Page,
        *,
        timeout_ms: int = 120_000,
    ) -> None:
        """Wait until the meeting admits us from the waiting room."""
        log.info("meeting.waiting_room.waiting", timeout_ms=timeout_ms)
        await page.wait_for_selector(_SEL_IN_MEETING, timeout=timeout_ms, state="visible")
        log.info("meeting.waiting_room.admitted")
