"""Tests for teams_attendant.browser.meeting module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from teams_attendant.browser.meeting import (
    MeetingController,
    MeetingInfo,
    MeetingState,
    _SEL_CAMERA_TOGGLE,
    _SEL_IN_MEETING,
    _SEL_JOIN_BUTTON,
    _SEL_MEETING_ENDED,
    _SEL_MIC_TOGGLE,
    _SEL_WAITING_ROOM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_element(
    *,
    aria_checked: str | None = None,
    aria_pressed: str | None = None,
    inner_text: str = "",
) -> AsyncMock:
    """Build a mock that works as both Playwright ElementHandle and Locator."""
    el = AsyncMock()
    el.click = AsyncMock()
    el.wait_for = AsyncMock()

    async def _get_attr(name: str) -> str | None:
        if name == "aria-checked":
            return aria_checked
        if name == "aria-pressed":
            return aria_pressed
        return None

    el.get_attribute = AsyncMock(side_effect=_get_attr)
    el.inner_text = AsyncMock(return_value=inner_text)
    return el


def _make_page(*, is_closed: bool = False) -> AsyncMock:
    """Build a mock Playwright Page with common stubs."""
    page = AsyncMock()
    type(page).is_closed = MagicMock(return_value=is_closed)
    page.goto = AsyncMock()
    page.close = AsyncMock()
    page.wait_for_timeout = AsyncMock()

    # Default: wait_for_selector returns a mock element
    page.wait_for_selector = AsyncMock(return_value=_make_element())

    # locator() is synchronous in Playwright; returns a Locator with .first
    def _make_locator(selector: str) -> MagicMock:
        loc = MagicMock()
        loc.first = _make_element()
        return loc

    page.locator = MagicMock(side_effect=_make_locator)
    return page


def _make_context(page: AsyncMock | None = None) -> AsyncMock:
    """Build a mock BrowserContext."""
    ctx = AsyncMock()
    ctx.new_page = AsyncMock(return_value=page or _make_page())
    return ctx


# ---------------------------------------------------------------------------
# join – navigation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_join_navigates_to_meeting_url() -> None:
    """join() should open a new page and navigate to the meeting URL."""
    page = _make_page()
    ctx = _make_context(page)
    ctrl = MeetingController(ctx)

    url = "https://teams.microsoft.com/l/meetup-join/test-meeting"
    await ctrl.join(url, timeout_seconds=10)

    page.goto.assert_awaited_once()
    actual_url = page.goto.call_args.args[0]
    assert actual_url == url


# ---------------------------------------------------------------------------
# join – camera & mic toggles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_join_toggles_camera_off_when_on() -> None:
    """join() should click camera toggle when camera is on."""
    camera_el = _make_element(aria_checked="true")
    mic_el = _make_element(aria_checked="true")  # mic already on

    def _locator_router(selector: str) -> MagicMock:
        loc = MagicMock()
        if selector == _SEL_CAMERA_TOGGLE:
            loc.first = camera_el
        elif selector == _SEL_MIC_TOGGLE:
            loc.first = mic_el
        else:
            loc.first = _make_element()
        return loc

    page = _make_page()
    page.locator = MagicMock(side_effect=_locator_router)
    ctx = _make_context(page)
    ctrl = MeetingController(ctx)

    await ctrl.join("https://example.com/meeting", timeout_seconds=10)

    camera_el.click.assert_awaited()


@pytest.mark.asyncio
async def test_join_toggles_mic_on_when_muted() -> None:
    """join() should click mic toggle when mic is muted."""
    camera_el = _make_element(aria_checked="false")  # camera already off
    mic_el = _make_element(aria_checked="false")  # mic muted → needs click

    def _locator_router(selector: str) -> MagicMock:
        loc = MagicMock()
        if selector == _SEL_CAMERA_TOGGLE:
            loc.first = camera_el
        elif selector == _SEL_MIC_TOGGLE:
            loc.first = mic_el
        else:
            loc.first = _make_element()
        return loc

    page = _make_page()
    page.locator = MagicMock(side_effect=_locator_router)
    ctx = _make_context(page)
    ctrl = MeetingController(ctx)

    await ctrl.join("https://example.com/meeting", timeout_seconds=10)

    mic_el.click.assert_awaited()


# ---------------------------------------------------------------------------
# join – join button
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_join_clicks_join_button() -> None:
    """join() should click the 'Join now' button."""
    join_el = _make_element()

    def _locator_router(selector: str) -> MagicMock:
        loc = MagicMock()
        if selector == _SEL_JOIN_BUTTON:
            loc.first = join_el
        else:
            loc.first = _make_element()
        return loc

    page = _make_page()
    page.locator = MagicMock(side_effect=_locator_router)
    ctx = _make_context(page)
    ctrl = MeetingController(ctx)

    await ctrl.join("https://example.com/meeting", timeout_seconds=10)

    join_el.click.assert_awaited()


# ---------------------------------------------------------------------------
# join – stores page reference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_join_stores_page() -> None:
    """join() should store the page as ctrl.page."""
    page = _make_page()
    ctx = _make_context(page)
    ctrl = MeetingController(ctx)

    await ctrl.join("https://example.com/meeting", timeout_seconds=10)

    assert ctrl.page is page


# ---------------------------------------------------------------------------
# leave
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leave_clicks_leave_and_closes_page() -> None:
    """leave() should click the leave button and close the page."""
    leave_el = _make_element()

    def _locator_router(selector: str) -> MagicMock:
        loc = MagicMock()
        loc.first = leave_el
        return loc

    page = _make_page()
    page.locator = MagicMock(side_effect=_locator_router)
    ctx = _make_context(page)

    ctrl = MeetingController(ctx)
    ctrl._page = page
    ctrl._state = MeetingState.JOINED

    await ctrl.leave()

    leave_el.click.assert_awaited()
    page.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_leave_noop_when_no_page() -> None:
    """leave() should not raise when there is no page."""
    ctx = _make_context()
    ctrl = MeetingController(ctx)
    ctrl._page = None

    await ctrl.leave()  # should not raise


# ---------------------------------------------------------------------------
# get_state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_state_ended_when_page_closed() -> None:
    """get_state() should return ENDED when the page is closed."""
    page = _make_page(is_closed=True)
    ctx = _make_context(page)
    ctrl = MeetingController(ctx)
    ctrl._page = page

    assert await ctrl.get_state() == MeetingState.ENDED


@pytest.mark.asyncio
async def test_get_state_detects_joined() -> None:
    """get_state() should return JOINED when in-meeting selectors are visible."""
    from playwright.async_api import TimeoutError as PlaywrightTimeout

    async def _selector_router(selector: str, **kwargs: object) -> AsyncMock | None:
        if _SEL_MEETING_ENDED in selector:
            raise PlaywrightTimeout("timeout")
        if _SEL_WAITING_ROOM in selector:
            raise PlaywrightTimeout("timeout")
        if _SEL_IN_MEETING in selector:
            return _make_element()
        raise PlaywrightTimeout("timeout")

    page = _make_page()
    page.wait_for_selector = AsyncMock(side_effect=_selector_router)
    ctx = _make_context(page)
    ctrl = MeetingController(ctx)
    ctrl._page = page

    assert await ctrl.get_state() == MeetingState.JOINED


@pytest.mark.asyncio
async def test_get_state_detects_ended() -> None:
    """get_state() should return ENDED when meeting-ended selector is visible."""

    async def _selector_router(selector: str, **kwargs: object) -> AsyncMock:
        if _SEL_MEETING_ENDED in selector:
            return _make_element()
        return _make_element()

    page = _make_page()
    page.wait_for_selector = AsyncMock(side_effect=_selector_router)
    ctx = _make_context(page)
    ctrl = MeetingController(ctx)
    ctrl._page = page

    assert await ctrl.get_state() == MeetingState.ENDED


# ---------------------------------------------------------------------------
# wait_for_meeting_end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_meeting_end_exits_when_ended() -> None:
    """wait_for_meeting_end() should return when state becomes ENDED."""
    page = _make_page()
    ctx = _make_context(page)
    ctrl = MeetingController(ctx)
    ctrl._page = page

    # Simulate: first poll → JOINED, second poll → ENDED
    call_count = 0

    async def _get_state() -> MeetingState:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            ctrl._state = MeetingState.ENDED
            return MeetingState.ENDED
        return MeetingState.JOINED

    ctrl.get_state = _get_state  # type: ignore[assignment]

    await ctrl.wait_for_meeting_end(poll_interval=0.01)
    assert call_count >= 2


@pytest.mark.asyncio
async def test_wait_for_meeting_end_exits_on_page_close() -> None:
    """wait_for_meeting_end() should return when the page is closed."""
    page = _make_page()
    ctx = _make_context(page)
    ctrl = MeetingController(ctx)
    ctrl._page = page

    # After first iteration, simulate page close
    call_count = 0

    def _dynamic_is_closed() -> bool:
        nonlocal call_count
        call_count += 1
        return call_count >= 2

    type(page).is_closed = MagicMock(side_effect=_dynamic_is_closed)

    await ctrl.wait_for_meeting_end(poll_interval=0.01)
    assert ctrl._state == MeetingState.ENDED


# ---------------------------------------------------------------------------
# MeetingInfo defaults
# ---------------------------------------------------------------------------


def test_meeting_info_defaults() -> None:
    """MeetingInfo should have sensible defaults."""
    info = MeetingInfo()
    assert info.title == ""
    assert info.state == MeetingState.NOT_STARTED
    assert info.participant_count == 0
    assert info.is_recording is False
    assert info.is_screen_sharing is False


# ---------------------------------------------------------------------------
# MeetingState enum values
# ---------------------------------------------------------------------------


def test_meeting_state_values() -> None:
    """MeetingState should have all expected members."""
    assert MeetingState.NOT_STARTED.value == "not_started"
    assert MeetingState.WAITING_ROOM.value == "waiting_room"
    assert MeetingState.JOINED.value == "joined"
    assert MeetingState.ENDED.value == "ended"
    assert MeetingState.ERROR.value == "error"
