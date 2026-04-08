"""Tests for teams_attendant.browser.screen module."""

from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock

from teams_attendant.browser.screen import (
    ScreenCaptureObserver,
    _SEL_SHARED_CONTENT,
    _SEL_SHARING_INDICATORS,
    _SEL_VIDEO_STREAM,
)
from teams_attendant.utils.events import EventBus, ScreenCaptureEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page(*, is_closed: bool = False) -> AsyncMock:
    """Build a mock Playwright Page."""
    page = AsyncMock()
    type(page).is_closed = MagicMock(return_value=is_closed)
    page.screenshot = AsyncMock(return_value=b"\x89PNG-full-page")
    page.wait_for_selector = AsyncMock(return_value=None)
    return page


def _make_element(screenshot_data: bytes = b"\x89PNG-element") -> AsyncMock:
    """Build a mock Playwright ElementHandle with screenshot support."""
    el = AsyncMock()
    el.screenshot = AsyncMock(return_value=screenshot_data)
    return el


def _make_event_bus() -> AsyncMock:
    """Build a mock EventBus."""
    bus = AsyncMock(spec=EventBus)
    bus.publish = AsyncMock()
    return bus


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    async def test_start_creates_task(self) -> None:
        """start() should set _running and create an asyncio task."""
        page = _make_page()
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus, capture_interval=100)

        await observer.start()
        assert observer._running is True
        assert observer._task is not None
        assert not observer._task.done()

        await observer.stop()

    async def test_start_is_idempotent(self) -> None:
        """Calling start() twice should not create a second task."""
        page = _make_page()
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus, capture_interval=100)

        await observer.start()
        task = observer._task
        await observer.start()
        assert observer._task is task

        await observer.stop()

    async def test_stop_cancels_task(self) -> None:
        """stop() should cancel the task and set _running to False."""
        page = _make_page()
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus, capture_interval=100)

        await observer.start()
        task = observer._task
        assert task is not None

        await observer.stop()
        assert observer._running is False
        assert observer._task is None
        assert task.cancelled()

    async def test_stop_when_not_started(self) -> None:
        """stop() should not raise when called before start()."""
        page = _make_page()
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus)

        await observer.stop()  # should not raise
        assert observer._running is False


# ---------------------------------------------------------------------------
# capture_now
# ---------------------------------------------------------------------------


class TestCaptureNow:
    async def test_screenshots_shared_content_element(self) -> None:
        """capture_now() should screenshot the shared-content element if found."""
        element = _make_element(b"\x89PNG-shared")
        page = _make_page()

        async def _selector_router(selector: str, **kwargs: object) -> AsyncMock | None:
            if selector == _SEL_SHARED_CONTENT:
                return element
            return None

        page.wait_for_selector = AsyncMock(side_effect=_selector_router)
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus)

        result = await observer.capture_now()

        assert result == b"\x89PNG-shared"
        element.screenshot.assert_awaited_once_with(type="png")

    async def test_falls_back_to_video_stream(self) -> None:
        """capture_now() should try video-stream if shared-content not found."""
        element = _make_element(b"\x89PNG-video")
        page = _make_page()

        async def _selector_router(selector: str, **kwargs: object) -> AsyncMock | None:
            if selector == _SEL_SHARED_CONTENT:
                raise Exception("not found")
            if selector == _SEL_VIDEO_STREAM:
                return element
            return None

        page.wait_for_selector = AsyncMock(side_effect=_selector_router)
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus)

        result = await observer.capture_now()

        assert result == b"\x89PNG-video"

    async def test_falls_back_to_full_page_screenshot(self) -> None:
        """capture_now() should fall back to full page screenshot if no element found."""
        page = _make_page()
        page.screenshot = AsyncMock(return_value=b"\x89PNG-fullpage")

        # All element selectors fail
        async def _selector_router(selector: str, **kwargs: object) -> None:
            raise Exception("not found")

        page.wait_for_selector = AsyncMock(side_effect=_selector_router)
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus)

        result = await observer.capture_now()

        assert result == b"\x89PNG-fullpage"
        page.screenshot.assert_awaited_once_with(type="png", full_page=False)

    async def test_returns_none_on_complete_failure(self) -> None:
        """capture_now() should return None if everything fails."""
        page = _make_page()

        async def _selector_fail(selector: str, **kwargs: object) -> None:
            raise Exception("not found")

        page.wait_for_selector = AsyncMock(side_effect=_selector_fail)
        page.screenshot = AsyncMock(side_effect=Exception("page closed"))
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus)

        result = await observer.capture_now()

        assert result is None


# ---------------------------------------------------------------------------
# _is_screen_sharing_active
# ---------------------------------------------------------------------------


class TestIsScreenSharingActive:
    async def test_detects_sharing_dom_elements(self) -> None:
        """_is_screen_sharing_active() returns True when sharing indicators found."""
        page = _make_page()
        element = _make_element()

        async def _selector_router(selector: str, **kwargs: object) -> AsyncMock:
            if _SEL_SHARING_INDICATORS in selector or selector == _SEL_SHARING_INDICATORS:
                return element
            raise Exception("not found")

        page.wait_for_selector = AsyncMock(side_effect=_selector_router)
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus)

        assert await observer._is_screen_sharing_active() is True

    async def test_returns_false_when_not_sharing(self) -> None:
        """_is_screen_sharing_active() returns False when no sharing indicators."""
        page = _make_page()

        page.wait_for_selector = AsyncMock(side_effect=Exception("timeout"))
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus)

        assert await observer._is_screen_sharing_active() is False


# ---------------------------------------------------------------------------
# _image_changed
# ---------------------------------------------------------------------------


class TestImageChanged:
    def test_detects_new_image(self) -> None:
        """_image_changed() should return True for a new image."""
        page = _make_page()
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus)

        assert observer._image_changed(b"image-data-1") is True

    def test_detects_duplicate_image(self) -> None:
        """_image_changed() should return False for the same image."""
        page = _make_page()
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus)

        observer._image_changed(b"image-data-1")
        assert observer._image_changed(b"image-data-1") is False

    def test_detects_changed_image_after_duplicate(self) -> None:
        """_image_changed() should return True when image changes again."""
        page = _make_page()
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus)

        observer._image_changed(b"image-data-1")
        observer._image_changed(b"image-data-1")
        assert observer._image_changed(b"image-data-2") is True

    def test_updates_hash_on_change(self) -> None:
        """_image_changed() should update stored hash when image changes."""
        page = _make_page()
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(page, bus)

        observer._image_changed(b"image-data-1")
        expected_hash = hashlib.md5(b"image-data-1").hexdigest()
        assert observer._last_capture_hash == expected_hash


# ---------------------------------------------------------------------------
# _capture_loop
# ---------------------------------------------------------------------------


class TestCaptureLoop:
    async def test_skips_when_not_sharing(self) -> None:
        """Capture loop should skip capture when only_when_sharing=True and not sharing."""
        page = _make_page()
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(
            page, bus, capture_interval=0.01, only_when_sharing=True
        )

        # Sharing detection returns False
        observer._is_screen_sharing_active = AsyncMock(return_value=False)
        observer.capture_now = AsyncMock(return_value=b"\x89PNG")

        observer._running = True
        # Run loop briefly then stop
        task = asyncio.create_task(observer._capture_loop())
        await asyncio.sleep(0.05)
        observer._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        observer._is_screen_sharing_active.assert_awaited()
        observer.capture_now.assert_not_awaited()

    async def test_captures_when_sharing(self) -> None:
        """Capture loop should capture and publish when sharing is active."""
        page = _make_page()
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(
            page, bus, capture_interval=0.01, only_when_sharing=True
        )

        observer._is_screen_sharing_active = AsyncMock(return_value=True)
        observer.capture_now = AsyncMock(return_value=b"\x89PNG-new")

        observer._running = True
        task = asyncio.create_task(observer._capture_loop())
        await asyncio.sleep(0.05)
        observer._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        observer.capture_now.assert_awaited()
        bus.publish.assert_awaited()
        # Verify the published event is a ScreenCaptureEvent
        published_event = bus.publish.call_args[0][0]
        assert isinstance(published_event, ScreenCaptureEvent)
        assert published_event.data["image_data"] == b"\x89PNG-new"

    async def test_publishes_screen_capture_event(self) -> None:
        """Capture loop should publish a ScreenCaptureEvent with correct data."""
        page = _make_page()
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(
            page, bus, capture_interval=0.01, only_when_sharing=False
        )

        image_bytes = b"\x89PNG-capture-test"
        observer.capture_now = AsyncMock(return_value=image_bytes)

        observer._running = True
        task = asyncio.create_task(observer._capture_loop())
        await asyncio.sleep(0.05)
        observer._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        bus.publish.assert_awaited()
        event = bus.publish.call_args[0][0]
        assert event.type == "screen_capture"
        assert event.data["image_data"] == image_bytes

    async def test_skips_duplicate_images(self) -> None:
        """Capture loop should not publish duplicate images."""
        page = _make_page()
        bus = _make_event_bus()
        observer = ScreenCaptureObserver(
            page, bus, capture_interval=0.01, only_when_sharing=False
        )

        # Always return the same image
        observer.capture_now = AsyncMock(return_value=b"\x89PNG-same")

        observer._running = True
        task = asyncio.create_task(observer._capture_loop())
        await asyncio.sleep(0.08)
        observer._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should only publish once despite multiple captures
        assert bus.publish.await_count == 1
