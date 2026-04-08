"""Chat message observation and interaction."""

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
# Selectors – Teams Web UI chat panel
# ---------------------------------------------------------------------------

_SEL_CHAT_BUTTON = (
    "[data-tid='chat-button'],"
    "button[aria-label*='Chat'],"
    "button:has-text('Chat')"
)

_SEL_CHAT_PANEL = (
    "[data-tid='chat-pane'],"
    "[data-tid='chat-pane-list'],"
    "[class*='chat-pane'],"
    "[role='log']"
)

_SEL_MESSAGE_CONTAINER = (
    "[data-tid='chat-pane-message'],"
    "[class*='message']"
)

_SEL_MESSAGE_AUTHOR = (
    "[data-tid='message-author'],"
    "[class*='sender'],"
    "[class*='author']"
)

_SEL_MESSAGE_BODY = (
    "[data-tid='message-body'],"
    "[class*='message-body'],"
    "[class*='message-text']"
)

_SEL_CHAT_INPUT = (
    "[data-tid='chat-pane-compose-input'],"
    "[aria-label*='Type a message'],"
    "div[contenteditable='true']"
)


@dataclass
class ChatMessage:
    """A parsed chat message."""

    id: str
    author: str
    text: str
    timestamp: datetime
    is_own: bool = False


class ChatObserver:
    """Observes and interacts with the Teams meeting chat."""

    def __init__(self, page: Page, event_bus: EventBus) -> None:
        self._page = page
        self._event_bus = event_bus
        self._running = False
        self._seen_messages: set[str] = set()
        self._poll_interval: float = 2.0
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start observing chat messages."""
        if self._running:
            return

        log.info("chat.observer.starting")
        await self._open_chat_panel()

        # Snapshot existing messages so we don't emit events for them
        existing = await self._parse_messages_from_dom()
        for msg in existing:
            self._seen_messages.add(msg.id)
        log.info("chat.observer.snapshot", count=len(existing))

        self._running = True
        self._task = asyncio.create_task(self._poll_messages())

    async def stop(self) -> None:
        """Stop observing chat messages."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("chat.observer.stopped")

    async def send_message(self, text: str) -> None:
        """Send a chat message."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        log.info("chat.send.start", text_length=len(text))
        await self._open_chat_panel()

        try:
            input_el = await self._page.wait_for_selector(
                _SEL_CHAT_INPUT, timeout=5_000, state="visible"
            )
            if input_el is None:
                log.error("chat.send.input_not_found")
                return

            await input_el.click()
            await self._page.keyboard.type(text, delay=30)
            await self._page.keyboard.press("Enter")
            log.info("chat.send.done", text=text[:80])
        except PlaywrightTimeout:
            log.error("chat.send.timeout", detail="Could not find chat input box")
        except Exception as exc:
            log.error("chat.send.error", error=str(exc))

    async def get_recent_messages(self, count: int = 20) -> list[ChatMessage]:
        """Get recent chat messages from the DOM."""
        messages = await self._parse_messages_from_dom()
        return messages[-count:]

    async def _open_chat_panel(self) -> None:
        """Ensure the chat panel is open."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        # Check if already open
        try:
            panel = await self._page.wait_for_selector(
                _SEL_CHAT_PANEL, timeout=1_000, state="visible"
            )
            if panel:
                log.debug("chat.panel.already_open")
                return
        except PlaywrightTimeout:
            pass

        # Click the chat button to open
        try:
            btn = await self._page.wait_for_selector(
                _SEL_CHAT_BUTTON, timeout=5_000, state="visible"
            )
            if btn:
                await btn.click()
                log.info("chat.panel.opened")
                # Wait for the panel to appear
                await self._page.wait_for_selector(
                    _SEL_CHAT_PANEL, timeout=5_000, state="visible"
                )
        except PlaywrightTimeout:
            log.error("chat.panel.open_failed", detail="Chat button or panel not found")

    async def _poll_messages(self) -> None:
        """Polling loop for new messages."""
        from teams_attendant.utils.events import ChatMessageEvent

        while self._running:
            try:
                messages = await self._parse_messages_from_dom()
                for msg in messages:
                    if msg.id not in self._seen_messages:
                        self._seen_messages.add(msg.id)
                        event = ChatMessageEvent(
                            text=msg.text,
                            author=msg.author,
                            message_id=msg.id,
                        )
                        await self._event_bus.publish(event)
                        log.info(
                            "chat.new_message",
                            author=msg.author,
                            text=msg.text[:80],
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("chat.poll.error", error=str(exc))

            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                raise

    async def _parse_messages_from_dom(self) -> list[ChatMessage]:
        """Parse chat messages from the Teams chat DOM."""
        messages: list[ChatMessage] = []

        try:
            elements = await self._page.query_selector_all(_SEL_MESSAGE_CONTAINER)
        except Exception as exc:
            log.warning("chat.parse.container_error", error=str(exc))
            return messages

        for idx, el in enumerate(elements):
            try:
                # Extract author
                author = ""
                author_el = await el.query_selector(_SEL_MESSAGE_AUTHOR)
                if author_el:
                    author = (await author_el.inner_text()).strip()

                # Extract body text
                text = ""
                body_el = await el.query_selector(_SEL_MESSAGE_BODY)
                if body_el:
                    text = (await body_el.inner_text()).strip()

                if not text:
                    continue

                msg_id = self._generate_message_id(author, text, idx)
                messages.append(
                    ChatMessage(
                        id=msg_id,
                        author=author,
                        text=text,
                        timestamp=datetime.now(timezone.utc),
                    )
                )
            except Exception as exc:
                log.warning("chat.parse.message_error", index=idx, error=str(exc))
                continue

        return messages

    def _generate_message_id(self, author: str, text: str, index: int) -> str:
        """Generate a unique-ish ID for a message."""
        raw = f"{author}:{text}:{index}"
        return hashlib.md5(raw.encode()).hexdigest()  # noqa: S324
