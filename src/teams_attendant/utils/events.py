"""Async event bus for inter-component communication."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import structlog

log = structlog.get_logger()


@dataclass
class Event:
    """Base event."""

    type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = field(default_factory=dict)


class TranscriptEvent(Event):
    """A speech transcript segment."""

    def __init__(
        self, text: str, speaker: str = "", is_final: bool = True, confidence: float = 1.0
    ) -> None:
        super().__init__(
            type="transcript",
            data={"text": text, "speaker": speaker, "is_final": is_final, "confidence": confidence},
        )


class ChatMessageEvent(Event):
    """A chat message received."""

    def __init__(self, text: str, author: str, message_id: str = "") -> None:
        super().__init__(
            type="chat_message",
            data={"text": text, "author": author, "message_id": message_id},
        )


class ScreenCaptureEvent(Event):
    """A screen capture taken."""

    def __init__(self, image_data: bytes, description: str = "") -> None:
        super().__init__(
            type="screen_capture",
            data={"image_data": image_data, "description": description},
        )


class AgentResponseEvent(Event):
    """Agent generated a response."""

    def __init__(self, text: str, channel: str = "chat") -> None:
        super().__init__(
            type="agent_response",
            data={"text": text, "channel": channel},
        )


EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Simple async pub/sub event bus."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task[None] | None = None

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register *handler* for events of *event_type* (use ``"*"`` for all)."""
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a previously registered handler."""
        try:
            self._handlers[event_type].remove(handler)
        except ValueError:
            pass

    async def publish(self, event: Event) -> None:
        """Enqueue an event for asynchronous processing."""
        await self._queue.put(event)

    async def start(self) -> None:
        """Start the background event-processing loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_events())

    async def stop(self) -> None:
        """Stop the event-processing loop and drain remaining events."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _process_events(self) -> None:
        """Internal loop: pull events from the queue and dispatch to handlers."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except (asyncio.TimeoutError, TimeoutError):
                continue

            log.debug("event_bus.dispatch", event_type=event.type)

            handlers: list[EventHandler] = list(self._handlers.get(event.type, []))
            handlers.extend(self._handlers.get("*", []))

            for handler in handlers:
                try:
                    await handler(event)
                except Exception:
                    log.exception("event_bus.handler_error", event_type=event.type)
