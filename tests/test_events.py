"""Tests for the async event bus."""

from __future__ import annotations

import asyncio


from teams_attendant.utils.events import (
    AgentResponseEvent,
    ChatMessageEvent,
    Event,
    EventBus,
    ScreenCaptureEvent,
    TranscriptEvent,
)


class TestEventDataclasses:
    """Tests for event dataclass construction."""

    def test_base_event(self) -> None:
        event = Event(type="test", data={"key": "value"})
        assert event.type == "test"
        assert event.data["key"] == "value"
        assert event.timestamp is not None

    def test_transcript_event(self) -> None:
        event = TranscriptEvent(text="hello world", speaker="Alice", confidence=0.95)
        assert event.type == "transcript"
        assert event.data["text"] == "hello world"
        assert event.data["speaker"] == "Alice"
        assert event.data["is_final"] is True
        assert event.data["confidence"] == 0.95

    def test_transcript_event_defaults(self) -> None:
        event = TranscriptEvent(text="hi")
        assert event.data["speaker"] == ""
        assert event.data["is_final"] is True
        assert event.data["confidence"] == 1.0

    def test_chat_message_event(self) -> None:
        event = ChatMessageEvent(text="msg", author="Bob", message_id="123")
        assert event.type == "chat_message"
        assert event.data["text"] == "msg"
        assert event.data["author"] == "Bob"
        assert event.data["message_id"] == "123"

    def test_screen_capture_event(self) -> None:
        event = ScreenCaptureEvent(image_data=b"\x89PNG", description="slide 1")
        assert event.type == "screen_capture"
        assert event.data["image_data"] == b"\x89PNG"
        assert event.data["description"] == "slide 1"

    def test_agent_response_event(self) -> None:
        event = AgentResponseEvent(text="reply", channel="voice")
        assert event.type == "agent_response"
        assert event.data["text"] == "reply"
        assert event.data["channel"] == "voice"


class TestEventBus:
    """Tests for EventBus subscribe/publish/lifecycle."""

    async def test_subscribe_and_publish(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test", handler)
        await bus.start()

        await bus.publish(Event(type="test", data={"x": 1}))
        await asyncio.sleep(0.2)

        await bus.stop()
        assert len(received) == 1
        assert received[0].data["x"] == 1

    async def test_multiple_handlers(self) -> None:
        bus = EventBus()
        results_a: list[Event] = []
        results_b: list[Event] = []

        async def handler_a(event: Event) -> None:
            results_a.append(event)

        async def handler_b(event: Event) -> None:
            results_b.append(event)

        bus.subscribe("test", handler_a)
        bus.subscribe("test", handler_b)
        await bus.start()

        await bus.publish(Event(type="test"))
        await asyncio.sleep(0.2)

        await bus.stop()
        assert len(results_a) == 1
        assert len(results_b) == 1

    async def test_wildcard_handler(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("*", handler)
        await bus.start()

        await bus.publish(Event(type="alpha"))
        await bus.publish(Event(type="beta"))
        await asyncio.sleep(0.2)

        await bus.stop()
        assert len(received) == 2
        assert received[0].type == "alpha"
        assert received[1].type == "beta"

    async def test_handler_exception_does_not_crash_bus(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def bad_handler(event: Event) -> None:
            raise RuntimeError("boom")

        async def good_handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test", bad_handler)
        bus.subscribe("test", good_handler)
        await bus.start()

        await bus.publish(Event(type="test"))
        await asyncio.sleep(0.2)

        await bus.stop()
        # good_handler still ran despite bad_handler raising
        assert len(received) == 1

    async def test_unsubscribe(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test", handler)
        bus.unsubscribe("test", handler)
        await bus.start()

        await bus.publish(Event(type="test"))
        await asyncio.sleep(0.2)

        await bus.stop()
        assert len(received) == 0

    async def test_unsubscribe_nonexistent_handler(self) -> None:
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        # Should not raise
        bus.unsubscribe("test", handler)

    async def test_start_stop_lifecycle(self) -> None:
        bus = EventBus()
        await bus.start()
        assert bus._running is True
        assert bus._task is not None

        await bus.stop()
        assert bus._running is False
        assert bus._task is None

    async def test_double_start_is_idempotent(self) -> None:
        bus = EventBus()
        await bus.start()
        task = bus._task
        await bus.start()  # should not create a second task
        assert bus._task is task
        await bus.stop()

    async def test_events_not_delivered_to_wrong_type(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("alpha", handler)
        await bus.start()

        await bus.publish(Event(type="beta"))
        await asyncio.sleep(0.2)

        await bus.stop()
        assert len(received) == 0
