"""Tests for the meeting context accumulator."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from teams_attendant.agent.context import (
    EventDrivenContext,
    MeetingContext,
)
from teams_attendant.utils.events import (
    AgentResponseEvent,
    ChatMessageEvent,
    EventBus,
    ScreenCaptureEvent,
    TranscriptEvent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx() -> MeetingContext:
    return MeetingContext(max_entries=100, summary_threshold=10, user_name="TestUser")


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


# ---------------------------------------------------------------------------
# add_transcript
# ---------------------------------------------------------------------------


class TestAddTranscript:
    def test_adds_entry(self, ctx: MeetingContext) -> None:
        ctx.add_transcript("Hello everyone", speaker="Alice")
        assert len(ctx._entries) == 1
        entry = ctx._entries[0]
        assert entry.type == "transcript"
        assert entry.speaker == "Alice"
        assert entry.content == "Hello everyone"

    def test_tracks_participant(self, ctx: MeetingContext) -> None:
        ctx.add_transcript("Hi", speaker="Bob")
        assert "Bob" in ctx.get_participants()

    def test_empty_speaker(self, ctx: MeetingContext) -> None:
        ctx.add_transcript("Some audio")
        assert len(ctx._entries) == 1
        assert ctx._entries[0].speaker == ""


# ---------------------------------------------------------------------------
# add_chat_message
# ---------------------------------------------------------------------------


class TestAddChatMessage:
    def test_adds_entry(self, ctx: MeetingContext) -> None:
        ctx.add_chat_message("Check the doc", author="Carol")
        assert len(ctx._entries) == 1
        entry = ctx._entries[0]
        assert entry.type == "chat"
        assert entry.speaker == "Carol"
        assert entry.content == "Check the doc"

    def test_tracks_participant(self, ctx: MeetingContext) -> None:
        ctx.add_chat_message("Hi", author="Dave")
        assert "Dave" in ctx.get_participants()


# ---------------------------------------------------------------------------
# add_agent_response
# ---------------------------------------------------------------------------


class TestAddAgentResponse:
    def test_adds_entry_with_user_name(self, ctx: MeetingContext) -> None:
        ctx.add_agent_response("I'll review that", channel="chat")
        assert len(ctx._entries) == 1
        entry = ctx._entries[0]
        assert entry.type == "agent_response"
        assert entry.speaker == "TestUser"
        assert entry.content == "I'll review that"
        assert entry.metadata == {"channel": "chat"}

    def test_defaults_to_agent_when_no_user_name(self) -> None:
        ctx = MeetingContext(user_name="")
        ctx.add_agent_response("Hello")
        assert ctx._entries[0].speaker == "Agent"

    def test_no_channel_metadata(self, ctx: MeetingContext) -> None:
        ctx.add_agent_response("Ok")
        assert ctx._entries[0].metadata == {}


# ---------------------------------------------------------------------------
# add_screen_description
# ---------------------------------------------------------------------------


class TestAddScreenDescription:
    def test_adds_entry(self, ctx: MeetingContext) -> None:
        ctx.add_screen_description("A slide showing Q1 revenue")
        entry = ctx._entries[0]
        assert entry.type == "screen_description"
        assert entry.speaker == ""
        assert entry.content == "A slide showing Q1 revenue"


# ---------------------------------------------------------------------------
# get_recent_context
# ---------------------------------------------------------------------------


class TestGetRecentContext:
    def test_returns_formatted_text(self, ctx: MeetingContext) -> None:
        ctx.add_transcript("Let's discuss the timeline.", speaker="Alice")
        ctx.add_chat_message("Shared the doc.", author="Bob")
        result = ctx.get_recent_context()
        assert "Alice (voice)" in result
        assert "Let's discuss the timeline." in result
        assert "Bob (chat)" in result
        assert "Shared the doc." in result

    def test_respects_max_entries_limit(self, ctx: MeetingContext) -> None:
        for i in range(20):
            ctx.add_transcript(f"Message {i}", speaker="Speaker")
        result = ctx.get_recent_context(max_entries=5)
        lines = [line for line in result.strip().split("\n") if line]
        assert len(lines) == 5
        # Should contain the last 5 messages
        assert "Message 15" in result
        assert "Message 19" in result

    def test_agent_response_shows_you(self, ctx: MeetingContext) -> None:
        ctx.add_agent_response("I'll handle it", channel="chat")
        result = ctx.get_recent_context()
        assert "You (chat)" in result

    def test_screen_description_format(self, ctx: MeetingContext) -> None:
        ctx.add_screen_description("A pie chart")
        result = ctx.get_recent_context()
        assert "[Screen]" in result
        assert "A pie chart" in result


# ---------------------------------------------------------------------------
# get_full_context
# ---------------------------------------------------------------------------


class TestGetFullContext:
    def test_includes_summaries(self, ctx: MeetingContext) -> None:
        ctx._summaries.append("The team discussed Q1 results.")
        ctx.add_transcript("Moving on to Q2.", speaker="Alice")
        result = ctx.get_full_context()
        assert "Earlier in the meeting (summarized)" in result
        assert "The team discussed Q1 results." in result
        assert "Recent conversation" in result
        assert "Moving on to Q2." in result

    def test_no_summaries_no_header(self, ctx: MeetingContext) -> None:
        ctx.add_transcript("Hello", speaker="Alice")
        result = ctx.get_full_context()
        assert "summarized" not in result
        assert "Hello" in result


# ---------------------------------------------------------------------------
# get_agent_contributions
# ---------------------------------------------------------------------------


class TestGetAgentContributions:
    def test_filters_correctly(self, ctx: MeetingContext) -> None:
        ctx.add_transcript("Hello", speaker="Alice")
        ctx.add_agent_response("Reply 1")
        ctx.add_chat_message("Chat msg", author="Bob")
        ctx.add_agent_response("Reply 2")

        contributions = ctx.get_agent_contributions()
        assert len(contributions) == 2
        assert all(c.type == "agent_response" for c in contributions)
        assert contributions[0].content == "Reply 1"
        assert contributions[1].content == "Reply 2"

    def test_empty_when_no_responses(self, ctx: MeetingContext) -> None:
        ctx.add_transcript("Hello", speaker="Alice")
        assert ctx.get_agent_contributions() == []


# ---------------------------------------------------------------------------
# get_participants
# ---------------------------------------------------------------------------


class TestGetParticipants:
    def test_returns_all_participants(self, ctx: MeetingContext) -> None:
        ctx.add_transcript("Hi", speaker="Alice")
        ctx.add_chat_message("Hello", author="Bob")
        ctx.add_transcript("Hey", speaker="Alice")
        participants = ctx.get_participants()
        assert participants == {"Alice", "Bob"}

    def test_returns_copy(self, ctx: MeetingContext) -> None:
        ctx.add_transcript("Hi", speaker="Alice")
        p = ctx.get_participants()
        p.add("Intruder")
        assert "Intruder" not in ctx.get_participants()


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_approximation(self, ctx: MeetingContext) -> None:
        text = "a" * 400
        assert ctx.estimate_tokens(text) == 100

    def test_empty_string(self, ctx: MeetingContext) -> None:
        assert ctx.estimate_tokens("") == 0

    def test_short_string(self, ctx: MeetingContext) -> None:
        assert ctx.estimate_tokens("abc") == 0  # 3 // 4 = 0


# ---------------------------------------------------------------------------
# compress_if_needed
# ---------------------------------------------------------------------------


class TestCompressIfNeeded:
    async def test_calls_llm_and_stores_summary(self, ctx: MeetingContext) -> None:
        for i in range(12):
            ctx.add_transcript(f"Statement {i}", speaker="Speaker")

        mock_client = AsyncMock()
        mock_client.complete.return_value = AsyncMock(
            content="Summary: 12 statements were made."
        )

        await ctx.compress_if_needed(mock_client)

        mock_client.complete.assert_awaited_once()
        assert len(ctx._summaries) == 1
        assert "Summary: 12 statements were made." in ctx._summaries[0]
        # Remaining entries should be roughly half
        assert len(ctx._entries) == 6

    async def test_no_compression_below_threshold(self, ctx: MeetingContext) -> None:
        for i in range(5):
            ctx.add_transcript(f"Statement {i}", speaker="Speaker")

        mock_client = AsyncMock()
        await ctx.compress_if_needed(mock_client)

        mock_client.complete.assert_not_awaited()
        assert len(ctx._summaries) == 0
        assert len(ctx._entries) == 5

    async def test_prompt_contains_entries(self, ctx: MeetingContext) -> None:
        for i in range(12):
            ctx.add_transcript(f"Important point {i}", speaker="Alice")

        mock_client = AsyncMock()
        mock_client.complete.return_value = AsyncMock(content="Summarized")

        await ctx.compress_if_needed(mock_client)

        call_args = mock_client.complete.call_args
        prompt = call_args.kwargs.get("messages") or call_args.args[0]
        prompt_text = prompt[0].content
        assert "Important point 0" in prompt_text
        assert "Summarize" in prompt_text


# ---------------------------------------------------------------------------
# to_meeting_record
# ---------------------------------------------------------------------------


class TestToMeetingRecord:
    def test_structure(self, ctx: MeetingContext) -> None:
        start = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        ctx.set_meeting_info(title="Sprint Planning", start_time=start)
        ctx.add_transcript("Let's plan", speaker="Alice")
        ctx.add_agent_response("Noted")
        ctx._summaries.append("Earlier discussion covered backlog.")

        record = ctx.to_meeting_record()

        assert record["title"] == "Sprint Planning"
        assert record["start_time"] == "2024-01-15T09:00:00+00:00"
        assert "Alice" in record["participants"]
        assert len(record["summaries"]) == 1
        assert "Earlier discussion covered backlog." in record["summaries"]
        assert len(record["entries"]) == 2
        assert len(record["agent_contributions"]) == 1


# ---------------------------------------------------------------------------
# set_meeting_info
# ---------------------------------------------------------------------------


class TestSetMeetingInfo:
    def test_sets_title_and_start_time(self, ctx: MeetingContext) -> None:
        t = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
        ctx.set_meeting_info(title="Standup", start_time=t)
        assert ctx._meeting_title == "Standup"
        assert ctx._start_time == t

    def test_partial_update(self, ctx: MeetingContext) -> None:
        ctx.set_meeting_info(title="Retro")
        assert ctx._meeting_title == "Retro"
        # start_time unchanged (still default)


# ---------------------------------------------------------------------------
# deque max_entries limit
# ---------------------------------------------------------------------------


class TestDequeMaxEntries:
    def test_enforces_limit(self) -> None:
        ctx = MeetingContext(max_entries=5)
        for i in range(10):
            ctx.add_transcript(f"Message {i}", speaker="Speaker")
        assert len(ctx._entries) == 5
        # Oldest entries should be dropped
        assert ctx._entries[0].content == "Message 5"
        assert ctx._entries[-1].content == "Message 9"


# ---------------------------------------------------------------------------
# EventDrivenContext
# ---------------------------------------------------------------------------


class TestEventDrivenContext:
    async def test_handles_transcript_event(self, event_bus: EventBus) -> None:
        ctx = EventDrivenContext(event_bus, user_name="Agent")
        event = TranscriptEvent(text="Good morning", speaker="Alice")
        await ctx._handle_transcript(event)
        assert len(ctx._entries) == 1
        assert ctx._entries[0].content == "Good morning"
        assert ctx._entries[0].speaker == "Alice"

    async def test_handles_chat_event(self, event_bus: EventBus) -> None:
        ctx = EventDrivenContext(event_bus, user_name="Agent")
        event = ChatMessageEvent(text="Link here", author="Bob")
        await ctx._handle_chat(event)
        assert len(ctx._entries) == 1
        assert ctx._entries[0].content == "Link here"
        assert ctx._entries[0].speaker == "Bob"

    async def test_handles_agent_response_event(self, event_bus: EventBus) -> None:
        ctx = EventDrivenContext(event_bus, user_name="TestUser")
        event = AgentResponseEvent(text="On it", channel="chat")
        await ctx._handle_agent_response(event)
        assert len(ctx._entries) == 1
        assert ctx._entries[0].content == "On it"
        assert ctx._entries[0].speaker == "TestUser"

    async def test_handles_screen_capture_event(self, event_bus: EventBus) -> None:
        ctx = EventDrivenContext(event_bus, user_name="Agent")
        event = ScreenCaptureEvent(image_data=b"fake", description="A bar chart")
        await ctx._handle_screen_capture(event)
        assert len(ctx._entries) == 1
        assert ctx._entries[0].content == "A bar chart"

    async def test_ignores_empty_screen_description(self, event_bus: EventBus) -> None:
        ctx = EventDrivenContext(event_bus, user_name="Agent")
        event = ScreenCaptureEvent(image_data=b"fake", description="")
        await ctx._handle_screen_capture(event)
        assert len(ctx._entries) == 0

    async def test_events_via_bus(self, event_bus: EventBus) -> None:
        """Verify handlers are called when events flow through the bus."""
        ctx = EventDrivenContext(event_bus, user_name="Agent")
        await event_bus.start()
        try:
            await event_bus.publish(TranscriptEvent(text="Via bus", speaker="Eve"))
            # Give the bus time to process
            import asyncio
            await asyncio.sleep(0.3)
        finally:
            await event_bus.stop()

        assert len(ctx._entries) == 1
        assert ctx._entries[0].content == "Via bus"

    def test_subscribes_to_all_event_types(self, event_bus: EventBus) -> None:
        _ctx = EventDrivenContext(event_bus, user_name="Agent")  # noqa: F841
        assert len(event_bus._handlers["transcript"]) == 1
        assert len(event_bus._handlers["chat_message"]) == 1
        assert len(event_bus._handlers["agent_response"]) == 1
        assert len(event_bus._handlers["screen_capture"]) == 1
