"""Integration-style tests verifying components work together."""

from __future__ import annotations

import asyncio

import pytest

from teams_attendant.agent.context import EventDrivenContext, MeetingContext
from teams_attendant.agent.profiles import ProfileEvaluator
from teams_attendant.config import BehaviorProfile
from teams_attendant.utils.events import (
    AgentResponseEvent,
    ChatMessageEvent,
    EventBus,
    TranscriptEvent,
)


class TestEventFlowIntegration:
    """EventBus → publish TranscriptEvent → verify context receives it."""

    @pytest.mark.asyncio
    async def test_transcript_event_flows_to_context(self):
        bus = EventBus()
        ctx = EventDrivenContext(event_bus=bus)

        await bus.start()
        try:
            event = TranscriptEvent(text="Hello everyone", speaker="Alice")
            await bus.publish(event)
            # Give the event loop time to process
            await asyncio.sleep(0.2)
        finally:
            await bus.stop()

        output = ctx.get_recent_context()
        assert "Hello everyone" in output
        assert "Alice" in output

    @pytest.mark.asyncio
    async def test_chat_message_event_flows_to_context(self):
        bus = EventBus()
        ctx = EventDrivenContext(event_bus=bus)

        await bus.start()
        try:
            event = ChatMessageEvent(text="Can you help?", author="Bob")
            await bus.publish(event)
            await asyncio.sleep(0.2)
        finally:
            await bus.stop()

        output = ctx.get_recent_context()
        assert "Can you help?" in output
        assert "Bob" in output

    @pytest.mark.asyncio
    async def test_agent_response_event_flows_to_context(self):
        bus = EventBus()
        ctx = EventDrivenContext(event_bus=bus)

        await bus.start()
        try:
            event = AgentResponseEvent(text="Sure, I can help!", channel="chat")
            await bus.publish(event)
            await asyncio.sleep(0.2)
        finally:
            await bus.stop()

        output = ctx.get_recent_context()
        assert "Sure, I can help!" in output

    @pytest.mark.asyncio
    async def test_multiple_events_maintain_order(self):
        bus = EventBus()
        ctx = EventDrivenContext(event_bus=bus)

        await bus.start()
        try:
            await bus.publish(TranscriptEvent(text="First message", speaker="Alice"))
            await bus.publish(ChatMessageEvent(text="Second message", author="Bob"))
            await bus.publish(TranscriptEvent(text="Third message", speaker="Carol"))
            await asyncio.sleep(0.3)
        finally:
            await bus.stop()

        output = ctx.get_recent_context()
        first_pos = output.index("First message")
        second_pos = output.index("Second message")
        third_pos = output.index("Third message")
        assert first_pos < second_pos < third_pos


class TestProfileEvaluationIntegration:
    """Create real BehaviorProfile + ProfileEvaluator → evaluate real events."""

    @pytest.fixture()
    def passive_profile(self):
        return BehaviorProfile(
            name="passive",
            proactivity=0.0,
            response_threshold=0.9,
            cooldown_seconds=30,
            response_length="minimal",
            prefer_voice=False,
        )

    @pytest.fixture()
    def active_profile(self):
        return BehaviorProfile(
            name="active",
            proactivity=0.8,
            response_threshold=0.3,
            cooldown_seconds=5,
            response_length="moderate",
            prefer_voice=True,
        )

    def test_passive_ignores_general_question(self, passive_profile):
        evaluator = ProfileEvaluator(passive_profile, user_name="Agent")
        event = TranscriptEvent(text="What do you think about this?", speaker="Alice")
        decision = evaluator.evaluate(event, time_since_last_response=60.0)
        # Passive profiles don't respond unless directly addressed
        assert decision.should_respond is False

    def test_passive_responds_when_addressed(self, passive_profile):
        evaluator = ProfileEvaluator(passive_profile, user_name="Agent")
        event = TranscriptEvent(text="Hey Agent, what's your take?", speaker="Alice")
        decision = evaluator.evaluate(event, time_since_last_response=60.0)
        assert decision.should_respond is True
        assert decision.confidence > 0.5

    def test_active_responds_to_questions(self, active_profile):
        evaluator = ProfileEvaluator(active_profile, user_name="Agent")
        event = TranscriptEvent(text="Does anyone have thoughts on this?", speaker="Alice")
        decision = evaluator.evaluate(event, time_since_last_response=60.0)
        assert decision.should_respond is True

    def test_active_prefers_voice_for_transcript(self, active_profile):
        evaluator = ProfileEvaluator(active_profile, user_name="Agent")
        event = TranscriptEvent(text="@Agent can you elaborate?", speaker="Alice")
        decision = evaluator.evaluate(event, time_since_last_response=60.0)
        assert decision.should_respond is True
        assert decision.channel == "voice"

    def test_cooldown_prevents_response(self, active_profile):
        evaluator = ProfileEvaluator(active_profile, user_name="Agent")
        event = TranscriptEvent(text="@Agent what about this?", speaker="Alice")
        decision = evaluator.evaluate(event, time_since_last_response=1.0)
        assert decision.should_respond is False
        assert "Cooldown" in decision.reason

    def test_chat_message_addressed_always_responds(self, passive_profile):
        evaluator = ProfileEvaluator(passive_profile, user_name="Agent")
        event = ChatMessageEvent(text="@Agent please share the notes", author="Bob")
        decision = evaluator.evaluate(event, time_since_last_response=120.0)
        assert decision.should_respond is True
        assert decision.channel == "chat"

    def test_system_prompt_varies_by_proactivity(self, passive_profile, active_profile):
        passive_eval = ProfileEvaluator(passive_profile, user_name="TestUser")
        active_eval = ProfileEvaluator(active_profile, user_name="TestUser")
        passive_prompt = passive_eval.get_system_prompt()
        active_prompt = active_eval.get_system_prompt()
        assert passive_prompt != active_prompt
        assert "only respond when directly addressed" in passive_prompt
        assert "Actively participate" in active_prompt


class TestContextFormattingIntegration:
    """Add multiple entries to MeetingContext → verify get_recent_context() format."""

    def test_mixed_entry_types_formatted(self):
        ctx = MeetingContext(user_name="TestBot")
        ctx.add_transcript("Let's begin the meeting", speaker="Alice")
        ctx.add_chat_message("Hi everyone!", author="Bob")
        ctx.add_agent_response("Hello, ready to help.", channel="chat")
        ctx.add_screen_description("Slide showing Q1 results")

        output = ctx.get_recent_context()

        # Verify all entries appear
        assert "Let's begin the meeting" in output
        assert "Hi everyone!" in output
        assert "Hello, ready to help." in output
        assert "Q1 results" in output

        # Verify formatting labels
        assert "(voice):" in output
        assert "(chat):" in output
        assert "[Screen]:" in output

    def test_recent_context_respects_max_entries(self):
        ctx = MeetingContext()
        for i in range(100):
            ctx.add_transcript(f"Message {i}", speaker="Speaker")

        output = ctx.get_recent_context(max_entries=10)
        lines = [line for line in output.strip().split("\n") if line.strip()]
        assert len(lines) == 10
        # Should contain the most recent messages
        assert "Message 99" in output
        assert "Message 90" in output

    def test_participants_tracked(self):
        ctx = MeetingContext()
        ctx.add_transcript("Hello", speaker="Alice")
        ctx.add_chat_message("Hi", author="Bob")
        ctx.add_transcript("Hey", speaker="Carol")

        participants = ctx.get_participants()
        assert participants == {"Alice", "Bob", "Carol"}

    def test_full_context_includes_summaries(self):
        ctx = MeetingContext()
        ctx._summaries.append("Earlier discussion about budgets.")
        ctx.add_transcript("Now let's talk about timelines", speaker="Alice")

        full = ctx.get_full_context()
        assert "Earlier in the meeting (summarized)" in full
        assert "budgets" in full
        assert "timelines" in full

    def test_meeting_record_export(self):
        ctx = MeetingContext(user_name="Bot")
        ctx.set_meeting_info(title="Sprint Retro")
        ctx.add_transcript("Good sprint overall", speaker="Alice")
        ctx.add_agent_response("I agree, metrics look positive.")

        record = ctx.to_meeting_record()
        assert record["title"] == "Sprint Retro"
        assert "Alice" in record["participants"]
        assert len(record["entries"]) == 2
        assert len(record["agent_contributions"]) == 1

    def test_get_agent_contributions(self):
        ctx = MeetingContext()
        ctx.add_transcript("Some discussion", speaker="Alice")
        ctx.add_agent_response("My input here")
        ctx.add_agent_response("And another thought")

        contributions = ctx.get_agent_contributions()
        assert len(contributions) == 2
        assert all(c.type == "agent_response" for c in contributions)
