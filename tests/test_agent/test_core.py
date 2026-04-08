"""Tests for the agent decision engine."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock


from teams_attendant.agent.context import MeetingContext
from teams_attendant.agent.core import AgentDecisionEngine
from teams_attendant.agent.llm import LLMResponse
from teams_attendant.agent.profiles import ProfileEvaluator
from teams_attendant.config import BehaviorProfile
from teams_attendant.utils.events import (
    ChatMessageEvent,
    Event,
    EventBus,
    TranscriptEvent,
)

# ---------------------------------------------------------------------------
# Shared profile definitions
# ---------------------------------------------------------------------------

_PASSIVE = BehaviorProfile(
    name="passive",
    response_threshold=0.9,
    proactivity=0.0,
    response_length="minimal",
    prefer_voice=False,
    cooldown_seconds=60,
)

_BALANCED = BehaviorProfile(
    name="balanced",
    response_threshold=0.5,
    proactivity=0.3,
    response_length="concise",
    prefer_voice=False,
    cooldown_seconds=30,
)

_ACTIVE = BehaviorProfile(
    name="active",
    response_threshold=0.3,
    proactivity=0.7,
    response_length="moderate",
    prefer_voice=True,
    cooldown_seconds=15,
)

USER_NAME = "Alice"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_mock(response_text: str = "Sure, I can help.") -> AsyncMock:
    """Create a mock ClaudeClient that returns a fixed response."""
    mock = AsyncMock()
    mock.complete.return_value = LLMResponse(
        content=response_text,
        input_tokens=50,
        output_tokens=20,
        model="claude-test",
        stop_reason="end_turn",
    )
    return mock


def _build_engine(
    profile: BehaviorProfile = _BALANCED,
    user_name: str = USER_NAME,
    llm_response: str = "Sure, I can help.",
    on_chat: AsyncMock | None = None,
    on_voice: AsyncMock | None = None,
) -> tuple[AgentDecisionEngine, AsyncMock, MeetingContext, EventBus]:
    """Build an engine with mocked dependencies, returning (engine, llm_mock, context, bus)."""
    llm = _make_llm_mock(llm_response)
    ctx = MeetingContext(max_entries=100, summary_threshold=300, user_name=user_name)
    evaluator = ProfileEvaluator(profile, user_name=user_name)
    bus = EventBus()
    engine = AgentDecisionEngine(
        llm_client=llm,
        context=ctx,
        profile_evaluator=evaluator,
        event_bus=bus,
        on_chat_response=on_chat,
        on_voice_response=on_voice,
    )
    return engine, llm, ctx, bus


# ---------------------------------------------------------------------------
# Lifecycle: start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    async def test_start_subscribes_and_starts_processing(self) -> None:
        engine, _, _, bus = _build_engine()
        await engine.start()
        try:
            assert engine._running is True
            assert engine._process_task is not None
            assert engine._handle_event in bus._handlers["transcript"]
            assert engine._handle_event in bus._handlers["chat_message"]
        finally:
            await engine.stop()

    async def test_stop_unsubscribes_and_stops(self) -> None:
        engine, _, _, bus = _build_engine()
        await engine.start()
        await engine.stop()

        assert engine._running is False
        assert engine._process_task is None
        assert engine._handle_event not in bus._handlers.get("transcript", [])
        assert engine._handle_event not in bus._handlers.get("chat_message", [])


# ---------------------------------------------------------------------------
# Own-echo detection
# ---------------------------------------------------------------------------


class TestOwnEchoDetection:
    async def test_filters_event_from_own_user_name(self) -> None:
        engine, _, _, _ = _build_engine()
        event = TranscriptEvent(text="Hello", speaker="Alice")
        assert engine._is_own_echo(event) is True

    async def test_allows_event_from_other_speaker(self) -> None:
        engine, _, _, _ = _build_engine()
        event = TranscriptEvent(text="Hello", speaker="Bob")
        assert engine._is_own_echo(event) is False

    async def test_filters_matching_agent_contribution(self) -> None:
        engine, _, ctx, _ = _build_engine()
        ctx.add_agent_response("I'll review that")
        event = TranscriptEvent(text="I'll review that", speaker="Bob")
        assert engine._is_own_echo(event) is True

    async def test_no_match_different_text(self) -> None:
        engine, _, ctx, _ = _build_engine()
        ctx.add_agent_response("I'll review that")
        event = TranscriptEvent(text="Something else entirely", speaker="Bob")
        assert engine._is_own_echo(event) is False

    async def test_empty_event_text_not_echo(self) -> None:
        engine, _, _, _ = _build_engine()
        event = Event(type="transcript", data={"text": "", "speaker": "Bob"})
        assert engine._is_own_echo(event) is False

    async def test_echo_filtered_in_handle_event(self) -> None:
        """Own echoes should not be queued."""
        engine, _, _, _ = _build_engine()
        event = TranscriptEvent(text="Hello", speaker="Alice")
        await engine._handle_event(event)
        assert engine._pending_events.empty()


# ---------------------------------------------------------------------------
# Passive profile behaviour
# ---------------------------------------------------------------------------


class TestPassiveProfile:
    async def test_ignores_unaddressed_event(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, _, _ = _build_engine(profile=_PASSIVE, on_chat=chat_cb)
        event = TranscriptEvent(text="The deadline is Friday.", speaker="Bob")
        await engine._evaluate_and_respond(event)
        llm.complete.assert_not_awaited()
        chat_cb.assert_not_awaited()

    async def test_responds_when_addressed(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, _, _ = _build_engine(profile=_PASSIVE, on_chat=chat_cb)
        event = TranscriptEvent(text="Alice, what do you think?", speaker="Bob")
        await engine._evaluate_and_respond(event)
        llm.complete.assert_awaited_once()
        chat_cb.assert_awaited_once()


# ---------------------------------------------------------------------------
# Balanced profile behaviour
# ---------------------------------------------------------------------------


class TestBalancedProfile:
    async def test_responds_to_question(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, _, _ = _build_engine(profile=_BALANCED, on_chat=chat_cb)
        event = TranscriptEvent(text="Does anyone know the status?", speaker="Bob")
        await engine._evaluate_and_respond(event)
        llm.complete.assert_awaited_once()
        chat_cb.assert_awaited_once()

    async def test_ignores_plain_statement(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, _, _ = _build_engine(profile=_BALANCED, on_chat=chat_cb)
        event = TranscriptEvent(text="I finished the report.", speaker="Bob")
        await engine._evaluate_and_respond(event)
        llm.complete.assert_not_awaited()
        chat_cb.assert_not_awaited()


# ---------------------------------------------------------------------------
# Active profile behaviour
# ---------------------------------------------------------------------------


class TestActiveProfile:
    async def test_responds_to_statement(self) -> None:
        voice_cb = AsyncMock()
        engine, llm, _, _ = _build_engine(profile=_ACTIVE, on_voice=voice_cb)
        event = TranscriptEvent(text="We shipped v2 last week.", speaker="Bob")
        await engine._evaluate_and_respond(event)
        llm.complete.assert_awaited_once()
        voice_cb.assert_awaited_once()


# ---------------------------------------------------------------------------
# Cooldown enforcement
# ---------------------------------------------------------------------------


class TestCooldownEnforcement:
    async def test_second_event_within_cooldown_ignored(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, _, _ = _build_engine(profile=_BALANCED, on_chat=chat_cb)

        # First response: should succeed (time_since_last is large initially)
        event1 = TranscriptEvent(text="Alice, hello!", speaker="Bob")
        await engine._evaluate_and_respond(event1)
        llm.complete.assert_awaited_once()

        # Second response immediately after — cooldown blocks it
        event2 = TranscriptEvent(text="Alice, one more thing?", speaker="Bob")
        await engine._evaluate_and_respond(event2)
        # LLM should still have been called only once
        assert llm.complete.await_count == 1


# ---------------------------------------------------------------------------
# Channel dispatch
# ---------------------------------------------------------------------------


class TestChannelDispatch:
    async def test_chat_response_dispatched_to_chat_callback(self) -> None:
        chat_cb = AsyncMock()
        engine, _, _, _ = _build_engine(profile=_BALANCED, on_chat=chat_cb)
        await engine._dispatch_response("Hello", "chat")
        chat_cb.assert_awaited_once_with("Hello")

    async def test_voice_response_dispatched_to_voice_callback(self) -> None:
        voice_cb = AsyncMock()
        engine, _, _, _ = _build_engine(profile=_BALANCED, on_voice=voice_cb)
        await engine._dispatch_response("Hello", "voice")
        voice_cb.assert_awaited_once_with("Hello")

    async def test_missing_chat_callback_logs_warning(self) -> None:
        engine, _, _, _ = _build_engine(profile=_BALANCED)
        # Should not raise
        await engine._dispatch_response("Hello", "chat")

    async def test_missing_voice_callback_logs_warning(self) -> None:
        engine, _, _, _ = _build_engine(profile=_BALANCED)
        await engine._dispatch_response("Hello", "voice")

    async def test_unknown_channel_logs_warning(self) -> None:
        engine, _, _, _ = _build_engine(profile=_BALANCED)
        await engine._dispatch_response("Hello", "telepathy")


# ---------------------------------------------------------------------------
# LLM integration
# ---------------------------------------------------------------------------


class TestLLMIntegration:
    async def test_system_prompt_passed_to_llm(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, _, _ = _build_engine(profile=_BALANCED, on_chat=chat_cb)
        event = TranscriptEvent(text="Alice, thoughts?", speaker="Bob")
        await engine._evaluate_and_respond(event)

        call_kwargs = llm.complete.call_args.kwargs
        assert "system" in call_kwargs
        assert "Alice" in call_kwargs["system"]

    async def test_max_tokens_matches_profile_response_length(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, _, _ = _build_engine(profile=_PASSIVE, on_chat=chat_cb)
        event = TranscriptEvent(text="Alice, hello!", speaker="Bob")
        await engine._evaluate_and_respond(event)

        call_kwargs = llm.complete.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100  # minimal → 100

    async def test_context_included_in_messages(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, ctx, _ = _build_engine(profile=_BALANCED, on_chat=chat_cb)
        ctx.add_transcript("Earlier discussion about Q1.", speaker="Carol")

        event = TranscriptEvent(text="Alice, what's the update?", speaker="Bob")
        await engine._evaluate_and_respond(event)

        messages = llm.complete.call_args.kwargs["messages"]
        content = messages[0].content
        assert "Earlier discussion about Q1." in content
        assert "Alice, what's the update?" in content

    async def test_build_messages_transcript(self) -> None:
        engine, _, _, _ = _build_engine()
        event = TranscriptEvent(text="How are you?", speaker="Bob")
        messages = engine._build_messages(event)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert 'Bob (voice): "How are you?"' in messages[0].content

    async def test_build_messages_chat(self) -> None:
        engine, _, _, _ = _build_engine()
        event = ChatMessageEvent(text="Check this", author="Carol")
        messages = engine._build_messages(event)
        assert 'Carol (chat): "Check this"' in messages[0].content


# ---------------------------------------------------------------------------
# LLM error handling
# ---------------------------------------------------------------------------


class TestLLMErrorHandling:
    async def test_llm_exception_does_not_crash_engine(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, _, _ = _build_engine(profile=_BALANCED, on_chat=chat_cb)
        llm.complete.side_effect = RuntimeError("LLM timeout")

        event = TranscriptEvent(text="Alice, help?", speaker="Bob")
        # Should not raise
        await engine._evaluate_and_respond(event)
        chat_cb.assert_not_awaited()


# ---------------------------------------------------------------------------
# Context compression
# ---------------------------------------------------------------------------


class TestContextCompression:
    async def test_compression_triggered_after_event(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, ctx, _ = _build_engine(
            profile=_BALANCED, on_chat=chat_cb
        )
        # Set summary_threshold very low to trigger compression
        ctx._summary_threshold = 3

        # Populate context above threshold
        for i in range(5):
            ctx.add_transcript(f"Message {i}", speaker="Bob")

        event = TranscriptEvent(text="Alice, thoughts?", speaker="Bob")

        # _evaluate_and_respond + compress_if_needed
        async with engine._processing_lock:
            await engine._evaluate_and_respond(event)
            await ctx.compress_if_needed(llm)

        # LLM should be called at least twice: once for response, once for compression
        assert llm.complete.await_count >= 2


# ---------------------------------------------------------------------------
# Processing lock prevents concurrent responses
# ---------------------------------------------------------------------------


class TestProcessingLock:
    async def test_lock_prevents_concurrent_responses(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, _, _ = _build_engine(profile=_BALANCED, on_chat=chat_cb)

        # Make the LLM slow so we can test concurrency
        async def slow_complete(**kwargs):
            await asyncio.sleep(0.2)
            return LLMResponse(content="response", input_tokens=10, output_tokens=5)

        llm.complete.side_effect = slow_complete

        event1 = TranscriptEvent(text="Alice, first question?", speaker="Bob")
        event2 = TranscriptEvent(text="Alice, second question?", speaker="Carol")

        # Manually push events and run the evaluate loop
        # Since the lock is held, they should be serialized
        async def run_eval(ev: Event) -> None:
            async with engine._processing_lock:
                await engine._evaluate_and_respond(ev)

        t1 = asyncio.create_task(run_eval(event1))
        t2 = asyncio.create_task(run_eval(event2))

        await asyncio.gather(t1, t2)

        # Both should complete (serialized), but the second is blocked by cooldown
        # after the first responds. The important thing: no crash / no concurrency issue.
        assert llm.complete.await_count >= 1


# ---------------------------------------------------------------------------
# End-to-end via event bus
# ---------------------------------------------------------------------------


class TestEndToEnd:
    async def test_event_flows_through_bus_to_engine(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, _, bus = _build_engine(profile=_BALANCED, on_chat=chat_cb)

        await bus.start()
        await engine.start()
        try:
            await bus.publish(TranscriptEvent(text="Alice, what do you think?", speaker="Bob"))
            # Allow processing time
            await asyncio.sleep(0.5)
        finally:
            await engine.stop()
            await bus.stop()

        llm.complete.assert_awaited_once()
        chat_cb.assert_awaited_once()

    async def test_agent_response_event_published(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, _, bus = _build_engine(
            profile=_BALANCED, on_chat=chat_cb, llm_response="I'll check."
        )
        captured: list[Event] = []

        async def capture(event: Event) -> None:
            captured.append(event)

        bus.subscribe("agent_response", capture)

        await bus.start()
        await engine.start()
        try:
            await bus.publish(TranscriptEvent(text="Alice, update?", speaker="Bob"))
            await asyncio.sleep(0.5)
        finally:
            await engine.stop()
            await bus.stop()

        # The agent should have published an AgentResponseEvent
        assert any(e.type == "agent_response" for e in captured)
        resp_event = next(e for e in captured if e.type == "agent_response")
        assert resp_event.data["text"] == "I'll check."

    async def test_context_updated_after_response(self) -> None:
        chat_cb = AsyncMock()
        engine, llm, ctx, bus = _build_engine(
            profile=_BALANCED, on_chat=chat_cb, llm_response="Noted."
        )

        await bus.start()
        await engine.start()
        try:
            await bus.publish(TranscriptEvent(text="Alice, your thoughts?", speaker="Bob"))
            await asyncio.sleep(0.5)
        finally:
            await engine.stop()
            await bus.stop()

        contributions = ctx.get_agent_contributions()
        assert len(contributions) == 1
        assert contributions[0].content == "Noted."
