"""Agent decision engine."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import structlog

from teams_attendant.agent.profiles import ParticipationDecision

if TYPE_CHECKING:
    from teams_attendant.agent.context import MeetingContext
    from teams_attendant.agent.llm import LLMClient, Message
    from teams_attendant.agent.profiles import ProfileEvaluator
    from teams_attendant.utils.events import Event, EventBus

log = structlog.get_logger()

# Maps profile response_length → max_tokens for LLM calls
_MAX_TOKENS_BY_LENGTH: dict[str, int] = {
    "minimal": 100,
    "concise": 200,
    "moderate": 400,
    "detailed": 800,
}


class AgentDecisionEngine:
    """Core decision loop that decides when and how the agent participates."""

    def __init__(
        self,
        llm_client: LLMClient,
        context: MeetingContext,
        profile_evaluator: ProfileEvaluator,
        event_bus: EventBus,
        on_chat_response: Callable[[str], Awaitable[None]] | None = None,
        on_voice_response: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._llm = llm_client
        self._context = context
        self._evaluator = profile_evaluator
        self._event_bus = event_bus
        self._on_chat_response = on_chat_response
        self._on_voice_response = on_voice_response
        self._running = False
        self._last_response_time: float = 0.0
        self._processing_lock = asyncio.Lock()
        self._pending_events: asyncio.Queue[Event] = asyncio.Queue()
        self._process_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the decision engine. Subscribe to events and begin processing."""
        self._running = True
        self._event_bus.subscribe("transcript", self._handle_event)
        self._event_bus.subscribe("chat_message", self._handle_event)
        self._process_task = asyncio.create_task(self._process_events())
        log.info("agent_engine.started")

    async def stop(self) -> None:
        """Stop the decision engine."""
        self._running = False
        self._event_bus.unsubscribe("transcript", self._handle_event)
        self._event_bus.unsubscribe("chat_message", self._handle_event)
        if self._process_task is not None:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None
        log.info("agent_engine.stopped")

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event: Event) -> None:
        """Handle an incoming event (transcript or chat message)."""
        if self._is_own_echo(event):
            log.debug("agent_engine.own_echo_filtered", event_type=event.type)
            return
        await self._pending_events.put(event)

    async def _process_events(self) -> None:
        """Main processing loop — consume events from queue and decide actions."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._pending_events.get(), timeout=0.5)
            except (asyncio.TimeoutError, TimeoutError):
                continue

            async with self._processing_lock:
                try:
                    await self._evaluate_and_respond(event)
                    await self._context.compress_if_needed(self._llm)
                except Exception:
                    log.exception("agent_engine.process_error")

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    async def _evaluate_and_respond(self, event: Event) -> None:
        """Evaluate an event and generate a response if appropriate."""
        time_since_last = time.monotonic() - self._last_response_time
        recent_context = self._context.get_recent_context(max_entries=30)
        decision = self._evaluator.evaluate(event, time_since_last, recent_context)

        if decision.should_respond:
            log.info(
                "agent_engine.responding",
                reason=decision.reason,
                confidence=decision.confidence,
                channel=decision.channel,
            )
            try:
                response = await self._generate_response(event, decision)
            except Exception:
                log.exception("agent_engine.generate_error")
                return

            await self._dispatch_response(response, decision.channel)
            self._last_response_time = time.monotonic()
            self._context.add_agent_response(response, decision.channel)

            from teams_attendant.utils.events import AgentResponseEvent

            await self._event_bus.publish(
                AgentResponseEvent(text=response, channel=decision.channel)
            )
        else:
            log.debug(
                "agent_engine.skip",
                reason=decision.reason,
                event_type=event.type,
            )

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    async def _generate_response(
        self, event: Event, decision: ParticipationDecision
    ) -> str:
        """Generate a response using the LLM."""
        messages = self._build_messages(event)
        system_prompt = self._evaluator.get_system_prompt()

        response_length = self._evaluator._profile.response_length
        max_tokens = _MAX_TOKENS_BY_LENGTH.get(response_length, 200)

        llm_response = await self._llm.complete(
            messages=messages,
            system=system_prompt,
            max_tokens=max_tokens,
        )
        return llm_response.content

    def _build_messages(self, event: Event) -> list[Message]:
        """Build the LLM message list with context and the triggering event."""
        from teams_attendant.agent.llm import Message

        recent_context = self._context.get_recent_context(max_entries=30)
        guidelines = self._evaluator.get_response_guidelines()

        text = event.data.get("text", "")
        event_type = event.type
        if event_type == "transcript":
            speaker = event.data.get("speaker", "Unknown")
            trigger = f'{speaker} (voice): "{text}"'
        else:
            author = event.data.get("author", "Unknown")
            trigger = f'{author} (chat): "{text}"'

        content = (
            f"Meeting context:\n{recent_context}\n\n"
            f"Latest message:\n{trigger}\n\n"
            f"Based on the meeting context above, respond to the latest message. "
            f"{guidelines}"
        )
        return [Message(role="user", content=content)]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch_response(self, text: str, channel: str) -> None:
        """Send the response through the appropriate channel."""
        if channel == "chat":
            if self._on_chat_response:
                await self._on_chat_response(text)
            else:
                log.warning("agent_engine.no_chat_callback")
        elif channel == "voice":
            if self._on_voice_response:
                await self._on_voice_response(text)
            else:
                log.warning("agent_engine.no_voice_callback")
        else:
            log.warning("agent_engine.unknown_channel", channel=channel)

    # ------------------------------------------------------------------
    # Echo detection
    # ------------------------------------------------------------------

    def _is_own_echo(self, event: Event) -> bool:
        """Check if an event is the agent's own output being echoed back."""
        text = event.data.get("text", "")
        if not text:
            return False

        # Check if speaker/author matches the user name
        speaker_or_author = event.data.get("speaker", "") or event.data.get("author", "")
        user_name = self._evaluator._user_name
        if user_name and speaker_or_author and speaker_or_author.lower() == user_name.lower():
            return True

        # Check against recent agent contributions
        contributions = self._context.get_agent_contributions()
        for entry in contributions[-5:]:
            if entry.content == text:
                return True

        return False
