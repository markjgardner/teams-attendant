"""Vision analysis for screen-shared content."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from teams_attendant.agent.context import MeetingContext
    from teams_attendant.agent.llm import LLMClient
    from teams_attendant.utils.events import Event, EventBus

log = structlog.get_logger()


VISION_SYSTEM_PROMPT = """You are analyzing screen-shared content from a Microsoft Teams meeting.
Describe what you see concisely and accurately. Focus on:
- Slide titles and key bullet points
- Code being shown (language, key functions, logic)
- Diagrams and their meaning
- Data/charts and key takeaways
- Any text that seems important for the meeting discussion

Be factual and concise. Do not speculate about things you can't see clearly."""


class VisionAnalyzer:
    """Analyzes screen captures using Claude's vision capabilities."""

    def __init__(
        self,
        llm_client: LLMClient,
        context: MeetingContext,
        event_bus: EventBus,
        analysis_prompt: str = VISION_SYSTEM_PROMPT,
    ) -> None:
        self._llm = llm_client
        self._context = context
        self._event_bus = event_bus
        self._analysis_prompt = analysis_prompt
        self._running = False
        self._task: asyncio.Task | None = None
        self._analysis_count: int = 0

    async def start(self) -> None:
        """Start listening for screen capture events."""
        self._event_bus.subscribe("screen_capture", self._handle_screen_capture)
        self._running = True
        log.info("vision_analyzer.started")

    async def stop(self) -> None:
        """Stop the vision analyzer."""
        self._event_bus.unsubscribe("screen_capture", self._handle_screen_capture)
        self._running = False
        log.info("vision_analyzer.stopped", analysis_count=self._analysis_count)

    async def analyze_image(self, image_data: bytes) -> str:
        """Analyze a single image and return a description."""
        if not image_data:
            log.warning("vision_analyzer.empty_image")
            return ""

        from teams_attendant.agent.llm import Message

        messages = [
            Message(
                role="user",
                content="Describe the content being shared in this meeting screen capture.",
            )
        ]

        try:
            response = await self._llm.complete_with_vision(
                messages=messages,
                images=[image_data],
                system=self._analysis_prompt,
                max_tokens=500,
            )
            description = response.content.strip()
            log.info("vision_analyzer.analyzed", description_length=len(description))
            return description
        except Exception:
            log.warning("vision_analyzer.analysis_failed", exc_info=True)
            return ""

    async def _handle_screen_capture(self, event: Event) -> None:
        """Handle a screen capture event from the event bus."""
        image_data = event.data.get("image_data", b"")
        if not image_data:
            log.debug("vision_analyzer.skip_no_image")
            return

        description = await self.analyze_image(image_data)
        if description:
            self._context.add_screen_description(description)
            log.info("vision_analyzer.description_added", description=description[:100])
            self._analysis_count += 1
