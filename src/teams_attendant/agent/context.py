"""Meeting context accumulator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from teams_attendant.agent.llm import ClaudeClient
    from teams_attendant.utils.events import Event

from teams_attendant.utils.events import EventBus

log = structlog.get_logger()


@dataclass
class ContextEntry:
    """A single entry in the meeting context."""

    timestamp: datetime
    type: str  # "transcript", "chat", "agent_response", "screen_description", "summary"
    speaker: str
    content: str
    metadata: dict = field(default_factory=dict)


class MeetingContext:
    """Accumulates and manages the meeting context for the AI agent."""

    def __init__(
        self,
        max_entries: int = 500,
        max_tokens_estimate: int = 50000,
        summary_threshold: int = 300,
        user_name: str = "",
    ) -> None:
        self._entries: deque[ContextEntry] = deque(maxlen=max_entries)
        self._summary_buffer: list[ContextEntry] = []
        self._summaries: list[str] = []
        self._max_tokens_estimate = max_tokens_estimate
        self._summary_threshold = summary_threshold
        self._user_name = user_name
        self._meeting_title: str = ""
        self._start_time: datetime = datetime.now(timezone.utc)
        self._participant_names: set[str] = set()

    def add_transcript(self, text: str, speaker: str = "") -> None:
        """Add a transcript segment."""
        entry = ContextEntry(
            timestamp=datetime.now(timezone.utc),
            type="transcript",
            speaker=speaker,
            content=text,
        )
        self._entries.append(entry)
        if speaker:
            self._participant_names.add(speaker)
        log.debug("context.add_transcript", speaker=speaker, length=len(text))

    def add_chat_message(self, text: str, author: str) -> None:
        """Add a chat message."""
        entry = ContextEntry(
            timestamp=datetime.now(timezone.utc),
            type="chat",
            speaker=author,
            content=text,
        )
        self._entries.append(entry)
        if author:
            self._participant_names.add(author)
        log.debug("context.add_chat_message", author=author, length=len(text))

    def add_agent_response(self, text: str, channel: str = "") -> None:
        """Add an agent response (for tracking what the agent said)."""
        entry = ContextEntry(
            timestamp=datetime.now(timezone.utc),
            type="agent_response",
            speaker=self._user_name or "Agent",
            content=text,
            metadata={"channel": channel} if channel else {},
        )
        self._entries.append(entry)
        log.debug("context.add_agent_response", channel=channel, length=len(text))

    def add_screen_description(self, description: str) -> None:
        """Add a description of screen-shared content."""
        entry = ContextEntry(
            timestamp=datetime.now(timezone.utc),
            type="screen_description",
            speaker="",
            content=description,
        )
        self._entries.append(entry)
        log.debug("context.add_screen_description", length=len(description))

    def get_recent_context(self, max_entries: int = 50) -> str:
        """Get a formatted string of recent context for the LLM."""
        recent = list(self._entries)[-max_entries:]
        return "\n".join(self._format_entry(e) for e in recent)

    def get_full_context(self) -> str:
        """Get the full formatted context including summaries."""
        parts: list[str] = []
        if self._summaries:
            parts.append("--- Earlier in the meeting (summarized) ---")
            parts.extend(self._summaries)
            parts.append("")
            parts.append("--- Recent conversation ---")
        parts.extend(self._format_entry(e) for e in self._entries)
        return "\n".join(parts)

    def get_agent_contributions(self) -> list[ContextEntry]:
        """Get all agent responses for summary generation."""
        return [e for e in self._entries if e.type == "agent_response"]

    def get_participants(self) -> set[str]:
        """Get names of all observed participants."""
        return set(self._participant_names)

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (chars / 4)."""
        return len(text) // 4

    async def compress_if_needed(self, llm_client: ClaudeClient) -> None:
        """If context is getting large, summarize older entries."""
        if len(self._entries) < self._summary_threshold:
            return

        from teams_attendant.agent.llm import Message

        # Take the oldest half of entries to summarize
        entries_list = list(self._entries)
        half = len(entries_list) // 2
        to_summarize = entries_list[:half]
        remaining = entries_list[half:]

        text_block = "\n".join(self._format_entry(e) for e in to_summarize)
        prompt = (
            "Summarize the following meeting segment concisely, preserving key decisions, "
            "action items, and important points:\n\n" + text_block
        )

        log.info("context.compressing", entries_summarized=len(to_summarize))
        response = await llm_client.complete(
            messages=[Message(role="user", content=prompt)],
            max_tokens=512,
            temperature=0.3,
        )

        self._summaries.append(response.content)

        # Rebuild the deque with only the remaining entries
        self._entries.clear()
        for entry in remaining:
            self._entries.append(entry)

        log.info(
            "context.compressed",
            summaries=len(self._summaries),
            remaining_entries=len(self._entries),
        )

    def _format_entry(self, entry: ContextEntry) -> str:
        """Format a single context entry as text."""
        ts = entry.timestamp.strftime("%H:%M:%S")
        if entry.type == "transcript":
            speaker = entry.speaker or "Unknown"
            return f"[{ts}] {speaker} (voice): \"{entry.content}\""
        if entry.type == "chat":
            label = "You" if entry.speaker == self._user_name and self._user_name else entry.speaker
            return f"[{ts}] {label} (chat): \"{entry.content}\""
        if entry.type == "agent_response":
            label = "You" if self._user_name else "Agent"
            return f"[{ts}] {label} (chat): \"{entry.content}\""
        if entry.type == "screen_description":
            return f"[{ts}] [Screen]: {entry.content}"
        # fallback for summary or unknown types
        return f"[{ts}] {entry.content}"

    def set_meeting_info(self, title: str = "", start_time: datetime | None = None) -> None:
        """Set meeting metadata."""
        if title:
            self._meeting_title = title
        if start_time is not None:
            self._start_time = start_time

    def to_meeting_record(self) -> dict:
        """Export the full context as a structured record for summary generation."""
        return {
            "title": self._meeting_title,
            "start_time": self._start_time.isoformat(),
            "participants": sorted(self._participant_names),
            "summaries": list(self._summaries),
            "entries": [self._format_entry(e) for e in self._entries],
            "agent_contributions": [
                self._format_entry(e) for e in self._entries if e.type == "agent_response"
            ],
        }


class EventDrivenContext(MeetingContext):
    """MeetingContext that auto-updates from EventBus events."""

    def __init__(self, event_bus: EventBus, **kwargs) -> None:
        super().__init__(**kwargs)
        self._event_bus = event_bus
        self._subscribe()

    def _subscribe(self) -> None:
        self._event_bus.subscribe("transcript", self._handle_transcript)
        self._event_bus.subscribe("chat_message", self._handle_chat)
        self._event_bus.subscribe("agent_response", self._handle_agent_response)
        self._event_bus.subscribe("screen_capture", self._handle_screen_capture)

    async def _handle_transcript(self, event: Event) -> None:
        self.add_transcript(
            text=event.data.get("text", ""),
            speaker=event.data.get("speaker", ""),
        )

    async def _handle_chat(self, event: Event) -> None:
        self.add_chat_message(
            text=event.data.get("text", ""),
            author=event.data.get("author", ""),
        )

    async def _handle_agent_response(self, event: Event) -> None:
        self.add_agent_response(
            text=event.data.get("text", ""),
            channel=event.data.get("channel", ""),
        )

    async def _handle_screen_capture(self, event: Event) -> None:
        description = event.data.get("description", "")
        if description:
            self.add_screen_description(description)
