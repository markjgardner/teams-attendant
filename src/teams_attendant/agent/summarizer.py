"""Post-meeting summary generation."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from teams_attendant.agent.context import MeetingContext
    from teams_attendant.agent.llm import LLMClient, Message

log = structlog.get_logger()


SUMMARY_SYSTEM_PROMPT = """You are generating a post-meeting summary. Create a well-structured, comprehensive summary in Markdown format.

Include these sections:
## Meeting Overview
Brief description of the meeting purpose and participants.

## Key Discussion Points
Numbered list of the main topics discussed.

## Decisions Made
Bullet list of any decisions that were reached.

## Action Items
Checklist of action items with assignees (if mentioned).

## Questions Raised
Any open questions that weren't fully resolved.

## Agent Participation
Summary of how the AI agent contributed to the meeting (what it said, questions it answered, value it added).

Be factual and concise. Use the actual meeting transcript and chat messages as your source. Don't fabricate information."""

_DETAIL_MAX_TOKENS = {
    "brief": 500,
    "standard": 1500,
    "detailed": 3000,
}


class MeetingSummarizer:
    """Generates post-meeting summaries using the LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        summaries_dir: Path = Path("summaries"),
    ) -> None:
        self._llm = llm_client
        self._summaries_dir = summaries_dir

    async def generate_summary(
        self,
        context: MeetingContext,
        detail_level: str = "standard",
    ) -> str:
        """Generate a meeting summary from the accumulated context."""
        messages = self._build_summary_prompt(context, detail_level)
        max_tokens = _DETAIL_MAX_TOKENS.get(detail_level, 1500)

        try:
            response = await self._llm.complete(
                messages=messages,
                system=SUMMARY_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            log.info(
                "summarizer.generated",
                detail_level=detail_level,
                output_tokens=response.output_tokens,
            )
            return response.content
        except Exception:
            log.exception("summarizer.generate_failed")
            return ""

    async def save_summary(
        self,
        summary: str,
        meeting_title: str = "",
        meeting_date: datetime | None = None,
        participants: list[str] | None = None,
    ) -> Path:
        """Save the summary as a Markdown file."""
        if meeting_date is None:
            meeting_date = datetime.now(timezone.utc)

        self._summaries_dir.mkdir(parents=True, exist_ok=True)

        filename = self._generate_filename(meeting_title, meeting_date)
        path = self._summaries_dir / filename

        title = meeting_title or "Meeting"
        date_str = meeting_date.strftime("%Y-%m-%d")
        participants_str = ", ".join(participants) if participants else ""

        header = f"---\ntitle: \"{title}\"\ndate: {date_str}\nparticipants: {participants_str}\n---\n\n"
        content = header + summary

        path.write_text(content, encoding="utf-8")
        log.info("summarizer.saved", path=str(path))
        return path

    def list_summaries(self) -> list[dict]:
        """List all saved summaries."""
        if not self._summaries_dir.is_dir():
            return []

        summaries: list[dict] = []
        for md_path in self._summaries_dir.glob("*.md"):
            entry: dict = {
                "id": md_path.stem,
                "path": str(md_path),
                "title": "",
                "date": "",
            }
            try:
                text = md_path.read_text(encoding="utf-8")
                meta = self._parse_frontmatter(text)
                if meta:
                    entry["title"] = meta.get("title", "")
                    entry["date"] = meta.get("date", "")
            except Exception:
                log.warning("summarizer.parse_failed", path=str(md_path))
            summaries.append(entry)

        summaries.sort(key=lambda s: s.get("date", ""), reverse=True)
        return summaries

    def load_summary(self, summary_id: str) -> str | None:
        """Load a summary by its ID (filename without extension)."""
        path = self._summaries_dir / f"{summary_id}.md"
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def _build_summary_prompt(
        self, context: MeetingContext, detail_level: str
    ) -> list[Message]:
        """Build the LLM prompt for summary generation."""
        from teams_attendant.agent.llm import Message

        record = context.to_meeting_record()

        parts: list[str] = []
        parts.append(f"Meeting title: {record.get('title') or 'Untitled'}")
        parts.append(f"Participants: {', '.join(record.get('participants', [])) or 'Unknown'}")
        parts.append(f"Detail level: {detail_level}")
        parts.append("")

        # Full context (summaries of earlier parts + recent entries)
        full_context = context.get_full_context()
        if full_context:
            parts.append("=== Meeting Transcript and Chat ===")
            parts.append(full_context)
            parts.append("")

        # Agent contributions
        contributions = record.get("agent_contributions", [])
        if contributions:
            parts.append("=== Agent Contributions ===")
            parts.extend(contributions)
            parts.append("")

        return [Message(role="user", content="\n".join(parts))]

    def _generate_filename(self, meeting_title: str, meeting_date: datetime) -> str:
        """Generate a summary filename."""
        date_prefix = meeting_date.strftime("%Y-%m-%d")

        if not meeting_title or not meeting_title.strip():
            return f"{date_prefix}-meeting.md"

        slug = meeting_title.lower().strip()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"[\s-]+", "-", slug)
        slug = slug.strip("-")
        slug = slug[:50]
        slug = slug.rstrip("-")

        return f"{date_prefix}-{slug}.md"

    @staticmethod
    def _parse_frontmatter(text: str) -> dict[str, str]:
        """Parse YAML frontmatter between --- markers."""
        match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if not match:
            return {}
        meta: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                value = value.strip().strip('"')
                meta[key.strip()] = value
        return meta
