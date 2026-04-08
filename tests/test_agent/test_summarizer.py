"""Tests for post-meeting summary generation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from teams_attendant.agent.llm import LLMResponse
from teams_attendant.agent.summarizer import MeetingSummarizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> AsyncMock:
    client = AsyncMock()
    client.complete.return_value = LLMResponse(
        content="## Meeting Overview\nA test summary.",
        input_tokens=100,
        output_tokens=50,
        model="claude-test",
        stop_reason="end_turn",
    )
    return client


@pytest.fixture
def mock_context() -> MagicMock:
    ctx = MagicMock()
    ctx.to_meeting_record.return_value = {
        "title": "Weekly Standup",
        "start_time": "2026-04-08T10:00:00+00:00",
        "participants": ["Alice", "Bob"],
        "summaries": [],
        "entries": ["[10:00:01] Alice (voice): \"Hello\""],
        "agent_contributions": ["[10:01:00] Agent (chat): \"Here are the notes\""],
    }
    ctx.get_full_context.return_value = (
        "[10:00:01] Alice (voice): \"Hello\"\n"
        "[10:01:00] Agent (chat): \"Here are the notes\""
    )
    ctx.get_agent_contributions.return_value = []
    ctx.get_participants.return_value = {"Alice", "Bob"}
    return ctx


@pytest.fixture
def summarizer(mock_llm: AsyncMock, tmp_path: Path) -> MeetingSummarizer:
    return MeetingSummarizer(llm_client=mock_llm, summaries_dir=tmp_path / "summaries")


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    async def test_calls_llm_with_correct_prompt(
        self,
        summarizer: MeetingSummarizer,
        mock_llm: AsyncMock,
        mock_context: MagicMock,
    ) -> None:
        result = await summarizer.generate_summary(mock_context)

        assert result == "## Meeting Overview\nA test summary."
        mock_llm.complete.assert_called_once()
        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs["system"] is not None
        assert "summary" in call_kwargs.kwargs["system"].lower()
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "Weekly Standup" in messages[0].content
        assert "Alice" in messages[0].content

    async def test_brief_detail_level(
        self,
        summarizer: MeetingSummarizer,
        mock_llm: AsyncMock,
        mock_context: MagicMock,
    ) -> None:
        await summarizer.generate_summary(mock_context, detail_level="brief")
        assert mock_llm.complete.call_args.kwargs["max_tokens"] == 500

    async def test_standard_detail_level(
        self,
        summarizer: MeetingSummarizer,
        mock_llm: AsyncMock,
        mock_context: MagicMock,
    ) -> None:
        await summarizer.generate_summary(mock_context, detail_level="standard")
        assert mock_llm.complete.call_args.kwargs["max_tokens"] == 1500

    async def test_detailed_detail_level(
        self,
        summarizer: MeetingSummarizer,
        mock_llm: AsyncMock,
        mock_context: MagicMock,
    ) -> None:
        await summarizer.generate_summary(mock_context, detail_level="detailed")
        assert mock_llm.complete.call_args.kwargs["max_tokens"] == 3000

    async def test_llm_error_returns_empty_string(
        self,
        summarizer: MeetingSummarizer,
        mock_llm: AsyncMock,
        mock_context: MagicMock,
    ) -> None:
        mock_llm.complete.side_effect = RuntimeError("LLM unavailable")
        result = await summarizer.generate_summary(mock_context)
        assert result == ""


# ---------------------------------------------------------------------------
# save_summary
# ---------------------------------------------------------------------------


class TestSaveSummary:
    async def test_creates_file_with_frontmatter(
        self,
        summarizer: MeetingSummarizer,
        tmp_path: Path,
    ) -> None:
        date = datetime(2026, 4, 8, tzinfo=timezone.utc)
        path = await summarizer.save_summary(
            "# Summary content",
            meeting_title="Weekly Standup",
            meeting_date=date,
            participants=["Alice", "Bob"],
        )

        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "---" in text
        assert 'title: "Weekly Standup"' in text
        assert "date: 2026-04-08" in text
        assert "participants: Alice, Bob" in text
        assert "# Summary content" in text

    async def test_creates_directory_if_missing(
        self,
        mock_llm: AsyncMock,
        tmp_path: Path,
    ) -> None:
        nested = tmp_path / "deep" / "nested" / "summaries"
        summarizer = MeetingSummarizer(llm_client=mock_llm, summaries_dir=nested)
        path = await summarizer.save_summary("content", meeting_title="Test")

        assert path.exists()
        assert nested.is_dir()


# ---------------------------------------------------------------------------
# list_summaries
# ---------------------------------------------------------------------------


class TestListSummaries:
    def test_finds_and_parses_saved_files(self, tmp_path: Path, mock_llm: AsyncMock) -> None:
        summaries_dir = tmp_path / "summaries"
        summaries_dir.mkdir()
        (summaries_dir / "2026-04-08-standup.md").write_text(
            '---\ntitle: "Standup"\ndate: 2026-04-08\nparticipants: Alice\n---\n\nContent',
            encoding="utf-8",
        )
        (summaries_dir / "2026-04-07-retro.md").write_text(
            '---\ntitle: "Retro"\ndate: 2026-04-07\nparticipants: Bob\n---\n\nContent',
            encoding="utf-8",
        )

        summarizer = MeetingSummarizer(llm_client=mock_llm, summaries_dir=summaries_dir)
        result = summarizer.list_summaries()

        assert len(result) == 2
        # Sorted by date descending
        assert result[0]["date"] == "2026-04-08"
        assert result[0]["title"] == "Standup"
        assert result[0]["id"] == "2026-04-08-standup"
        assert result[1]["date"] == "2026-04-07"
        assert result[1]["title"] == "Retro"

    def test_returns_empty_list_when_no_summaries(
        self, tmp_path: Path, mock_llm: AsyncMock
    ) -> None:
        summarizer = MeetingSummarizer(
            llm_client=mock_llm, summaries_dir=tmp_path / "nonexistent"
        )
        assert summarizer.list_summaries() == []


# ---------------------------------------------------------------------------
# load_summary
# ---------------------------------------------------------------------------


class TestLoadSummary:
    def test_returns_contents(self, tmp_path: Path, mock_llm: AsyncMock) -> None:
        summaries_dir = tmp_path / "summaries"
        summaries_dir.mkdir()
        content = "---\ntitle: \"Test\"\n---\n\n# Summary"
        (summaries_dir / "2026-04-08-test.md").write_text(content, encoding="utf-8")

        summarizer = MeetingSummarizer(llm_client=mock_llm, summaries_dir=summaries_dir)
        result = summarizer.load_summary("2026-04-08-test")
        assert result == content

    def test_returns_none_for_missing_id(self, tmp_path: Path, mock_llm: AsyncMock) -> None:
        summarizer = MeetingSummarizer(llm_client=mock_llm, summaries_dir=tmp_path)
        assert summarizer.load_summary("does-not-exist") is None


# ---------------------------------------------------------------------------
# _generate_filename
# ---------------------------------------------------------------------------


class TestGenerateFilename:
    def test_slugifies_correctly(self, summarizer: MeetingSummarizer) -> None:
        date = datetime(2026, 4, 8, tzinfo=timezone.utc)
        result = summarizer._generate_filename("Weekly Team Standup", date)
        assert result == "2026-04-08-weekly-team-standup.md"

    def test_handles_missing_title(self, summarizer: MeetingSummarizer) -> None:
        date = datetime(2026, 4, 8, tzinfo=timezone.utc)
        assert summarizer._generate_filename("", date) == "2026-04-08-meeting.md"
        assert summarizer._generate_filename("   ", date) == "2026-04-08-meeting.md"

    def test_handles_special_characters(self, summarizer: MeetingSummarizer) -> None:
        date = datetime(2026, 4, 8, tzinfo=timezone.utc)
        result = summarizer._generate_filename("Q1 Review: Sales & Marketing!!!", date)
        assert result == "2026-04-08-q1-review-sales-marketing.md"

    def test_truncates_long_titles(self, summarizer: MeetingSummarizer) -> None:
        date = datetime(2026, 4, 8, tzinfo=timezone.utc)
        long_title = "a" * 100
        result = summarizer._generate_filename(long_title, date)
        # date prefix + dash + slug(50) + .md
        slug = result.removeprefix("2026-04-08-").removesuffix(".md")
        assert len(slug) <= 50
