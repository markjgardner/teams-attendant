"""Tests for the CLI interface."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from teams_attendant.audio.devices import AudioSetupStatus
from teams_attendant.cli import app
from teams_attendant.config import AppConfig, BehaviorProfile

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: object) -> AppConfig:
    return AppConfig(**overrides)  # type: ignore[arg-type]


def _make_profile(**overrides: object) -> BehaviorProfile:
    defaults = {
        "name": "balanced",
        "description": "Default balanced profile",
        "response_threshold": 0.5,
        "proactivity": 0.5,
        "response_length": "concise",
        "prefer_voice": False,
        "cooldown_seconds": 30.0,
    }
    defaults.update(overrides)
    return BehaviorProfile(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


class TestHelp:
    def test_help_shows_all_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "join" in result.output
        assert "login" in result.output
        assert "config" in result.output
        assert "profiles" in result.output
        assert "summaries" in result.output
        assert "audio-check" in result.output
        assert "version" in result.output


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_shows_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Teams Attendant v" in result.output


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------


class TestJoin:
    def test_join_invalid_url(self) -> None:
        with patch("teams_attendant.cli.console"):
            result = runner.invoke(app, ["join", "https://example.com/meeting"])
        assert result.exit_code == 1

    def test_join_valid_url_calls_orchestrator(self) -> None:
        mock_orchestrator = MagicMock()
        mock_orchestrator.join_meeting = AsyncMock()

        with (
            patch("teams_attendant.config.load_app_config", return_value=_make_config()),
            patch("teams_attendant.utils.logging.setup_logging"),
            patch(
                "teams_attendant.orchestrator.MeetingOrchestrator",
                return_value=mock_orchestrator,
            ),
        ):
            result = runner.invoke(
                app,
                ["join", "https://teams.microsoft.com/l/meetup/abc123"],
            )
        assert result.exit_code == 0
        mock_orchestrator.join_meeting.assert_awaited_once()


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------


class TestLogin:
    def test_login_clear_calls_clear_session(self) -> None:
        mock_clear = AsyncMock()
        mock_is_valid = AsyncMock(return_value=True)

        with (
            patch("teams_attendant.config.load_app_config", return_value=_make_config()),
            patch("teams_attendant.browser.auth.clear_session", mock_clear),
            patch("teams_attendant.browser.auth.is_session_valid", mock_is_valid),
        ):
            result = runner.invoke(app, ["login", "--clear"])
        assert result.exit_code == 0
        mock_clear.assert_awaited_once()

    def test_login_already_logged_in(self) -> None:
        mock_is_valid = AsyncMock(return_value=True)

        with (
            patch("teams_attendant.config.load_app_config", return_value=_make_config()),
            patch("teams_attendant.browser.auth.is_session_valid", mock_is_valid),
        ):
            result = runner.invoke(app, ["login"])
        assert result.exit_code == 0
        assert "Already logged in" in result.output


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_config_displays_configuration(self) -> None:
        cfg = _make_config()
        with (
            patch("teams_attendant.config.load_app_config", return_value=cfg),
            patch("teams_attendant.config.get_config_dir", return_value=Path("/fake/config")),
        ):
            result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "Configuration" in result.output
        assert "balanced" in result.output
        assert "eastus" in result.output


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------


class TestProfiles:
    def test_profiles_list_shows_table(self) -> None:
        profiles = [
            _make_profile(name="passive", proactivity=0.1),
            _make_profile(name="balanced", proactivity=0.5),
            _make_profile(name="active", proactivity=0.9),
        ]
        with patch("teams_attendant.config.list_profiles", return_value=profiles):
            result = runner.invoke(app, ["profiles", "list"])
        assert result.exit_code == 0
        assert "passive" in result.output
        assert "balanced" in result.output
        assert "active" in result.output

    def test_profiles_show_displays_details(self) -> None:
        profile = _make_profile(name="balanced", description="A balanced profile")
        with patch("teams_attendant.config.load_profile", return_value=profile):
            result = runner.invoke(app, ["profiles", "show", "balanced"])
        assert result.exit_code == 0
        assert "balanced" in result.output
        assert "A balanced profile" in result.output

    def test_profiles_show_without_name_errors(self) -> None:
        result = runner.invoke(app, ["profiles", "show"])
        assert result.exit_code == 1

    def test_profiles_list_empty(self) -> None:
        with patch("teams_attendant.config.list_profiles", return_value=[]):
            result = runner.invoke(app, ["profiles", "list"])
        assert result.exit_code == 0
        assert "No profiles found" in result.output


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


class TestSummaries:
    def test_summaries_list_empty(self) -> None:
        mock_summarizer = MagicMock()
        mock_summarizer.list_summaries.return_value = []

        with (
            patch("teams_attendant.config.load_app_config", return_value=_make_config()),
            patch(
                "teams_attendant.agent.summarizer.MeetingSummarizer",
                return_value=mock_summarizer,
            ),
        ):
            result = runner.invoke(app, ["summaries", "list"])
        assert result.exit_code == 0
        assert "No meeting summaries found" in result.output

    def test_summaries_show_without_id_errors(self) -> None:
        mock_summarizer = MagicMock()

        with (
            patch("teams_attendant.config.load_app_config", return_value=_make_config()),
            patch(
                "teams_attendant.agent.summarizer.MeetingSummarizer",
                return_value=mock_summarizer,
            ),
        ):
            result = runner.invoke(app, ["summaries", "show"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# audio-check
# ---------------------------------------------------------------------------


class TestAudioCheck:
    def test_audio_check_ready(self) -> None:
        status = AudioSetupStatus(
            is_ready=True,
            platform="linux",
            capture_device="virtual_capture",
            playback_device="virtual_playback",
        )
        with patch("teams_attendant.audio.devices.check_audio_setup", return_value=status):
            result = runner.invoke(app, ["audio-check"])
        assert result.exit_code == 0
        assert "Audio Setup Check" in result.output
        assert "linux" in result.output
        assert "virtual_capture" in result.output
        assert "virtual_playback" in result.output

    def test_audio_check_not_ready(self) -> None:
        status = AudioSetupStatus(
            is_ready=False,
            platform="linux",
            issues=["No virtual audio device found."],
            suggestions=["Install a virtual audio cable."],
        )
        with patch("teams_attendant.audio.devices.check_audio_setup", return_value=status):
            result = runner.invoke(app, ["audio-check"])
        assert result.exit_code == 1
        assert "No virtual audio device found" in result.output
