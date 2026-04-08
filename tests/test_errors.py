"""Tests for teams_attendant.errors — custom exception hierarchy."""

from __future__ import annotations

from teams_attendant.errors import (
    AudioDeviceNotFoundError,
    AudioError,
    AuthenticationError,
    BrowserError,
    ConfigError,
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    MeetingDisconnectedError,
    MeetingJoinError,
    STTError,
    TTSError,
    TeamsAttendantError,
    VisionError,
)


# ---------------------------------------------------------------------------
# Hierarchy checks
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Every custom exception must be a TeamsAttendantError."""

    def test_browser_error_is_teams_attendant_error(self) -> None:
        assert issubclass(BrowserError, TeamsAttendantError)

    def test_authentication_error_is_browser_error(self) -> None:
        assert issubclass(AuthenticationError, BrowserError)

    def test_meeting_join_error_is_browser_error(self) -> None:
        assert issubclass(MeetingJoinError, BrowserError)

    def test_meeting_disconnected_error_is_browser_error(self) -> None:
        assert issubclass(MeetingDisconnectedError, BrowserError)

    def test_audio_error_is_teams_attendant_error(self) -> None:
        assert issubclass(AudioError, TeamsAttendantError)

    def test_audio_device_not_found_is_audio_error(self) -> None:
        assert issubclass(AudioDeviceNotFoundError, AudioError)

    def test_stt_error_is_audio_error(self) -> None:
        assert issubclass(STTError, AudioError)

    def test_tts_error_is_audio_error(self) -> None:
        assert issubclass(TTSError, AudioError)

    def test_llm_error_is_teams_attendant_error(self) -> None:
        assert issubclass(LLMError, TeamsAttendantError)

    def test_llm_auth_error_is_llm_error(self) -> None:
        assert issubclass(LLMAuthError, LLMError)

    def test_llm_rate_limit_error_is_llm_error(self) -> None:
        assert issubclass(LLMRateLimitError, LLMError)

    def test_config_error_is_teams_attendant_error(self) -> None:
        assert issubclass(ConfigError, TeamsAttendantError)

    def test_vision_error_is_teams_attendant_error(self) -> None:
        assert issubclass(VisionError, TeamsAttendantError)


# ---------------------------------------------------------------------------
# Recoverable flag
# ---------------------------------------------------------------------------


class TestRecoverableFlag:
    """Exceptions carry the ``recoverable`` flag correctly."""

    def test_base_defaults_to_not_recoverable(self) -> None:
        exc = TeamsAttendantError("boom")
        assert exc.recoverable is False

    def test_base_accepts_recoverable_true(self) -> None:
        exc = TeamsAttendantError("boom", recoverable=True)
        assert exc.recoverable is True

    def test_meeting_disconnected_defaults_to_recoverable(self) -> None:
        exc = MeetingDisconnectedError()
        assert exc.recoverable is True

    def test_meeting_disconnected_custom_message(self) -> None:
        exc = MeetingDisconnectedError("custom msg")
        assert str(exc) == "custom msg"
        assert exc.recoverable is True

    def test_llm_rate_limit_defaults_to_recoverable(self) -> None:
        exc = LLMRateLimitError()
        assert exc.recoverable is True

    def test_vision_error_defaults_to_recoverable(self) -> None:
        exc = VisionError()
        assert exc.recoverable is True

    def test_browser_error_not_recoverable_by_default(self) -> None:
        exc = BrowserError("fail")
        assert exc.recoverable is False

    def test_audio_error_not_recoverable_by_default(self) -> None:
        exc = AudioError("fail")
        assert exc.recoverable is False


# ---------------------------------------------------------------------------
# LLMRateLimitError.retry_after
# ---------------------------------------------------------------------------


class TestLLMRateLimitError:
    """LLMRateLimitError carries a ``retry_after`` attribute."""

    def test_default_retry_after(self) -> None:
        exc = LLMRateLimitError()
        assert exc.retry_after == 0

    def test_custom_retry_after(self) -> None:
        exc = LLMRateLimitError("slow down", retry_after=30.5)
        assert exc.retry_after == 30.5
        assert str(exc) == "slow down"

    def test_message_preserved(self) -> None:
        exc = LLMRateLimitError("custom message")
        assert str(exc) == "custom message"


# ---------------------------------------------------------------------------
# Catch-all: all exceptions are instances of Exception
# ---------------------------------------------------------------------------


class TestAllAreExceptions:
    """Sanity: every custom exception can be caught as Exception."""

    def test_catch_as_exception(self) -> None:
        exceptions = [
            TeamsAttendantError("a"),
            BrowserError("b"),
            AuthenticationError("c"),
            MeetingJoinError("d"),
            MeetingDisconnectedError(),
            AudioError("f"),
            AudioDeviceNotFoundError("g"),
            STTError("h"),
            TTSError("i"),
            LLMError("j"),
            LLMAuthError("k"),
            LLMRateLimitError(),
            ConfigError("m"),
            VisionError(),
        ]
        for exc in exceptions:
            assert isinstance(exc, Exception)
            assert isinstance(exc, TeamsAttendantError)
