"""Custom exceptions for Teams Attendant."""

from __future__ import annotations


class TeamsAttendantError(Exception):
    """Base exception for all Teams Attendant errors."""

    def __init__(self, message: str, recoverable: bool = False) -> None:
        super().__init__(message)
        self.recoverable = recoverable


class BrowserError(TeamsAttendantError):
    """Browser automation errors."""

    pass


class AuthenticationError(BrowserError):
    """Authentication/session errors."""

    pass


class MeetingJoinError(BrowserError):
    """Failed to join a meeting."""

    pass


class MeetingDisconnectedError(BrowserError):
    """Disconnected from meeting unexpectedly."""

    def __init__(self, message: str = "Disconnected from meeting") -> None:
        super().__init__(message, recoverable=True)


class AudioError(TeamsAttendantError):
    """Audio pipeline errors."""

    pass


class AudioDeviceNotFoundError(AudioError):
    """Required audio device not found."""

    pass


class STTError(AudioError):
    """Speech-to-text errors."""

    pass


class TTSError(AudioError):
    """Text-to-speech errors."""

    pass


class LLMError(TeamsAttendantError):
    """LLM/AI errors."""

    pass


class LLMAuthError(LLMError):
    """LLM authentication error."""

    pass


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float = 0) -> None:
        super().__init__(message, recoverable=True)
        self.retry_after = retry_after


class ConfigError(TeamsAttendantError):
    """Configuration errors."""

    pass


class VisionError(TeamsAttendantError):
    """Vision analysis errors."""

    def __init__(self, message: str = "Vision analysis failed") -> None:
        super().__init__(message, recoverable=True)
