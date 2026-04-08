"""Browser automation for Teams Attendant."""

from teams_attendant.browser.auth import (
    clear_session,
    get_authenticated_context,
    is_session_valid,
    login,
)
from teams_attendant.browser.chat import (
    ChatMessage,
    ChatObserver,
)
from teams_attendant.browser.meeting import (
    MeetingController,
    MeetingInfo,
    MeetingState,
)
from teams_attendant.browser.screen import ScreenCaptureObserver

__all__ = [
    "ChatMessage",
    "ChatObserver",
    "MeetingController",
    "MeetingInfo",
    "MeetingState",
    "ScreenCaptureObserver",
    "clear_session",
    "get_authenticated_context",
    "is_session_valid",
    "login",
]