"""Behavior profile management."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from teams_attendant.config import BehaviorProfile
    from teams_attendant.utils.events import Event

log = structlog.get_logger()

# Question words that typically start interrogative sentences
_QUESTION_STARTERS = re.compile(
    r"^\s*(who|what|when|where|why|how|can|could|would|should|do|does|is|are|will)\b",
    re.IGNORECASE,
)

# Patterns that indicate a question directed at the group
_GROUP_QUESTION = re.compile(
    r"\b(anyone|anybody|somebody|someone|thoughts\??|opinions?\??)\b",
    re.IGNORECASE,
)

_RESPONSE_GUIDELINES: dict[str, str] = {
    "minimal": "Keep responses to 1-2 sentences maximum. Be direct and brief.",
    "concise": "Respond in 2-3 sentences. Get straight to the point.",
    "moderate": "Use a short paragraph. You may elaborate where helpful.",
    "detailed": "Provide a full explanation with examples if relevant.",
}


@dataclass
class ParticipationDecision:
    """Decision on whether and how to participate."""

    should_respond: bool
    confidence: float  # 0.0 to 1.0
    channel: str  # "voice", "chat", or "none"
    reason: str  # Human-readable explanation
    urgency: str  # "immediate", "normal", "low"


class ProfileEvaluator:
    """Evaluates events against a behavior profile to decide participation."""

    def __init__(self, profile: BehaviorProfile, user_name: str = "") -> None:
        self._profile = profile
        self._user_name = user_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        event: Event,
        time_since_last_response: float,
        recent_context: str = "",
    ) -> ParticipationDecision:
        """Evaluate whether the agent should respond to an event."""
        if time_since_last_response < self._profile.cooldown_seconds:
            log.debug(
                "cooldown_active",
                elapsed=time_since_last_response,
                cooldown=self._profile.cooldown_seconds,
            )
            return ParticipationDecision(
                should_respond=False,
                confidence=0.0,
                channel="none",
                reason="Cooldown period has not elapsed",
                urgency="low",
            )

        text: str = event.data.get("text", "")
        if event.type == "transcript":
            speaker: str = event.data.get("speaker", "")
            return self._evaluate_transcript(text, speaker, time_since_last_response)
        if event.type == "chat_message":
            author: str = event.data.get("author", "")
            return self._evaluate_chat_message(text, author, time_since_last_response)

        return ParticipationDecision(
            should_respond=False,
            confidence=0.0,
            channel="none",
            reason=f"Unhandled event type: {event.type}",
            urgency="low",
        )

    def get_system_prompt(self) -> str:
        """Generate LLM system prompt based on the active profile."""
        user = self._user_name or "the user"
        proactivity = self._profile.proactivity

        if proactivity < 0.1:
            persona = (
                f"You are attending a meeting on behalf of {user}. "
                "You should only respond when directly addressed. "
                "Keep responses minimal and to the point."
            )
        elif proactivity <= 0.5:
            persona = (
                f"You are attending a meeting on behalf of {user}. "
                "Respond helpfully when addressed or when you can add clear value. "
                "Be concise but thorough."
            )
        else:
            persona = (
                f"You are attending a meeting on behalf of {user}. "
                "Actively participate in discussions, ask clarifying questions, "
                "and contribute your knowledge. Be engaging and collaborative."
            )

        guidelines = self.get_response_guidelines()
        return f"{persona}\n\nResponse guidelines: {guidelines}"

    def get_response_guidelines(self) -> str:
        """Get response length/style guidelines for the LLM."""
        return _RESPONSE_GUIDELINES.get(
            self._profile.response_length,
            _RESPONSE_GUIDELINES["concise"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_transcript(
        self, text: str, speaker: str, time_since_last: float
    ) -> ParticipationDecision:
        addressed = self._is_addressed_to_user(text)
        is_question = self._is_question(text)
        channel = self._determine_channel("transcript")
        proactivity = self._profile.proactivity
        threshold = self._profile.response_threshold

        # Passive (proactivity < 0.1): only respond if directly addressed
        if proactivity < 0.1:
            if addressed:
                return ParticipationDecision(
                    should_respond=True,
                    confidence=0.95,
                    channel=channel,
                    reason="Directly addressed by name",
                    urgency="immediate" if channel == "voice" else "normal",
                )
            return ParticipationDecision(
                should_respond=False,
                confidence=0.0,
                channel="none",
                reason="Passive profile: not directly addressed",
                urgency="low",
            )

        # Balanced (0.1 <= proactivity <= 0.5)
        if proactivity <= 0.5:
            if addressed:
                return ParticipationDecision(
                    should_respond=True,
                    confidence=0.9,
                    channel=channel,
                    reason="Directly addressed by name",
                    urgency="immediate" if channel == "voice" else "normal",
                )
            if is_question:
                confidence = 0.5 + proactivity
                if confidence >= threshold:
                    return ParticipationDecision(
                        should_respond=True,
                        confidence=confidence,
                        channel=channel,
                        reason="Question detected that may be relevant",
                        urgency="normal",
                    )
            return ParticipationDecision(
                should_respond=False,
                confidence=0.0,
                channel="none",
                reason="Balanced profile: not addressed and no relevant question",
                urgency="low",
            )

        # Active (proactivity > 0.5)
        if addressed:
            return ParticipationDecision(
                should_respond=True,
                confidence=0.95,
                channel=channel,
                reason="Directly addressed by name",
                urgency="immediate" if channel == "voice" else "normal",
            )
        if is_question:
            return ParticipationDecision(
                should_respond=True,
                confidence=0.85,
                channel=channel,
                reason="Question detected — active profile contributes",
                urgency="normal",
            )
        # Proactively contribute even to statements
        confidence = 0.3 + proactivity * 0.4
        if confidence >= threshold:
            return ParticipationDecision(
                should_respond=True,
                confidence=confidence,
                channel=channel,
                reason="Active profile: proactively contributing",
                urgency="low",
            )
        return ParticipationDecision(
            should_respond=False,
            confidence=confidence,
            channel="none",
            reason="Active profile: confidence below threshold",
            urgency="low",
        )

    def _evaluate_chat_message(
        self, text: str, author: str, time_since_last: float
    ) -> ParticipationDecision:
        addressed = self._is_addressed_to_user(text)
        is_question = self._is_question(text)
        channel = self._determine_channel("chat_message")

        # In chat, direct @mentions always warrant a response
        if addressed:
            return ParticipationDecision(
                should_respond=True,
                confidence=0.95,
                channel=channel,
                reason="Directly addressed in chat",
                urgency="normal",
            )

        # Even passive profiles respond to questions addressed to them in chat
        proactivity = self._profile.proactivity

        if proactivity < 0.1:
            return ParticipationDecision(
                should_respond=False,
                confidence=0.0,
                channel="none",
                reason="Passive profile: not addressed in chat",
                urgency="low",
            )

        if is_question:
            confidence = 0.5 + proactivity
            threshold = self._profile.response_threshold
            if confidence >= threshold:
                return ParticipationDecision(
                    should_respond=True,
                    confidence=confidence,
                    channel=channel,
                    reason="Question in chat — may be relevant",
                    urgency="normal",
                )

        if proactivity > 0.5:
            confidence = 0.3 + proactivity * 0.3
            if confidence >= self._profile.response_threshold:
                return ParticipationDecision(
                    should_respond=True,
                    confidence=confidence,
                    channel=channel,
                    reason="Active profile: proactively responding in chat",
                    urgency="low",
                )

        return ParticipationDecision(
            should_respond=False,
            confidence=0.0,
            channel="none",
            reason="Not addressed and no relevant question in chat",
            urgency="low",
        )

    def _is_addressed_to_user(self, text: str) -> bool:
        """Check if *text* is directed at the user."""
        if not self._user_name:
            return False
        name_lower = self._user_name.lower()
        text_lower = text.lower()

        # Direct @mention
        if f"@{name_lower}" in text_lower:
            return True
        # "name," at the beginning or after whitespace
        if re.search(rf"(?:^|\s){re.escape(name_lower)}\s*,", text_lower):
            return True
        # "hey name", "hi name", etc.
        if re.search(rf"\b(?:hey|hi|hello|yo)\s+{re.escape(name_lower)}\b", text_lower):
            return True
        # Name appears as a standalone word in the text
        if re.search(rf"\b{re.escape(name_lower)}\b", text_lower):
            return True
        return False

    def _is_question(self, text: str) -> bool:
        """Detect whether *text* is a question."""
        if "?" in text:
            return True
        if _QUESTION_STARTERS.search(text):
            return True
        if _GROUP_QUESTION.search(text):
            return True
        return False

    def _determine_channel(self, event_type: str) -> str:
        """Pick the response channel based on event origin and profile preferences."""
        if event_type == "chat_message":
            return "chat"
        # Transcript (voice) events
        if self._profile.prefer_voice:
            return "voice"
        return "chat"
