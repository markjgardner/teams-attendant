"""Tests for behavior profile evaluation."""

from __future__ import annotations

import pytest

from teams_attendant.agent.profiles import ProfileEvaluator
from teams_attendant.config import BehaviorProfile
from teams_attendant.utils.events import Event

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_PASSIVE = BehaviorProfile(
    name="passive",
    description="Listen only.",
    response_threshold=0.9,
    proactivity=0.0,
    response_length="minimal",
    prefer_voice=False,
    cooldown_seconds=60,
)

_BALANCED = BehaviorProfile(
    name="balanced",
    description="Respond when addressed.",
    response_threshold=0.5,
    proactivity=0.3,
    response_length="concise",
    prefer_voice=False,
    cooldown_seconds=30,
)

_ACTIVE = BehaviorProfile(
    name="active",
    description="Proactively contribute.",
    response_threshold=0.3,
    proactivity=0.7,
    response_length="moderate",
    prefer_voice=True,
    cooldown_seconds=15,
)

USER_NAME = "Alice"


def _transcript(text: str, speaker: str = "Bob") -> Event:
    return Event(type="transcript", data={"text": text, "speaker": speaker})


def _chat(text: str, author: str = "Bob") -> Event:
    return Event(type="chat_message", data={"text": text, "author": author})


# ---------------------------------------------------------------------------
# Passive profile
# ---------------------------------------------------------------------------


class TestPassiveProfile:
    def setup_method(self) -> None:
        self.evaluator = ProfileEvaluator(_PASSIVE, user_name=USER_NAME)

    def test_responds_when_addressed_by_name(self) -> None:
        decision = self.evaluator.evaluate(_transcript("Alice, what do you think?"), 120.0)
        assert decision.should_respond is True
        assert decision.confidence > 0.8

    def test_ignores_general_question(self) -> None:
        decision = self.evaluator.evaluate(_transcript("What time is the meeting?"), 120.0)
        assert decision.should_respond is False

    def test_ignores_statement(self) -> None:
        decision = self.evaluator.evaluate(_transcript("The deadline is Friday."), 120.0)
        assert decision.should_respond is False

    def test_responds_to_at_mention_in_chat(self) -> None:
        decision = self.evaluator.evaluate(_chat("@Alice can you check this?"), 120.0)
        assert decision.should_respond is True

    def test_ignores_unaddressed_chat(self) -> None:
        decision = self.evaluator.evaluate(_chat("Has anyone seen the report?"), 120.0)
        assert decision.should_respond is False


# ---------------------------------------------------------------------------
# Balanced profile
# ---------------------------------------------------------------------------


class TestBalancedProfile:
    def setup_method(self) -> None:
        self.evaluator = ProfileEvaluator(_BALANCED, user_name=USER_NAME)

    def test_responds_to_direct_address(self) -> None:
        decision = self.evaluator.evaluate(_transcript("Hey Alice, can you explain?"), 60.0)
        assert decision.should_respond is True
        assert decision.confidence >= 0.9

    def test_responds_to_question(self) -> None:
        decision = self.evaluator.evaluate(_transcript("Does anyone know the status?"), 60.0)
        assert decision.should_respond is True

    def test_respects_cooldown(self) -> None:
        decision = self.evaluator.evaluate(_transcript("Alice, thoughts?"), 10.0)
        assert decision.should_respond is False
        assert "cooldown" in decision.reason.lower()

    def test_ignores_plain_statement(self) -> None:
        decision = self.evaluator.evaluate(_transcript("I finished the report."), 60.0)
        assert decision.should_respond is False


# ---------------------------------------------------------------------------
# Active profile
# ---------------------------------------------------------------------------


class TestActiveProfile:
    def setup_method(self) -> None:
        self.evaluator = ProfileEvaluator(_ACTIVE, user_name=USER_NAME)

    def test_responds_to_question(self) -> None:
        decision = self.evaluator.evaluate(_transcript("What's the next step?"), 30.0)
        assert decision.should_respond is True

    def test_responds_proactively(self) -> None:
        decision = self.evaluator.evaluate(_transcript("We shipped v2 last week."), 30.0)
        assert decision.should_respond is True
        assert "proactiv" in decision.reason.lower()

    def test_responds_when_addressed(self) -> None:
        decision = self.evaluator.evaluate(_transcript("Alice, your input?"), 30.0)
        assert decision.should_respond is True

    def test_prefers_voice_channel(self) -> None:
        decision = self.evaluator.evaluate(_transcript("Alice, what do you think?"), 30.0)
        assert decision.channel == "voice"


# ---------------------------------------------------------------------------
# _is_addressed_to_user
# ---------------------------------------------------------------------------


class TestIsAddressedToUser:
    def setup_method(self) -> None:
        self.evaluator = ProfileEvaluator(_BALANCED, user_name=USER_NAME)

    @pytest.mark.parametrize(
        "text",
        [
            "Alice, can you look at this?",
            "@Alice please review",
            "Hey Alice what's up",
            "Hi Alice!",
            "I think Alice should decide",
        ],
    )
    def test_addressed(self, text: str) -> None:
        assert self.evaluator._is_addressed_to_user(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Can someone review this?",
            "The meeting is at noon.",
            "Bob, your turn",
        ],
    )
    def test_not_addressed(self, text: str) -> None:
        assert self.evaluator._is_addressed_to_user(text) is False

    def test_empty_user_name(self) -> None:
        evaluator = ProfileEvaluator(_BALANCED, user_name="")
        assert evaluator._is_addressed_to_user("Alice are you there?") is False


# ---------------------------------------------------------------------------
# _is_question
# ---------------------------------------------------------------------------


class TestIsQuestion:
    def setup_method(self) -> None:
        self.evaluator = ProfileEvaluator(_BALANCED, user_name=USER_NAME)

    @pytest.mark.parametrize(
        "text",
        [
            "What time is it?",
            "Can we move forward",
            "Does this make sense",
            "Anyone have the link?",
            "thoughts?",
            "How should we proceed",
            "Who is responsible",
            "Is this ready",
        ],
    )
    def test_detected_as_question(self, text: str) -> None:
        assert self.evaluator._is_question(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "The report is done.",
            "I finished the task",
            "Let me know later",
        ],
    )
    def test_not_question(self, text: str) -> None:
        assert self.evaluator._is_question(text) is False


# ---------------------------------------------------------------------------
# _determine_channel
# ---------------------------------------------------------------------------


class TestDetermineChannel:
    def test_chat_event_returns_chat(self) -> None:
        evaluator = ProfileEvaluator(_ACTIVE, user_name=USER_NAME)
        assert evaluator._determine_channel("chat_message") == "chat"

    def test_transcript_with_prefer_voice(self) -> None:
        evaluator = ProfileEvaluator(_ACTIVE, user_name=USER_NAME)
        assert evaluator._determine_channel("transcript") == "voice"

    def test_transcript_without_prefer_voice(self) -> None:
        evaluator = ProfileEvaluator(_BALANCED, user_name=USER_NAME)
        assert evaluator._determine_channel("transcript") == "chat"


# ---------------------------------------------------------------------------
# get_system_prompt
# ---------------------------------------------------------------------------


class TestGetSystemPrompt:
    def test_passive_prompt(self) -> None:
        prompt = ProfileEvaluator(_PASSIVE, user_name="Alice").get_system_prompt()
        assert "only respond when directly addressed" in prompt
        assert "Alice" in prompt

    def test_balanced_prompt(self) -> None:
        prompt = ProfileEvaluator(_BALANCED, user_name="Alice").get_system_prompt()
        assert "concise but thorough" in prompt.lower()

    def test_active_prompt(self) -> None:
        prompt = ProfileEvaluator(_ACTIVE, user_name="Alice").get_system_prompt()
        assert "actively participate" in prompt.lower()

    def test_different_profiles_produce_different_prompts(self) -> None:
        prompts = {
            p.name: ProfileEvaluator(p, user_name="Alice").get_system_prompt()
            for p in [_PASSIVE, _BALANCED, _ACTIVE]
        }
        assert len(set(prompts.values())) == 3


# ---------------------------------------------------------------------------
# get_response_guidelines
# ---------------------------------------------------------------------------


class TestGetResponseGuidelines:
    @pytest.mark.parametrize(
        ("length", "fragment"),
        [
            ("minimal", "1-2 sentences"),
            ("concise", "2-3 sentences"),
            ("moderate", "paragraph"),
            ("detailed", "full explanation"),
        ],
    )
    def test_guidelines(self, length: str, fragment: str) -> None:
        profile = BehaviorProfile(name="test", response_length=length)  # type: ignore[arg-type]
        guidelines = ProfileEvaluator(profile).get_response_guidelines()
        assert fragment in guidelines.lower()


# ---------------------------------------------------------------------------
# Cooldown enforcement
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_cooldown_blocks_response(self) -> None:
        evaluator = ProfileEvaluator(_BALANCED, user_name=USER_NAME)
        decision = evaluator.evaluate(_transcript("Alice, hello!"), 5.0)
        assert decision.should_respond is False

    def test_after_cooldown_allows_response(self) -> None:
        evaluator = ProfileEvaluator(_BALANCED, user_name=USER_NAME)
        decision = evaluator.evaluate(_transcript("Alice, hello!"), 60.0)
        assert decision.should_respond is True


# ---------------------------------------------------------------------------
# Chat message evaluation
# ---------------------------------------------------------------------------


class TestChatMessageEvaluation:
    def test_addressed_chat_always_responds(self) -> None:
        evaluator = ProfileEvaluator(_PASSIVE, user_name=USER_NAME)
        decision = evaluator.evaluate(_chat("@Alice check this please"), 120.0)
        assert decision.should_respond is True
        assert decision.channel == "chat"

    def test_unaddressed_chat_passive_ignores(self) -> None:
        evaluator = ProfileEvaluator(_PASSIVE, user_name=USER_NAME)
        decision = evaluator.evaluate(_chat("Anyone free for a sync?"), 120.0)
        assert decision.should_respond is False

    def test_question_in_chat_balanced(self) -> None:
        evaluator = ProfileEvaluator(_BALANCED, user_name=USER_NAME)
        decision = evaluator.evaluate(_chat("Does someone have the design doc?"), 60.0)
        assert decision.should_respond is True

    def test_unhandled_event_type(self) -> None:
        evaluator = ProfileEvaluator(_BALANCED, user_name=USER_NAME)
        decision = evaluator.evaluate(Event(type="unknown", data={}), 60.0)
        assert decision.should_respond is False
