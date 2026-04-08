"""Tests for the meeting lifecycle orchestrator."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from teams_attendant.browser.meeting import MeetingInfo, MeetingState
from teams_attendant.orchestrator import (
    MeetingOrchestrator,
    MeetingSession,
    _MAX_REJOIN_ATTEMPTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> MagicMock:
    """Return a mock AppConfig."""
    config = MagicMock()
    config.browser_data_dir = ".browser-data"
    config.azure.speech.key = "fake-key"
    config.azure.speech.region = "eastus"
    config.azure.foundry.endpoint = "https://fake.endpoint"
    config.azure.foundry.api_key = "fake-api-key"
    config.azure.foundry.model_deployment = "claude-test"
    return config


def _make_session(
    *,
    voice_responder: AsyncMock | None = None,
) -> MeetingSession:
    """Build a MeetingSession with mocked components."""
    event_bus = MagicMock()
    event_bus.start = AsyncMock()
    event_bus.stop = AsyncMock()

    context = MagicMock()
    context.get_agent_contributions = MagicMock(return_value=[])

    controller = MagicMock()
    controller.page = MagicMock()
    controller.leave = AsyncMock()
    controller.join = AsyncMock()
    controller.wait_for_meeting_end = AsyncMock()
    controller.get_state = AsyncMock(return_value=MeetingState.JOINED)
    controller.get_info = AsyncMock(
        return_value=MeetingInfo(
            title="Test Meeting",
            state=MeetingState.JOINED,
            participant_count=3,
            is_recording=False,
            is_screen_sharing=False,
        )
    )

    chat_observer = MagicMock()
    chat_observer.start = AsyncMock()
    chat_observer.stop = AsyncMock()
    chat_observer.send_message = AsyncMock()

    session = MeetingSession(
        event_bus=event_bus,
        context=context,
        meeting_controller=controller,
        chat_observer=chat_observer,
        voice_responder=voice_responder,
    )
    return session


# ---------------------------------------------------------------------------
# Chat callback
# ---------------------------------------------------------------------------


class TestChatCallback:
    """Test that the chat callback sends via ChatObserver."""

    async def test_chat_callback_sends_message(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = _make_session()

        cb = orchestrator._create_chat_callback()
        await cb("Hello from the agent!")

        orchestrator._session.chat_observer.send_message.assert_awaited_once_with(
            "Hello from the agent!"
        )

    async def test_chat_callback_no_session(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = None

        cb = orchestrator._create_chat_callback()
        # Should not raise when session is None
        await cb("test")


# ---------------------------------------------------------------------------
# Voice callback
# ---------------------------------------------------------------------------


class TestVoiceCallback:
    """Test that the voice callback sends via VoiceResponder."""

    async def test_voice_callback_speaks(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        voice_responder = AsyncMock()
        voice_responder.speak = AsyncMock()
        orchestrator._session = _make_session(voice_responder=voice_responder)

        cb = orchestrator._create_voice_callback()
        await cb("Speaking this text")

        voice_responder.speak.assert_awaited_once_with("Speaking this text")

    async def test_voice_callback_falls_back_to_chat(self) -> None:
        """When voice_responder is None, the callback falls back to chat."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = _make_session(voice_responder=None)

        cb = orchestrator._create_voice_callback()
        await cb("Fallback text")

        orchestrator._session.chat_observer.send_message.assert_awaited_once_with(
            "Fallback text"
        )

    async def test_voice_callback_no_session(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = None

        cb = orchestrator._create_voice_callback()
        # Should not raise when session is None
        await cb("test")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """Test that cleanup stops all components in order."""

    async def test_cleanup_stops_all_components(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)

        decision_engine = MagicMock()
        decision_engine.stop = AsyncMock()

        vision_analyzer = MagicMock()
        vision_analyzer.stop = AsyncMock()

        screen_observer = MagicMock()
        screen_observer.stop = AsyncMock()

        transcriber = MagicMock()
        transcriber.stop = AsyncMock()

        audio_capture = MagicMock()
        audio_capture.stop = AsyncMock()

        browser_context = MagicMock()
        browser_context.close = AsyncMock()

        llm_client = MagicMock()
        llm_client.close = AsyncMock()

        session = _make_session()
        session.decision_engine = decision_engine
        session.vision_analyzer = vision_analyzer
        session.screen_observer = screen_observer
        session.transcriber = transcriber
        session.audio_capture = audio_capture
        session.browser_context = browser_context
        session.llm_client = llm_client

        orchestrator._session = session

        await orchestrator._cleanup()

        decision_engine.stop.assert_awaited_once()
        vision_analyzer.stop.assert_awaited_once()
        screen_observer.stop.assert_awaited_once()
        transcriber.stop.assert_awaited_once()
        audio_capture.stop.assert_awaited_once()
        session.chat_observer.stop.assert_awaited_once()
        session.meeting_controller.leave.assert_awaited_once()
        browser_context.close.assert_awaited_once()
        session.event_bus.stop.assert_awaited_once()
        llm_client.close.assert_awaited_once()

        assert orchestrator._session is None

    async def test_cleanup_no_session(self) -> None:
        """Cleanup should be a no-op when there is no session."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = None
        # Should not raise
        await orchestrator._cleanup()

    async def test_cleanup_resilient_to_errors(self) -> None:
        """Cleanup continues even if individual stop calls raise."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)

        decision_engine = MagicMock()
        decision_engine.stop = AsyncMock(side_effect=RuntimeError("engine stop failed"))

        transcriber = MagicMock()
        transcriber.stop = AsyncMock(side_effect=RuntimeError("transcriber stop failed"))

        session = _make_session()
        session.decision_engine = decision_engine
        session.transcriber = transcriber

        orchestrator._session = session

        # Should not raise despite errors
        await orchestrator._cleanup()

        decision_engine.stop.assert_awaited_once()
        transcriber.stop.assert_awaited_once()
        session.chat_observer.stop.assert_awaited_once()
        session.meeting_controller.leave.assert_awaited_once()


# ---------------------------------------------------------------------------
# Chat-only mode (no audio)
# ---------------------------------------------------------------------------


class TestChatOnlyMode:
    """Test graceful handling when audio is not available."""

    async def test_setup_audio_skips_when_not_ready(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = _make_session()

        with patch(
            "teams_attendant.audio.devices.check_audio_setup"
        ) as mock_check:
            mock_status = MagicMock()
            mock_status.is_ready = False
            mock_status.issues = ["No virtual audio cable detected."]
            mock_check.return_value = mock_status

            await orchestrator._setup_audio()

        assert orchestrator._session.audio_capture is None
        assert orchestrator._session.transcriber is None
        assert orchestrator._session.voice_responder is None

    async def test_setup_audio_handles_exception(self) -> None:
        """Audio setup failure should not crash the orchestrator."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = _make_session()

        with patch(
            "teams_attendant.audio.devices.check_audio_setup",
            side_effect=RuntimeError("audio exploded"),
        ):
            # Should not raise — audio failure is handled gracefully
            await orchestrator._setup_audio()

        # The session should still exist with no audio components
        assert orchestrator._session.audio_capture is None

    async def test_setup_audio_no_session(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = None
        # Should not raise
        await orchestrator._setup_audio()


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------


class TestAgentSetup:
    """Test agent setup creates decision engine and wires callbacks."""

    async def test_setup_agent_creates_engine(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = _make_session()

        with (
            patch("teams_attendant.config.load_profile") as mock_load,
            patch("teams_attendant.agent.profiles.ProfileEvaluator"),
            patch("teams_attendant.agent.llm.ClaudeClient") as mock_llm_cls,
            patch(
                "teams_attendant.agent.core.AgentDecisionEngine"
            ) as mock_engine_cls,
        ):
            mock_profile = MagicMock()
            mock_load.return_value = mock_profile

            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm

            mock_engine = MagicMock()
            mock_engine.start = AsyncMock()
            mock_engine_cls.return_value = mock_engine

            await orchestrator._setup_agent("balanced", "TestUser", vision_enabled=False)

        mock_load.assert_called_once_with("balanced")
        mock_engine.start.assert_awaited_once()
        assert orchestrator._session.decision_engine is mock_engine
        assert orchestrator._session.llm_client is mock_llm

    async def test_setup_agent_no_session(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = None
        # Should not raise
        await orchestrator._setup_agent("balanced", "User", vision_enabled=False)


# ---------------------------------------------------------------------------
# Vision components
# ---------------------------------------------------------------------------


class TestVisionSetup:
    """Test vision components created/omitted based on flag."""

    async def test_vision_enabled_creates_components(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = _make_session()

        with (
            patch("teams_attendant.config.load_profile") as mock_load,
            patch("teams_attendant.agent.profiles.ProfileEvaluator"),
            patch("teams_attendant.agent.llm.ClaudeClient") as mock_llm_cls,
            patch(
                "teams_attendant.agent.core.AgentDecisionEngine"
            ) as mock_engine_cls,
            patch(
                "teams_attendant.agent.vision.VisionAnalyzer"
            ) as mock_vision_cls,
            patch(
                "teams_attendant.browser.screen.ScreenCaptureObserver"
            ) as mock_screen_cls,
        ):
            mock_load.return_value = MagicMock()
            mock_llm_cls.return_value = MagicMock()

            mock_engine = MagicMock()
            mock_engine.start = AsyncMock()
            mock_engine_cls.return_value = mock_engine

            mock_vision = MagicMock()
            mock_vision.start = AsyncMock()
            mock_vision_cls.return_value = mock_vision

            mock_screen = MagicMock()
            mock_screen.start = AsyncMock()
            mock_screen_cls.return_value = mock_screen

            await orchestrator._setup_agent("balanced", "User", vision_enabled=True)

        assert orchestrator._session.vision_analyzer is mock_vision
        assert orchestrator._session.screen_observer is mock_screen
        mock_vision.start.assert_awaited_once()
        mock_screen.start.assert_awaited_once()

    async def test_vision_disabled_skips_components(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = _make_session()

        with (
            patch("teams_attendant.config.load_profile") as mock_load,
            patch("teams_attendant.agent.profiles.ProfileEvaluator"),
            patch("teams_attendant.agent.llm.ClaudeClient") as mock_llm_cls,
            patch(
                "teams_attendant.agent.core.AgentDecisionEngine"
            ) as mock_engine_cls,
        ):
            mock_load.return_value = MagicMock()
            mock_llm_cls.return_value = MagicMock()

            mock_engine = MagicMock()
            mock_engine.start = AsyncMock()
            mock_engine_cls.return_value = mock_engine

            await orchestrator._setup_agent("balanced", "User", vision_enabled=False)

        assert orchestrator._session.vision_analyzer is None
        assert orchestrator._session.screen_observer is None

    async def test_vision_setup_failure_continues(self) -> None:
        """Vision setup failure should not crash the orchestrator."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = _make_session()

        with (
            patch("teams_attendant.config.load_profile") as mock_load,
            patch("teams_attendant.agent.profiles.ProfileEvaluator"),
            patch("teams_attendant.agent.llm.ClaudeClient") as mock_llm_cls,
            patch(
                "teams_attendant.agent.core.AgentDecisionEngine"
            ) as mock_engine_cls,
            patch(
                "teams_attendant.agent.vision.VisionAnalyzer",
                side_effect=RuntimeError("vision init failed"),
            ),
        ):
            mock_load.return_value = MagicMock()
            mock_llm_cls.return_value = MagicMock()

            mock_engine = MagicMock()
            mock_engine.start = AsyncMock()
            mock_engine_cls.return_value = mock_engine

            # Should not raise
            await orchestrator._setup_agent("balanced", "User", vision_enabled=True)

        assert orchestrator._session.vision_analyzer is None
        assert orchestrator._session.screen_observer is None


# ---------------------------------------------------------------------------
# Meeting loop
# ---------------------------------------------------------------------------


class TestMeetingLoop:
    """Test the main meeting loop."""

    async def test_run_meeting_loop_detects_ended_state(self) -> None:
        """Loop exits when meeting state becomes ENDED."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        session = _make_session()
        session.meeting_controller.get_state = AsyncMock(return_value=MeetingState.ENDED)
        orchestrator._session = session
        orchestrator._running = True

        await orchestrator._run_meeting_loop()

        assert orchestrator._running is False

    async def test_run_meeting_loop_no_session(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = None
        # Should not raise
        await orchestrator._run_meeting_loop()

    async def test_run_meeting_loop_handles_cancelled_error(self) -> None:
        """Loop handles CancelledError gracefully."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        session = _make_session()
        session.meeting_controller.get_state = AsyncMock(side_effect=asyncio.CancelledError)
        orchestrator._session = session
        orchestrator._running = True

        await orchestrator._run_meeting_loop()
        assert orchestrator._running is False

    async def test_run_meeting_loop_handles_generic_exception(self) -> None:
        """Loop handles unexpected exceptions without crashing."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        session = _make_session()
        session.meeting_controller.get_state = AsyncMock(
            side_effect=RuntimeError("unexpected")
        )
        orchestrator._session = session
        orchestrator._running = True
        orchestrator._meeting_url = "https://example.com/meeting"

        # ERROR state triggers rejoin, which will also fail → eventually exceeds max
        session.meeting_controller.join = AsyncMock(side_effect=RuntimeError("rejoin fail"))

        await orchestrator._run_meeting_loop()
        assert orchestrator._running is False


# ---------------------------------------------------------------------------
# Leave meeting
# ---------------------------------------------------------------------------


class TestLeaveMeeting:
    """Test leave_meeting triggers cleanup."""

    async def test_leave_meeting_cleans_up(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = _make_session()
        orchestrator._running = True

        await orchestrator.leave_meeting()

        assert orchestrator._running is False
        assert orchestrator._session is None


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


class TestLoggingSetup:
    """Test structured logging configuration."""

    def test_setup_logging_runs(self) -> None:
        from teams_attendant.utils.logging import setup_logging

        # Should not raise
        setup_logging(level="DEBUG", json_output=False)

    def test_setup_logging_json(self) -> None:
        from teams_attendant.utils.logging import setup_logging

        # Should not raise
        setup_logging(level="INFO", json_output=True)

    def test_setup_logging_invalid_level_defaults(self) -> None:
        from teams_attendant.utils.logging import setup_logging

        # Invalid level should default to INFO
        setup_logging(level="INVALID")


# ---------------------------------------------------------------------------
# Auto-rejoin on ERROR state
# ---------------------------------------------------------------------------


class TestAutoRejoin:
    """Test auto-rejoin behaviour on ERROR state."""

    async def test_rejoin_on_error_state(self) -> None:
        """Loop attempts to rejoin when state becomes ERROR, then recovers."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        session = _make_session()
        orchestrator._session = session
        orchestrator._running = True
        orchestrator._meeting_url = "https://example.com/meeting"

        call_count = 0

        async def _state_sequence() -> MeetingState:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MeetingState.ERROR
            if call_count == 2:
                return MeetingState.JOINED
            return MeetingState.ENDED

        session.meeting_controller.get_state = AsyncMock(side_effect=_state_sequence)

        with patch("teams_attendant.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            await orchestrator._run_meeting_loop()

        session.meeting_controller.join.assert_awaited_once_with(
            "https://example.com/meeting"
        )
        assert orchestrator._running is False

    async def test_max_rejoin_attempts_respected(self) -> None:
        """Loop stops after exceeding max rejoin attempts."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        session = _make_session()
        orchestrator._session = session
        orchestrator._running = True
        orchestrator._meeting_url = "https://example.com/meeting"

        # Always return ERROR
        session.meeting_controller.get_state = AsyncMock(return_value=MeetingState.ERROR)
        session.meeting_controller.join = AsyncMock(side_effect=RuntimeError("fail"))

        with patch("teams_attendant.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            await orchestrator._run_meeting_loop()

        assert orchestrator._running is False
        assert session.meeting_controller.join.await_count == _MAX_REJOIN_ATTEMPTS

    async def test_rejoin_counter_resets_on_success(self) -> None:
        """Rejoin counter resets after successful rejoin and JOINED state."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        session = _make_session()
        orchestrator._session = session
        orchestrator._running = True
        orchestrator._meeting_url = "https://example.com/meeting"

        call_count = 0

        async def _state_sequence() -> MeetingState:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MeetingState.ERROR  # 1st error, triggers rejoin #1
            if call_count == 2:
                return MeetingState.JOINED  # Rejoin success, counter resets
            if call_count == 3:
                return MeetingState.ERROR  # 2nd error, triggers rejoin #1 again
            if call_count == 4:
                return MeetingState.JOINED  # Success again
            return MeetingState.ENDED

        session.meeting_controller.get_state = AsyncMock(side_effect=_state_sequence)

        with patch("teams_attendant.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            await orchestrator._run_meeting_loop()

        # Should have rejoined twice (counter reset between them)
        assert session.meeting_controller.join.await_count == 2


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    """Test meeting state transition detection."""

    async def test_state_transitions_logged(self) -> None:
        """Loop detects and handles state transitions."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        session = _make_session()
        orchestrator._session = session
        orchestrator._running = True

        states = [MeetingState.JOINED, MeetingState.JOINED, MeetingState.ENDED]
        session.meeting_controller.get_state = AsyncMock(side_effect=states)

        with patch("teams_attendant.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            await orchestrator._run_meeting_loop()

        assert orchestrator._running is False

    async def test_get_state_exception_treated_as_error(self) -> None:
        """When get_state raises, treat as ERROR state."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        session = _make_session()
        orchestrator._session = session
        orchestrator._running = True
        orchestrator._meeting_url = "https://example.com/meeting"

        # get_state always raises → treated as ERROR → max rejoin exceeded
        session.meeting_controller.get_state = AsyncMock(
            side_effect=RuntimeError("page crash")
        )
        session.meeting_controller.join = AsyncMock(
            side_effect=RuntimeError("rejoin fail")
        )

        with patch("teams_attendant.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            await orchestrator._run_meeting_loop()

        assert orchestrator._running is False
        assert session.meeting_controller.join.await_count == _MAX_REJOIN_ATTEMPTS


# ---------------------------------------------------------------------------
# Duration tracking
# ---------------------------------------------------------------------------


class TestDurationTracking:
    """Test meeting duration tracking."""

    async def test_duration_tracked_in_status(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = _make_session()
        orchestrator._start_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        with patch(
            "teams_attendant.orchestrator.datetime",
        ) as mock_dt:
            mock_dt.now.return_value = datetime(
                2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc
            )
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            status = await orchestrator.get_meeting_status()

        assert status["duration_seconds"] == 300.0

    async def test_duration_zero_when_no_start_time(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = _make_session()
        orchestrator._start_time = None

        status = await orchestrator.get_meeting_status()
        assert status["duration_seconds"] == 0.0


# ---------------------------------------------------------------------------
# get_meeting_status
# ---------------------------------------------------------------------------


class TestGetMeetingStatus:
    """Test get_meeting_status returns correct structure."""

    async def test_not_in_meeting(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        orchestrator._session = None

        status = await orchestrator.get_meeting_status()
        assert status == {"status": "not_in_meeting"}

    async def test_returns_correct_structure(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        session = _make_session()
        orchestrator._session = session
        orchestrator._start_time = datetime.now(timezone.utc)

        status = await orchestrator.get_meeting_status()

        assert status["status"] == "joined"
        assert status["title"] == "Test Meeting"
        assert status["participants"] == 3
        assert isinstance(status["duration_seconds"], float)
        assert status["is_recording"] is False
        assert status["is_screen_sharing"] is False
        assert status["agent_responses"] == 0
        assert status["audio_enabled"] is False
        assert status["vision_enabled"] is False

    async def test_with_audio_and_vision(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        session = _make_session()
        session.transcriber = MagicMock()
        session.vision_analyzer = MagicMock()
        orchestrator._session = session
        orchestrator._start_time = datetime.now(timezone.utc)

        status = await orchestrator.get_meeting_status()
        assert status["audio_enabled"] is True
        assert status["vision_enabled"] is True

    async def test_handles_get_info_failure(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        session = _make_session()
        session.meeting_controller.get_info = AsyncMock(
            side_effect=RuntimeError("page crashed")
        )
        orchestrator._session = session

        status = await orchestrator.get_meeting_status()
        assert status["status"] == "error"

    async def test_agent_responses_counted(self) -> None:
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)
        session = _make_session()
        session.context.get_agent_contributions = MagicMock(
            return_value=[MagicMock(), MagicMock(), MagicMock()]
        )
        orchestrator._session = session
        orchestrator._start_time = datetime.now(timezone.utc)

        status = await orchestrator.get_meeting_status()
        assert status["agent_responses"] == 3


# ---------------------------------------------------------------------------
# Cleanup resilience
# ---------------------------------------------------------------------------


class TestCleanupResilience:
    """Test cleanup handles individual component failures."""

    async def test_cleanup_continues_after_component_failure(self) -> None:
        """Each component failure is isolated; subsequent components still stop."""
        config = _make_config()
        orchestrator = MeetingOrchestrator(config)

        decision_engine = MagicMock()
        decision_engine.stop = AsyncMock(side_effect=RuntimeError("engine boom"))

        transcriber = MagicMock()
        transcriber.stop = AsyncMock(side_effect=RuntimeError("transcriber boom"))

        audio_capture = MagicMock()
        audio_capture.stop = AsyncMock()

        browser_context = MagicMock()
        browser_context.close = AsyncMock(side_effect=RuntimeError("browser boom"))

        llm_client = MagicMock()
        llm_client.close = AsyncMock()

        session = _make_session()
        session.decision_engine = decision_engine
        session.transcriber = transcriber
        session.audio_capture = audio_capture
        session.browser_context = browser_context
        session.llm_client = llm_client

        orchestrator._session = session

        # Should not raise despite 3 component failures
        await orchestrator._cleanup()

        # All components attempted to stop
        decision_engine.stop.assert_awaited_once()
        transcriber.stop.assert_awaited_once()
        audio_capture.stop.assert_awaited_once()
        session.chat_observer.stop.assert_awaited_once()
        session.meeting_controller.leave.assert_awaited_once()
        browser_context.close.assert_awaited_once()
        session.event_bus.stop.assert_awaited_once()
        llm_client.close.assert_awaited_once()

        assert orchestrator._session is None
