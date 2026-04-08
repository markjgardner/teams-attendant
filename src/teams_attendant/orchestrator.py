"""Meeting lifecycle orchestrator."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext

    from teams_attendant.agent.context import EventDrivenContext
    from teams_attendant.agent.core import AgentDecisionEngine
    from teams_attendant.agent.llm import ClaudeClient
    from teams_attendant.agent.vision import VisionAnalyzer
    from teams_attendant.audio.capture import AudioCaptureStream
    from teams_attendant.audio.playback import AudioPlayer
    from teams_attendant.audio.stt import SpeechTranscriber
    from teams_attendant.audio.tts import SpeechSynthesizer, VoiceResponder
    from teams_attendant.browser.chat import ChatObserver
    from teams_attendant.browser.meeting import MeetingController
    from teams_attendant.browser.screen import ScreenCaptureObserver
    from teams_attendant.config import AppConfig
    from teams_attendant.utils.events import EventBus

log = structlog.get_logger()

_POLL_INTERVAL_SECONDS = 5.0
_MAX_REJOIN_ATTEMPTS = 3
_REJOIN_DELAY_SECONDS = 5.0
_STATUS_LOG_INTERVAL_CHECKS = 12  # Log status every 12 polls (~60s)


@dataclass
class MeetingSession:
    """Holds all components for an active meeting session."""

    event_bus: EventBus
    context: EventDrivenContext
    meeting_controller: MeetingController
    chat_observer: ChatObserver
    browser_context: BrowserContext | None = None
    llm_client: ClaudeClient | None = None
    audio_capture: AudioCaptureStream | None = None
    transcriber: SpeechTranscriber | None = None
    synthesizer: SpeechSynthesizer | None = None
    player: AudioPlayer | None = None
    voice_responder: VoiceResponder | None = None
    decision_engine: AgentDecisionEngine | None = None
    screen_observer: ScreenCaptureObserver | None = None
    vision_analyzer: VisionAnalyzer | None = None


class MeetingOrchestrator:
    """Orchestrates all components for a meeting session."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._session: MeetingSession | None = None
        self._running = False
        self._start_time: datetime | None = None
        self._meeting_url: str = ""
        self._profile_name: str = ""
        self._user_name: str = ""
        self._vision_enabled: bool = False

    async def join_meeting(
        self,
        meeting_url: str,
        profile_name: str = "balanced",
        vision_enabled: bool = False,
        user_name: str = "",
    ) -> None:
        """Join a meeting and start the agent."""
        from teams_attendant.utils.logging import setup_logging

        setup_logging()
        log.info("orchestrator.starting", meeting_url=meeting_url, profile=profile_name)

        self._meeting_url = meeting_url
        self._profile_name = profile_name
        self._user_name = user_name
        self._vision_enabled = vision_enabled

        from teams_attendant.agent.context import EventDrivenContext
        from teams_attendant.utils.events import EventBus

        log.info("orchestrator.setting_up_event_bus")
        event_bus = EventBus()
        await event_bus.start()

        context = EventDrivenContext(event_bus=event_bus, user_name=user_name)

        try:
            log.info("orchestrator.authenticating")
            browser_context, controller = await self._setup_browser(meeting_url)

            from teams_attendant.browser.chat import ChatObserver

            page = controller.page
            chat_observer = ChatObserver(page=page, event_bus=event_bus)

            self._session = MeetingSession(
                event_bus=event_bus,
                context=context,
                meeting_controller=controller,
                chat_observer=chat_observer,
                browser_context=browser_context,
            )

            self._start_time = datetime.now(timezone.utc)

            meeting_info = await controller.get_info()
            log.info(
                "orchestrator.meeting_joined",
                title=meeting_info.title,
                participants=meeting_info.participant_count,
            )

            log.info("orchestrator.starting_chat_observer")
            await chat_observer.start()

            log.info("orchestrator.setting_up_audio")
            await self._setup_audio()
            audio_available = self._session.transcriber is not None
            log.info("orchestrator.audio_status", audio_available=audio_available)

            log.info(
                "orchestrator.starting_agent",
                profile=profile_name,
                vision=vision_enabled,
            )
            await self._setup_agent(profile_name, user_name, vision_enabled)
            log.info(
                "orchestrator.agent_ready_attending",
                profile=profile_name,
            )

            self._running = True
            await self._run_meeting_loop()
        except KeyboardInterrupt:
            log.info("orchestrator.interrupted")
        except asyncio.CancelledError:
            log.info("orchestrator.cancelled")
        except Exception:
            log.exception("orchestrator.error")
        finally:
            await self._cleanup()

    async def leave_meeting(self) -> None:
        """Leave the meeting and clean up."""
        log.info("orchestrator.leave_requested")
        self._running = False
        await self._cleanup()

    async def _setup_browser(
        self, meeting_url: str
    ) -> tuple[BrowserContext, MeetingController]:
        """Set up browser and join the meeting."""
        from playwright.async_api import async_playwright

        from teams_attendant.browser.auth import get_authenticated_context
        from teams_attendant.browser.meeting import MeetingController

        audio_env: dict[str, str] = {}
        try:
            from teams_attendant.audio.devices import (
                check_audio_setup,
                get_browser_audio_env,
                resolve_devices,
            )
            from teams_attendant.config import AudioDeviceConfig

            status = check_audio_setup()
            if status.is_ready:
                devices = resolve_devices(AudioDeviceConfig())
                audio_env = get_browser_audio_env(devices)
        except Exception:
            log.debug("orchestrator.audio_env_skip", reason="could not resolve devices")

        pw = await async_playwright().start()
        browser_context = await get_authenticated_context(
            pw,
            browser_data_dir=self._config.browser_data_dir,
            headless=True,
            audio_env=audio_env or None,
        )

        controller = MeetingController(context=browser_context)
        await controller.join(meeting_url)
        log.info("orchestrator.meeting_joined")

        return browser_context, controller

    async def _setup_audio(self) -> None:
        """Set up audio pipeline (capture, STT, TTS, playback)."""
        if self._session is None:
            return

        try:
            from teams_attendant.audio.devices import check_audio_setup

            status = check_audio_setup()
            if not status.is_ready:
                log.warning(
                    "orchestrator.audio_not_available",
                    issues=status.issues,
                    detail="Continuing in chat-only mode",
                )
                return

            from teams_attendant.audio.capture import AudioCaptureStream
            from teams_attendant.audio.devices import resolve_devices
            from teams_attendant.audio.playback import AudioPlayer
            from teams_attendant.audio.stt import SpeechTranscriber
            from teams_attendant.audio.tts import SpeechSynthesizer, VoiceResponder
            from teams_attendant.config import AudioDeviceConfig

            devices = resolve_devices(AudioDeviceConfig())

            self._session.audio_capture = AudioCaptureStream(devices=devices)
            self._session.transcriber = SpeechTranscriber(
                config=self._config.azure.speech,
                devices=devices,
                event_bus=self._session.event_bus,
            )
            self._session.synthesizer = SpeechSynthesizer(config=self._config.azure.speech)
            self._session.player = AudioPlayer(devices=devices)
            self._session.voice_responder = VoiceResponder(
                synthesizer=self._session.synthesizer,
                player=self._session.player,
            )

            await self._session.audio_capture.start()
            await self._session.transcriber.start()

            log.info("orchestrator.audio_ready")
        except Exception:
            log.warning(
                "orchestrator.audio_setup_failed",
                detail="Continuing in chat-only mode",
                exc_info=True,
            )
            self._session.audio_capture = None
            self._session.transcriber = None
            self._session.synthesizer = None
            self._session.player = None
            self._session.voice_responder = None

    async def _setup_agent(
        self,
        profile_name: str,
        user_name: str,
        vision_enabled: bool,
    ) -> None:
        """Set up the AI agent (decision engine, vision analyzer)."""
        if self._session is None:
            return

        from teams_attendant.agent.core import AgentDecisionEngine
        from teams_attendant.agent.llm import ClaudeClient
        from teams_attendant.agent.profiles import ProfileEvaluator
        from teams_attendant.config import load_profile

        profile = load_profile(profile_name)
        evaluator = ProfileEvaluator(profile, user_name=user_name)

        llm_client = ClaudeClient(config=self._config.azure.foundry)
        self._session.llm_client = llm_client

        chat_cb = self._create_chat_callback()
        voice_cb = self._create_voice_callback()

        self._session.decision_engine = AgentDecisionEngine(
            llm_client=llm_client,
            context=self._session.context,
            profile_evaluator=evaluator,
            event_bus=self._session.event_bus,
            on_chat_response=chat_cb,
            on_voice_response=voice_cb,
        )
        await self._session.decision_engine.start()

        if vision_enabled:
            try:
                from teams_attendant.agent.vision import VisionAnalyzer
                from teams_attendant.browser.screen import ScreenCaptureObserver

                self._session.vision_analyzer = VisionAnalyzer(
                    llm_client=llm_client,
                    context=self._session.context,
                    event_bus=self._session.event_bus,
                )
                await self._session.vision_analyzer.start()

                page = self._session.meeting_controller.page
                if page is not None:
                    self._session.screen_observer = ScreenCaptureObserver(
                        page=page,
                        event_bus=self._session.event_bus,
                    )
                    await self._session.screen_observer.start()

                log.info("orchestrator.vision_enabled")
            except Exception:
                log.warning(
                    "orchestrator.vision_setup_failed",
                    detail="Continuing without vision",
                    exc_info=True,
                )
                self._session.vision_analyzer = None
                self._session.screen_observer = None

        log.info("orchestrator.agent_ready")

    def _create_chat_callback(self) -> Callable[[str], Awaitable[None]]:
        """Create the chat response callback for the decision engine."""

        async def _chat_callback(text: str) -> None:
            log.info("orchestrator.chat_response", text=text[:100])
            if self._session is not None:
                await self._session.chat_observer.send_message(text)

        return _chat_callback

    def _create_voice_callback(self) -> Callable[[str], Awaitable[None]]:
        """Create the voice response callback for the decision engine."""

        async def _voice_callback(text: str) -> None:
            log.info("orchestrator.voice_response", text=text[:100])
            if self._session is not None and self._session.voice_responder is not None:
                await self._session.voice_responder.speak(text)
            else:
                log.warning("orchestrator.voice_not_available", detail="Falling back to chat")
                if self._session is not None:
                    await self._session.chat_observer.send_message(text)

        return _voice_callback

    async def _run_meeting_loop(self) -> None:
        """Main meeting attendance loop with state monitoring and auto-rejoin."""
        if self._session is None:
            return

        from teams_attendant.browser.meeting import MeetingState

        log.info("orchestrator.meeting_loop_started")
        rejoin_attempts = 0
        previous_state: MeetingState | None = None
        poll_count = 0

        try:
            while self._running:
                if self._session is None:
                    break

                try:
                    current_state = await self._session.meeting_controller.get_state()
                except Exception:
                    log.warning("orchestrator.state_check_failed", exc_info=True)
                    current_state = MeetingState.ERROR

                # Detect and log state transitions
                if current_state != previous_state:
                    log.info(
                        "orchestrator.state_transition",
                        previous=previous_state.value if previous_state else "none",
                        current=current_state.value,
                    )
                    previous_state = current_state

                if current_state == MeetingState.ENDED:
                    log.info("orchestrator.meeting_ended")
                    self._running = False
                    break

                if current_state == MeetingState.ERROR:
                    rejoin_attempts += 1
                    if rejoin_attempts > _MAX_REJOIN_ATTEMPTS:
                        log.error(
                            "orchestrator.max_rejoin_attempts_exceeded",
                            attempts=rejoin_attempts - 1,
                        )
                        self._running = False
                        break

                    log.warning(
                        "orchestrator.attempting_rejoin",
                        attempt=rejoin_attempts,
                        max_attempts=_MAX_REJOIN_ATTEMPTS,
                    )
                    await asyncio.sleep(_REJOIN_DELAY_SECONDS)

                    try:
                        await self._attempt_rejoin()
                        previous_state = None  # Reset to detect new state
                        log.info("orchestrator.rejoin_succeeded", attempt=rejoin_attempts)
                    except Exception:
                        log.warning(
                            "orchestrator.rejoin_failed",
                            attempt=rejoin_attempts,
                            exc_info=True,
                        )
                    continue

                # Successful state check resets rejoin counter
                if current_state == MeetingState.JOINED:
                    rejoin_attempts = 0

                # Periodic status logging
                poll_count += 1
                if poll_count % _STATUS_LOG_INTERVAL_CHECKS == 0:
                    await self._log_meeting_status()

                await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        except asyncio.CancelledError:
            log.info("orchestrator.shutdown_requested")
        except KeyboardInterrupt:
            log.info("orchestrator.manual_exit")
        except Exception:
            log.exception("orchestrator.meeting_loop_error")
        finally:
            self._running = False
            log.info("orchestrator.meeting_loop_ended")

    async def _attempt_rejoin(self) -> None:
        """Attempt to rejoin the meeting after a disconnect."""
        if self._session is None or not self._meeting_url:
            raise RuntimeError("No session or meeting URL for rejoin")

        await self._session.meeting_controller.join(self._meeting_url)

    async def _log_meeting_status(self) -> None:
        """Log a periodic status snapshot."""
        try:
            status = await self.get_meeting_status()
            log.info(
                "orchestrator.status",
                status=status.get("status"),
                participants=status.get("participants"),
                duration_seconds=int(status.get("duration_seconds", 0)),
                agent_responses=status.get("agent_responses"),
            )
        except Exception:
            log.debug("orchestrator.status_log_failed", exc_info=True)

    async def _cleanup(self) -> None:
        """Clean up all resources in reverse order of creation."""
        if self._session is None:
            return

        log.info("orchestrator.cleanup_start")
        session = self._session
        self._session = None

        # Stop decision engine
        if session.decision_engine is not None:
            try:
                await session.decision_engine.stop()
                log.info("orchestrator.cleanup.decision_engine_stopped")
            except Exception:
                log.warning("orchestrator.cleanup.decision_engine_error", exc_info=True)

        # Stop vision components
        if session.vision_analyzer is not None:
            try:
                await session.vision_analyzer.stop()
                log.info("orchestrator.cleanup.vision_analyzer_stopped")
            except Exception:
                log.warning("orchestrator.cleanup.vision_analyzer_error", exc_info=True)

        if session.screen_observer is not None:
            try:
                await session.screen_observer.stop()
                log.info("orchestrator.cleanup.screen_observer_stopped")
            except Exception:
                log.warning("orchestrator.cleanup.screen_observer_error", exc_info=True)

        # Stop audio components
        if session.transcriber is not None:
            try:
                await session.transcriber.stop()
                log.info("orchestrator.cleanup.transcriber_stopped")
            except Exception:
                log.warning("orchestrator.cleanup.transcriber_error", exc_info=True)

        if session.audio_capture is not None:
            try:
                await session.audio_capture.stop()
                log.info("orchestrator.cleanup.audio_capture_stopped")
            except Exception:
                log.warning("orchestrator.cleanup.audio_capture_error", exc_info=True)

        # Stop chat observer
        try:
            await session.chat_observer.stop()
            log.info("orchestrator.cleanup.chat_observer_stopped")
        except Exception:
            log.warning("orchestrator.cleanup.chat_observer_error", exc_info=True)

        # Leave meeting
        try:
            await session.meeting_controller.leave()
            log.info("orchestrator.cleanup.meeting_left")
        except Exception:
            log.warning("orchestrator.cleanup.meeting_leave_error", exc_info=True)

        # Close browser context
        if session.browser_context is not None:
            try:
                await session.browser_context.close()
                log.info("orchestrator.cleanup.browser_closed")
            except Exception:
                log.warning("orchestrator.cleanup.browser_close_error", exc_info=True)

        # Stop event bus
        try:
            await session.event_bus.stop()
            log.info("orchestrator.cleanup.event_bus_stopped")
        except Exception:
            log.warning("orchestrator.cleanup.event_bus_error", exc_info=True)

        # Close LLM client
        if session.llm_client is not None:
            try:
                await session.llm_client.close()
                log.info("orchestrator.cleanup.llm_closed")
            except Exception:
                log.warning("orchestrator.cleanup.llm_close_error", exc_info=True)

        log.info("orchestrator.cleanup_done")

    async def get_meeting_status(self) -> dict[str, Any]:
        """Get current meeting status for CLI display."""
        if not self._session:
            return {"status": "not_in_meeting"}

        try:
            info = await self._session.meeting_controller.get_info()
        except Exception:
            return {"status": "error", "error": "Failed to get meeting info"}

        duration_seconds = 0.0
        if self._start_time is not None:
            duration_seconds = (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds()

        try:
            agent_responses = len(self._session.context.get_agent_contributions())
        except Exception:
            agent_responses = 0

        return {
            "status": info.state.value,
            "title": info.title,
            "participants": info.participant_count,
            "duration_seconds": duration_seconds,
            "is_recording": info.is_recording,
            "is_screen_sharing": info.is_screen_sharing,
            "agent_responses": agent_responses,
            "audio_enabled": self._session.transcriber is not None,
            "vision_enabled": self._session.vision_analyzer is not None,
        }
