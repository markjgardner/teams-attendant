"""Azure Speech-to-Text integration."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:  # pragma: no cover
    speechsdk = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from teams_attendant.audio.devices import ResolvedDevices
    from teams_attendant.config import AzureSpeechConfig
    from teams_attendant.utils.events import EventBus

log = structlog.get_logger()


class SpeechTranscriber:
    """Continuous speech-to-text using Azure Speech Services."""

    def __init__(
        self,
        config: AzureSpeechConfig,
        devices: ResolvedDevices,
        event_bus: EventBus,
    ) -> None:
        if speechsdk is None:
            raise ImportError(
                "azure-cognitiveservices-speech is required for speech-to-text. "
                "Install it with: pip install azure-cognitiveservices-speech"
            )

        self._config = config
        self._devices = devices
        self._event_bus = event_bus
        self._recognizer: speechsdk.SpeechRecognizer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        from teams_attendant.audio.speech_auth import build_speech_config

        speech_config = build_speech_config(config)
        speech_config.speech_recognition_language = "en-US"

        # Enable diarization if available
        try:
            speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
                "Continuous",
            )
        except Exception:
            log.debug("speech.diarization_not_available")

        # Configure audio input
        if devices.capture_device_index is not None:
            audio_config = speechsdk.audio.AudioConfig(
                device_name=str(devices.capture_device_index),
            )
        else:
            audio_config = speechsdk.audio.AudioConfig(
                use_default_microphone=True,
            )

        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        # Wire up callbacks
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.recognizing.connect(self._on_recognizing)
        self._recognizer.canceled.connect(self._on_canceled)

    async def start(self) -> None:
        """Start continuous speech recognition (non-blocking)."""
        if self._recognizer is None:
            raise RuntimeError("SpeechTranscriber not properly initialized")

        self._loop = asyncio.get_running_loop()
        self._recognizer.start_continuous_recognition_async()
        log.info("speech.recognition_started")

    async def stop(self) -> None:
        """Stop speech recognition and clean up."""
        if self._recognizer is None:
            return

        self._recognizer.stop_continuous_recognition_async()
        log.info("speech.recognition_stopped")

    def _publish_event(self, text: str, speaker: str, is_final: bool, confidence: float) -> None:
        """Bridge SDK callbacks to async event bus."""
        from teams_attendant.utils.events import TranscriptEvent

        event = TranscriptEvent(
            text=text,
            speaker=speaker,
            is_final=is_final,
            confidence=confidence,
        )

        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future,
                self._event_bus.publish(event),
            )

    def _on_recognized(self, evt: Any) -> None:
        """Handle final recognition result."""
        result = evt.result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech and result.text:
            speaker = ""
            try:
                if hasattr(result, "speaker_id"):
                    speaker = result.speaker_id or ""
            except Exception:
                pass

            confidence = 1.0
            try:
                if result.best() and len(result.best()) > 0:
                    confidence = result.best()[0].confidence
            except Exception:
                pass

            log.info("speech.recognized", text=result.text, speaker=speaker)
            self._publish_event(
                text=result.text,
                speaker=speaker,
                is_final=True,
                confidence=confidence,
            )

    def _on_recognizing(self, evt: Any) -> None:
        """Handle interim recognition result."""
        result = evt.result
        if result.text:
            log.debug("speech.recognizing", text=result.text)
            self._publish_event(
                text=result.text,
                speaker="",
                is_final=False,
                confidence=0.0,
            )

    def _on_canceled(self, evt: Any) -> None:
        """Handle recognition cancellation/error."""
        cancellation = evt.result
        reason = cancellation.reason

        if reason == speechsdk.CancellationReason.Error:
            error_code = cancellation.error_code
            error_details = cancellation.error_details

            if error_code == speechsdk.CancellationErrorCode.AuthenticationFailure:
                log.error(
                    "speech.auth_failure",
                    details="Check your Azure Speech key and region.",
                )
            elif error_code == speechsdk.CancellationErrorCode.ConnectionFailure:
                log.error(
                    "speech.connection_failure",
                    details=error_details,
                )
            else:
                log.error(
                    "speech.canceled_error",
                    error_code=str(error_code),
                    details=error_details,
                )
        elif reason == speechsdk.CancellationReason.EndOfStream:
            log.info("speech.end_of_stream")
        else:
            log.warning("speech.canceled", reason=str(reason))
