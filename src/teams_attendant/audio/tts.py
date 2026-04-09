"""Azure Text-to-Speech integration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from teams_attendant.errors import TTSError

if TYPE_CHECKING:
    from teams_attendant.audio.playback import AudioPlayer
    from teams_attendant.config import AzureSpeechConfig

log = structlog.get_logger()


@dataclass
class TTSResult:
    """Result of a TTS synthesis."""

    audio_data: bytes
    duration_ms: int
    text: str


class SpeechSynthesizer:
    """Text-to-speech using Azure Speech Services."""

    def __init__(
        self,
        config: AzureSpeechConfig,
        voice_name: str = "en-US-JennyNeural",
    ) -> None:
        import azure.cognitiveservices.speech as speechsdk

        from teams_attendant.audio.speech_auth import build_speech_config

        self._speech_config = build_speech_config(config)
        self._speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
        )
        self._speech_config.speech_synthesis_voice_name = voice_name
        self._speechsdk = speechsdk

    def _create_synthesizer(self) -> object:
        """Create a synthesizer that writes to an in-memory stream."""
        return self._speechsdk.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=None,
        )

    def _handle_result(
        self,
        result: object,
        original_text: str,
    ) -> TTSResult:
        """Extract audio data from a synthesis result or raise on error."""
        cancellation_details = None
        if result.reason == self._speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            error_code = cancellation_details.error_code
            if error_code == self._speechsdk.CancellationErrorCode.AuthenticationFailure:
                raise TTSError(
                    f"TTS authentication failed: {cancellation_details.error_details}"
                )
            if error_code == self._speechsdk.CancellationErrorCode.ConnectionFailure:
                raise TTSError(
                    f"TTS connection failed: {cancellation_details.error_details}"
                )
            raise TTSError(
                f"TTS synthesis canceled: {cancellation_details.reason} – "
                f"{cancellation_details.error_details}"
            )

        if result.reason != self._speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise TTSError(f"TTS synthesis failed with reason: {result.reason}")

        audio_data = result.audio_data
        # 16 kHz, 16-bit mono → 32 000 bytes/sec
        duration_ms = int(len(audio_data) / 32_000 * 1000)

        log.info(
            "tts_synthesis_complete",
            text_length=len(original_text),
            audio_bytes=len(audio_data),
            duration_ms=duration_ms,
        )
        return TTSResult(audio_data=audio_data, duration_ms=duration_ms, text=original_text)

    async def synthesize(self, text: str) -> TTSResult:
        """Convert text to speech audio."""

        def _run() -> TTSResult:
            synth = self._create_synthesizer()
            result = synth.speak_text(text)
            return self._handle_result(result, text)

        return await asyncio.to_thread(_run)

    async def synthesize_ssml(self, ssml: str) -> TTSResult:
        """Convert SSML to speech audio."""

        def _run() -> TTSResult:
            synth = self._create_synthesizer()
            result = synth.speak_ssml(ssml)
            return self._handle_result(result, ssml)

        return await asyncio.to_thread(_run)

    def set_voice(self, voice_name: str) -> None:
        """Change the TTS voice."""
        self._speech_config.speech_synthesis_voice_name = voice_name
        log.info("tts_voice_changed", voice=voice_name)

    async def list_voices(self) -> list[dict[str, str]]:
        """List available voices."""

        def _run() -> list[dict[str, str]]:
            synth = self._create_synthesizer()
            result = synth.get_voices_async().get()
            if result.reason == self._speechsdk.ResultReason.VoicesListRetrieved:
                return [
                    {
                        "name": v.short_name,
                        "locale": v.locale,
                        "gender": str(v.gender),
                        "local_name": v.local_name,
                    }
                    for v in result.voices
                ]
            raise TTSError(f"Failed to list voices: {result.reason}")

        return await asyncio.to_thread(_run)


class VoiceResponder:
    """Combines TTS and audio playback for voice responses."""

    def __init__(self, synthesizer: SpeechSynthesizer, player: AudioPlayer) -> None:
        self._synthesizer = synthesizer
        self._player = player

    async def speak(self, text: str) -> None:
        """Synthesize text and play through virtual mic."""
        log.info("voice_responder_speaking", text_length=len(text))
        result = await self._synthesizer.synthesize(text)
        await self._player.play_with_fade(result.audio_data, sample_rate=16000)
        log.info("voice_responder_done", duration_ms=result.duration_ms)

    @property
    def is_speaking(self) -> bool:
        """Whether audio is currently being played."""
        return self._player.is_playing
