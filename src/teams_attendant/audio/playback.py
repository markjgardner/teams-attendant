"""Audio playback to virtual microphone."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import structlog

if TYPE_CHECKING:
    from teams_attendant.audio.devices import ResolvedDevices

log = structlog.get_logger()


class AudioPlayer:
    """Plays audio through a virtual microphone device."""

    def __init__(
        self,
        devices: ResolvedDevices,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self._device_index = devices.playback_device_index
        self._sample_rate = sample_rate
        self._channels = channels
        self._playing = False

    async def play(self, audio_data: bytes, sample_rate: int = 16000) -> None:
        """Play raw PCM audio through the virtual mic."""
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if self._channels == 1:
            samples = samples.reshape(-1, 1)

        log.info(
            "audio_play_start",
            samples=len(samples),
            sample_rate=sample_rate,
            device=self._device_index,
        )
        self._playing = True

        def _run() -> None:
            try:
                sd.play(samples, samplerate=sample_rate, device=self._device_index)
                sd.wait()
            finally:
                self._playing = False

        await asyncio.to_thread(_run)

    async def play_with_fade(
        self,
        audio_data: bytes,
        fade_ms: int = 50,
        sample_rate: int = 16000,
    ) -> None:
        """Play audio with fade-in/out to avoid clicks."""
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        fade_samples = int(sample_rate * fade_ms / 1000)
        if fade_samples > 0 and len(samples) > 2 * fade_samples:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            samples[:fade_samples] *= fade_in
            samples[-fade_samples:] *= fade_out

        if self._channels == 1:
            samples = samples.reshape(-1, 1)

        log.info(
            "audio_play_with_fade",
            samples=len(samples),
            fade_ms=fade_ms,
            device=self._device_index,
        )
        self._playing = True

        def _run() -> None:
            try:
                sd.play(samples, samplerate=sample_rate, device=self._device_index)
                sd.wait()
            finally:
                self._playing = False

        await asyncio.to_thread(_run)

    def stop(self) -> None:
        """Stop any currently playing audio."""
        sd.stop()
        self._playing = False
        log.info("audio_playback_stopped")

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently playing."""
        return self._playing
