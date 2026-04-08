"""Meeting audio capture from virtual devices."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable

import numpy as np
import sounddevice as sd
import structlog

if TYPE_CHECKING:
    from teams_attendant.audio.devices import ResolvedDevices

log = structlog.get_logger()


class AudioCaptureStream:
    """Captures audio from a virtual audio device (meeting audio output)."""

    def __init__(
        self,
        devices: ResolvedDevices,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1600,  # 100ms at 16kHz
    ) -> None:
        self._device_index = devices.capture_device_index
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._stream: sd.InputStream | None = None
        self._callbacks: list[Callable[[bytes], None]] = []
        self._lock = threading.Lock()
        self._capturing = False

    async def start(self) -> None:
        """Start capturing audio."""
        if self._capturing:
            log.warning("audio_capture.already_started")
            return

        log.info(
            "audio_capture.start",
            device=self._device_index,
            sample_rate=self._sample_rate,
            channels=self._channels,
            chunk_size=self._chunk_size,
        )
        self._stream = sd.InputStream(
            device=self._device_index,
            samplerate=self._sample_rate,
            channels=self._channels,
            blocksize=self._chunk_size,
            dtype="int16",
            callback=self._audio_callback,
        )
        self._stream.start()
        self._capturing = True
        log.info("audio_capture.started")

    async def stop(self) -> None:
        """Stop capturing audio."""
        if not self._capturing:
            log.warning("audio_capture.already_stopped")
            return

        log.info("audio_capture.stopping")
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._capturing = False
        log.info("audio_capture.stopped")

    def register_callback(self, callback: Callable[[bytes], None]) -> None:
        """Register a callback that receives raw PCM audio chunks."""
        with self._lock:
            self._callbacks.append(callback)
        log.debug("audio_capture.callback_registered", total=len(self._callbacks))

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        """sounddevice input stream callback (runs in a background thread)."""
        if status:
            log.warning("audio_capture.stream_status", status=str(status))

        pcm_bytes = indata.tobytes()

        with self._lock:
            callbacks = list(self._callbacks)

        for cb in callbacks:
            try:
                cb(pcm_bytes)
            except Exception:
                log.exception("audio_capture.callback_error")

    @property
    def is_capturing(self) -> bool:
        """Whether the stream is currently capturing audio."""
        return self._capturing
