"""Tests for teams_attendant.audio.capture module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from teams_attendant.audio.capture import AudioCaptureStream
from teams_attendant.audio.devices import ResolvedDevices


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_resolved_devices(**overrides) -> ResolvedDevices:
    defaults = dict(
        capture_device_index=3,
        playback_device_index=4,
        capture_device_name="virtual_sink",
        playback_device_name="virtual_source",
        sample_rate=16000,
        channels=1,
    )
    defaults.update(overrides)
    return ResolvedDevices(**defaults)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestAudioCaptureStreamInit:
    def test_defaults(self):
        devices = _make_resolved_devices()
        stream = AudioCaptureStream(devices)

        assert stream._device_index == 3
        assert stream._sample_rate == 16000
        assert stream._channels == 1
        assert stream._chunk_size == 1600
        assert stream._stream is None
        assert stream._callbacks == []
        assert stream.is_capturing is False

    def test_custom_params(self):
        devices = _make_resolved_devices(capture_device_index=7)
        stream = AudioCaptureStream(
            devices, sample_rate=48000, channels=2, chunk_size=4800
        )

        assert stream._device_index == 7
        assert stream._sample_rate == 48000
        assert stream._channels == 2
        assert stream._chunk_size == 4800


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestAudioCaptureStreamStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_input_stream(self):
        devices = _make_resolved_devices()
        stream = AudioCaptureStream(devices)

        mock_input_stream = MagicMock()
        with patch("teams_attendant.audio.capture.sd.InputStream", return_value=mock_input_stream):
            await stream.start()

        assert stream.is_capturing is True
        mock_input_stream.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_passes_correct_args(self):
        devices = _make_resolved_devices()
        stream = AudioCaptureStream(devices, sample_rate=48000, channels=2, chunk_size=4800)

        mock_input_stream = MagicMock()
        with patch(
            "teams_attendant.audio.capture.sd.InputStream", return_value=mock_input_stream
        ) as mock_cls:
            await stream.start()

        mock_cls.assert_called_once_with(
            device=3,
            samplerate=48000,
            channels=2,
            blocksize=4800,
            dtype="int16",
            callback=stream._audio_callback,
        )

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        devices = _make_resolved_devices()
        stream = AudioCaptureStream(devices)

        mock_input_stream = MagicMock()
        with patch("teams_attendant.audio.capture.sd.InputStream", return_value=mock_input_stream):
            await stream.start()
            await stream.start()  # second call should be a no-op

        mock_input_stream.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_closes_stream(self):
        devices = _make_resolved_devices()
        stream = AudioCaptureStream(devices)

        mock_input_stream = MagicMock()
        with patch("teams_attendant.audio.capture.sd.InputStream", return_value=mock_input_stream):
            await stream.start()
            await stream.stop()

        assert stream.is_capturing is False
        mock_input_stream.stop.assert_called_once()
        mock_input_stream.close.assert_called_once()
        assert stream._stream is None

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        devices = _make_resolved_devices()
        stream = AudioCaptureStream(devices)

        # Stop without starting should be a no-op
        await stream.stop()
        assert stream.is_capturing is False


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------


class TestCallbackRegistration:
    def test_register_single_callback(self):
        devices = _make_resolved_devices()
        stream = AudioCaptureStream(devices)

        cb = MagicMock()
        stream.register_callback(cb)

        assert len(stream._callbacks) == 1
        assert stream._callbacks[0] is cb

    def test_register_multiple_callbacks(self):
        devices = _make_resolved_devices()
        stream = AudioCaptureStream(devices)

        cb1 = MagicMock()
        cb2 = MagicMock()
        cb3 = MagicMock()
        stream.register_callback(cb1)
        stream.register_callback(cb2)
        stream.register_callback(cb3)

        assert len(stream._callbacks) == 3


# ---------------------------------------------------------------------------
# _audio_callback
# ---------------------------------------------------------------------------


class TestAudioCallback:
    def test_converts_numpy_to_bytes_and_dispatches(self):
        devices = _make_resolved_devices()
        stream = AudioCaptureStream(devices)

        cb = MagicMock()
        stream.register_callback(cb)

        # Simulate audio data: 10 samples of int16
        indata = np.array([[100], [-200], [300], [0], [32767], [-32768], [1], [2], [3], [4]],
                          dtype=np.int16)
        status = MagicMock()
        status.__bool__ = MagicMock(return_value=False)

        stream._audio_callback(indata, frames=10, time_info=None, status=status)

        cb.assert_called_once()
        received = cb.call_args[0][0]
        assert isinstance(received, bytes)
        assert received == indata.tobytes()

    def test_dispatches_to_multiple_callbacks(self):
        devices = _make_resolved_devices()
        stream = AudioCaptureStream(devices)

        cb1 = MagicMock()
        cb2 = MagicMock()
        stream.register_callback(cb1)
        stream.register_callback(cb2)

        indata = np.zeros((10, 1), dtype=np.int16)
        status = MagicMock()
        status.__bool__ = MagicMock(return_value=False)

        stream._audio_callback(indata, frames=10, time_info=None, status=status)

        cb1.assert_called_once()
        cb2.assert_called_once()
        assert cb1.call_args[0][0] == cb2.call_args[0][0]

    def test_callback_error_does_not_break_others(self):
        devices = _make_resolved_devices()
        stream = AudioCaptureStream(devices)

        failing_cb = MagicMock(side_effect=ValueError("boom"))
        good_cb = MagicMock()
        stream.register_callback(failing_cb)
        stream.register_callback(good_cb)

        indata = np.zeros((10, 1), dtype=np.int16)
        status = MagicMock()
        status.__bool__ = MagicMock(return_value=False)

        stream._audio_callback(indata, frames=10, time_info=None, status=status)

        failing_cb.assert_called_once()
        good_cb.assert_called_once()

    def test_pcm_format_int16(self):
        """Verify that the callback delivers int16 PCM bytes."""
        devices = _make_resolved_devices()
        stream = AudioCaptureStream(devices)

        received: list[bytes] = []
        stream.register_callback(lambda data: received.append(data))

        samples = np.array([[1000], [-1000]], dtype=np.int16)
        status = MagicMock()
        status.__bool__ = MagicMock(return_value=False)

        stream._audio_callback(samples, frames=2, time_info=None, status=status)

        assert len(received) == 1
        reconstructed = np.frombuffer(received[0], dtype=np.int16)
        np.testing.assert_array_equal(reconstructed, [1000, -1000])


# ---------------------------------------------------------------------------
# is_capturing property
# ---------------------------------------------------------------------------


class TestIsCapturing:
    def test_false_initially(self):
        stream = AudioCaptureStream(_make_resolved_devices())
        assert stream.is_capturing is False

    @pytest.mark.asyncio
    async def test_true_after_start(self):
        stream = AudioCaptureStream(_make_resolved_devices())
        with patch("teams_attendant.audio.capture.sd.InputStream", return_value=MagicMock()):
            await stream.start()
        assert stream.is_capturing is True

    @pytest.mark.asyncio
    async def test_false_after_stop(self):
        stream = AudioCaptureStream(_make_resolved_devices())
        with patch("teams_attendant.audio.capture.sd.InputStream", return_value=MagicMock()):
            await stream.start()
            await stream.stop()
        assert stream.is_capturing is False
