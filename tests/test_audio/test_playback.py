"""Tests for teams_attendant.audio.playback."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from teams_attendant.audio.devices import ResolvedDevices

# Ensure sounddevice is mockable even when PortAudio is absent.
if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = MagicMock()
    sys.modules["_sounddevice"] = MagicMock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_devices(*, playback_index: int = 5) -> ResolvedDevices:
    return ResolvedDevices(
        capture_device_index=1,
        playback_device_index=playback_index,
        capture_device_name="Virtual Capture",
        playback_device_name="Virtual Playback",
        sample_rate=16000,
        channels=1,
    )


# 0.5 s of silence at 16 kHz 16-bit mono → 16 000 bytes
_SILENCE = b"\x00\x00" * 8000


# ---------------------------------------------------------------------------
# AudioPlayer init
# ---------------------------------------------------------------------------


class TestAudioPlayerInit:
    def test_stores_device_index(self):
        from teams_attendant.audio.playback import AudioPlayer

        player = AudioPlayer(_make_devices(playback_index=7))
        assert player._device_index == 7

    def test_default_sample_rate_and_channels(self):
        from teams_attendant.audio.playback import AudioPlayer

        player = AudioPlayer(_make_devices())
        assert player._sample_rate == 16000
        assert player._channels == 1

    def test_custom_sample_rate(self):
        from teams_attendant.audio.playback import AudioPlayer

        player = AudioPlayer(_make_devices(), sample_rate=48000, channels=2)
        assert player._sample_rate == 48000
        assert player._channels == 2


# ---------------------------------------------------------------------------
# play
# ---------------------------------------------------------------------------


class TestPlay:
    @pytest.mark.asyncio
    async def test_converts_bytes_to_float32(self):
        from teams_attendant.audio.playback import AudioPlayer

        player = AudioPlayer(_make_devices(playback_index=3))

        with patch("teams_attendant.audio.playback.sd") as mock_sd:
            mock_sd.play = MagicMock()
            mock_sd.wait = MagicMock()
            await player.play(_SILENCE)

        call_args = mock_sd.play.call_args
        samples = call_args[0][0]
        assert samples.dtype == np.float32

    @pytest.mark.asyncio
    async def test_passes_correct_device_and_rate(self):
        from teams_attendant.audio.playback import AudioPlayer

        player = AudioPlayer(_make_devices(playback_index=9))

        with patch("teams_attendant.audio.playback.sd") as mock_sd:
            mock_sd.play = MagicMock()
            mock_sd.wait = MagicMock()
            await player.play(_SILENCE, sample_rate=44100)

        call_args = mock_sd.play.call_args
        assert call_args.kwargs["samplerate"] == 44100
        assert call_args.kwargs["device"] == 9

    @pytest.mark.asyncio
    async def test_reshapes_mono(self):
        from teams_attendant.audio.playback import AudioPlayer

        player = AudioPlayer(_make_devices(), channels=1)

        with patch("teams_attendant.audio.playback.sd") as mock_sd:
            mock_sd.play = MagicMock()
            mock_sd.wait = MagicMock()
            await player.play(_SILENCE)

        samples = mock_sd.play.call_args[0][0]
        assert samples.ndim == 2
        assert samples.shape[1] == 1

    @pytest.mark.asyncio
    async def test_is_playing_false_after_play(self):
        from teams_attendant.audio.playback import AudioPlayer

        player = AudioPlayer(_make_devices())

        with patch("teams_attendant.audio.playback.sd") as mock_sd:
            mock_sd.play = MagicMock()
            mock_sd.wait = MagicMock()
            await player.play(_SILENCE)

        assert player.is_playing is False


# ---------------------------------------------------------------------------
# play_with_fade
# ---------------------------------------------------------------------------


class TestPlayWithFade:
    @pytest.mark.asyncio
    async def test_applies_fade(self):
        from teams_attendant.audio.playback import AudioPlayer

        # 1 second of mid-amplitude signal
        raw = (np.ones(16000, dtype=np.int16) * 16384).tobytes()
        player = AudioPlayer(_make_devices())

        with patch("teams_attendant.audio.playback.sd") as mock_sd:
            mock_sd.play = MagicMock()
            mock_sd.wait = MagicMock()
            await player.play_with_fade(raw, fade_ms=100)

        samples = mock_sd.play.call_args[0][0].flatten()
        # First sample should be near zero (faded in from 0)
        assert abs(samples[0]) < 0.01
        # Last sample should be near zero (faded out to 0)
        assert abs(samples[-1]) < 0.01
        # Middle should be unaffected
        mid = len(samples) // 2
        assert abs(samples[mid] - 16384 / 32768.0) < 0.01

    @pytest.mark.asyncio
    async def test_no_fade_when_audio_too_short(self):
        from teams_attendant.audio.playback import AudioPlayer

        # Very short audio: only 10 samples
        raw = (np.ones(10, dtype=np.int16) * 16384).tobytes()
        player = AudioPlayer(_make_devices())

        with patch("teams_attendant.audio.playback.sd") as mock_sd:
            mock_sd.play = MagicMock()
            mock_sd.wait = MagicMock()
            # fade_ms=100 → 1600 fade samples at 16 kHz, but only 10 samples total
            await player.play_with_fade(raw, fade_ms=100)

        samples = mock_sd.play.call_args[0][0].flatten()
        # No fade applied; all samples should be the original value
        expected = 16384 / 32768.0
        np.testing.assert_allclose(samples, expected, atol=0.001)


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_calls_sd_stop(self):
        from teams_attendant.audio.playback import AudioPlayer

        player = AudioPlayer(_make_devices())

        with patch("teams_attendant.audio.playback.sd") as mock_sd:
            mock_sd.stop = MagicMock()
            player.stop()
            mock_sd.stop.assert_called_once()

    def test_clears_playing_flag(self):
        from teams_attendant.audio.playback import AudioPlayer

        player = AudioPlayer(_make_devices())
        player._playing = True

        with patch("teams_attendant.audio.playback.sd") as mock_sd:
            mock_sd.stop = MagicMock()
            player.stop()

        assert player.is_playing is False


# ---------------------------------------------------------------------------
# is_playing
# ---------------------------------------------------------------------------


class TestIsPlaying:
    def test_initially_false(self):
        from teams_attendant.audio.playback import AudioPlayer

        player = AudioPlayer(_make_devices())
        assert player.is_playing is False
