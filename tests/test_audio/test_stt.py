"""Tests for Azure Speech-to-Text integration."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Build a fake azure.cognitiveservices.speech module for testing
# ---------------------------------------------------------------------------

def _build_mock_speechsdk() -> MagicMock:
    """Create a mock that mimics the azure.cognitiveservices.speech module."""
    sdk = MagicMock()

    # Enum-like attributes
    sdk.ResultReason.RecognizedSpeech = "RecognizedSpeech"
    sdk.ResultReason.NoMatch = "NoMatch"
    sdk.CancellationReason.Error = "Error"
    sdk.CancellationReason.EndOfStream = "EndOfStream"
    sdk.CancellationErrorCode.AuthenticationFailure = "AuthenticationFailure"
    sdk.CancellationErrorCode.ConnectionFailure = "ConnectionFailure"
    sdk.PropertyId.SpeechServiceConnection_LanguageIdMode = "LanguageIdMode"

    # SpeechConfig
    speech_config_instance = MagicMock()
    sdk.SpeechConfig.return_value = speech_config_instance

    # AudioConfig
    audio_config_instance = MagicMock()
    sdk.audio.AudioConfig.return_value = audio_config_instance

    # SpeechRecognizer with event hooks
    recognizer = MagicMock()
    recognizer.recognized = MagicMock()
    recognizer.recognizing = MagicMock()
    recognizer.canceled = MagicMock()
    recognizer.recognized.connect = MagicMock()
    recognizer.recognizing.connect = MagicMock()
    recognizer.canceled.connect = MagicMock()
    recognizer.start_continuous_recognition_async = MagicMock()
    recognizer.stop_continuous_recognition_async = MagicMock()
    sdk.SpeechRecognizer.return_value = recognizer

    return sdk


@pytest.fixture()
def mock_speechsdk():
    """Patch azure.cognitiveservices.speech into sys.modules."""
    sdk = _build_mock_speechsdk()
    with patch.dict(sys.modules, {"azure.cognitiveservices.speech": sdk, "azure": MagicMock(), "azure.cognitiveservices": MagicMock()}):
        # Force re-import so the module picks up our mock
        import importlib

        import teams_attendant.audio.stt as stt_mod

        stt_mod.speechsdk = sdk
        importlib.reload(stt_mod)
        stt_mod.speechsdk = sdk
        yield sdk, stt_mod


@pytest.fixture()
def speech_config():
    """Minimal AzureSpeechConfig stub."""
    cfg = SimpleNamespace(key="test-key", region="eastus")
    return cfg


@pytest.fixture()
def devices():
    """Minimal ResolvedDevices stub."""
    return SimpleNamespace(
        capture_device_index=1,
        playback_device_index=2,
        capture_device_name="TestMic",
        playback_device_name="TestSpk",
        sample_rate=16000,
        channels=1,
    )


@pytest.fixture()
def event_bus():
    """A real EventBus instance."""
    from teams_attendant.utils.events import EventBus

    return EventBus()


class TestSpeechTranscriberInit:
    """Test SpeechTranscriber initialization."""

    def test_creates_recognizer(self, mock_speechsdk, speech_config, devices, event_bus) -> None:
        sdk, stt_mod = mock_speechsdk
        transcriber = stt_mod.SpeechTranscriber(speech_config, devices, event_bus)

        sdk.SpeechConfig.assert_called_once_with(subscription="test-key", region="eastus")
        sdk.SpeechRecognizer.assert_called_once()
        assert transcriber._recognizer is not None

    def test_connects_event_handlers(self, mock_speechsdk, speech_config, devices, event_bus) -> None:
        sdk, stt_mod = mock_speechsdk
        _transcriber = stt_mod.SpeechTranscriber(speech_config, devices, event_bus)  # noqa: F841

        recognizer = sdk.SpeechRecognizer.return_value
        recognizer.recognized.connect.assert_called_once()
        recognizer.recognizing.connect.assert_called_once()
        recognizer.canceled.connect.assert_called_once()

    def test_audio_config_with_device_index(
        self, mock_speechsdk, speech_config, devices, event_bus
    ) -> None:
        sdk, stt_mod = mock_speechsdk
        stt_mod.SpeechTranscriber(speech_config, devices, event_bus)

        sdk.audio.AudioConfig.assert_called_once_with(device_name="1")

    def test_audio_config_default_mic(
        self, mock_speechsdk, speech_config, event_bus
    ) -> None:
        sdk, stt_mod = mock_speechsdk
        dev = SimpleNamespace(
            capture_device_index=None,
            playback_device_index=2,
            capture_device_name="",
            playback_device_name="",
            sample_rate=16000,
            channels=1,
        )
        stt_mod.SpeechTranscriber(speech_config, dev, event_bus)

        sdk.audio.AudioConfig.assert_called_once_with(use_default_microphone=True)


class TestSpeechTranscriberStartStop:
    """Test start/stop methods."""

    async def test_start_calls_sdk(self, mock_speechsdk, speech_config, devices, event_bus) -> None:
        sdk, stt_mod = mock_speechsdk
        transcriber = stt_mod.SpeechTranscriber(speech_config, devices, event_bus)

        await transcriber.start()

        recognizer = sdk.SpeechRecognizer.return_value
        recognizer.start_continuous_recognition_async.assert_called_once()

    async def test_stop_calls_sdk(self, mock_speechsdk, speech_config, devices, event_bus) -> None:
        sdk, stt_mod = mock_speechsdk
        transcriber = stt_mod.SpeechTranscriber(speech_config, devices, event_bus)

        await transcriber.start()
        await transcriber.stop()

        recognizer = sdk.SpeechRecognizer.return_value
        recognizer.stop_continuous_recognition_async.assert_called_once()


class TestSpeechTranscriberCallbacks:
    """Test that recognition callbacks create correct TranscriptEvents."""

    async def test_on_recognized_publishes_event(
        self, mock_speechsdk, speech_config, devices, event_bus
    ) -> None:
        sdk, stt_mod = mock_speechsdk

        received: list = []

        async def handler(event):
            received.append(event)

        event_bus.subscribe("transcript", handler)
        await event_bus.start()

        transcriber = stt_mod.SpeechTranscriber(speech_config, devices, event_bus)
        await transcriber.start()

        # Simulate a recognized event
        evt = MagicMock()
        evt.result.reason = sdk.ResultReason.RecognizedSpeech
        evt.result.text = "Hello world"
        evt.result.best.return_value = []

        transcriber._on_recognized(evt)
        await asyncio.sleep(0.3)

        await event_bus.stop()
        await transcriber.stop()

        assert len(received) == 1
        assert received[0].data["text"] == "Hello world"
        assert received[0].data["is_final"] is True

    async def test_on_recognizing_publishes_interim(
        self, mock_speechsdk, speech_config, devices, event_bus
    ) -> None:
        sdk, stt_mod = mock_speechsdk
        received: list = []

        async def handler(event):
            received.append(event)

        event_bus.subscribe("transcript", handler)
        await event_bus.start()

        transcriber = stt_mod.SpeechTranscriber(speech_config, devices, event_bus)
        await transcriber.start()

        evt = MagicMock()
        evt.result.text = "Hel"

        transcriber._on_recognizing(evt)
        await asyncio.sleep(0.3)

        await event_bus.stop()
        await transcriber.stop()

        assert len(received) == 1
        assert received[0].data["text"] == "Hel"
        assert received[0].data["is_final"] is False

    async def test_on_recognized_ignores_empty_text(
        self, mock_speechsdk, speech_config, devices, event_bus
    ) -> None:
        sdk, stt_mod = mock_speechsdk
        received: list = []

        async def handler(event):
            received.append(event)

        event_bus.subscribe("transcript", handler)
        await event_bus.start()

        transcriber = stt_mod.SpeechTranscriber(speech_config, devices, event_bus)
        await transcriber.start()

        evt = MagicMock()
        evt.result.reason = sdk.ResultReason.RecognizedSpeech
        evt.result.text = ""

        transcriber._on_recognized(evt)
        await asyncio.sleep(0.3)

        await event_bus.stop()
        assert len(received) == 0


class TestSpeechTranscriberErrorHandling:
    """Test error handling on canceled recognition."""

    def test_on_canceled_auth_failure(
        self, mock_speechsdk, speech_config, devices, event_bus
    ) -> None:
        sdk, stt_mod = mock_speechsdk
        transcriber = stt_mod.SpeechTranscriber(speech_config, devices, event_bus)

        evt = MagicMock()
        evt.result.reason = sdk.CancellationReason.Error
        evt.result.error_code = sdk.CancellationErrorCode.AuthenticationFailure
        evt.result.error_details = "Bad key"

        # Should not raise
        transcriber._on_canceled(evt)

    def test_on_canceled_connection_failure(
        self, mock_speechsdk, speech_config, devices, event_bus
    ) -> None:
        sdk, stt_mod = mock_speechsdk
        transcriber = stt_mod.SpeechTranscriber(speech_config, devices, event_bus)

        evt = MagicMock()
        evt.result.reason = sdk.CancellationReason.Error
        evt.result.error_code = sdk.CancellationErrorCode.ConnectionFailure
        evt.result.error_details = "Network unreachable"

        transcriber._on_canceled(evt)

    def test_on_canceled_end_of_stream(
        self, mock_speechsdk, speech_config, devices, event_bus
    ) -> None:
        sdk, stt_mod = mock_speechsdk
        transcriber = stt_mod.SpeechTranscriber(speech_config, devices, event_bus)

        evt = MagicMock()
        evt.result.reason = sdk.CancellationReason.EndOfStream

        transcriber._on_canceled(evt)
