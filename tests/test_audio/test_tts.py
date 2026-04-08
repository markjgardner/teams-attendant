"""Tests for teams_attendant.audio.tts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import azure.cognitiveservices.speech as speechsdk

from teams_attendant.config import AzureSpeechConfig
from teams_attendant.errors import TTSError


# 16 kHz 16-bit mono → 32 000 bytes/sec.  0.5 s = 16 000 bytes
_FAKE_AUDIO = b"\x00\x01" * 8000  # 16 000 bytes → 500 ms

_SDK_PATH = "azure.cognitiveservices.speech"


# ---------------------------------------------------------------------------
# SpeechSynthesizer init
# ---------------------------------------------------------------------------


class TestSpeechSynthesizerInit:
    @patch(f"{_SDK_PATH}.SpeechConfig")
    def test_creates_speech_config(self, mock_cfg_cls: MagicMock):
        from teams_attendant.audio.tts import SpeechSynthesizer

        cfg = AzureSpeechConfig(key="test-key", region="westus")
        SpeechSynthesizer(cfg)

        mock_cfg_cls.assert_called_once_with(subscription="test-key", region="westus")

    @patch(f"{_SDK_PATH}.SpeechConfig")
    def test_sets_output_format(self, mock_cfg_cls: MagicMock):
        from teams_attendant.audio.tts import SpeechSynthesizer

        cfg = AzureSpeechConfig(key="k", region="r")
        SpeechSynthesizer(cfg)

        speech_config = mock_cfg_cls.return_value
        speech_config.set_speech_synthesis_output_format.assert_called_once_with(
            speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
        )

    @patch(f"{_SDK_PATH}.SpeechConfig")
    def test_default_voice(self, mock_cfg_cls: MagicMock):
        from teams_attendant.audio.tts import SpeechSynthesizer

        cfg = AzureSpeechConfig(key="k", region="r")
        SpeechSynthesizer(cfg)

        speech_config = mock_cfg_cls.return_value
        assert speech_config.speech_synthesis_voice_name == "en-US-JennyNeural"

    @patch(f"{_SDK_PATH}.SpeechConfig")
    def test_custom_voice(self, mock_cfg_cls: MagicMock):
        from teams_attendant.audio.tts import SpeechSynthesizer

        cfg = AzureSpeechConfig(key="k", region="r")
        SpeechSynthesizer(cfg, voice_name="en-US-GuyNeural")

        speech_config = mock_cfg_cls.return_value
        assert speech_config.speech_synthesis_voice_name == "en-US-GuyNeural"


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    @pytest.mark.asyncio
    @patch(f"{_SDK_PATH}.SpeechSynthesizer")
    @patch(f"{_SDK_PATH}.SpeechConfig")
    async def test_returns_tts_result(
        self, mock_cfg_cls: MagicMock, mock_synth_cls: MagicMock
    ):
        from teams_attendant.audio.tts import SpeechSynthesizer, TTSResult

        result_mock = MagicMock()
        result_mock.reason = speechsdk.ResultReason.SynthesizingAudioCompleted
        result_mock.audio_data = _FAKE_AUDIO

        mock_synth_cls.return_value.speak_text.return_value = result_mock

        cfg = AzureSpeechConfig(key="k", region="r")
        synth = SpeechSynthesizer(cfg)
        out = await synth.synthesize("hello world")

        assert isinstance(out, TTSResult)
        assert out.audio_data == _FAKE_AUDIO
        assert out.duration_ms == 500
        assert out.text == "hello world"

    @pytest.mark.asyncio
    @patch(f"{_SDK_PATH}.SpeechSynthesizer")
    @patch(f"{_SDK_PATH}.SpeechConfig")
    async def test_auth_error(
        self, mock_cfg_cls: MagicMock, mock_synth_cls: MagicMock
    ):
        from teams_attendant.audio.tts import SpeechSynthesizer

        result_mock = MagicMock()
        result_mock.reason = speechsdk.ResultReason.Canceled
        result_mock.cancellation_details.error_code = (
            speechsdk.CancellationErrorCode.AuthenticationFailure
        )
        result_mock.cancellation_details.error_details = "Invalid key"

        mock_synth_cls.return_value.speak_text.return_value = result_mock

        cfg = AzureSpeechConfig(key="bad", region="r")
        synth = SpeechSynthesizer(cfg)
        with pytest.raises(TTSError, match="authentication failed"):
            await synth.synthesize("hi")

    @pytest.mark.asyncio
    @patch(f"{_SDK_PATH}.SpeechSynthesizer")
    @patch(f"{_SDK_PATH}.SpeechConfig")
    async def test_connection_error(
        self, mock_cfg_cls: MagicMock, mock_synth_cls: MagicMock
    ):
        from teams_attendant.audio.tts import SpeechSynthesizer

        result_mock = MagicMock()
        result_mock.reason = speechsdk.ResultReason.Canceled
        result_mock.cancellation_details.error_code = (
            speechsdk.CancellationErrorCode.ConnectionFailure
        )
        result_mock.cancellation_details.error_details = "Network down"

        mock_synth_cls.return_value.speak_text.return_value = result_mock

        cfg = AzureSpeechConfig(key="k", region="r")
        synth = SpeechSynthesizer(cfg)
        with pytest.raises(TTSError, match="connection failed"):
            await synth.synthesize("hi")


# ---------------------------------------------------------------------------
# synthesize_ssml
# ---------------------------------------------------------------------------


class TestSynthesizeSsml:
    @pytest.mark.asyncio
    @patch(f"{_SDK_PATH}.SpeechSynthesizer")
    @patch(f"{_SDK_PATH}.SpeechConfig")
    async def test_returns_tts_result(
        self, mock_cfg_cls: MagicMock, mock_synth_cls: MagicMock
    ):
        from teams_attendant.audio.tts import SpeechSynthesizer

        result_mock = MagicMock()
        result_mock.reason = speechsdk.ResultReason.SynthesizingAudioCompleted
        result_mock.audio_data = _FAKE_AUDIO

        mock_synth_cls.return_value.speak_ssml.return_value = result_mock

        cfg = AzureSpeechConfig(key="k", region="r")
        synth = SpeechSynthesizer(cfg)
        ssml = "<speak>hello</speak>"
        out = await synth.synthesize_ssml(ssml)

        assert out.audio_data == _FAKE_AUDIO
        assert out.text == ssml


# ---------------------------------------------------------------------------
# set_voice
# ---------------------------------------------------------------------------


class TestSetVoice:
    @patch(f"{_SDK_PATH}.SpeechConfig")
    def test_changes_voice(self, mock_cfg_cls: MagicMock):
        from teams_attendant.audio.tts import SpeechSynthesizer

        cfg = AzureSpeechConfig(key="k", region="r")
        synth = SpeechSynthesizer(cfg)
        synth.set_voice("en-GB-SoniaNeural")

        speech_config = mock_cfg_cls.return_value
        assert speech_config.speech_synthesis_voice_name == "en-GB-SoniaNeural"


# ---------------------------------------------------------------------------
# list_voices
# ---------------------------------------------------------------------------


class TestListVoices:
    @pytest.mark.asyncio
    @patch(f"{_SDK_PATH}.SpeechSynthesizer")
    @patch(f"{_SDK_PATH}.SpeechConfig")
    async def test_returns_voice_list(
        self, mock_cfg_cls: MagicMock, mock_synth_cls: MagicMock
    ):
        from teams_attendant.audio.tts import SpeechSynthesizer

        voice = MagicMock()
        voice.short_name = "en-US-JennyNeural"
        voice.locale = "en-US"
        voice.gender = "Female"
        voice.local_name = "Jenny"

        voices_result = MagicMock()
        voices_result.reason = speechsdk.ResultReason.VoicesListRetrieved
        voices_result.voices = [voice]

        get_future = MagicMock()
        get_future.get.return_value = voices_result
        mock_synth_cls.return_value.get_voices_async.return_value = get_future

        cfg = AzureSpeechConfig(key="k", region="r")
        synth = SpeechSynthesizer(cfg)
        result = await synth.list_voices()

        assert len(result) == 1
        assert result[0]["name"] == "en-US-JennyNeural"
        assert result[0]["locale"] == "en-US"


# ---------------------------------------------------------------------------
# VoiceResponder
# ---------------------------------------------------------------------------


class TestVoiceResponder:
    @pytest.mark.asyncio
    @patch(f"{_SDK_PATH}.SpeechSynthesizer")
    @patch(f"{_SDK_PATH}.SpeechConfig")
    async def test_speak_synthesizes_and_plays(
        self, mock_cfg_cls: MagicMock, mock_synth_cls: MagicMock
    ):
        from teams_attendant.audio.tts import SpeechSynthesizer, VoiceResponder

        result_mock = MagicMock()
        result_mock.reason = speechsdk.ResultReason.SynthesizingAudioCompleted
        result_mock.audio_data = _FAKE_AUDIO

        mock_synth_cls.return_value.speak_text.return_value = result_mock

        cfg = AzureSpeechConfig(key="k", region="r")
        synth = SpeechSynthesizer(cfg)

        player = MagicMock()

        async def mock_play(data: bytes, sample_rate: int = 16000) -> None:
            pass

        player.play_with_fade = mock_play
        player.is_playing = False

        responder = VoiceResponder(synth, player)
        await responder.speak("hello")
        assert responder.is_speaking is False

    @patch(f"{_SDK_PATH}.SpeechConfig")
    def test_is_speaking_delegates_to_player(self, mock_cfg_cls: MagicMock):
        from teams_attendant.audio.tts import SpeechSynthesizer, VoiceResponder

        cfg = AzureSpeechConfig(key="k", region="r")
        synth = SpeechSynthesizer(cfg)

        player = MagicMock()
        player.is_playing = True

        responder = VoiceResponder(synth, player)
        assert responder.is_speaking is True

