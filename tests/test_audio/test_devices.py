"""Tests for teams_attendant.audio.devices."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from teams_attendant.audio.devices import (
    AudioDevice,
    AudioDeviceConfig,
    AudioSetupStatus,
    DeviceType,
    ResolvedDevices,
    VirtualCableDevices,
    check_audio_setup,
    find_virtual_cable_devices,
    get_browser_audio_env,
    list_audio_devices,
    resolve_devices,
    setup_pulseaudio_virtual_devices,
    teardown_pulseaudio_virtual_devices,
)
from teams_attendant.errors import AudioDeviceNotFoundError

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_DEVICES = [
    {
        "name": "Built-in Microphone",
        "hostapi": 0,
        "max_input_channels": 2,
        "max_output_channels": 0,
    },
    {
        "name": "Built-in Speaker",
        "hostapi": 0,
        "max_input_channels": 0,
        "max_output_channels": 2,
    },
    {
        "name": "CABLE Output (VB-Audio Virtual Cable)",
        "hostapi": 0,
        "max_input_channels": 2,
        "max_output_channels": 0,
    },
    {
        "name": "CABLE Input (VB-Audio Virtual Cable)",
        "hostapi": 0,
        "max_input_channels": 0,
        "max_output_channels": 2,
    },
]

FAKE_LINUX_DEVICES = [
    {
        "name": "PulseAudio default input",
        "hostapi": 0,
        "max_input_channels": 2,
        "max_output_channels": 0,
    },
    {
        "name": "PulseAudio default output",
        "hostapi": 0,
        "max_input_channels": 0,
        "max_output_channels": 2,
    },
]


def _patch_sd(devices: list[dict]):
    """Return a patch that makes sounddevice.query_devices() return *devices*."""
    return patch("teams_attendant.audio.devices.sd.query_devices", return_value=devices)


def _patch_sd_import(devices: list[dict]):
    """Patch the dynamic ``import sounddevice`` inside _query_devices_safe."""
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = devices
    return patch.dict("sys.modules", {"sounddevice": mock_sd})


# ---------------------------------------------------------------------------
# list_audio_devices
# ---------------------------------------------------------------------------


class TestListAudioDevices:
    def test_returns_audio_devices(self):
        with _patch_sd_import(FAKE_DEVICES):
            devices = list_audio_devices()

        assert len(devices) == 4
        assert all(isinstance(d, AudioDevice) for d in devices)

    def test_splits_input_output(self):
        with _patch_sd_import(FAKE_DEVICES):
            devices = list_audio_devices()

        inputs = [d for d in devices if d.type == DeviceType.INPUT]
        outputs = [d for d in devices if d.type == DeviceType.OUTPUT]
        assert len(inputs) == 2
        assert len(outputs) == 2

    def test_marks_virtual_devices(self):
        with _patch_sd_import(FAKE_DEVICES):
            devices = list_audio_devices()

        virtual = [d for d in devices if d.is_virtual]
        assert len(virtual) == 2
        assert all("CABLE" in d.name for d in virtual)

    def test_returns_empty_on_failure(self):
        broken = MagicMock()
        broken.query_devices.side_effect = OSError("no audio")
        with patch.dict("sys.modules", {"sounddevice": broken}):
            devices = list_audio_devices()
        assert devices == []


# ---------------------------------------------------------------------------
# find_virtual_cable_devices
# ---------------------------------------------------------------------------


class TestFindVirtualCableDevices:
    def test_detects_vb_cable_on_windows(self):
        with _patch_sd_import(FAKE_DEVICES), patch("teams_attendant.audio.devices.sys") as mock_sys:
            mock_sys.platform = "win32"
            result = find_virtual_cable_devices()

        assert result is not None
        assert isinstance(result, VirtualCableDevices)
        assert "CABLE" in result.capture_device.name
        assert "CABLE" in result.playback_device.name

    def test_detects_pulse_on_linux(self):
        with _patch_sd_import(FAKE_LINUX_DEVICES), patch(
            "teams_attendant.audio.devices.sys"
        ) as mock_sys:
            mock_sys.platform = "linux"
            result = find_virtual_cable_devices()

        assert result is not None
        assert result.capture_device.type == DeviceType.INPUT
        assert result.playback_device.type == DeviceType.OUTPUT

    def test_returns_none_when_no_virtual_devices(self):
        no_virtual = [
            {
                "name": "Microphone",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
            {
                "name": "Speaker",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 1,
            },
        ]
        with _patch_sd_import(no_virtual), patch(
            "teams_attendant.audio.devices.sys"
        ) as mock_sys:
            mock_sys.platform = "win32"
            result = find_virtual_cable_devices()

        assert result is None


# ---------------------------------------------------------------------------
# AudioDeviceConfig validation
# ---------------------------------------------------------------------------


class TestAudioDeviceConfig:
    def test_defaults(self):
        cfg = AudioDeviceConfig()
        assert cfg.sample_rate == 16000
        assert cfg.channels == 1
        assert cfg.auto_detect is True
        assert cfg.capture_device_name is None
        assert cfg.playback_device_name is None

    def test_custom_values(self):
        cfg = AudioDeviceConfig(
            capture_device_name="Mic",
            playback_device_name="Spk",
            sample_rate=48000,
            channels=2,
            auto_detect=False,
        )
        assert cfg.capture_device_name == "Mic"
        assert cfg.playback_device_name == "Spk"
        assert cfg.sample_rate == 48000
        assert cfg.channels == 2
        assert cfg.auto_detect is False


# ---------------------------------------------------------------------------
# resolve_devices
# ---------------------------------------------------------------------------


class TestResolveDevices:
    def test_auto_detect(self):
        with _patch_sd_import(FAKE_DEVICES), patch(
            "teams_attendant.audio.devices.sys"
        ) as mock_sys:
            mock_sys.platform = "win32"
            resolved = resolve_devices(AudioDeviceConfig())

        assert isinstance(resolved, ResolvedDevices)
        assert "CABLE" in resolved.capture_device_name
        assert "CABLE" in resolved.playback_device_name
        assert resolved.sample_rate == 16000
        assert resolved.channels == 1

    def test_explicit_device_names(self):
        with _patch_sd_import(FAKE_DEVICES), patch(
            "teams_attendant.audio.devices.sys"
        ) as mock_sys:
            mock_sys.platform = "win32"
            cfg = AudioDeviceConfig(
                capture_device_name="Built-in Microphone",
                playback_device_name="Built-in Speaker",
                auto_detect=False,
            )
            resolved = resolve_devices(cfg)

        assert resolved.capture_device_name == "Built-in Microphone"
        assert resolved.playback_device_name == "Built-in Speaker"

    def test_raises_when_device_not_found(self):
        with _patch_sd_import(FAKE_DEVICES), patch(
            "teams_attendant.audio.devices.sys"
        ) as mock_sys:
            mock_sys.platform = "win32"
            cfg = AudioDeviceConfig(
                capture_device_name="NonexistentDevice",
                auto_detect=False,
            )
            with pytest.raises(AudioDeviceNotFoundError, match="not found"):
                resolve_devices(cfg)

    def test_raises_when_no_capture_resolved(self):
        no_virtual = [
            {
                "name": "Speaker",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
        ]
        with _patch_sd_import(no_virtual), patch(
            "teams_attendant.audio.devices.sys"
        ) as mock_sys:
            mock_sys.platform = "win32"
            with pytest.raises(AudioDeviceNotFoundError, match="No capture"):
                resolve_devices(AudioDeviceConfig())


# ---------------------------------------------------------------------------
# check_audio_setup
# ---------------------------------------------------------------------------


class TestCheckAudioSetup:
    def test_ready_with_virtual_cable(self):
        with _patch_sd_import(FAKE_DEVICES), patch(
            "teams_attendant.audio.devices.sys"
        ) as mock_sys:
            mock_sys.platform = "win32"
            status = check_audio_setup()

        assert isinstance(status, AudioSetupStatus)
        assert status.is_ready is True
        assert status.platform == "win32"
        assert status.capture_device is not None
        assert status.playback_device is not None
        assert status.issues == []

    def test_not_ready_windows_no_cable(self):
        no_virtual = [
            {
                "name": "Mic",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
            {
                "name": "Spk",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 1,
            },
        ]
        with _patch_sd_import(no_virtual), patch(
            "teams_attendant.audio.devices.sys"
        ) as mock_sys:
            mock_sys.platform = "win32"
            status = check_audio_setup()

        assert status.is_ready is False
        assert len(status.issues) > 0
        assert any("VB-Audio" in i for i in status.issues)

    def test_not_ready_no_devices(self):
        broken = MagicMock()
        broken.query_devices.side_effect = OSError("no audio")
        with patch.dict("sys.modules", {"sounddevice": broken}), patch(
            "teams_attendant.audio.devices.sys"
        ) as mock_sys:
            mock_sys.platform = "linux"
            status = check_audio_setup()

        assert status.is_ready is False
        assert any("No audio devices" in i for i in status.issues)

    def test_linux_no_virtual_sink(self):
        no_virtual = [
            {
                "name": "HDA Intel",
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
            },
            {
                "name": "HDA Intel Output",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
        ]
        with _patch_sd_import(no_virtual), patch(
            "teams_attendant.audio.devices.sys"
        ) as mock_sys:
            mock_sys.platform = "linux"
            with patch("teams_attendant.audio.devices.subprocess") as mock_sp:
                mock_sp.run.return_value = MagicMock(returncode=0)
                mock_sp.CalledProcessError = subprocess.CalledProcessError
                status = check_audio_setup()

        assert status.is_ready is False
        assert any("virtual" in i.lower() for i in status.issues)


# ---------------------------------------------------------------------------
# PulseAudio helpers
# ---------------------------------------------------------------------------


class TestPulseAudioHelpers:
    def test_setup_only_runs_on_linux(self):
        with patch("teams_attendant.audio.devices.sys") as mock_sys, patch(
            "teams_attendant.audio.devices.subprocess"
        ) as mock_sp:
            mock_sys.platform = "win32"
            setup_pulseaudio_virtual_devices()
            mock_sp.run.assert_not_called()

    def test_teardown_only_runs_on_linux(self):
        with patch("teams_attendant.audio.devices.sys") as mock_sys, patch(
            "teams_attendant.audio.devices.subprocess"
        ) as mock_sp:
            mock_sys.platform = "win32"
            teardown_pulseaudio_virtual_devices()
            mock_sp.run.assert_not_called()

    def test_setup_calls_pactl(self):
        with patch("teams_attendant.audio.devices.sys") as mock_sys, patch(
            "teams_attendant.audio.devices.subprocess"
        ) as mock_sp:
            mock_sys.platform = "linux"
            mock_sp.run.return_value = MagicMock(returncode=0)
            setup_pulseaudio_virtual_devices()
            assert mock_sp.run.call_count == 2
            calls = mock_sp.run.call_args_list
            assert "module-null-sink" in calls[0].args[0]
            assert "module-virtual-source" in calls[1].args[0]

    def test_teardown_calls_pactl(self):
        with patch("teams_attendant.audio.devices.sys") as mock_sys, patch(
            "teams_attendant.audio.devices.subprocess"
        ) as mock_sp:
            mock_sys.platform = "linux"
            mock_sp.run.return_value = MagicMock(returncode=0)
            mock_sp.CalledProcessError = subprocess.CalledProcessError
            teardown_pulseaudio_virtual_devices()
            assert mock_sp.run.call_count == 2


# ---------------------------------------------------------------------------
# get_browser_audio_env
# ---------------------------------------------------------------------------


class TestGetBrowserAudioEnv:
    def test_linux_returns_pulse_vars(self):
        devices = ResolvedDevices(
            capture_device_index=0,
            playback_device_index=1,
            capture_device_name="my_sink",
            playback_device_name="my_source",
            sample_rate=16000,
            channels=1,
        )
        with patch("teams_attendant.audio.devices.sys") as mock_sys:
            mock_sys.platform = "linux"
            env = get_browser_audio_env(devices)

        assert env == {"PULSE_SINK": "my_sink", "PULSE_SOURCE": "my_source"}

    def test_non_linux_returns_empty(self):
        devices = ResolvedDevices(
            capture_device_index=0,
            playback_device_index=1,
            capture_device_name="my_sink",
            playback_device_name="my_source",
            sample_rate=16000,
            channels=1,
        )
        with patch("teams_attendant.audio.devices.sys") as mock_sys:
            mock_sys.platform = "win32"
            env = get_browser_audio_env(devices)

        assert env == {}

    def test_partial_device_names(self):
        devices = ResolvedDevices(
            capture_device_index=0,
            playback_device_index=1,
            capture_device_name="sink_only",
            playback_device_name="",
            sample_rate=16000,
            channels=1,
        )
        with patch("teams_attendant.audio.devices.sys") as mock_sys:
            mock_sys.platform = "linux"
            env = get_browser_audio_env(devices)

        assert env == {"PULSE_SINK": "sink_only"}
