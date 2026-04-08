"""Audio subsystem – virtual device management, capture, playback, STT & TTS."""

from teams_attendant.audio.devices import (
    AudioDevice,
    AudioDeviceConfig,
    AudioSetupStatus,
    ResolvedDevices,
    check_audio_setup,
    find_virtual_cable_devices,
    list_audio_devices,
)

__all__ = [
    "AudioDevice",
    "AudioDeviceConfig",
    "AudioSetupStatus",
    "ResolvedDevices",
    "check_audio_setup",
    "find_virtual_cable_devices",
    "list_audio_devices",
]
