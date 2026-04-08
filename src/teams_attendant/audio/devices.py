"""Virtual audio device management.

Cross-platform virtual audio device discovery, configuration, and lifecycle
management using sounddevice (PortAudio wrapper).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

import structlog
from pydantic import BaseModel

from teams_attendant.errors import AudioDeviceNotFoundError

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Browser audio environment
# ---------------------------------------------------------------------------

def get_browser_audio_env(devices: ResolvedDevices) -> dict[str, str]:
    """Get environment variables for routing browser audio through virtual devices.

    On Linux, sets ``PULSE_SINK`` and ``PULSE_SOURCE`` so the browser's
    PulseAudio streams are routed through the virtual cable devices.
    """
    env: dict[str, str] = {}
    if sys.platform == "linux":
        if devices.capture_device_name:
            # Browser audio output → our capture sink
            env["PULSE_SINK"] = devices.capture_device_name
        if devices.playback_device_name:
            # Our playback → browser microphone input
            env["PULSE_SOURCE"] = devices.playback_device_name
    return env


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class DeviceType(str, Enum):
    """Audio device direction."""

    INPUT = "input"
    OUTPUT = "output"


@dataclass
class AudioDevice:
    """Represents a single audio device reported by the host."""

    id: int
    name: str
    type: DeviceType
    is_virtual: bool = False
    host_api: str = ""


@dataclass
class VirtualCableDevices:
    """A matched pair of virtual-cable capture and playback devices."""

    capture_device: AudioDevice
    playback_device: AudioDevice


@dataclass
class ResolvedDevices:
    """Concrete device indices ready to be passed to an audio stream."""

    capture_device_index: int
    playback_device_index: int
    capture_device_name: str
    playback_device_name: str
    sample_rate: int
    channels: int


class AudioDeviceConfig(BaseModel):
    """User-facing configuration for audio device selection."""

    capture_device_name: str | None = None
    playback_device_name: str | None = None
    sample_rate: int = 16000
    channels: int = 1
    auto_detect: bool = True


class AudioSetupStatus(BaseModel):
    """Result of an audio-readiness check."""

    is_ready: bool
    platform: str
    capture_device: str | None = None
    playback_device: str | None = None
    issues: list[str] = []
    suggestions: list[str] = []


# ---------------------------------------------------------------------------
# PulseAudio virtual-device names
# ---------------------------------------------------------------------------

_PA_SINK = "teams_attendant_sink"
_PA_SOURCE = "teams_attendant_source"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_virtual_device(name: str) -> bool:
    """Heuristic: does this device name look like a virtual audio device?"""
    markers = [
        "cable",
        "virtual",
        "vb-audio",
        "pulse",
        "pipewire",
        _PA_SINK,
        _PA_SOURCE,
    ]
    lower = name.lower()
    return any(m in lower for m in markers)


def _query_devices_safe() -> list[dict]:
    """Call ``sounddevice.query_devices()`` and return a list of dicts.

    Returns an empty list when no audio subsystem is available.
    """
    try:
        import sounddevice as sd  # noqa: PLC0415

        raw = sd.query_devices()
    except Exception:
        logger.warning("sounddevice.query_devices() failed – no audio subsystem?")
        return []

    if isinstance(raw, dict):
        return [raw]
    return list(raw)  # type: ignore[arg-type]


def _device_dicts_to_objects(raw_devices: list[dict]) -> list[AudioDevice]:
    """Convert raw sounddevice dicts into ``AudioDevice`` objects."""
    devices: list[AudioDevice] = []
    for idx, dev in enumerate(raw_devices):
        name: str = dev.get("name", "")
        host_api: str = str(dev.get("hostapi", ""))
        max_in: int = dev.get("max_input_channels", 0)
        max_out: int = dev.get("max_output_channels", 0)
        is_virtual = _is_virtual_device(name)
        if max_in > 0:
            devices.append(
                AudioDevice(
                    id=idx,
                    name=name,
                    type=DeviceType.INPUT,
                    is_virtual=is_virtual,
                    host_api=host_api,
                )
            )
        if max_out > 0:
            devices.append(
                AudioDevice(
                    id=idx,
                    name=name,
                    type=DeviceType.OUTPUT,
                    is_virtual=is_virtual,
                    host_api=host_api,
                )
            )
    return devices


# ---------------------------------------------------------------------------
# Public API – device discovery
# ---------------------------------------------------------------------------

def list_audio_devices() -> list[AudioDevice]:
    """List all audio devices visible to the host."""
    return _device_dicts_to_objects(_query_devices_safe())


def find_virtual_cable_devices() -> VirtualCableDevices | None:
    """Auto-detect a virtual-cable capture/playback pair.

    * **Windows** – look for VB-Audio Virtual Cable (``"CABLE"`` in name).
    * **Linux** – look for PulseAudio / PipeWire virtual devices, including
      the ones created by :func:`setup_pulseaudio_virtual_devices`.
    """
    devices = list_audio_devices()

    if sys.platform == "win32":
        capture = next(
            (d for d in devices if "cable" in d.name.lower() and d.type == DeviceType.INPUT),
            None,
        )
        playback = next(
            (d for d in devices if "cable" in d.name.lower() and d.type == DeviceType.OUTPUT),
            None,
        )
    else:  # linux / other
        capture = next(
            (
                d
                for d in devices
                if d.is_virtual and d.type == DeviceType.INPUT
            ),
            None,
        )
        playback = next(
            (
                d
                for d in devices
                if d.is_virtual and d.type == DeviceType.OUTPUT
            ),
            None,
        )

    if capture and playback:
        return VirtualCableDevices(capture_device=capture, playback_device=playback)
    return None


# ---------------------------------------------------------------------------
# Public API – device resolution
# ---------------------------------------------------------------------------

def _find_device_by_name(
    name: str,
    device_type: DeviceType,
    devices: list[AudioDevice],
) -> AudioDevice:
    """Look up a device by (sub-)name and type, raising on failure."""
    lower = name.lower()
    match = next(
        (d for d in devices if lower in d.name.lower() and d.type == device_type),
        None,
    )
    if match is None:
        available = [d.name for d in devices if d.type == device_type]
        raise AudioDeviceNotFoundError(
            f"Audio device '{name}' ({device_type.value}) not found. "
            f"Available: {available}"
        )
    return match


def resolve_devices(config: AudioDeviceConfig) -> ResolvedDevices:
    """Resolve an :class:`AudioDeviceConfig` to concrete device indices."""
    devices = list_audio_devices()

    capture: AudioDevice | None = None
    playback: AudioDevice | None = None

    # --- auto-detect ---
    if config.auto_detect:
        pair = find_virtual_cable_devices()
        if pair:
            capture = pair.capture_device
            playback = pair.playback_device

    # --- explicit names override auto-detect ---
    if config.capture_device_name:
        capture = _find_device_by_name(config.capture_device_name, DeviceType.INPUT, devices)

    if config.playback_device_name:
        playback = _find_device_by_name(config.playback_device_name, DeviceType.OUTPUT, devices)

    if capture is None:
        raise AudioDeviceNotFoundError(
            "No capture (input) device resolved. "
            "Set capture_device_name or install a virtual audio cable."
        )
    if playback is None:
        raise AudioDeviceNotFoundError(
            "No playback (output) device resolved. "
            "Set playback_device_name or install a virtual audio cable."
        )

    return ResolvedDevices(
        capture_device_index=capture.id,
        playback_device_index=playback.id,
        capture_device_name=capture.name,
        playback_device_name=playback.name,
        sample_rate=config.sample_rate,
        channels=config.channels,
    )


# ---------------------------------------------------------------------------
# Public API – platform setup check
# ---------------------------------------------------------------------------

def check_audio_setup() -> AudioSetupStatus:
    """Check whether the audio environment is ready for the agent."""
    platform = sys.platform
    issues: list[str] = []
    suggestions: list[str] = []
    capture_name: str | None = None
    playback_name: str | None = None

    devices = list_audio_devices()
    if not devices:
        issues.append("No audio devices detected (sounddevice/PortAudio unavailable).")
        suggestions.append("Install PortAudio and ensure an audio subsystem is running.")
        return AudioSetupStatus(
            is_ready=False,
            platform=platform,
            issues=issues,
            suggestions=suggestions,
        )

    pair = find_virtual_cable_devices()
    if pair:
        capture_name = pair.capture_device.name
        playback_name = pair.playback_device.name
    else:
        if platform == "win32":
            issues.append("VB-Audio Virtual Cable not detected.")
            suggestions.append(
                "Install VB-Cable from https://vb-audio.com/Cable/ and restart."
            )
        elif platform == "linux":
            # Check for PulseAudio
            try:
                subprocess.run(
                    ["pactl", "info"],
                    capture_output=True,
                    check=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                issues.append("PulseAudio/PipeWire not running or pactl not found.")
                suggestions.append("Install and start PulseAudio or PipeWire.")

            issues.append("No virtual audio sink/source found.")
            suggestions.append(
                "Run `teams-attendant audio-setup` or call "
                "setup_pulseaudio_virtual_devices() to create them."
            )
        else:
            issues.append("No virtual audio cable detected.")
            suggestions.append("Install a virtual audio cable for your platform.")

    return AudioSetupStatus(
        is_ready=len(issues) == 0,
        platform=platform,
        capture_device=capture_name,
        playback_device=playback_name,
        issues=issues,
        suggestions=suggestions,
    )


# ---------------------------------------------------------------------------
# PulseAudio helpers (Linux only)
# ---------------------------------------------------------------------------

def setup_pulseaudio_virtual_devices() -> None:
    """Create PulseAudio virtual sink and source for the agent.

    Idempotent – silently succeeds if the modules already exist.
    """
    if sys.platform != "linux":
        logger.info("setup_pulseaudio_virtual_devices is a no-op on non-Linux platforms")
        return

    logger.info("Creating PulseAudio virtual devices")

    # Virtual sink (where Teams audio will be routed)
    subprocess.run(
        [
            "pactl",
            "load-module",
            "module-null-sink",
            f"sink_name={_PA_SINK}",
            f"sink_properties=device.description={_PA_SINK}",
        ],
        check=True,
        capture_output=True,
    )

    # Virtual source (loopback from the sink's monitor)
    subprocess.run(
        [
            "pactl",
            "load-module",
            "module-virtual-source",
            f"source_name={_PA_SOURCE}",
            f"master={_PA_SINK}.monitor",
            f"source_properties=device.description={_PA_SOURCE}",
        ],
        check=True,
        capture_output=True,
    )

    logger.info("PulseAudio virtual devices created", sink=_PA_SINK, source=_PA_SOURCE)


def teardown_pulseaudio_virtual_devices() -> None:
    """Remove PulseAudio virtual devices created by :func:`setup_pulseaudio_virtual_devices`."""
    if sys.platform != "linux":
        logger.info("teardown_pulseaudio_virtual_devices is a no-op on non-Linux platforms")
        return

    logger.info("Removing PulseAudio virtual devices")

    for module in ("module-virtual-source", "module-null-sink"):
        try:
            subprocess.run(
                ["pactl", "unload-module", module],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            logger.warning("Failed to unload PulseAudio module", module=module)

    logger.info("PulseAudio virtual devices removed")


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def managed_audio_devices(
    config: AudioDeviceConfig,
) -> AsyncGenerator[ResolvedDevices, None]:
    """Set up and tear down audio devices for a meeting session."""
    created_pa = False

    if sys.platform == "linux" and config.auto_detect:
        pair = find_virtual_cable_devices()
        if pair is None:
            setup_pulseaudio_virtual_devices()
            created_pa = True

    resolved = resolve_devices(config)
    logger.info(
        "Audio devices acquired",
        capture=resolved.capture_device_name,
        playback=resolved.playback_device_name,
    )

    try:
        yield resolved
    finally:
        if created_pa:
            teardown_pulseaudio_virtual_devices()
        logger.info("Audio devices released")
