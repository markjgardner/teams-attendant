"""Configuration models for Teams Attendant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


_PROJECT_ROOT = _find_project_root()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AzureSpeechConfig(BaseModel):
    """Azure Speech Services configuration."""

    key: str = ""
    region: str = "eastus"
    voice: str = Field(
        default="en-US-JennyNeural",
        description="Azure TTS voice name (e.g. 'en-US-GuyNeural', 'en-GB-SoniaNeural')",
    )


class AzureFoundryConfig(BaseModel):
    """Azure Foundry configuration for LLM models (Claude or GPT).

    When ``api_key`` is empty, identity-based authentication is used
    via ``DefaultAzureCredential`` from the ``azure-identity`` package.
    """

    endpoint: str = ""
    api_key: str = ""
    model_deployment: str = "claude-sonnet"


class AzureConfig(BaseModel):
    """Azure services configuration."""

    speech: AzureSpeechConfig = Field(default_factory=AzureSpeechConfig)
    foundry: AzureFoundryConfig = Field(default_factory=AzureFoundryConfig)


class BehaviorProfile(BaseModel):
    """Behavior profile controlling agent participation."""

    name: str
    description: str = ""
    response_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="How confident the agent must be to respond"
    )
    proactivity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="How proactively the agent contributes"
    )
    response_length: Literal["minimal", "concise", "moderate", "detailed"] = "concise"
    prefer_voice: bool = Field(
        default=False, description="Whether to prefer voice over chat responses"
    )
    cooldown_seconds: float = Field(
        default=30.0, description="Minimum seconds between unprompted responses"
    )


class MeetingConfig(BaseModel):
    """Per-meeting configuration."""

    meeting_url: str
    profile: str = "balanced"
    vision_enabled: bool = False
    vision_interval_seconds: float = 15.0


class AppConfig(BaseModel):
    """Top-level application configuration."""

    azure: AzureConfig = Field(default_factory=AzureConfig)
    browser_data_dir: Path = Path(".browser-data")
    summaries_dir: Path = Path("summaries")
    default_profile: str = "balanced"
    transcript_source: Literal["auto", "ui", "audio"] = Field(
        default="auto",
        description=(
            "Where to get meeting transcripts: "
            "'auto' tries UI captions first then falls back to audio STT, "
            "'ui' uses only Teams live captions, "
            "'audio' uses only Azure Speech-to-Text"
        ),
    )
    browser: Literal["chromium", "msedge"] = Field(
        default="chromium",
        description=(
            "Browser to use: "
            "'chromium' uses bundled Chromium, "
            "'msedge' uses system-installed Microsoft Edge"
        ),
    )
    llm_provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description=(
            "LLM API format to use with Azure Foundry: "
            "'anthropic' for Claude models (Messages API), "
            "'openai' for GPT models (Chat Completions API)"
        ),
    )

    def validate_for_meeting(self) -> list[str]:
        """Check that all credentials required for joining a meeting are present.

        Returns a list of human-readable error strings. An empty list means the
        configuration is valid.
        """
        errors: list[str] = []
        if self.transcript_source in ("audio", "auto"):
            if not self.azure.speech.key:
                errors.append(
                    "Azure Speech key is required for audio transcription "
                    "(set azure.speech.key or AZURE_SPEECH_KEY, "
                    "or use transcript_source='ui' to skip)"
                )
            if not self.azure.speech.region:
                errors.append(
                    "Azure Speech region is required for audio transcription "
                    "(set azure.speech.region or AZURE_SPEECH_REGION, "
                    "or use transcript_source='ui' to skip)"
                )
        if not self.azure.foundry.endpoint:
            errors.append(
                "Azure Foundry endpoint is required "
                "(set azure.foundry.endpoint or AZURE_FOUNDRY_ENDPOINT)"
            )
        return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_MAP: dict[tuple[str, ...], str] = {
    ("azure", "speech", "key"): "AZURE_SPEECH_KEY",
    ("azure", "speech", "region"): "AZURE_SPEECH_REGION",
    ("azure", "speech", "voice"): "AZURE_SPEECH_VOICE",
    ("azure", "foundry", "endpoint"): "AZURE_FOUNDRY_ENDPOINT",
    ("azure", "foundry", "api_key"): "AZURE_FOUNDRY_API_KEY",
    ("azure", "foundry", "model_deployment"): "AZURE_FOUNDRY_MODEL",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_nested(data: dict[str, Any], keys: tuple[str, ...], value: str) -> None:
    """Set a value in a nested dict using a tuple of keys."""
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Layer environment variable overrides onto the config dict."""
    for keys, env_var in _ENV_MAP.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            _set_nested(data, keys, env_value)
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as fh:
        result = yaml.safe_load(fh)
    return result if isinstance(result, dict) else {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_config_dir() -> Path:
    """Return the configuration directory.

    Checks (in order):
    1. ``<project_root>/config/``
    2. ``~/.config/teams-attendant/``
    """
    project_config = _PROJECT_ROOT / "config"
    if project_config.is_dir():
        return project_config
    user_config = Path.home() / ".config" / "teams-attendant"
    return user_config


def load_app_config(config_path: Path | None = None) -> AppConfig:
    """Load application config from a YAML file, merged with env-var overrides.

    If *config_path* is ``None`` the loader looks for ``default.yaml`` inside
    :func:`get_config_dir`.  If the file does not exist, defaults are used.
    """
    data: dict[str, Any] = {}

    if config_path is not None:
        if config_path.is_file():
            data = _load_yaml(config_path)
    else:
        default_path = get_config_dir() / "default.yaml"
        if default_path.is_file():
            data = _load_yaml(default_path)

    data = _apply_env_overrides(data)
    return AppConfig(**data)


def load_profile(name: str, profiles_dir: Path | None = None) -> BehaviorProfile:
    """Load a single behaviour profile by *name* from a YAML file."""
    if profiles_dir is None:
        profiles_dir = get_config_dir() / "profiles"
    profile_path = profiles_dir / f"{name}.yaml"
    if not profile_path.is_file():
        raise FileNotFoundError(f"Profile '{name}' not found at {profile_path}")
    data = _load_yaml(profile_path)
    return BehaviorProfile(**data)


def list_profiles(profiles_dir: Path | None = None) -> list[BehaviorProfile]:
    """Return all behaviour profiles found in *profiles_dir*."""
    if profiles_dir is None:
        profiles_dir = get_config_dir() / "profiles"
    if not profiles_dir.is_dir():
        return []
    profiles: list[BehaviorProfile] = []
    for path in sorted(profiles_dir.glob("*.yaml")):
        data = _load_yaml(path)
        profiles.append(BehaviorProfile(**data))
    return profiles


def merge_configs(base: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    """Return a new :class:`AppConfig` by merging *overrides* into *base*."""
    base_data = base.model_dump()
    merged = _deep_merge(base_data, overrides)
    return AppConfig(**merged)
