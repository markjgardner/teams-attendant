"""Tests for the configuration system."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from teams_attendant.config import (
    AppConfig,
    list_profiles,
    load_app_config,
    load_profile,
    merge_configs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data))
    return path


# ---------------------------------------------------------------------------
# Default config (no YAML file)
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_load_default_when_no_yaml(self, tmp_path: Path) -> None:
        """load_app_config returns defaults when the YAML file does not exist."""
        cfg = load_app_config(tmp_path / "nonexistent.yaml")
        assert cfg.azure.speech.region == "eastus"
        assert cfg.azure.speech.key == ""
        assert cfg.default_profile == "balanced"

    def test_default_config_has_expected_paths(self) -> None:
        cfg = AppConfig()
        assert cfg.browser_data_dir == Path(".browser-data")
        assert cfg.summaries_dir == Path("summaries")


# ---------------------------------------------------------------------------
# Loading from YAML
# ---------------------------------------------------------------------------


class TestLoadFromYaml:
    def test_load_config_from_yaml(self, tmp_path: Path) -> None:
        data = {
            "azure": {
                "speech": {"key": "yaml-key", "region": "westus2"},
                "foundry": {
                    "endpoint": "https://example.com",
                    "api_key": "yaml-foundry-key",
                },
            },
            "default_profile": "active",
        }
        config_path = _write_yaml(tmp_path / "default.yaml", data)
        cfg = load_app_config(config_path)

        assert cfg.azure.speech.key == "yaml-key"
        assert cfg.azure.speech.region == "westus2"
        assert cfg.azure.foundry.endpoint == "https://example.com"
        assert cfg.azure.foundry.api_key == "yaml-foundry-key"
        assert cfg.default_profile == "active"

    def test_partial_yaml_fills_defaults(self, tmp_path: Path) -> None:
        data = {"azure": {"speech": {"key": "partial-key"}}}
        config_path = _write_yaml(tmp_path / "partial.yaml", data)
        cfg = load_app_config(config_path)

        assert cfg.azure.speech.key == "partial-key"
        assert cfg.azure.speech.region == "eastus"  # default
        assert cfg.azure.foundry.model_deployment == "claude-sonnet"  # default


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    def test_env_vars_override_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        data = {"azure": {"speech": {"key": "yaml-key", "region": "yaml-region"}}}
        config_path = _write_yaml(tmp_path / "default.yaml", data)

        monkeypatch.setenv("AZURE_SPEECH_KEY", "env-key")
        monkeypatch.setenv("AZURE_SPEECH_REGION", "env-region")

        cfg = load_app_config(config_path)
        assert cfg.azure.speech.key == "env-key"
        assert cfg.azure.speech.region == "env-region"

    def test_env_vars_without_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://env-endpoint.com")
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "env-foundry-key")
        monkeypatch.setenv("AZURE_FOUNDRY_MODEL", "claude-opus")

        cfg = load_app_config(tmp_path / "nonexistent.yaml")
        assert cfg.azure.foundry.endpoint == "https://env-endpoint.com"
        assert cfg.azure.foundry.api_key == "env-foundry-key"
        assert cfg.azure.foundry.model_deployment == "claude-opus"

    def test_env_var_not_set_keeps_yaml_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = {"azure": {"speech": {"key": "yaml-key"}}}
        config_path = _write_yaml(tmp_path / "default.yaml", data)

        monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
        cfg = load_app_config(config_path)
        assert cfg.azure.speech.key == "yaml-key"


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------


class TestProfileLoading:
    def test_load_profile_from_yaml(self, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "profiles"
        profile_data = {
            "name": "custom",
            "description": "A custom profile",
            "response_threshold": 0.7,
            "proactivity": 0.2,
            "response_length": "detailed",
            "prefer_voice": True,
            "cooldown_seconds": 45,
        }
        _write_yaml(profiles_dir / "custom.yaml", profile_data)

        profile = load_profile("custom", profiles_dir)
        assert profile.name == "custom"
        assert profile.description == "A custom profile"
        assert profile.response_threshold == 0.7
        assert profile.proactivity == 0.2
        assert profile.response_length == "detailed"
        assert profile.prefer_voice is True
        assert profile.cooldown_seconds == 45

    def test_load_missing_profile_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Profile 'nope' not found"):
            load_profile("nope", tmp_path)

    def test_load_profile_defaults(self, tmp_path: Path) -> None:
        """A profile with only name should get sensible defaults."""
        profiles_dir = tmp_path / "profiles"
        _write_yaml(profiles_dir / "minimal.yaml", {"name": "minimal"})

        profile = load_profile("minimal", profiles_dir)
        assert profile.response_threshold == 0.5
        assert profile.cooldown_seconds == 30.0


# ---------------------------------------------------------------------------
# Listing profiles
# ---------------------------------------------------------------------------


class TestListProfiles:
    def test_list_profiles(self, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "profiles"
        _write_yaml(profiles_dir / "alpha.yaml", {"name": "alpha", "proactivity": 0.1})
        _write_yaml(profiles_dir / "beta.yaml", {"name": "beta", "proactivity": 0.9})

        profiles = list_profiles(profiles_dir)
        assert len(profiles) == 2
        names = [p.name for p in profiles]
        assert "alpha" in names
        assert "beta" in names

    def test_list_profiles_empty_dir(self, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        assert list_profiles(profiles_dir) == []

    def test_list_profiles_nonexistent_dir(self, tmp_path: Path) -> None:
        assert list_profiles(tmp_path / "does-not-exist") == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_for_meeting_all_missing(self) -> None:
        cfg = AppConfig()
        errors = cfg.validate_for_meeting()
        # When azure-identity is installed, speech key is not required
        # (identity-based auth can be used instead).
        # Only foundry endpoint is required.
        assert any("Foundry endpoint" in e for e in errors)

    @patch.dict("sys.modules", {"azure.identity": None, "azure": MagicMock()})
    def test_validate_for_meeting_all_missing_no_identity(self) -> None:
        cfg = AppConfig()
        errors = cfg.validate_for_meeting()
        # Without azure-identity, speech key IS required
        assert len(errors) == 2
        assert any("Speech key" in e or "azure-identity" in e for e in errors)
        assert any("Foundry endpoint" in e for e in errors)

    def test_validate_for_meeting_all_present(self) -> None:
        cfg = AppConfig(
            azure={
                "speech": {"key": "k", "region": "r"},
                "foundry": {"endpoint": "https://e", "api_key": "a"},
            }
        )
        errors = cfg.validate_for_meeting()
        assert errors == []

    def test_validate_for_meeting_partial(self) -> None:
        cfg = AppConfig(azure={"speech": {"key": "k"}})
        errors = cfg.validate_for_meeting()
        assert any("Foundry endpoint" in e for e in errors)
        assert not any("Speech key" in e for e in errors)


# ---------------------------------------------------------------------------
# Config merging
# ---------------------------------------------------------------------------


class TestMergeConfigs:
    def test_merge_overrides_nested(self) -> None:
        base = AppConfig()
        merged = merge_configs(base, {"azure": {"speech": {"key": "merged-key"}}})
        assert merged.azure.speech.key == "merged-key"
        assert merged.azure.speech.region == "eastus"  # preserved from base

    def test_merge_overrides_top_level(self) -> None:
        base = AppConfig()
        merged = merge_configs(base, {"default_profile": "passive"})
        assert merged.default_profile == "passive"

    def test_merge_does_not_mutate_base(self) -> None:
        base = AppConfig()
        merge_configs(base, {"azure": {"speech": {"key": "new"}}})
        assert base.azure.speech.key == ""  # unchanged


# ---------------------------------------------------------------------------
# Transcript source configuration
# ---------------------------------------------------------------------------


def _valid_foundry_creds() -> dict:
    """Return minimal Azure config with valid foundry credentials."""
    return {
        "azure": {
            "foundry": {"endpoint": "https://e", "api_key": "a"},
        }
    }


class TestBrowserField:
    def test_browser_defaults_to_chromium(self) -> None:
        cfg = AppConfig()
        assert cfg.browser == "chromium"

    def test_browser_accepts_msedge(self) -> None:
        cfg = AppConfig(browser="msedge")
        assert cfg.browser == "msedge"

    def test_browser_rejects_invalid(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AppConfig(browser="firefox")


class TestTranscriptSource:
    def test_transcript_source_defaults_to_auto(self) -> None:
        cfg = AppConfig()
        assert cfg.transcript_source == "auto"

    @pytest.mark.parametrize("value", ["auto", "ui", "audio"])
    def test_transcript_source_accepts_valid_values(self, value: str) -> None:
        cfg = AppConfig(transcript_source=value)
        assert cfg.transcript_source == value

    def test_transcript_source_rejects_invalid(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AppConfig(transcript_source="invalid")

    def test_validate_ui_mode_skips_speech_creds(self) -> None:
        cfg = AppConfig(transcript_source="ui", **_valid_foundry_creds())
        errors = cfg.validate_for_meeting()
        assert errors == []

    @patch.dict("sys.modules", {"azure.identity": None, "azure": MagicMock()})
    def test_validate_audio_mode_requires_speech_creds(self) -> None:
        cfg = AppConfig(
            transcript_source="audio",
            azure={"speech": {"key": "", "region": "eastus"}, **_valid_foundry_creds()["azure"]},
        )
        errors = cfg.validate_for_meeting()
        assert any("Speech key" in e or "azure-identity" in e for e in errors)

    @patch.dict("sys.modules", {"azure.identity": None, "azure": MagicMock()})
    def test_validate_auto_mode_requires_speech_creds(self) -> None:
        cfg = AppConfig(
            transcript_source="auto",
            azure={"speech": {"key": "", "region": "eastus"}, **_valid_foundry_creds()["azure"]},
        )
        errors = cfg.validate_for_meeting()
        assert any("Speech key" in e or "azure-identity" in e for e in errors)

    def test_validate_audio_mode_identity_auth_ok(self) -> None:
        """No speech key error when azure-identity is available."""
        cfg = AppConfig(
            transcript_source="audio",
            azure={"speech": {"key": "", "region": "eastus"}, **_valid_foundry_creds()["azure"]},
        )
        errors = cfg.validate_for_meeting()
        assert not any("Speech key" in e for e in errors)

    def test_load_config_with_transcript_source(self, tmp_path: Path) -> None:
        data = {"transcript_source": "ui"}
        config_path = _write_yaml(tmp_path / "cfg.yaml", data)
        cfg = load_app_config(config_path)
        assert cfg.transcript_source == "ui"


# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------


class TestLLMProviderConfig:
    def test_llm_provider_defaults_to_anthropic(self) -> None:
        cfg = AppConfig()
        assert cfg.llm_provider == "anthropic"

    def test_llm_provider_accepts_openai(self) -> None:
        cfg = AppConfig(llm_provider="openai")
        assert cfg.llm_provider == "openai"

    def test_llm_provider_rejects_invalid(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AppConfig(llm_provider="gemini")

    def test_validate_anthropic_requires_foundry_endpoint(self) -> None:
        cfg = AppConfig(llm_provider="anthropic")
        errors = cfg.validate_for_meeting()
        assert any("Foundry endpoint" in e for e in errors)

    def test_validate_openai_requires_foundry_endpoint(self) -> None:
        cfg = AppConfig(llm_provider="openai")
        errors = cfg.validate_for_meeting()
        assert any("Foundry endpoint" in e for e in errors)

    def test_validate_no_api_key_is_ok(self) -> None:
        """API key is optional — identity-based auth is the alternative."""
        cfg = AppConfig(
            transcript_source="ui",
            azure={"foundry": {"endpoint": "https://e.com"}},
        )
        errors = cfg.validate_for_meeting()
        assert len(errors) == 0
