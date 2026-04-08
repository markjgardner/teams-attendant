"""Tests for teams_attendant.browser.auth module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from teams_attendant.browser.auth import (
    TEAMS_URL,
    _CHROME_USER_AGENT,
    _CHROMIUM_ARGS,
    _STEALTH_JS,
    clear_session,
    get_authenticated_context,
    login,
)


# ---------------------------------------------------------------------------
# clear_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_session_removes_directory(tmp_path: Path) -> None:
    """clear_session should delete the browser data directory."""
    data_dir = tmp_path / "browser-data"
    data_dir.mkdir()
    (data_dir / "cookies.json").write_text("{}")

    await clear_session(data_dir)

    assert not data_dir.exists()


@pytest.mark.asyncio
async def test_clear_session_noop_when_missing(tmp_path: Path) -> None:
    """clear_session should not raise when the directory doesn't exist."""
    data_dir = tmp_path / "nonexistent"
    await clear_session(data_dir)  # should not raise
    assert not data_dir.exists()


# ---------------------------------------------------------------------------
# get_authenticated_context
# ---------------------------------------------------------------------------


def _make_mock_playwright() -> MagicMock:
    """Build a mock Playwright instance with a mock persistent context."""
    mock_context = AsyncMock()
    mock_context.add_init_script = AsyncMock()

    pw = MagicMock()
    pw.chromium.launch_persistent_context = AsyncMock(return_value=mock_context)
    return pw


@pytest.mark.asyncio
async def test_get_authenticated_context_returns_context(tmp_path: Path) -> None:
    """get_authenticated_context should return a BrowserContext."""
    pw = _make_mock_playwright()
    ctx = await get_authenticated_context(pw, tmp_path / "data", headless=True)

    pw.chromium.launch_persistent_context.assert_awaited_once()
    call_kwargs = pw.chromium.launch_persistent_context.call_args.kwargs
    assert call_kwargs["headless"] is True
    assert call_kwargs["user_agent"] == _CHROME_USER_AGENT
    assert call_kwargs["viewport"] == {"width": 1920, "height": 1080}
    assert call_kwargs["locale"] == "en-US"
    assert call_kwargs["args"] == _CHROMIUM_ARGS

    ctx.add_init_script.assert_awaited_once_with(_STEALTH_JS)


@pytest.mark.asyncio
async def test_get_authenticated_context_creates_dir(tmp_path: Path) -> None:
    """get_authenticated_context should create the data dir if it doesn't exist."""
    pw = _make_mock_playwright()
    data_dir = tmp_path / "new-dir"
    assert not data_dir.exists()

    await get_authenticated_context(pw, data_dir)
    assert data_dir.exists()


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_navigates_to_teams(tmp_path: Path) -> None:
    """login should open a browser and navigate to Teams URL."""
    mock_page = AsyncMock()
    mock_context = AsyncMock()
    mock_context.pages = [mock_page]
    mock_context.add_init_script = AsyncMock()
    mock_context.close = AsyncMock()

    mock_pw = MagicMock()
    mock_pw.chromium.launch_persistent_context = AsyncMock(return_value=mock_context)

    async_pw_cm = AsyncMock()
    async_pw_cm.__aenter__ = AsyncMock(return_value=mock_pw)
    async_pw_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("teams_attendant.browser.auth.async_playwright", return_value=async_pw_cm):
        await login(tmp_path / "data")

    mock_page.goto.assert_awaited_once()
    url_arg = mock_page.goto.call_args.args[0]
    assert url_arg == TEAMS_URL

    mock_page.wait_for_selector.assert_awaited_once()
    mock_context.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_login_launches_non_headless(tmp_path: Path) -> None:
    """login should launch browser in non-headless mode."""
    mock_context = AsyncMock()
    mock_context.pages = []
    mock_context.new_page = AsyncMock(return_value=AsyncMock())
    mock_context.add_init_script = AsyncMock()
    mock_context.close = AsyncMock()

    mock_pw = MagicMock()
    mock_pw.chromium.launch_persistent_context = AsyncMock(return_value=mock_context)

    async_pw_cm = AsyncMock()
    async_pw_cm.__aenter__ = AsyncMock(return_value=mock_pw)
    async_pw_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("teams_attendant.browser.auth.async_playwright", return_value=async_pw_cm):
        await login(tmp_path / "data")

    call_kwargs = mock_pw.chromium.launch_persistent_context.call_args.kwargs
    assert call_kwargs["headless"] is False


# ---------------------------------------------------------------------------
# audio_env parameter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_authenticated_context_without_audio_env(tmp_path: Path) -> None:
    """Without audio_env, no env kwarg should be passed."""
    pw = _make_mock_playwright()
    await get_authenticated_context(pw, tmp_path / "data", headless=True)

    call_kwargs = pw.chromium.launch_persistent_context.call_args.kwargs
    assert "env" not in call_kwargs


@pytest.mark.asyncio
async def test_get_authenticated_context_with_audio_env(tmp_path: Path) -> None:
    """audio_env should be merged with os.environ and passed to launch."""
    pw = _make_mock_playwright()
    audio = {"PULSE_SINK": "my_sink", "PULSE_SOURCE": "my_source"}

    await get_authenticated_context(
        pw, tmp_path / "data", headless=True, audio_env=audio
    )

    call_kwargs = pw.chromium.launch_persistent_context.call_args.kwargs
    assert "env" in call_kwargs
    env = call_kwargs["env"]
    # Should contain the audio vars
    assert env["PULSE_SINK"] == "my_sink"
    assert env["PULSE_SOURCE"] == "my_source"
    # Should also contain existing os.environ entries
    assert "PATH" in env or len(env) > 2
