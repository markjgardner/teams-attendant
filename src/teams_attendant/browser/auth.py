"""Teams authentication and session management."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import structlog
from playwright.async_api import BrowserContext, Playwright, async_playwright

from teams_attendant.errors import AuthenticationError

log = structlog.get_logger()

TEAMS_URL = "https://teams.microsoft.com"
LOGIN_TIMEOUT_MS = 5 * 60 * 1000  # 5 minutes

# Selectors that indicate a successful Teams login
_LOGGED_IN_SELECTORS = [
    "[data-tid='app-layout']",
    "[data-tid='app-bar']",
    "#app",
]

_CHROME_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
"""

_CHROMIUM_ARGS = [
    "--disable-blink-features=AutomationControlled",
]


def _ensure_dir(path: Path) -> Path:
    """Ensure the directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _create_persistent_context(
    playwright: Playwright,
    browser_data_dir: Path,
    *,
    headless: bool = True,
    audio_env: dict[str, str] | None = None,
) -> BrowserContext:
    """Create a persistent browser context with stealth settings."""
    _ensure_dir(browser_data_dir)

    launch_kwargs: dict = dict(
        user_data_dir=str(browser_data_dir),
        headless=headless,
        user_agent=_CHROME_USER_AGENT,
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
        permissions=[],
        args=_CHROMIUM_ARGS,
    )

    if audio_env:
        merged_env = {**os.environ, **audio_env}
        launch_kwargs["env"] = merged_env
        log.info("browser.context.audio_env", extra_keys=list(audio_env.keys()))

    context = await playwright.chromium.launch_persistent_context(**launch_kwargs)
    await context.add_init_script(_STEALTH_JS)
    return context


async def login(browser_data_dir: Path = Path(".browser-data")) -> None:
    """Launch a visible browser for the user to log in to Teams.

    The browser stays open until login is detected or the timeout is reached.
    Session data is persisted automatically via the persistent context.
    """
    log.info("browser.login.start", browser_data_dir=str(browser_data_dir))

    async with async_playwright() as pw:
        context = await _create_persistent_context(
            pw, browser_data_dir, headless=False
        )
        page = context.pages[0] if context.pages else await context.new_page()

        await page.goto(TEAMS_URL, wait_until="domcontentloaded")
        log.info("browser.login.waiting", url=TEAMS_URL)

        # Wait for any of the logged-in selectors to appear
        selector = ", ".join(_LOGGED_IN_SELECTORS)
        try:
            await page.wait_for_selector(selector, timeout=LOGIN_TIMEOUT_MS)
            log.info("browser.login.success")
        except Exception as exc:
            log.warning("browser.login.timeout")
            raise AuthenticationError(
                "Login timed out – no logged-in indicator detected"
            ) from exc

        await context.close()


async def is_session_valid(browser_data_dir: Path = Path(".browser-data")) -> bool:
    """Check whether the persisted session is still authenticated.

    Launches a headless browser, navigates to Teams, and checks for
    logged-in indicators.
    """
    if not browser_data_dir.exists():
        log.info("browser.session.no_data", browser_data_dir=str(browser_data_dir))
        return False

    log.info("browser.session.checking", browser_data_dir=str(browser_data_dir))
    try:
        async with async_playwright() as pw:
            context = await _create_persistent_context(
                pw, browser_data_dir, headless=True
            )
            page = await context.new_page()
            await page.goto(TEAMS_URL, wait_until="domcontentloaded")

            selector = ", ".join(_LOGGED_IN_SELECTORS)
            try:
                await page.wait_for_selector(selector, timeout=15_000)
                log.info("browser.session.valid")
                return True
            except Exception:
                log.info("browser.session.invalid")
                return False
            finally:
                await context.close()
    except Exception:
        log.exception("browser.session.check_error")
        return False


async def get_authenticated_context(
    playwright: Playwright,
    browser_data_dir: Path = Path(".browser-data"),
    *,
    headless: bool = True,
    audio_env: dict[str, str] | None = None,
) -> BrowserContext:
    """Return a Playwright BrowserContext with the persisted session.

    Parameters
    ----------
    audio_env:
        Extra environment variables for audio routing (e.g. ``PULSE_SINK``,
        ``PULSE_SOURCE``).  Merged with ``os.environ`` before launching the
        browser.

    The caller is responsible for closing the returned context.
    """
    log.info(
        "browser.context.create",
        browser_data_dir=str(browser_data_dir),
        headless=headless,
        audio_env=bool(audio_env),
    )
    return await _create_persistent_context(
        playwright, browser_data_dir, headless=headless, audio_env=audio_env
    )


async def clear_session(browser_data_dir: Path = Path(".browser-data")) -> None:
    """Delete the browser data directory to force re-login."""
    if browser_data_dir.exists():
        shutil.rmtree(browser_data_dir)
        log.info("browser.session.cleared", browser_data_dir=str(browser_data_dir))
    else:
        log.info("browser.session.already_clean", browser_data_dir=str(browser_data_dir))
