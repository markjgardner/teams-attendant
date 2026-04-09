"""Teams authentication and session management."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

import structlog
from playwright.async_api import BrowserContext, Playwright, async_playwright

from teams_attendant.errors import AuthenticationError, BrowserProfileLockedError

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

_EDGE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
)

_STEALTH_JS = """
// Hide webdriver property
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});

// Mock chrome.runtime to look like a real browser
if (!window.chrome) { window.chrome = {}; }
if (!window.chrome.runtime) {
    window.chrome.runtime = {
        connect: function() {},
        sendMessage: function() {},
    };
}

// Fix permissions query for notifications (automation detection)
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) =>
    parameters.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : originalQuery(parameters);

// Ensure navigator.plugins is non-empty
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5],
});

// Ensure navigator.languages returns a proper array
Object.defineProperty(navigator, 'languages', {
    get: () => ['en-US', 'en'],
});

// Remove automation-related properties from document
delete Object.getPrototypeOf(navigator).webdriver;
"""

_CHROMIUM_ARGS = [
    "--disable-blink-features=AutomationControlled",
]

# Playwright default args that reveal automation to bot detectors
_IGNORE_DEFAULT_ARGS = [
    "--enable-automation",
]


def _ensure_dir(path: Path) -> Path:
    """Ensure the directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_system_browser_profile(browser: str = "msedge") -> Path:
    """Return the path to the system browser's user data directory.

    Raises ``FileNotFoundError`` if the profile directory does not exist.

    .. note:: The system browser must be **closed** before Playwright can
       use its profile — Chromium locks the user-data directory.
    """
    system = platform.system()

    if system == "Windows":
        local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
        paths = {
            "msedge": local_app_data / "Microsoft" / "Edge" / "User Data",
            "chromium": local_app_data / "Google" / "Chrome" / "User Data",
        }
    elif system == "Darwin":
        support = Path.home() / "Library" / "Application Support"
        paths = {
            "msedge": support / "Microsoft Edge",
            "chromium": support / "Google" / "Chrome",
        }
    else:  # Linux
        config = Path.home() / ".config"
        paths = {
            "msedge": config / "microsoft-edge",
            "chromium": config / "google-chrome",
        }

    profile_dir = paths.get(browser, paths.get("chromium", Path()))
    if not profile_dir or not profile_dir.exists():
        raise FileNotFoundError(
            f"System browser profile not found at {profile_dir}. "
            f"Is {browser} installed?"
        )

    log.info("browser.system_profile", path=str(profile_dir))
    return profile_dir


def _check_profile_lock(browser_data_dir: Path) -> None:
    """Raise if the browser profile directory is locked by a running browser.

    Chromium-based browsers create a ``SingletonLock`` (Linux/macOS) or
    ``lockfile`` (Windows) inside the user-data directory.  If such a file
    exists, another browser instance owns the profile and Playwright will
    fail with ``TargetClosedError``.
    """
    lock_names = ("SingletonLock", "lockfile", "Lock")
    for name in lock_names:
        lock_path = browser_data_dir / name
        if lock_path.exists():
            raise BrowserProfileLockedError(
                f"The browser profile at {browser_data_dir} is locked "
                f"(found {name}). Close all browser windows using this "
                "profile and try again."
            )


async def _create_persistent_context(
    playwright: Playwright,
    browser_data_dir: Path,
    *,
    headless: bool = True,
    audio_env: dict[str, str] | None = None,
    browser: str = "chromium",
) -> BrowserContext:
    """Create a persistent browser context with stealth settings."""
    _check_profile_lock(browser_data_dir)
    _ensure_dir(browser_data_dir)

    user_agent = _EDGE_USER_AGENT if browser == "msedge" else _CHROME_USER_AGENT

    launch_kwargs: dict = dict(
        user_data_dir=str(browser_data_dir),
        headless=headless,
        user_agent=user_agent,
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
        permissions=["microphone", "camera"],
        args=_CHROMIUM_ARGS,
        ignore_default_args=_IGNORE_DEFAULT_ARGS,
    )

    if browser == "msedge":
        launch_kwargs["channel"] = "msedge"

    if audio_env:
        merged_env = {**os.environ, **audio_env}
        launch_kwargs["env"] = merged_env
        log.info("browser.context.audio_env", extra_keys=list(audio_env.keys()))

    context = await playwright.chromium.launch_persistent_context(**launch_kwargs)
    await context.add_init_script(_STEALTH_JS)
    return context


async def login(
    browser_data_dir: Path = Path(".browser-data"),
    browser: str = "chromium",
) -> None:
    """Launch a visible browser for the user to log in to Teams.

    The browser stays open until login is detected or the timeout is reached.
    Session data is persisted automatically via the persistent context.
    """
    log.info("browser.login.start", browser_data_dir=str(browser_data_dir), browser=browser)

    async with async_playwright() as pw:
        context = await _create_persistent_context(
            pw, browser_data_dir, headless=False, browser=browser
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


async def is_session_valid(
    browser_data_dir: Path = Path(".browser-data"),
    browser: str = "chromium",
) -> bool:
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
                pw, browser_data_dir, headless=True, browser=browser
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
    browser: str = "chromium",
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
        browser=browser,
    )
    return await _create_persistent_context(
        playwright, browser_data_dir, headless=headless, audio_env=audio_env, browser=browser
    )


async def clear_session(browser_data_dir: Path = Path(".browser-data")) -> None:
    """Delete the browser data directory to force re-login."""
    if browser_data_dir.exists():
        shutil.rmtree(browser_data_dir)
        log.info("browser.session.cleared", browser_data_dir=str(browser_data_dir))
    else:
        log.info("browser.session.already_clean", browser_data_dir=str(browser_data_dir))
