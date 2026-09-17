"""Playwright browser fetching with persistent per-domain profiles.

Reserved for JS-heavy domains where plain httpx gets an empty shell
(task T301). Each domain gets a persistent browser profile under
``data/browser_profiles/<domain>`` (gitignored, like the rest of
``data/``): cookies and local storage survive across runs, so a captcha
solved once via ``cli.py captcha <domain>`` keeps the headless fetcher
unblocked.

The companion CLI command :func:`captcha_command` opens the same profile
in a **headed** browser so a human can solve the captcha interactively;
pressing Enter in the terminal closes the browser, and the profile keeps
the session cookies on disk.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

from playwright.async_api import BrowserContext, Playwright, async_playwright

from idea_finder.fetch.httpx_fetcher import FetchError, sanitize_domain

logger = logging.getLogger(__name__)

__all__ = [
    "BROWSER_PROFILES_DIRNAME",
    "BrowserFetcher",
    "BrowserNotInstalledError",
    "captcha_command",
    "default_profile_root",
]

#: Browser profiles live under the gitignored runtime directory.
BROWSER_PROFILES_DIRNAME: Final = "browser_profiles"

#: Playwright's error text when the browser bundle is missing.
_NOT_INSTALLED_HINT: Final = "Executable doesn't exist"

#: Human-facing fix for a missing browser bundle.
_NOT_INSTALLED_MESSAGE: Final = (
    "Playwright browser binaries are not installed. "
    "Run: uv run playwright install chromium"
)

#: Profile directory for URLs without a host (``data:``, ``about:``).
_LOCAL_PROFILE: Final = "local"


class BrowserNotInstalledError(FetchError):
    """Playwright browser binaries are missing (see the message for the fix)."""


def default_profile_root() -> Path:
    """Return ``<repo>/data/browser_profiles`` (the documented location)."""
    return Path(__file__).resolve().parents[2] / "data" / BROWSER_PROFILES_DIRNAME


def _is_not_installed(exc: BaseException) -> bool:
    """Detect Playwright's 'browser bundle missing' failure mode."""
    return _NOT_INSTALLED_HINT in str(exc)


def _url_domain(url: str) -> str:
    """Extract the URL host for profile selection; ``local`` when absent."""
    host = urlsplit(url).hostname
    return host.lower() if host else _LOCAL_PROFILE


class BrowserFetcher:
    """Headless Chromium fetcher with persistent per-domain profiles.

    Usage::

        fetcher = BrowserFetcher()  # profiles under data/browser_profiles
        html = await fetcher.fetch("https://some-js-site.ru/")

    Each domain keeps its own persistent context (cookies survive runs);
    contexts are created lazily and reused until :meth:`aclose`. The
    browser bundle must be installed once per machine:
    ``uv run playwright install chromium`` — without it every fetch raises
    :class:`BrowserNotInstalledError`.
    """

    def __init__(self, *, profile_root: Path | None = None, headless: bool = True) -> None:
        self._profile_root = profile_root if profile_root is not None else default_profile_root()
        self._default_headless = headless
        self._playwright: Playwright | None = None
        self._contexts: dict[str, BrowserContext] = {}

    def profile_dir(self, domain: str) -> Path:
        """Return the persistent profile directory for ``domain``.

        Names are lowercased and stripped to filesystem-safe characters,
        so ``Habr.com`` and ``habr.com`` share one profile.
        """
        return self._profile_root / sanitize_domain(domain)

    async def _context_for(self, url: str, *, headed: bool) -> BrowserContext:
        """Return the persistent context for the URL's domain, lazily."""
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        domain = _url_domain(url)
        context = self._contexts.get(domain)
        if context is not None:
            return context
        profile = self.profile_dir(domain)
        profile.mkdir(parents=True, exist_ok=True)
        try:
            context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile),
                headless=not headed,
            )
        except Exception as e:
            if _is_not_installed(e):
                raise BrowserNotInstalledError(_NOT_INSTALLED_MESSAGE) from e
            raise
        self._contexts[domain] = context
        return context

    async def fetch(self, url: str, *, headed: bool = False) -> str:
        """Render ``url`` in Chromium and return the resulting HTML.

        Waits for the ``networkidle`` lifecycle event so JS-driven pages
        finish loading before extraction. ``headed=True`` launches the
        domain's context with a visible window (used by the captcha flow).
        """
        context = await self._context_for(url, headed=headed)
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="networkidle")
            return await page.content()
        finally:
            await page.close()

    async def aclose(self) -> None:
        """Close every open context and stop the Playwright driver."""
        for context in self._contexts.values():
            await context.close()
        self._contexts.clear()
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None


def captcha_command(domain: str, profile_root: Path | None = None) -> int:
    """Open a headed browser on ``https://<domain>`` for manual captcha solving.

    Synchronous CLI entry point: launches the domain's persistent profile
    in a visible window, waits for Enter on the terminal, then closes the
    browser. The profile directory keeps the session cookies, so the
    headless :class:`BrowserFetcher` stays unblocked afterwards.

    Returns a process exit code: 0 on success, 1 when the browser bundle
    is missing or stdin is unavailable.
    """
    root = profile_root if profile_root is not None else default_profile_root()
    profile = root / sanitize_domain(domain)
    print(f"opening headed browser for {domain}; solve captcha then press Enter")

    async def _run() -> None:
        async with async_playwright() as p:
            try:
                context = await p.chromium.launch_persistent_context(
                    user_data_dir=str(profile),
                    headless=False,
                )
            except Exception as e:
                if _is_not_installed(e):
                    raise BrowserNotInstalledError(_NOT_INSTALLED_MESSAGE) from e
                raise
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(f"https://{domain}", wait_until="domcontentloaded")
            await asyncio.to_thread(input)
            await context.close()

    try:
        asyncio.run(_run())
    except BrowserNotInstalledError as e:
        print(str(e))
        return 1
    except EOFError:
        print("no interactive stdin; closing without waiting")
        return 1
    logger.info("captcha session for %s saved to %s", domain, profile)
    return 0
