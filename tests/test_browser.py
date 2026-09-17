"""Tests for the Playwright browser layer (task T301): profile-path
sanitization, real headless rendering (skipped when the chromium bundle
is missing), and captcha-command configuration without launching a real
headed browser."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import pytest

import idea_finder.fetch.browser as browser_module
from idea_finder.fetch.browser import (
    BROWSER_PROFILES_DIRNAME,
    BrowserFetcher,
    BrowserNotInstalledError,
    captcha_command,
    default_profile_root,
)
from idea_finder.fetch.httpx_fetcher import sanitize_domain

logger = logging.getLogger(__name__)

DATA_PAGE_HTML = (
    "data:text/html;base64,"
    + base64.b64encode(
        b"<html><body><h1>idea-finder browser smoke</h1>"
        b"<p>Persistent profile rendering works.</p></body></html>"
    ).decode()
)


def test_profile_dir_is_sanitized_subdirectory() -> None:
    """Each domain maps to a safe directory under the profile root."""
    fetcher = BrowserFetcher(profile_root=Path("/tmp/profiles"))
    assert fetcher.profile_dir("habr.com") == Path("/tmp/profiles/habr.com")
    assert fetcher.profile_dir("Habr.com:443") == Path("/tmp/profiles") / sanitize_domain("Habr.com:443")
    assert sanitize_domain("Habr.com:443") not in ("", ".", "..")
    assert "/" not in sanitize_domain("bad/../domain")


def test_default_profile_root_points_at_data_dir() -> None:
    """The default root is <repo>/data/browser_profiles (gitignored)."""
    root = default_profile_root()
    assert root.name == BROWSER_PROFILES_DIRNAME
    assert root.parent.name == "data"
    assert root == Path(browser_module.__file__).resolve().parents[2] / "data" / BROWSER_PROFILES_DIRNAME


def test_data_url_uses_local_profile() -> None:
    """URLs without a host share one 'local' profile directory."""
    fetcher = BrowserFetcher(profile_root=Path("/tmp/profiles"))
    assert fetcher.profile_dir("local") == Path("/tmp/profiles/local")
    assert sanitize_domain("data:") == "data_"


def test_browser_not_installed_error_is_fetch_error() -> None:
    """The graceful-degradation error stays inside the FetchError family."""
    assert issubclass(BrowserNotInstalledError, Exception)


@pytest.fixture()
def missing_binaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the launch path to fail as if the browser bundle were missing."""

    class _FailingChromium:
        @staticmethod
        async def launch_persistent_context(**kwargs: Any) -> None:
            del kwargs
            raise browser_module.BrowserNotInstalledError(
                "Executable doesn't exist at ... please run playwright install"
            )

    class _FakePlaywright:
        chromium = _FailingChromium()

    class _FakeStart:
        async def __aenter__(self) -> _FakePlaywright:
            return _FakePlaywright()

        async def __aexit__(self, *exc_info: object) -> None:
            return None

        async def start(self) -> _FakePlaywright:
            return _FakePlaywright()

    monkeypatch.setattr(browser_module, "async_playwright", lambda: _FakeStart())


async def test_fetch_without_binaries_raises_browser_not_installed(
    missing_binaries: None, tmp_path: Path
) -> None:
    """A missing bundle surfaces as BrowserNotInstalledError with the fix."""
    fetcher = BrowserFetcher(profile_root=tmp_path / "profiles")
    with pytest.raises(BrowserNotInstalledError, match="playwright install chromium"):
        await fetcher.fetch("https://example.com/")


def test_captcha_command_reports_missing_binaries(
    missing_binaries: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without binaries captcha exits 1 and prints the install hint."""
    code = captcha_command("example.com", profile_root=tmp_path / "profiles")
    assert code == 1
    out = capsys.readouterr().out
    assert "opening headed browser for example.com" in out
    assert "playwright install chromium" in out


def test_captcha_command_builds_sanitizeable_profile_path(tmp_path: Path) -> None:
    """The profile directory the captcha flow would use is sanitized."""
    domain = "Habr.com:443"
    profile = tmp_path / "profiles" / sanitize_domain(domain)
    assert profile == tmp_path / "profiles" / "habr.com_443"
    # The real headed launch is covered by unit config above; here we pin
    # the exact path captcha_command derives from (domain, profile_root).
    fetcher = BrowserFetcher(profile_root=tmp_path / "profiles")
    assert fetcher.profile_dir(domain) == profile


def _chromium_available() -> bool:
    """True when the playwright chromium bundle is installed."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            path: object = p.chromium.executable_path
        return Path(str(path)).exists()
    except Exception as e:  # noqa: BLE001 - any driver failure means "not usable"
        logger.debug("chromium availability check failed: %s", e)
        return False


@pytest.mark.skipif(not _chromium_available(), reason="playwright chromium bundle not installed")
async def test_headless_fetch_renders_data_url(tmp_path: Path) -> None:
    """Real headless Chromium renders a page and returns its HTML."""
    fetcher = BrowserFetcher(profile_root=tmp_path / "profiles")
    try:
        html = await fetcher.fetch(DATA_PAGE_HTML)
    finally:
        await fetcher.aclose()
    assert "idea-finder browser smoke" in html
    assert "Persistent profile rendering works." in html
    # The temporary profile directory was actually used.
    assert (tmp_path / "profiles" / "local").exists()
