"""Tests for the HTTP fetch layer (task T301): per-domain aiolimiter
isolation, robots.txt respect, and FetchError on HTTP/network failures.
All network interaction goes through httpx.MockTransport (injected via
the fetcher's ``transport`` seam) — no real requests are made."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict, cast

import httpx
import pytest

from idea_finder.fetch.httpx_fetcher import (
    DEFAULT_RATE_RPS,
    REALISTIC_USER_AGENT,
    FetchError,
    HttpFetcher,
    RobotsDisallowedError,
    sanitize_domain,
)


class HttpFetcherOptions(TypedDict, total=False):
    """Keyword options of :class:`HttpFetcher` accepted by the test helper."""

    timeout_s: float
    rate_limits: dict[str, float]
    respect_robots: bool


Handler = Callable[[httpx.Request], httpx.Response]




async def _fetch_with(handler: Handler, url: str, **kwargs: Any) -> str:
    """Run one fetch against a mock transport and return the body text."""
    options = cast(HttpFetcherOptions, kwargs)
    async with HttpFetcher(transport=httpx.MockTransport(handler), **options) as fetcher:
        return await fetcher.fetch(url)


async def test_fetch_returns_text_with_realistic_ua() -> None:
    """A 200 response comes back as text, requested with the disclosed UA."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"] == REALISTIC_USER_AGENT
        assert "idea-finder/0.1" in request.headers["user-agent"]
        return httpx.Response(200, text="<html>ok</html>")

    assert await _fetch_with(handler, "https://example.com/page") == "<html>ok</html>"


async def test_fetch_404_raises_fetch_error() -> None:
    """HTTP 404 surfaces as FetchError, not as an httpx exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(404, text="gone")

    with pytest.raises(FetchError, match="404"):
        await _fetch_with(handler, "https://example.com/missing")


async def test_fetch_500_raises_fetch_error() -> None:
    """HTTP 5xx surfaces as FetchError too."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(500)

    with pytest.raises(FetchError, match="500"):
        await _fetch_with(handler, "https://example.com/broken")


async def test_network_error_wrapped_with_raise_from() -> None:
    """Connect errors are wrapped in FetchError with the cause attached."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ConnectError("dns failure")

    with pytest.raises(FetchError) as excinfo:
        await _fetch_with(handler, "https://down.example.com/x")
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)


async def test_fetch_outside_context_manager_raises() -> None:
    """Using the fetcher without 'async with' is a programming error."""
    fetcher = HttpFetcher()
    with pytest.raises(FetchError, match="async with"):
        await fetcher.fetch("https://example.com/")


def test_per_domain_limiters_are_isolated_and_lazy() -> None:
    """Two domains get two limiter instances, created on first use only."""
    fetcher = HttpFetcher(rate_limits={"habr.com": 2.0})
    assert fetcher._limiters == {}  # nothing created until first request

    first = fetcher._limiter("habr.com")
    second = fetcher._limiter("fl.ru")
    assert first is not second  # per-domain isolation
    assert fetcher._limiter("habr.com") is first  # cached per domain

    # The rate comes from the constructor mapping, default elsewhere.
    assert first.max_rate == 2.0
    assert second.max_rate == DEFAULT_RATE_RPS
    assert fetcher._limiter("unknown.example").max_rate == DEFAULT_RATE_RPS


async def test_robots_disallow_raises() -> None:
    """A robots.txt Disallow turns the page fetch into RobotsDisallowedError."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, text="secret")

    async with HttpFetcher(transport=httpx.MockTransport(handler)) as fetcher:
        assert await fetcher.fetch("https://example.com/public/page") == "secret"
        with pytest.raises(RobotsDisallowedError):
            await fetcher.fetch("https://example.com/private/page")


async def test_robots_fetched_once_per_domain() -> None:
    """robots.txt is cached: one request across many page fetches."""
    robots_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_requests
        if request.url.path == "/robots.txt":
            robots_requests += 1
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, text="page")

    async with HttpFetcher(transport=httpx.MockTransport(handler)) as fetcher:
        for n in range(3):
            assert await fetcher.fetch(f"https://example.com/page{n}") == "page"
    assert robots_requests == 1


async def test_robots_5xx_treated_as_absent() -> None:
    """An unreachable robots endpoint does not block page fetching."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(500)
        return httpx.Response(200, text="page")

    async with HttpFetcher(transport=httpx.MockTransport(handler)) as fetcher:
        assert await fetcher.fetch("https://example.com/page") == "page"


async def test_rate_limit_maps_onto_mock_handler_requests() -> None:
    """The per-domain limiter gate sits in front of every page request."""
    page_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal page_requests
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        page_requests += 1
        return httpx.Response(200, text="page")

    async with HttpFetcher(
        rate_limits={"example.com": 50.0},
        transport=httpx.MockTransport(handler),
    ) as fetcher:
        assert fetcher._limiter("example.com").max_rate == 50.0
        for _ in range(3):
            assert await fetcher.fetch("https://example.com/x") == "page"
    assert page_requests == 3


def test_sanitize_domain_maps_to_safe_dirname() -> None:
    """Domains map to one safe, lowercased directory name."""
    assert sanitize_domain("habr.com") == "habr.com"
    assert sanitize_domain("Habr.com") == "habr.com"  # case folding
    assert sanitize_domain("Habr.com:443") == "habr.com_443"  # port kept
    assert "/" not in sanitize_domain("bad/../domain")
    assert sanitize_domain("a b.c") == "a_b.c"
