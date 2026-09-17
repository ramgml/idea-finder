"""HTTP fetching with per-domain rate limiting and robots.txt respect.

The async fetch layer under every source adapter (task T301). One
:class:`HttpFetcher` serves the whole pipeline; adapters call
:meth:`HttpFetcher.fetch` and receive page text ready for trafilatura
extraction (:mod:`idea_finder.fetch.extract`).

Politeness rules (AGENTS.md, sources rule 3):

- requests to distinct hosts are throttled independently (aiolimiter
  ``AsyncLimiter`` per domain, default 1 request/second, overridable per domain
  via the ``rate_limits`` constructor argument sourced from
  ``source.rate_limit_rps``);
- ``robots.txt`` is fetched once per domain (with its own short timeout)
  and honored before any page request;
- requests identify honestly with a realistic browser UA that discloses
  the bot in a comment.
"""

from __future__ import annotations

import logging
import re
from typing import Final, Self
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import aiolimiter
import httpx

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_RATE_RPS", "REALISTIC_USER_AGENT", "FetchError", "HttpFetcher", "RobotsDisallowedError"]

#: Politeness default when a domain has no entry in ``source.rate_limit_rps``.
DEFAULT_RATE_RPS: Final[float] = 1.0

#: Realistic desktop browser UA with an honest bot disclosure comment.
REALISTIC_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 "
    "idea-finder/0.1 (+https://github.com/ramgml/idea-finder)"
)

#: Timeout for the robots.txt fetch itself (kept short so a hung robots
#: endpoint cannot stall every page request to that domain).
_ROBOTS_TIMEOUT_S: Final[float] = 5.0


class FetchError(Exception):
    """Base error for the fetch layer: network, HTTP status, or robots."""


class RobotsDisallowedError(FetchError):
    """robots.txt disallows fetching the requested URL."""


class HttpFetcher:
    """Async HTTP client with per-domain rate limits and robots.txt checks.

    Create one fetcher per pipeline run::

        async with HttpFetcher(rate_limits={"habr.com": 2.0}) as fetcher:
            html = await fetcher.fetch("https://habr.com/ru/feed/")

    ``rate_limits`` maps a domain to its ``source.rate_limit_rps`` value;
    domains missing from the mapping use :data:`DEFAULT_RATE_RPS`.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 20.0,
        rate_limits: dict[str, float] | None = None,
        respect_robots: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """``transport`` is an injection seam for tests (httpx.MockTransport);
        production code leaves it as None for real network I/O."""
        self._timeout_s = timeout_s
        self._rate_limits = dict(rate_limits) if rate_limits else {}
        self._respect_robots = respect_robots
        self._transport = transport
        # Lazy per-domain state: a domain's limiter/robots entries are
        # created on first request to it, so an unused source costs nothing.
        self._limiters: dict[str, aiolimiter.AsyncLimiter] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            headers={"User-Agent": REALISTIC_USER_AGENT},
            timeout=httpx.Timeout(self._timeout_s),
            follow_redirects=True,
            transport=self._transport,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _limiter(self, domain: str) -> aiolimiter.AsyncLimiter:
        """Return the per-domain limiter, creating it lazily."""
        limiter = self._limiters.get(domain)
        if limiter is None:
            rate = self._rate_limits.get(domain, DEFAULT_RATE_RPS)
            limiter = aiolimiter.AsyncLimiter(rate, 1.0)
            self._limiters[domain] = limiter
        return limiter

    def _host(self, url: str) -> str:
        """Extract the lowercase host (with port, if any) from ``url``."""
        host = urlsplit(url).hostname
        if host is None:
            raise FetchError(f"cannot parse host from url: {url}")
        return host.lower()

    async def _robots_parser(self, domain: str) -> RobotFileParser | None:
        """Fetch and parse ``robots.txt`` for ``domain`` (cached per domain).

        Returns ``None`` when robots.txt is unreachable or unparsable: the
        conservative reading is not "block everything" but "treat the site
        as not having expressed a preference".
        """
        if domain in self._robots:
            return self._robots[domain]
        assert self._client is not None, "HttpFetcher used outside async context"
        parser: RobotFileParser | None = None
        try:
            response = await self._client.get(
                f"https://{domain}/robots.txt",
                timeout=httpx.Timeout(_ROBOTS_TIMEOUT_S),
            )
            if response.status_code == 200:
                parser = RobotFileParser()
                parser.parse(response.text.splitlines())
        except httpx.HTTPError as e:
            logger.info("robots.txt fetch failed for %s: %s", domain, e)
        self._robots[domain] = parser
        return parser

    async def fetch(self, url: str) -> str:
        """Rate-limited, robots-checked GET returning the response text.

        Raises:
            RobotsDisallowedError: robots.txt disallows this URL.
            FetchError: HTTP 4xx/5xx, DNS/connect/timeout failure, or
                malformed URL (always with the original cause attached).
        """
        if self._client is None:
            raise FetchError("HttpFetcher used outside 'async with' context")
        domain = self._host(url)
        async with self._limiter(domain):
            if self._respect_robots:
                parser = await self._robots_parser(domain)
                if parser is not None and not parser.can_fetch(REALISTIC_USER_AGENT, url):
                    raise RobotsDisallowedError(f"robots.txt disallows: {url}")
            try:
                response = await self._client.get(url)
            except httpx.HTTPError as e:
                logger.info("fetch failed url=%s: %s", url, e)
                raise FetchError(f"fetch failed: {url}") from e
        if response.is_error:
            raise FetchError(f"HTTP {response.status_code} for {url}")
        return response.text


#: Split "https://host[:port]/..." into a filesystem-safe directory name.
_HOST_SANITIZE_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9.-]")


def sanitize_domain(domain: str) -> str:
    """Map a domain to a safe directory name for its browser profile.

    Keeps lowercase alphanumerics, dots, and dashes; everything else
    (``:``, ``/``, spaces, non-ASCII) collapses to ``_``. Lowercasing also
    makes ``Habr.com`` and ``habr.com`` share one profile.
    """
    return _HOST_SANITIZE_RE.sub("_", domain.strip().lower())
