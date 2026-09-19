"""Yandex Wordstat client: protocol + fake + real adapter (task T321).

The validate stage (core/pipeline.py) programs against :class:`WordstatClient`
only: it never knows whether the answer came from the deterministic fake or
the real Yandex Search API. The real client is built by
:func:`build_wordstat_client` from the ``wordstat_settings`` row (separate
table — never ``llm_provider``, different provider and validation);
:no-key configuration yields the fake so the mock mode works with zero
setup, exactly like the LLM factory's fake provider.

Secrets contract: the token lives only in the database; nothing here logs
or renders the raw value — masking is :func:`mask_token` (``****`` + last
4 characters, mirroring ``settings_view.mask_key``).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol, cast

import httpx
from psycopg import Connection

logger = logging.getLogger(__name__)

__all__ = [
    "FakeWordstatClient",
    "WordstatApiError",
    "WordstatClient",
    "YandexWordstatClient",
    "build_wordstat_client",
    "mask_token",
]

#: HTTP request timeout in seconds.
_DEFAULT_TIMEOUT: Final[float] = 30.0

#: Retries for transient failures (429 / 5xx / network errors).
_MAX_ATTEMPTS: Final[int] = 3

#: Yandex Wordstat API endpoint (Search API, reports endpoint).
_API_URL: Final[str] = "https://api.direct.yandex.com/v5/wordstat/reports"

#: Fake frequency band: deterministic hash-derived value in this range.
_FAKE_MAX_FREQUENCY: Final[int] = 10_000


class WordstatApiError(Exception):
    """Raised when the real Wordstat call fails after the retry schedule."""


@dataclass(frozen=True, slots=True)
class PhraseFrequency:
    """One phrase's Wordstat answer as the stage consumes it."""

    phrase: str
    frequency: int


class WordstatClient(Protocol):
    """Port every Wordstat consumer programs against (SYSTEM_DESIGN.md)."""

    def frequency(self, phrase: str) -> int:
        """Return the monthly search frequency for ``phrase``."""
        ...


def mask_token(token: str) -> str:
    """Mask a token to ``****`` + its last 4 characters.

    Tokens of four characters or fewer (and empty ones) are hidden
    entirely: the tail of a short token would be the whole token.
    Mirrors :func:`idea_finder.web.settings_view.mask_key` so both
    settings pages show secrets the same way.
    """
    if len(token) > 4:
        return f"****{token[-4:]}"
    return "****"


class FakeWordstatClient:
    """Deterministic, network-free client for tests and mock mode.

    The frequency is derived from a stable SHA-256 hash of the phrase
    mapped into ``[0, _FAKE_MAX_FREQUENCY]``: identical phrases always
    yield identical values, so validation runs are reproducible and the
    dashboard shows stable numbers without any network or API key.
    """

    def frequency(self, phrase: str) -> int:
        """Return the hash-derived deterministic frequency."""
        digest = hashlib.sha256(phrase.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % (_FAKE_MAX_FREQUENCY + 1)


class YandexWordstatClient:
    """Real Yandex Wordstat (Search API) client behind the same port.

    One synchronous POST per phrase with bearer auth; transient failures
    (429, 5xx, network errors) are retried up to three times. Rate
    limiting (10 rps cap) is the caller's job (``aiolimiter`` at the
    stage boundary), matching how the fetch layer composes limiters.
    """

    def __init__(
        self,
        token: str,
        *,
        base_url: str = _API_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._token = token
        self._base_url = base_url
        self._timeout = timeout

    def frequency(self, phrase: str) -> int:
        """POST one phrase report; return its total search frequency.

        Raises:
            WordstatApiError: when the API answers non-2xx after all
                retries or the body cannot be interpreted.
        """
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = httpx.post(
                    self._base_url,
                    headers={"Authorization": f"Bearer {self._token}"},
                    json={"Phrases": [phrase]},
                    timeout=self._timeout,
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = WordstatApiError(
                        f"attempt {attempt}/{_MAX_ATTEMPTS}: wordstat returned "
                        f"HTTP {response.status_code}"
                    )
                    logger.warning("%s", last_error)
                else:
                    return self._parse_response(phrase, response)
            except httpx.HTTPError as error:
                last_error = WordstatApiError(
                    f"attempt {attempt}/{_MAX_ATTEMPTS}: network error: {error}"
                )
                logger.warning("%s", last_error)
        msg = f"wordstat: all {_MAX_ATTEMPTS} attempts failed for phrase {phrase!r}"
        raise WordstatApiError(msg) from last_error

    def _parse_response(self, phrase: str, response: httpx.Response) -> int:
        """Extract the total frequency from a 2xx report body."""
        if not response.is_success:
            msg = f"wordstat returned HTTP {response.status_code}: {response.text[:200]}"
            raise WordstatApiError(msg)
        try:
            body = cast(Mapping[str, object], response.json())
        except ValueError as error:
            msg = "wordstat returned non-JSON body"
            raise WordstatApiError(msg) from error
        results = body.get("results")
        if not isinstance(results, list) or not results:
            msg = f"wordstat report for {phrase!r} has no results"
            raise WordstatApiError(msg)
        first = results[0]
        if not isinstance(first, dict):
            msg = f"wordstat report for {phrase!r} has malformed results"
            raise WordstatApiError(msg)
        total = first.get("total")
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            msg = f"wordstat report for {phrase!r} has no numeric total"
            raise WordstatApiError(msg)
        return total


def _load_stored_token(conn: Connection) -> str | None:
    """Return the stored 'yandex' token, or None when absent/empty.

    The raw token never leaves this function's return value scope: callers
    either build the real client with it or fall back to the fake.
    """
    row = conn.execute(
        "SELECT api_key FROM wordstat_settings WHERE name = 'yandex'"
    ).fetchone()
    if row is None:
        return None
    token = str(row[0])
    return token or None


def build_wordstat_client(conn: Connection) -> WordstatClient:
    """Construct the Wordstat client from stored settings.

    A stored non-empty 'yandex' token builds the real client; anything
    else (no row, empty token) builds the deterministic fake — mock mode
    works with zero setup, mirroring the LLM factory's fake provider.
    """
    token = _load_stored_token(conn)
    if token is None:
        logger.info("wordstat client: no token, using fake (mock mode)")
        return FakeWordstatClient()
    logger.info("wordstat client: real Yandex API (token %s)", mask_token(token))
    return YandexWordstatClient(token)
