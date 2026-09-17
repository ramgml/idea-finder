"""LlmClient factory: every LLM call in idea-finder goes through here.

The factory (:func:`build_llm_client`) reads the single active provider from
the ``llm_provider`` table (via :mod:`idea_finder.db.repo`) and constructs the
matching client:

* ``kind='fake'`` -> :class:`FakeLlmClient`: deterministic, network-free,
  zero-cost answers from the fixture dataset (mock mode, activated by
  ``is_active`` on the fake provider row).
* ``kind='openai_compat'`` -> :class:`OpenAiCompatClient`: OpenAI-compatible
  ``POST {base_url}/chat/completions`` with bearer auth, timeout, and
  exponential-backoff retries on 429/5xx/network errors.

``api_key`` lives only in the database and is never logged. Base URLs are
never hardcoded — they come from the provider row.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol, Self, cast

import httpx
from psycopg import Connection

from idea_finder.db.repo import LlmProviderRecord, get_active_llm_provider

logger = logging.getLogger(__name__)

__all__ = [
    "Completion",
    "FakeLlmClient",
    "LlmClient",
    "LlmError",
    "OpenAiCompatClient",
    "build_llm_client",
    "estimate_cost",
]

#: Retries for transient failures (429 / 5xx / network errors).
_MAX_ATTEMPTS = 3

#: Sleep seconds between attempts: exponential backoff 0.5 / 1 / 2.
_BACKOFF_SCHEDULE: tuple[float, ...] = (0.5, 1.0, 2.0)

#: HTTP request timeout in seconds (configurable per client instance).
_DEFAULT_TIMEOUT = 30.0


class LlmError(Exception):
    """Raised when an LLM call fails or no provider is configured."""


@dataclass(frozen=True, slots=True)
class Completion:
    """A single LLM completion plus the accounting data consumers need."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    provider_name: str


class LlmClient(Protocol):
    """Port every LLM consumer must program against (SYSTEM_DESIGN.md)."""

    def complete(self, prompt: str) -> Completion:
        """Return one completion for ``prompt``."""
        ...


def estimate_cost(completion: Completion, price_per_mtok: Decimal | float | None) -> float:
    """Return the completion's cost in the account currency.

    ``(prompt_tokens + completion_tokens) * price_per_mtok / 1e6``; a missing
    tariff or a fake-provider completion (always free) yields 0.0. This is the
    single cost-accounting point; the extract stage writes the value into
    ``run.stats``.
    """
    if price_per_mtok is None or completion.prompt_tokens == 0 and completion.completion_tokens == 0:
        return 0.0
    total_tokens = completion.prompt_tokens + completion.completion_tokens
    cost = Decimal(total_tokens) * Decimal(str(price_per_mtok)) / Decimal(1_000_000)
    return float(cost)


def _load_fixture_completion(prompt: str) -> str:
    """Return a deterministic fixture-derived answer for ``prompt``.

    Hashes the prompt to a stable index into the expected-pains fixture
    dataset, so identical prompts always get identical answers and different
    prompts spread across the dataset. Network-free by construction.
    """
    fixtures_dir = Path(__file__).resolve().parents[2] / "fixtures"
    with (fixtures_dir / "expected_pains.json").open(encoding="utf-8") as file:
        entries = cast(list[dict[str, object]], json.load(file))
    pains: list[str] = []
    for entry in entries:
        entry_pains = entry.get("pains")
        if isinstance(entry_pains, list):
            for pain in entry_pains:
                if isinstance(pain, dict) and isinstance(pain.get("body"), str):
                    pains.append(pain["body"])
    if not pains:
        msg = "fixtures/expected_pains.json contains no pain bodies"
        raise LlmError(msg)
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % len(pains)
    return pains[index]


class FakeLlmClient:
    """Deterministic, network-free client bound to the fake provider kind.

    Answers are canonical pain texts from ``fixtures/expected_pains.json``,
    selected by a stable hash of the prompt: two identical prompts yield two
    identical :class:`Completion` objects, and the cost is always zero.
    """

    def __init__(self, *, model: str = "fake", provider_name: str = "fake") -> None:
        self.model = model
        self.provider_name = provider_name

    def complete(self, prompt: str) -> Completion:
        """Return the fixture answer for ``prompt`` (deterministic)."""
        text = _load_fixture_completion(prompt)
        return Completion(
            text=text,
            prompt_tokens=0,
            completion_tokens=0,
            model=self.model,
            provider_name=self.provider_name,
        )


class OpenAiCompatClient:
    """Client for OpenAI-compatible ``/chat/completions`` endpoints.

    One ``httpx.Client`` per instance; ``complete`` makes a synchronous POST
    with bearer auth and retries transient failures (429, 5xx, network errors)
    up to three attempts with 0.5/1/2 s exponential backoff.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        provider_name: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        backoff_schedule: Sequence[float] = _BACKOFF_SCHEDULE,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider_name = provider_name
        self.timeout = timeout
        self._backoff_schedule = tuple(backoff_schedule)
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def __enter__(self) -> Self:
        """Return self so the client can wrap ``with`` blocks in tests/CLI."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release the underlying HTTP connection pool."""
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()

    def complete(self, prompt: str) -> Completion:
        """POST one chat completion, retrying transient failures.

        Token usage comes from ``response.usage`` (fallback 0). Any failure
        that survives the retry schedule raises :class:`LlmError` with the
        underlying exception or response attached via ``raise from``.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._http.post("/chat/completions", json=payload)
                if response.status_code in (429, *(500, 502, 503, 504)):
                    last_error = LlmError(
                        f"attempt {attempt}/{_MAX_ATTEMPTS}: provider returned "
                        f"HTTP {response.status_code}"
                    )
                    logger.warning("%s", last_error)
                else:
                    return self._parse_response(response)
            except httpx.HTTPError as error:
                last_error = LlmError(f"attempt {attempt}/{_MAX_ATTEMPTS}: network error: {error}")
                logger.warning("%s", last_error)
            if attempt < _MAX_ATTEMPTS:
                delay = self._backoff_schedule[min(attempt - 1, len(self._backoff_schedule) - 1)]
                logger.info("retrying in %.1fs (provider=%s model=%s)",
                            delay, self.provider_name, self.model)
                time.sleep(delay)
        msg = f"provider={self.provider_name} model={self.model}: all {_MAX_ATTEMPTS} attempts failed"
        raise LlmError(msg) from last_error

    def _parse_response(self, response: httpx.Response) -> Completion:
        """Extract the completion text and token usage from a 2xx response."""
        if not response.is_success:
            raise LlmError(
                f"provider={self.provider_name} returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            body = cast(Mapping[str, object], response.json())
        except ValueError as error:
            msg = f"provider={self.provider_name} returned non-JSON body"
            raise LlmError(msg) from error
        text = ""
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    text = message["content"]
        usage = body.get("usage")
        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(usage, dict):
            raw_prompt = usage.get("prompt_tokens")
            if isinstance(raw_prompt, int) and not isinstance(raw_prompt, bool):
                prompt_tokens = raw_prompt
            raw_completion = usage.get("completion_tokens")
            if isinstance(raw_completion, int) and not isinstance(raw_completion, bool):
                completion_tokens = raw_completion
        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=str(body.get("model", self.model)),
            provider_name=self.provider_name,
        )


def build_llm_client(conn: Connection) -> LlmClient:
    """Construct the client for the single active provider.

    Reads :func:`idea_finder.db.repo.get_active_llm_provider`; ``fake`` builds
    :class:`FakeLlmClient`, ``openai_compat`` builds
    :class:`OpenAiCompatClient` from the row's ``base_url``/``api_key``/
    ``model``. No active row is a configuration error.
    """
    provider: LlmProviderRecord | None = get_active_llm_provider(conn)
    if provider is None:
        msg = (
            "no active LLM provider: insert one into llm_provider and set "
            "is_active (dashboard 'LLM Settings' or repo.upsert_llm_provider)"
        )
        raise LlmError(msg)
    if provider.kind == "fake":
        logger.info("LLM client: kind=fake provider=%s (mock mode)", provider.name)
        return FakeLlmClient(model=provider.model or "fake", provider_name=provider.name)
    if provider.kind == "openai_compat":
        if not provider.base_url:
            msg = f"provider={provider.name!r}: kind=openai_compat requires base_url"
            raise LlmError(msg)
        logger.info("LLM client: kind=openai_compat provider=%s model=%s",
                    provider.name, provider.model)
        return OpenAiCompatClient(
            provider.base_url,
            provider.api_key,
            provider.model,
            provider.name,
        )
    msg = f"provider={provider.name!r}: unknown kind {provider.kind!r}"
    raise LlmError(msg)
