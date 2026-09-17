"""Tests for idea_finder.llm.client: factory, FakeLlmClient, OpenAiCompatClient.

The factory tests run against a real embedded postgres (unix socket only) and
cover the acceptance criterion "switching the active provider changes the
factory's behavior": flipping ``is_active`` between a fake and an
openai_compat row changes what :func:`build_llm_client` returns.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from decimal import Decimal
from typing import Any, override

import httpx
import pytest
from psycopg import Connection

from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import upsert_llm_provider
from idea_finder.llm.client import (
    Completion,
    FakeLlmClient,
    LlmError,
    OpenAiCompatClient,
    build_llm_client,
    estimate_cost,
)


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """One embedded postgres cluster for the whole module."""
    data_dir = tmp_path_factory.mktemp("pgllm") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture(scope="module")
def conn(pg: PgHandle) -> Iterator[Connection]:
    """Connection to the migrated application database."""
    with pg.get_conn() as connection:
        apply_migrations(connection)
        yield connection


# ---------------------------------------------------------------------------
# Factory: provider switch changes behavior (acceptance DoD)
# ---------------------------------------------------------------------------


def test_factory_fake_provider_builds_fake_client(conn: Connection) -> None:
    """Active kind=fake row -> FakeLlmClient."""
    upsert_llm_provider(conn, "mock", "fake", "", "mock-model", "", is_active=True)
    client = build_llm_client(conn)
    assert isinstance(client, FakeLlmClient)


def test_factory_switch_to_openai_compat_changes_behavior(conn: Connection) -> None:
    """Flipping the active row to openai_compat -> OpenAiCompatClient (DoD)."""
    upsert_llm_provider(conn, "mock", "fake", "", "mock-model", "", is_active=True)
    assert isinstance(build_llm_client(conn), FakeLlmClient)
    # "stub" openai_compat provider: base_url points at a closed port on
    # purpose — this test must never do real network I/O.
    upsert_llm_provider(conn, "stub", "openai_compat",
                        "http://localhost:1", "stub-model", "", is_active=True)
    active_client = build_llm_client(conn)
    assert isinstance(active_client, OpenAiCompatClient)
    assert active_client.base_url == "http://localhost:1"
    assert active_client.model == "stub-model"


def test_factory_no_active_provider_raises(conn: Connection) -> None:
    """No active row is a configuration error, not a silent default."""
    conn.execute("UPDATE llm_provider SET is_active = false")
    conn.commit()
    with pytest.raises(LlmError, match="no active LLM provider"):
        build_llm_client(conn)
    # restore an active provider for the remaining module tests
    upsert_llm_provider(conn, "mock", "fake", "", "mock-model", "", is_active=True)


def test_factory_openai_compat_without_base_url_raises(conn: Connection) -> None:
    """kind=openai_compat with an empty base_url cannot build a client."""
    upsert_llm_provider(conn, "broken", "openai_compat", "", "m", "", is_active=True)
    with pytest.raises(LlmError, match="requires base_url"):
        build_llm_client(conn)
    upsert_llm_provider(conn, "mock", "fake", "", "mock-model", "", is_active=True)


# ---------------------------------------------------------------------------
# FakeLlmClient: determinism and zero cost
# ---------------------------------------------------------------------------


def test_fake_client_is_deterministic() -> None:
    """Two identical prompts yield byte-identical completions."""
    client = FakeLlmClient()
    first = client.complete("Extract pains from this post: ...")
    second = client.complete("Extract pains from this post: ...")
    assert first == second
    assert first.text  # non-empty fixture-derived answer


def test_fake_client_zero_cost_and_zero_tokens() -> None:
    """Fake mode is free: zero tokens, zero cost regardless of tariff."""
    client = FakeLlmClient()
    completion = client.complete("any prompt")
    assert completion.prompt_tokens == 0
    assert completion.completion_tokens == 0
    assert estimate_cost(completion, Decimal("1000.0")) == 0.0


def test_fake_client_comes_from_fixtures() -> None:
    """The answer is one of the canonical pain bodies in the fixture dataset."""
    with open("fixtures/expected_pains.json", encoding="utf-8") as file:
        bodies = [
            pain["body"]
            for entry in json.load(file)
            for pain in entry["pains"]
        ]
    completion = FakeLlmClient().complete("prompt under test")
    assert completion.text in bodies


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


def test_estimate_cost_computes_tokens_times_tariff() -> None:
    """(prompt + completion) tokens * price_per_mtok / 1e6."""
    completion = Completion(text="x", prompt_tokens=100, completion_tokens=900,
                            model="m", provider_name="p")
    assert estimate_cost(completion, 2.0) == pytest.approx(0.002)
    assert estimate_cost(completion, Decimal("2.0")) == pytest.approx(0.002)


def test_estimate_cost_without_tariff_is_zero() -> None:
    """Missing tariff -> 0.0 cost."""
    completion = Completion(text="x", prompt_tokens=10, completion_tokens=20,
                            model="m", provider_name="p")
    assert estimate_cost(completion, None) == 0.0


# ---------------------------------------------------------------------------
# OpenAiCompatClient: retry/backoff/timeout/usage over a mocked transport
# ---------------------------------------------------------------------------


class _StubTransport(httpx.BaseTransport):
    """Scripted httpx transport: pops one canned response per request."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    @override
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("unexpected extra request")
        return self._responses.pop(0)


def _client_with_transport(
    transport: _StubTransport,
    backoff_schedule: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> OpenAiCompatClient:
    client = OpenAiCompatClient(
        "http://llm.test/api", "secret-key", "test-model", "stub",
        backoff_schedule=backoff_schedule,
    )
    # Test seam: swap the real pool for a stub transport, keeping the
    # constructor's auth header so requests are asserted end to end.
    client._http = httpx.Client(
        base_url=client.base_url,
        headers={"Authorization": "Bearer secret-key"},
        transport=transport,
        timeout=0.5,
    )
    return client


def test_openai_compat_success_parses_text_and_usage() -> None:
    """2xx response: text from choices[0].message.content, tokens from usage."""
    body = {
        "model": "server-model",
        "choices": [{"message": {"role": "assistant", "content": "pain: no VIN search"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }
    transport = _StubTransport([httpx.Response(200, json=body)])
    with _client_with_transport(transport) as client:
        completion = client.complete("hello")
    assert completion.text == "pain: no VIN search"
    assert completion.prompt_tokens == 11
    assert completion.completion_tokens == 7
    assert completion.model == "server-model"
    assert completion.provider_name == "stub"
    sent = json.loads(transport.requests[0].content)
    assert sent["model"] == "test-model"
    assert sent["messages"] == [{"role": "user", "content": "hello"}]
    assert transport.requests[0].headers["Authorization"] == "Bearer secret-key"


def test_openai_compat_usage_missing_falls_back_to_zero() -> None:
    """Absent or malformed usage block -> zero tokens, no crash."""
    body: dict[str, Any] = {"choices": [{"message": {"content": "ok"}}]}
    transport = _StubTransport([httpx.Response(200, json=body)])
    with _client_with_transport(transport) as client:
        completion = client.complete("hello")
    assert completion.prompt_tokens == 0
    assert completion.completion_tokens == 0


def test_openai_compat_500_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5xx burns attempts with backoff, then a 2xx completes the call."""
    body = {"choices": [{"message": {"content": "recovered"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    transport = _StubTransport([
        httpx.Response(500, text="boom"),
        httpx.Response(503, text="busy"),
        httpx.Response(200, json=body),
    ])
    sleeps: list[float] = []
    monkeypatch.setattr("idea_finder.llm.client.time.sleep", sleeps.append)
    with _client_with_transport(transport, backoff_schedule=(0.5, 1.0, 2.0)) as client:
        completion = client.complete("hello")
    assert completion.text == "recovered"
    assert len(transport.requests) == 3
    assert sleeps == [0.5, 1.0]


def test_openai_compat_429_exhausts_retries_and_raises() -> None:
    """Persistent 429 survives the whole schedule -> LlmError with cause."""
    transport = _StubTransport([httpx.Response(429, text="rate limited")] * 3)
    with _client_with_transport(transport) as client, pytest.raises(
        LlmError, match="all 3 attempts failed"
    ) as excinfo:
        client.complete("hello")
    assert len(transport.requests) == 3
    assert "HTTP 429" in str(excinfo.value.__cause__)


def test_openai_compat_timeout_maps_to_llm_error() -> None:
    """httpx.TimeoutException (a network error) exhausts retries -> LlmError."""

    class _TimeoutTransport(httpx.BaseTransport):
        @override
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timed out")

    client = OpenAiCompatClient(
        "http://llm.test", "k", "m", "stub",
        backoff_schedule=(0.0, 0.0, 0.0),
    )
    client._http = httpx.Client(
        base_url=client.base_url, transport=_TimeoutTransport(), timeout=0.5,
    )
    with client, pytest.raises(LlmError, match="all 3 attempts failed"):
        client.complete("hello")


def test_openai_compat_non_recoverable_status_raises_immediately() -> None:
    """4xx (e.g. 401 bad key) is a permanent error: no retries at all."""
    transport = _StubTransport([httpx.Response(401, text="unauthorized")])
    with _client_with_transport(transport) as client, pytest.raises(
        LlmError, match="HTTP 401"
    ):
        client.complete("hello")
    assert len(transport.requests) == 1
