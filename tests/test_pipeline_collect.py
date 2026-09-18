"""Collect stage tests: registry routing, isolation, idempotency (T335).

One embedded postgres cluster per module (same pattern as
``test_pipeline_cluster.py``). Real network is never touched: the
``pipeline._ADAPTERS`` registry is monkeypatched with fake factories, the
same seam the stage exposes to operators.

Test map (T335 DoD):
a. one fake enabled source -> posts inserted via ``repo`` with the source
   row id remapped, counters ``{collected: 2, skipped: 0, warnings: 0}``;
b. mixed success/failure: failing source bumps ``warnings`` only, the
   healthy source still delivers (isolation, gplay-degrade pattern);
c. idempotency: second run over unchanged feeds collects nothing
   (``collected=0``) and does not duplicate rows;
d. a ``source`` row without a registered adapter counts as a warning;
e. a run row is written with ``stages_json.collect == 'done'``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from psycopg import Connection

from idea_finder.core.models import RawPost
from idea_finder.core.pipeline import _ADAPTERS, run_collect
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import list_enabled_sources, table_counts
from idea_finder.sources.base import seed_sources


class _FakeAdapter:
    """In-test adapter returning two demand posts, no network."""

    name = "fake_source"

    async def fetch_new(self, since: datetime | None) -> list[RawPost]:
        del since
        return [
            RawPost(
                source_id=self.name,
                url_canon=f"https://example.com/pain-{n}",
                url=f"https://example.com/pain-{n}",
                title=f"Demand post {n}",
                text=f"Body of demand post {n}",
                published_at=datetime(2026, 9, n, tzinfo=UTC),
                kind="demand",
            )
            for n in (1, 2)
        ]


class _FailingAdapter:
    """In-test adapter whose feed fetch always raises."""

    name = "failing_source"

    async def fetch_new(self, since: datetime | None) -> list[RawPost]:
        del since
        raise RuntimeError("network unreachable")


class _HangingAdapter:
    """In-test adapter whose feed fetch never returns (pure coroutine)."""

    name = "hanging_source"

    async def fetch_new(self, since: datetime | None) -> list[RawPost]:
        del since
        await asyncio.sleep(120)
        return []


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """One embedded postgres cluster for the whole module."""
    data_dir = tmp_path_factory.mktemp("pgcluster-collect") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture()
def conn(pg: PgHandle) -> Iterator[Connection]:
    """Fresh migrated, seeded database for every test."""
    with pg.get_conn() as connection:
        apply_migrations(connection)
        seed_sources(connection)
        yield connection
        connection.execute(
            "TRUNCATE source, prompt_version, run, llm_provider, pain, raw_post, cluster CASCADE"
        )


@pytest.fixture()
def patch_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Route all three seeded names to fakes; no real adapter is built."""
    monkeypatch.setitem(_ADAPTERS, "fl_ru", lambda rate: _FakeAdapter())
    monkeypatch.setitem(_ADAPTERS, "habr", lambda rate: _FailingAdapter())

    def _never(rate: float) -> _FakeAdapter:
        del rate
        raise AssertionError("disabled-source adapter must not be constructed")

    monkeypatch.setitem(_ADAPTERS, "gplay", _never)
    yield


# --- success path -----------------------------------------------------------


def test_collect_inserts_posts_with_source_id_remap(conn: Connection, patch_registry: None) -> None:
    """Fake fl_ru feed lands in raw_post via repo, counters match DoD."""
    conn.execute("UPDATE source SET enabled = false WHERE name != 'fl_ru'")
    stats = run_collect(conn)
    assert stats == {"collected": 2, "skipped": 0, "warnings": 0}
    counts = table_counts(conn)
    assert counts["raw_post"] == 2
    row = conn.execute(
        """
        SELECT p.url_canon, s.name
        FROM raw_post p JOIN source s ON s.id = p.source_id
        ORDER BY p.url_canon
        """
    ).fetchall()
    assert [tuple(r) for r in row] == [
        ("https://example.com/pain-1", "fl_ru"),
        ("https://example.com/pain-2", "fl_ru"),
    ]


def test_collect_run_row_records_done_and_stats(conn: Connection, patch_registry: None) -> None:
    """The stage owns its run row: stats_json and stage status are written."""
    conn.execute("UPDATE source SET enabled = false WHERE name != 'fl_ru'")
    run_collect(conn)
    runs = conn.execute("SELECT stages_json, stats_json FROM run").fetchall()
    assert len(runs) == 1
    stages = runs[0][0] if isinstance(runs[0][0], dict) else json.loads(str(runs[0][0]))
    assert stages["collect"] == "done"
    stored = runs[0][1]
    if not isinstance(stored, dict):
        stored = json.loads(str(stored))
    assert stored["collected"] == 2
    assert json.dumps(stored)  # stats survive a JSON round trip


# --- per-source isolation ---------------------------------------------------


def test_failing_source_does_not_block_healthy_one(conn: Connection, patch_registry: None) -> None:
    """gplay-degrade pattern at stage level: warning, rest still delivers."""
    conn.execute("UPDATE source SET enabled = false WHERE name = 'gplay'")
    stats = run_collect(conn)
    assert stats == {"collected": 2, "skipped": 0, "warnings": 1}
    counts = table_counts(conn)
    assert counts["raw_post"] == 2


# --- idempotency ------------------------------------------------------------


def test_second_run_collects_nothing_new(conn: Connection, patch_registry: None) -> None:
    """Rerun over unchanged feeds: collected=0, no duplicated rows."""
    conn.execute("UPDATE source SET enabled = false WHERE name = 'gplay'")
    assert run_collect(conn)["collected"] == 2
    second = run_collect(conn)
    # The two feed items come back again but hit the url_canon UNIQUE
    # contract: recorded as skipped, raw_post stays at 2 rows.
    assert second == {"collected": 0, "skipped": 2, "warnings": 1}
    assert table_counts(conn)["raw_post"] == 2


# --- stage-level bound -------------------------------------------------------


def test_slow_source_cancelled_by_stage_budget(
    conn: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source hanging past SOURCE_FETCH_TIMEOUT_S becomes one warning.

    Proves the stage's own ``asyncio.wait_for`` (the isolation boundary
    the stage owns), with a fake adapter — no network involved.
    """
    monkeypatch.setitem(_ADAPTERS, "fl_ru", lambda rate: _HangingAdapter())
    conn.execute("UPDATE source SET enabled = false WHERE name != 'fl_ru'")
    monkeypatch.setattr("idea_finder.core.pipeline.SOURCE_FETCH_TIMEOUT_S", 1.0)
    t0 = time.monotonic()
    stats = run_collect(conn)
    elapsed = time.monotonic() - t0
    assert stats == {"collected": 0, "skipped": 0, "warnings": 1}
    # Far below the 300s default: the budget is the one under test.
    assert elapsed < 30


# --- registry mismatch ------------------------------------------------------


def test_unknown_source_counts_as_warning(
    conn: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An enabled row without an adapter warns and does not crash."""
    monkeypatch.delitem(_ADAPTERS, "fl_ru")
    conn.execute("INSERT INTO source (name) VALUES ('no_adapter_for_me')")
    conn.execute("UPDATE source SET enabled = false")
    conn.execute("UPDATE source SET enabled = true WHERE name = 'no_adapter_for_me'")
    stats = run_collect(conn)
    assert stats == {"collected": 0, "skipped": 0, "warnings": 1}
    assert table_counts(conn)["raw_post"] == 0


def test_disabled_sources_are_never_touched(conn: Connection) -> None:
    """The stage iterates only enabled rows (operator switch respected)."""
    conn.execute("UPDATE source SET enabled = false")
    stats = run_collect(conn)
    assert stats == {"collected": 0, "skipped": 0, "warnings": 0}
    assert list_enabled_sources(conn) == []
    assert table_counts(conn)["raw_post"] == 0
