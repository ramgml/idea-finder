"""Tests for the SourceAdapter contract and source seeding (task T301).

Two layers:

* Protocol contract (pure Python): a class with ``name`` and ``async
  fetch_new`` satisfies ``SourceAdapter``; an object missing either member
  does not. Uses a fake adapter returning two demand RawPosts.
* Database layer (module-scoped embedded postgres): migration 003 applies on
  a clean database and is a no-op on re-run, ``seed_sources`` is idempotent
  (UNIQUE name), and ``list_enabled_sources`` returns the seeded trio at
  the default 1 rps.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from psycopg import Connection

from idea_finder.core.models import RawPost
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import applied_versions, apply_migrations
from idea_finder.db.repo import list_enabled_sources
from idea_finder.sources.base import SEED_SOURCES, SourceAdapter, seed_sources


class _FakeAdapter:
    """In-test adapter: registry name plus two demand posts, no network."""

    name = "fake_source"

    async def fetch_new(self, since: datetime | None) -> list[RawPost]:
        return [
            RawPost(
                source_id=self.name,
                url_canon="https://example.com/pain-1",
                url="https://example.com/pain-1",
                title="Looking for a contractor",
                text="Need a cheap renovation crew this month",
                published_at=datetime(2026, 9, 1, tzinfo=UTC),
                kind="demand",
            ),
            RawPost(
                source_id=self.name,
                url_canon="https://example.com/pain-2",
                url="https://example.com/pain-2",
                title="Second demand post",
                text="No one delivers building materials on weekends",
                published_at=datetime(2026, 9, 2, tzinfo=UTC),
                kind="demand",
            ),
        ]


class _NoFetch:
    """Object with only a name: must NOT satisfy the protocol."""

    name = "broken"


class _NoName:
    """Object with only fetch_new: must NOT satisfy the protocol."""

    async def fetch_new(self, since: datetime | None) -> list[RawPost]:
        return []


def test_fake_adapter_satisfies_protocol() -> None:
    """A class with name + async fetch_new is a SourceAdapter instance."""
    adapter: SourceAdapter = _FakeAdapter()
    assert isinstance(_FakeAdapter(), SourceAdapter)
    assert adapter.name == "fake_source"


def test_broken_objects_do_not_satisfy_protocol() -> None:
    """Missing fetch_new or name disqualifies an object."""
    assert not isinstance(_NoFetch(), SourceAdapter)
    assert not isinstance(_NoName(), SourceAdapter)


def test_fetch_new_returns_two_demand_posts() -> None:
    """The fake adapter yields two RawPosts, both kind=demand.

    Driven synchronously with asyncio.run — the same bridge the B4 collect
    stage will use, since adapters are async per the fetch-layer stack.
    """
    posts = asyncio.run(_FakeAdapter().fetch_new(None))
    assert len(posts) == 2
    assert all(post.kind == "demand" for post in posts)


def test_seed_registry_matches_assignment() -> None:
    """Seeded trio and default rates are pinned by the assignment."""
    assert SEED_SOURCES == (("fl_ru", 1.0), ("habr", 1.0), ("gplay", 1.0))


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """One embedded postgres cluster for the whole module."""
    data_dir = tmp_path_factory.mktemp("pgsources") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture(scope="module")
def conn(pg: PgHandle) -> Iterator[Connection]:
    """Connection to the migrated application database."""
    with pg.get_conn() as connection:
        apply_migrations(connection)
        yield connection


def test_migration_003_applies_on_clean_database(pg: PgHandle) -> None:
    """Fresh database gets exactly versions 1, 2, 3, 4, 5."""
    with pg.get_conn() as fresh:
        assert apply_migrations(fresh) == [1, 2, 3, 4, 5, 6, 7]
        assert applied_versions(fresh) == {1, 2, 3, 4, 5, 6, 7}
        columns = {
            row[0]
            for row in fresh.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'source'
                """
            ).fetchall()
        }
        assert {"enabled", "rate_limit_rps"} <= columns


def test_migration_reapply_is_noop(conn: Connection) -> None:
    """Re-running migrations changes nothing once everything is applied."""
    assert apply_migrations(conn) == []
    assert applied_versions(conn) == {1, 2, 3, 4, 5, 6, 7}


def test_seed_sources_idempotent_and_default_rate(conn: Connection) -> None:
    """Double seed leaves exactly one row per seeded name, at 1 rps."""
    seed_sources(conn)
    seed_sources(conn)
    enabled = list_enabled_sources(conn)
    assert enabled == [("fl_ru", 1.0), ("gplay", 1.0), ("habr", 1.0)]
    count = conn.execute("SELECT count(*) FROM source").fetchone()
    assert count is not None and count[0] == len(SEED_SOURCES)


def test_seed_sources_preserves_operator_tuning(conn: Connection) -> None:
    """Re-seeding must not re-enable or re-rate an already-present row."""
    with conn.transaction():
        conn.execute(
            "UPDATE source SET enabled = false, rate_limit_rps = 0.5 WHERE name = 'fl_ru'"
        )
    seed_sources(conn)
    enabled = dict(list_enabled_sources(conn))
    assert "fl_ru" not in enabled
    all_rows = conn.execute(
        "SELECT rate_limit_rps FROM source WHERE name = 'fl_ru'"
    ).fetchone()
    assert all_rows is not None and float(all_rows[0]) == 0.5
    # Restore for any later test relying on the seeded state.
    with conn.transaction():
        conn.execute(
            "UPDATE source SET enabled = true, rate_limit_rps = 1.0 WHERE name = 'fl_ru'"
        )


def test_list_enabled_sources_skips_disabled(conn: Connection) -> None:
    """A disabled source disappears from the collect-stage view."""
    with conn.transaction():
        conn.execute("UPDATE source SET enabled = false WHERE name = 'gplay'")
    enabled = [name for name, _rate in list_enabled_sources(conn)]
    assert "gplay" not in enabled
    # Restore the seeded state.
    with conn.transaction():
        conn.execute("UPDATE source SET enabled = true WHERE name = 'gplay'")
