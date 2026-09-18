"""Repository and migration tests: schema v1, idempotency, repo API.

A single module-scoped embedded postgres cluster serves all tests; migrations
apply to it once, and every test exercises the repo contract against real
Postgres (unix socket only).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from psycopg import Connection

from idea_finder.core.canonical import canonical_url
from idea_finder.core.models import Pain, RawPost, Score
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import applied_versions, apply_migrations
from idea_finder.db.repo import (
    create_run,
    ensure_source,
    finish_run,
    get_active_llm_provider,
    get_run,
    insert_pain,
    insert_raw_post,
    insert_score,
    table_counts,
    update_cluster_stats,
    update_run_stage,
    upsert_cluster,
    upsert_llm_provider,
    upsert_prompt_version,
)


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """One embedded postgres cluster for the whole module."""
    data_dir = tmp_path_factory.mktemp("pgrepo") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture(scope="module")
def conn(pg: PgHandle) -> Iterator[Connection]:
    """Connection to the migrated application database."""
    with pg.get_conn() as connection:
        apply_migrations(connection)
        yield connection


def _make_post(source_id: str, url: str) -> RawPost:
    """Build a RawPost domain object for tests."""
    return RawPost(
        source_id=source_id,
        url_canon=canonical_url(url),
        url=url,
        title="Post",
        text="Body text mentioning a cheap plumber",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        kind="demand",
    )


def test_migrations_apply_to_clean_database(pg: PgHandle) -> None:
    """All migrations on a fresh database: schema objects + versions exist."""
    with pg.get_conn() as fresh:
        newly = apply_migrations(fresh)
        assert newly == [1, 2, 3, 4, 5, 6]
        tables = {
            row[0]
            for row in fresh.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                """
            ).fetchall()
        }
        expected = {
            "schema_migrations", "source", "raw_post", "pain", "cluster",
            "score", "run", "prompt_version", "llm_provider",
        }
        assert expected <= tables
        ext = fresh.execute(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        assert ext is not None


def test_migrations_reapply_is_noop(conn: Connection) -> None:
    """Re-running migrations on an already-migrated database changes nothing."""
    assert apply_migrations(conn) == []
    assert applied_versions(conn) == {1, 2, 3, 4, 5, 6}


def test_insert_raw_post_deduplicates_on_url_canon(conn: Connection) -> None:
    """Second insert with the same url_canon returns None and adds no row."""
    source_id = ensure_source(conn, "dedup-source")  
    post = _make_post(source_id, "https://example.com/dup?utm_campaign=x")
    first = insert_raw_post(conn, post)  
    assert first is not None
    before = conn.execute(  
        "SELECT count(*) FROM raw_post WHERE url_canon = %s", (post.url_canon,)
    ).fetchone()
    assert before is not None and before[0] == 1
    again = insert_raw_post(conn, post)  
    assert again is None
    after = conn.execute(  
        "SELECT count(*) FROM raw_post WHERE url_canon = %s", (post.url_canon,)
    ).fetchone()
    assert after is not None and after[0] == 1


def test_full_pipeline_cycle(conn: Connection) -> None:
    """source -> raw_post -> pain(+embedding) -> cluster -> score -> run."""
    connection = conn  
    source_id = ensure_source(connection, "cycle-source")
    post = _make_post(source_id, "https://example.com/cycle")
    post_id = insert_raw_post(connection, post)
    assert post_id is not None
    prompt_id = upsert_prompt_version(
        connection, "extract_pains", 1, "Extract pains", "file"
    )
    pain = Pain(
        source_post_id=post_id,
        body="Cannot find a cheap plumber",
        audience="renters",
        quote="cheap plumber",
    )
    pain_id = insert_pain(connection, pain, [0.01 * (i % 5) for i in range(384)],
                          prompt_id)
    assert pain_id
    cluster_id = upsert_cluster(connection, "Plumbing", 1, {"demand": 1})
    connection.execute(  
        "UPDATE pain SET cluster_id = %s WHERE id = %s", (cluster_id, pain_id)
    )
    update_cluster_stats(connection, cluster_id, 1, {"demand": 1})
    score_id = insert_score(
        connection,
        Score(cluster_id=cluster_id, total=6.5, rationale_md="decent",
              quotes=["cheap plumber"]),
    )
    assert score_id
    run_id = create_run(connection, {"collect": "running"})
    update_run_stage(connection, run_id, "collect", "done", {"bricked": 1}, 0.25)
    finish_run(connection, run_id)
    row = connection.execute(  
        """
        SELECT r.stages_json, r.stats_json, r.cost, r.finished_at,
               p.embedding IS NOT NULL, s.quotes_json
        FROM run r
        CROSS JOIN pain p
        CROSS JOIN score s
        WHERE r.id = %s AND p.id = %s AND s.id = %s
        """,
        (run_id, pain_id, score_id),
    ).fetchone()
    assert row is not None
    stages, stats, cost, finished_at, has_embedding, quotes = row
    assert stages == {"collect": "done"}
    assert stats == {"bricked": 1}
    assert float(cost) == 0.25
    assert finished_at is not None
    assert has_embedding is True
    assert list(quotes) == ["cheap plumber"]


def test_run_domain_roundtrip(conn: Connection) -> None:
    """get_run returns a domain Run reflecting the staged updates."""
    connection = conn  
    run_id = create_run(connection)
    update_run_stage(connection, run_id, "extract", "done", {"bricked": 2})
    update_run_stage(connection, run_id, "extract", "done", {"bricked": 1}, 0.5)
    run = get_run(connection, run_id)
    assert run is not None
    assert run.id == run_id
    assert run.stages == {"extract": "done"}
    assert run.stats == {"bricked": 3}  # deltas accumulate
    assert run.cost == 0.5


def test_table_counts_reflect_inserts(conn: Connection) -> None:
    """table_counts counts rows per table after inserts."""
    connection = conn  
    source_id = ensure_source(connection, "counts-source")
    insert_raw_post(connection, _make_post(source_id, "https://example.com/c1"))
    insert_raw_post(connection, _make_post(source_id, "https://example.com/c2"))
    counts = table_counts(connection)
    assert counts["source"] >= 3
    assert counts["raw_post"] >= 4
    assert set(counts) == {
        "source", "raw_post", "pain", "cluster", "score", "run",
        "prompt_version", "llm_provider",
    }


def test_prompt_version_unique_name_version(conn: Connection) -> None:
    """(name, version) is unique; re-upsert returns the same id."""
    connection = conn  
    first = upsert_prompt_version(connection, "score_cluster", 1, "A", "file")
    second = upsert_prompt_version(connection, "score_cluster", 1, "A", "file")
    third = upsert_prompt_version(connection, "score_cluster", 2, "B", "file")
    assert first == second
    assert third != first


def test_single_active_llm_provider(conn: Connection) -> None:
    """Exactly one provider is active; activation flips the previous one."""
    connection = conn  
    upsert_llm_provider(connection, "deepseek", "openai_compat",
                        "https://api.deepseek.com", "deepseek-chat", "k1",
                        is_active=True)
    upsert_llm_provider(connection, "glm", "openai_compat",
                        "https://open.bigmodel.cn", "glm-4", "k2",
                        is_active=True)
    upsert_llm_provider(connection, "mock", "fake", "", "mock", "",
                        is_active=True)
    active = get_active_llm_provider(connection)
    assert active is not None
    assert active.name == "mock"
    rows = connection.execute(  
        "SELECT count(*) FROM llm_provider WHERE is_active"
    ).fetchone()
    assert rows is not None and rows[0] == 1
    assert active.api_key == ""  # stored in db only, returned as-is here
