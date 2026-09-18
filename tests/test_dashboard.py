"""Clusters-page tests: read layer + Streamlit script contract (task 311).

Two module-scoped embedded postgres clusters (same pattern as test_repo.py):
one for the read-layer tests, a fresh one for the AppTest script runs so the
empty-state assertion is not polluted by earlier seeds. The script runs
in-process under Streamlit's AppTest with PGDATA_DIR pointed at the test
cluster (bootstrap_pgserver.default_data_dir resolves it per call).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from psycopg import Connection
from streamlit.testing.v1 import AppTest

from idea_finder.core.canonical import canonical_url
from idea_finder.core.models import Pain, PostKind, RawPost, Score
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import (
    ensure_source,
    insert_pain,
    insert_raw_post,
    insert_score,
    upsert_cluster,
    upsert_prompt_version,
)
from idea_finder.web.clusters_view import ClusterRow, count_clusters, list_clusters

ROOT = Path(__file__).resolve().parents[1]

_APP = str(ROOT / "idea_finder" / "web" / "streamlit_app.py")


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """Embedded postgres for the read-layer tests."""
    data_dir = tmp_path_factory.mktemp("pgweb") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture(scope="module")
def conn(pg: PgHandle) -> Iterator[Connection]:
    """Connection to the migrated application database."""
    with pg.get_conn() as connection:
        apply_migrations(connection)
        yield connection


@pytest.fixture(scope="module")
def app_pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """Fresh embedded postgres for dashboard script tests (own empty state)."""
    data_dir = tmp_path_factory.mktemp("pgwebapp") / "pg"
    handle = ensure_pgserver(data_dir)
    with handle.get_conn() as connection:
        apply_migrations(connection)
    yield handle
    handle.stop()


def _post(
    conn: Connection, url: str, source_name: str = "test-source", kind: PostKind = "complaint"
) -> str:
    """Insert a raw post, return its id."""
    post = RawPost(
        source_id=ensure_source(conn, source_name),
        url_canon=canonical_url(url),
        url=url,
        title="Post",
        text="Приложение постоянно падает при открытии профиля",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        kind=kind,
    )
    inserted = insert_raw_post(conn, post)
    assert inserted is not None
    return inserted


def _pain(conn: Connection, post_id: str, prompt_id: str) -> str:
    """Insert one pain, return its id."""
    pain = Pain(
        source_post_id=post_id,
        body="падает при открытии профиля",
        audience="пользователи",
        quote="постоянно падает",
    )
    return insert_pain(conn, pain, None, prompt_id)


def _seed(
    conn: Connection,
    title: str,
    size: int,
    total: float | None,
    *,
    source_name: str = "test-source",
) -> str:
    """Seed one cluster (+optional score, +pains) with a fresh post per pain."""
    cluster_id = upsert_cluster(conn, title, size, {"demand": size})
    if total is not None:
        insert_score(
            conn,
            Score(cluster_id=cluster_id, total=total, rationale_md="r", quotes=[]),
        )
    prompt_id = upsert_prompt_version(conn, "extract", 1, "body", "file")
    for n in range(size):
        post_id = _post(
            conn,
            f"https://example.com/p/{title}-{n}",
            source_name,
        )
        pain_id = _pain(conn, post_id, prompt_id)
        # Post-cluster state: pains belong to the cluster (pipeline assigns
        # this; raw UPDATE here because repo has no reassignment API yet).
        # Explicit commit: each repo call here runs as an implicit
        # transaction; a leftover open one would be rolled back by close().
        with conn.transaction():
            conn.execute("UPDATE pain SET cluster_id = %s WHERE id = %s", (cluster_id, pain_id))
    return cluster_id


def test_list_clusters_empty(conn: Connection) -> None:
    """Fresh schema: no clusters; count is zero and the list is empty."""
    assert count_clusters(conn) == 0
    assert list_clusters(conn) == []


def test_list_clusters_sorted_by_score(conn: Connection) -> None:
    """Order follows score desc; unscored clusters come after scored ones."""
    _seed(conn, "low", 1, 3.5)
    _seed(conn, "high", 3, 9.1)
    _seed(conn, "unscored", 2, None)

    rows = list_clusters(conn)
    assert [row.title for row in rows] == ["high", "low", "unscored"]
    assert [row.score for row in rows] == [9.1, 3.5, None]


def test_list_clusters_row_fields(conn: Connection) -> None:
    """Row carries size, kind mix, score, and distinct source names."""
    _seed(conn, "full", 2, 7.0, source_name="site-a")
    _seed(conn, "full-b", 1, 2.0, source_name="site-b")

    by_title = {row.title: row for row in list_clusters(conn)}
    top = by_title["full"]
    assert isinstance(top, ClusterRow)
    assert top.size == 2
    assert top.kinds == {"demand": 2}
    assert top.score == 7.0
    assert top.sources == ("site-a",)


def _run_app(pg_handle: PgHandle, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Run streamlit_app under AppTest against the given cluster."""
    monkeypatch.setenv("PGDATA_DIR", str(pg_handle.data_dir))
    return AppTest.from_file(_APP, default_timeout=180).run()


def test_dashboard_script_empty_state(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """streamlit_app under AppTest on a fresh schema: empty state, no error."""
    at = _run_app(app_pg, monkeypatch)
    assert not at.exception
    texts = [element.value for element in at.info]
    assert any("Нет кластеров" in text for text in texts)
    assert at.sidebar.radio[0].value == "Кластеры"


def test_dashboard_script_with_data(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """With seeded clusters the dataframe renders, still exception-free."""
    connection = app_pg.get_conn()
    try:
        _seed(connection, "dash-high", 2, 8.8)
        _seed(connection, "dash-low", 1, 1.2)
    finally:
        connection.close()

    at = _run_app(app_pg, monkeypatch)
    assert not at.exception
    dataframes = at.dataframe
    assert len(dataframes) == 1
    frame = dataframes[0].value
    assert list(frame["title"]) == ["dash-high", "dash-low"]
    assert list(frame["score"]) == [8.8, 1.2]
