"""Cluster card + filters tests: read layer and Streamlit page (task 312).

One module-scoped embedded postgres for read-layer assertions (same pattern
as test_repo.py / test_dashboard.py), one fresh cluster for the AppTest runs.
The seed helper mirrors test_dashboard._seed but adds quotes/rationale (the
score rows the card renders) and per-pain kind/source control.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import psycopg
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
from idea_finder.web.clusters_view import (
    ClusterDetail,
    ClusterFilters,
    get_cluster_detail,
    list_clusters_filtered,
    list_source_names,
)

ROOT = Path(__file__).resolve().parents[1]

_APP = str(ROOT / "idea_finder" / "web" / "streamlit_app.py")


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """Embedded postgres for the read-layer tests."""
    data_dir = tmp_path_factory.mktemp("pgcard") / "pg"
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
def app_pg_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """Shared pg server for dashboard script tests."""
    data_dir = tmp_path_factory.mktemp("pgcardapp") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture()
def app_pg(app_pg_server: PgHandle) -> Iterator[PgHandle]:
    """Fresh ``idea_finder`` database per script test.

    The app resolves its database by name (bootstrap_pgserver), so isolation
    comes from dropping and recreating that database between tests; the
    script test seeds and the app then see the same empty schema.
    """
    with psycopg.connect(
        app_pg_server._server.get_uri("postgres"),
        autocommit=True,
    ) as admin:
        admin.execute("DROP DATABASE IF EXISTS idea_finder WITH (FORCE)")
        admin.execute("CREATE DATABASE idea_finder")
    handle = PgHandle(app_pg_server._server, "idea_finder")
    with handle.get_conn() as connection:
        apply_migrations(connection)
    yield handle


def _post(
    conn: Connection,
    url: str,
    source_name: str = "test-source",
    kind: PostKind = "complaint",
    published_at: datetime | None = None,
) -> str:
    """Insert a raw post, return its id."""
    post = RawPost(
        source_id=ensure_source(conn, source_name),
        url_canon=canonical_url(url),
        url=url,
        title=f"Post {url.rsplit('/', 1)[-1]}",
        text="Приложение постоянно падает при открытии профиля",
        published_at=published_at or datetime(2026, 1, 1, tzinfo=UTC),
        kind=kind,
    )
    inserted = insert_raw_post(conn, post)
    assert inserted is not None
    return inserted


def _seed(
    conn: Connection,
    title: str,
    *,
    total: float | None = None,
    kinds: dict[str, int] | None = None,
    pains: list[tuple[str, str]] | None = None,
    created_at: datetime | None = None,
    rationale: str = "Спрос подтверждён заказами.",
    quotes: list[str] | None = None,
) -> str:
    """Seed one cluster with score and pains; return the cluster id.

    ``pains`` entries are ``(source_name, kind)`` pairs; each gets a fresh
    post so source/kind filters have real rows to match.
    """
    kind_mix = kinds if kinds is not None else {"complaint": 1}
    cluster_id = upsert_cluster(conn, title, sum(kind_mix.values()), kind_mix)
    if created_at is not None:
        with conn.transaction():
            conn.execute(
                "UPDATE cluster SET created_at = %s WHERE id = %s",
                (created_at, cluster_id),
            )
    if total is not None:
        insert_score(
            conn,
            Score(
                cluster_id=cluster_id,
                total=total,
                rationale_md=rationale,
                quotes=quotes if quotes is not None else ["цитата раз"],
            ),
        )
    prompt_id = upsert_prompt_version(conn, "extract", 1, "body", "file")
    for n, (source_name, kind) in enumerate(pains or [("test-source", "complaint")]):
        kind = cast(PostKind, kind)
        post_id = _post(conn, f"https://example.com/p/{title}-{n}", source_name, kind)
        pain_id = insert_pain(
            conn,
            Pain(
                source_post_id=post_id,
                body=f"Боль: {title} #{n}",
                audience="пользователи",
                quote="постоянно падает",
            ),
            None,
            prompt_id,
        )
        # Post-cluster state: pipeline assigns pains; repo has no API yet.
        with conn.transaction():
            conn.execute("UPDATE pain SET cluster_id = %s WHERE id = %s", (cluster_id, pain_id))
    return cluster_id


@pytest.fixture(scope="module")
def seeded(conn: Connection) -> None:
    """Shared read-layer dataset: five clusters across filters."""
    _seed(
        conn,
        "high-dem",
        total=9.0,
        kinds={"demand": 2},
        pains=[("fl.ru", "demand"), ("habr", "demand")],
    )
    _seed(conn, "mid-complaint", total=5.0, kinds={"complaint": 1}, pains=[("fl.ru", "complaint")])
    _seed(
        conn,
        "old-low",
        total=1.5,
        kinds={"discussion": 1},
        pains=[("habr", "discussion")],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    _seed(conn, "unscored", kinds={"complaint": 1}, pains=[("gplay", "complaint")])
    _seed(
        conn,
        "recent-mid",
        total=4.0,
        kinds={"demand": 1},
        pains=[("gplay", "demand")],
        created_at=datetime.now(UTC) - timedelta(hours=1),
    )


def _today() -> date:
    """Today in the local zone (DTZ011-safe wrapper for date filters)."""
    return datetime.now(UTC).date()


def test_filter_by_kind(conn: Connection, seeded: None) -> None:
    """Kind filter passes clusters whose kind_mix intersects the set."""
    rows = list_clusters_filtered(conn, ClusterFilters(kinds=frozenset({"demand"})))
    assert [row.title for row in rows] == ["high-dem", "recent-mid"]

    rows = list_clusters_filtered(
        conn, ClusterFilters(kinds=frozenset({"complaint", "discussion"}))
    )
    # Score DESC then NULLS LAST: mid-complaint (5.0), old-low (1.5),
    # unscored (no score row at all).
    assert [row.title for row in rows] == ["mid-complaint", "old-low", "unscored"]


def test_filter_by_source(conn: Connection, seeded: None) -> None:
    """Source filter follows the cluster's pains, not the cluster itself."""
    rows = list_clusters_filtered(conn, ClusterFilters(sources=frozenset({"gplay"})))
    # Score DESC then NULLS LAST: recent-mid (4.0), unscored (None).
    assert [row.title for row in rows] == ["recent-mid", "unscored"]


def test_filter_min_score_excludes_unscored(conn: Connection, seeded: None) -> None:
    """Score floor drops unscored clusters regardless of their size."""
    rows = list_clusters_filtered(conn, ClusterFilters(min_score=4.0))
    assert [row.title for row in rows] == ["high-dem", "mid-complaint", "recent-mid"]

    rows = list_clusters_filtered(conn, ClusterFilters(min_score=8.0))
    assert [row.title for row in rows] == ["high-dem"]


def test_filter_by_date_range(conn: Connection, seeded: None) -> None:
    """Date range bounds created_at inclusively on both ends."""
    today = _today()
    rows = list_clusters_filtered(conn, ClusterFilters(from_date=today, to_date=today))
    # Created today: three scored (9.0, 5.0, 4.0) plus unscored (NULLS LAST);
    # old-low is pinned to 2026-01-01 and must be excluded.
    assert [row.title for row in rows] == [
        "high-dem",
        "mid-complaint",
        "recent-mid",
        "unscored",
    ]

    old_date = date(2026, 1, 1)
    rows = list_clusters_filtered(conn, ClusterFilters(from_date=old_date, to_date=old_date))
    assert [row.title for row in rows] == ["old-low"]


def test_filters_combine_with_and(conn: Connection, seeded: None) -> None:
    """A cluster must satisfy every set filter to stay visible."""
    rows = list_clusters_filtered(
        conn,
        ClusterFilters(
            kinds=frozenset({"demand"}),
            sources=frozenset({"gplay"}),
            min_score=3.0,
            from_date=_today(),
        ),
    )
    assert [row.title for row in rows] == ["recent-mid"]

    rows = list_clusters_filtered(
        conn,
        ClusterFilters(kinds=frozenset({"demand"}), sources=frozenset({"habr"})),
    )
    assert [row.title for row in rows] == ["high-dem"]


def test_empty_filters_return_everything(conn: Connection, seeded: None) -> None:
    """All-None filters behave exactly like the unfiltered listing."""
    rows = list_clusters_filtered(conn, ClusterFilters())
    # Score DESC, NULLS LAST: 9.0, 5.0, 4.0, 1.5, then unscored.
    assert [row.title for row in rows] == [
        "high-dem",
        "mid-complaint",
        "recent-mid",
        "old-low",
        "unscored",
    ]


def test_list_source_names_only_with_pains(conn: Connection, seeded: None) -> None:
    """Dropdown lists sources that actually have pains attached."""
    assert list_source_names(conn) == ["fl.ru", "gplay", "habr"]


def test_cluster_detail_full_payload(conn: Connection, seeded: None) -> None:
    """Card payload: score block, pains with source refs, distinct sources."""
    high = next(
        row for row in list_clusters_filtered(conn, ClusterFilters()) if row.title == "high-dem"
    )
    detail = get_cluster_detail(conn, high.id)
    assert detail is not None
    assert isinstance(detail, ClusterDetail)
    assert detail.score == 9.0
    assert detail.rationale_md == "Спрос подтверждён заказами."
    assert detail.quotes == ("цитата раз",)
    assert detail.kinds == {"demand": 2}
    assert detail.sources == ("fl.ru", "habr")
    assert [pain.source_name for pain in detail.pains] == ["fl.ru", "habr"]
    assert [pain.kind for pain in detail.pains] == ["demand", "demand"]
    assert detail.pains[0].post_url.startswith("https://example.com/p/high-dem")
    assert detail.pains[0].quote == "постоянно падает"


def test_cluster_detail_unknown_id(conn: Connection, seeded: None) -> None:
    """Unknown id: None (the page shows a warning, not a traceback)."""
    assert get_cluster_detail(conn, "00000000-0000-0000-0000-000000000000") is None


def test_dashboard_card_flow(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full page flow: table renders, card selectbox shows cluster labels."""
    connection = app_pg.get_conn()
    try:
        _seed(
            connection,
            "card-high",
            total=8.0,
            kinds={"demand": 1},
            pains=[("fl.ru", "demand")],
            rationale="Рубрика подтверждает.",
            quotes=["q1", "q2"],
        )
        _seed(
            connection,
            "card-low",
            total=2.0,
            kinds={"complaint": 1},
            pains=[("gplay", "complaint")],
        )
    finally:
        connection.close()

    monkeypatch.setenv("PGDATA_DIR", str(app_pg.data_dir))
    at = AppTest.from_file(_APP, default_timeout=180)
    at.run()
    assert not at.exception

    # Table: both clusters, score order.
    frame = at.dataframe[0].value
    assert list(frame["title"]) == ["card-high", "card-low"]

    # Card selectbox defaults to the top cluster's label.
    boxes = at.selectbox
    assert len(boxes) == 1
    top_label = boxes[0].options[0]
    assert "card-high" in top_label

    # Card body: rationale markdown is on the page.
    markdown_texts = [el.value for el in at.markdown]
    assert any("Рубрика подтверждает." in text for text in markdown_texts)


def test_dashboard_filters_flow(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting filters reruns the page and shrinks the table."""
    connection = app_pg.get_conn()
    try:
        _seed(
            connection,
            "filt-a",
            total=7.0,
            kinds={"demand": 1},
            pains=[("fl.ru", "demand")],
        )
        _seed(
            connection,
            "filt-b",
            total=3.0,
            kinds={"complaint": 1},
            pains=[("gplay", "complaint")],
        )
    finally:
        connection.close()

    monkeypatch.setenv("PGDATA_DIR", str(app_pg.data_dir))
    at = AppTest.from_file(_APP, default_timeout=180)
    at.run()
    assert not at.exception

    # No filters: both visible.
    assert list(at.dataframe[0].value["title"]) == ["filt-a", "filt-b"]

    # Kind filter = demand: only filt-a remains.
    at.multiselect[0].set_value(["demand"]).run()
    assert not at.exception
    assert list(at.dataframe[0].value["title"]) == ["filt-a"]

    # Score floor 5.0 keeps filt-a (7.0); combined with kind filter still one.
    at.slider[0].set_value(5.0).run()
    assert not at.exception
    assert list(at.dataframe[0].value["title"]) == ["filt-a"]

    # Lower the floor to 1.0 with kind=demand: still filt-a only.
    at.slider[0].set_value(1.0).run()
    assert not at.exception
    assert list(at.dataframe[0].value["title"]) == ["filt-a"]
