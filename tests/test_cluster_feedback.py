"""Cluster feedback tests: labels, split flag, hidden filter (T315).

One module-scoped embedded postgres for the read/write-layer assertions
(same pattern as test_sources_page.py), one fresh database per AppTest run.
Covers the DoD chain: «метки пишутся и фильтруются» — set_feedback /
set_split_flag write the migration-006 columns, the default cluster list
drops hidden clusters and include_hidden brings them back, and the page's
feedback buttons write through to the database.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import psycopg
import pytest
from psycopg import Connection
from streamlit.testing.v1 import AppTest

from idea_finder.core.canonical import canonical_url
from idea_finder.core.models import Pain, RawPost
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import (
    ensure_source,
    insert_pain,
    insert_raw_post,
    upsert_cluster,
    upsert_prompt_version,
)
from idea_finder.web.cluster_feedback_view import (
    FeedbackLabel,
    set_feedback,
    set_split_flag,
)
from idea_finder.web.clusters_view import list_clusters

ROOT = Path(__file__).resolve().parents[1]

_APP = str(ROOT / "idea_finder" / "web" / "streamlit_app.py")


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """Embedded postgres for the read/write-layer tests."""
    data_dir = tmp_path_factory.mktemp("pgfeedback") / "pg"
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
    data_dir = tmp_path_factory.mktemp("pgfeedbackapp") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture()
def app_pg(app_pg_server: PgHandle) -> Iterator[PgHandle]:
    """Fresh ``idea_finder`` database per script test (drop + recreate)."""
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


def _seed_cluster(
    conn: Connection,
    title: str,
    body: str,
) -> str:
    """Seed one cluster with a single attached pain; return the cluster id."""
    cluster_id = upsert_cluster(conn, title, 1, {"complaint": 1})
    source_id = ensure_source(conn, "fl_ru")
    post = RawPost(
        source_id=source_id,
        url_canon=canonical_url(f"https://fl.ru/p/{title.replace(' ', '-')}"),
        url=f"https://fl.ru/p/{title.replace(' ', '-')}",
        title=f"Post {title}",
        text=body,
        published_at=datetime(2026, 9, 12, tzinfo=UTC),
        kind="complaint",
    )
    inserted = insert_raw_post(conn, post, fetch_status="fetched")
    assert inserted is not None
    prompt_id = upsert_prompt_version(conn, "extract_pains", 1, "body", "file")
    insert_pain(
        conn,
        Pain(source_post_id=inserted, body=body, audience="частные пользователи", quote=body),
        None,
        prompt_id,
    )
    conn.execute(
        "UPDATE pain SET cluster_id = %s WHERE id IN (SELECT id FROM pain WHERE body = %s)",
        (cluster_id, body),
    )
    return cluster_id


def _feedback_of(conn: Connection, cluster_id: str) -> str | None:
    row = conn.execute(
        "SELECT feedback FROM cluster WHERE id = %s",
        (cluster_id,),
    ).fetchone()
    assert row is not None
    return None if row[0] is None else str(row[0])


def _split_of(conn: Connection, cluster_id: str) -> bool:
    row = conn.execute(
        "SELECT split_flag FROM cluster WHERE id = %s",
        (cluster_id,),
    ).fetchone()
    assert row is not None
    return bool(row[0])


def test_set_feedback_write_and_clear(conn: Connection) -> None:
    """set_feedback writes 'interesting'/'hidden' and clears back to NULL."""
    cluster_id = _seed_cluster(conn, "Кластер фидбек", "боль номер один")
    assert _feedback_of(conn, cluster_id) is None

    set_feedback(conn, cluster_id, "interesting")
    assert _feedback_of(conn, cluster_id) == "interesting"

    set_feedback(conn, cluster_id, "hidden")
    assert _feedback_of(conn, cluster_id) == "hidden"

    set_feedback(conn, cluster_id, None)
    assert _feedback_of(conn, cluster_id) is None


def test_set_feedback_rejects_unknown_label(conn: Connection) -> None:
    """Unknown labels fail fast client-side (DB CHECK is the backstop)."""
    cluster_id = _seed_cluster(conn, "Кластер бэд-лейбл", "боль номер два")
    bad = "meh" + str(id(conn))[:0]  # str, not a Literal — runtime-only value
    bad_label = cast(FeedbackLabel, str(bad))
    with pytest.raises(ValueError, match="unknown feedback label"):
        set_feedback(conn, cluster_id, bad_label)
    assert _feedback_of(conn, cluster_id) is None


def test_set_feedback_unknown_id_noop(conn: Connection) -> None:
    """Unknown cluster id matches no row: no error, no new rows."""
    count_before = conn.execute("SELECT count(*) FROM cluster").fetchone()
    set_feedback(conn, "00000000-0000-0000-0000-000000000000", "interesting")
    count_after = conn.execute("SELECT count(*) FROM cluster").fetchone()
    assert count_before == count_after


def test_set_split_flag_both_ways(conn: Connection) -> None:
    """set_split_flag flips «это не одна боль» in both directions."""
    cluster_id = _seed_cluster(conn, "Кластер сплит", "боль номер три")
    assert _split_of(conn, cluster_id) is False

    set_split_flag(conn, cluster_id, True)
    assert _split_of(conn, cluster_id) is True

    set_split_flag(conn, cluster_id, False)
    assert _split_of(conn, cluster_id) is False


def test_hidden_clusters_filtered_by_default(conn: Connection) -> None:
    """DoD «скрытые не видны без фильтра»: hidden drops out, filter returns."""
    kept = _seed_cluster(conn, "Видимый кластер", "боль видимая")
    hidden = _seed_cluster(conn, "Скрытый кластер", "боль скрытая")
    set_feedback(conn, hidden, "hidden")

    visible_ids = {row.id for row in list_clusters(conn)}
    assert kept in visible_ids
    assert hidden not in visible_ids

    all_ids = {row.id for row in list_clusters(conn, include_hidden=True)}
    assert {kept, hidden} <= all_ids

    # Clearing the label restores default visibility.
    set_feedback(conn, hidden, None)
    assert hidden in {row.id for row in list_clusters(conn)}


def test_db_check_backstops_bad_label(conn: Connection) -> None:
    """The 006 CHECK constraint rejects a bad label written directly."""
    cluster_id = _seed_cluster(conn, "Кластер чек", "боль под чеком")
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "UPDATE cluster SET feedback = 'nope' WHERE id = %s",
            (cluster_id,),
        )
    conn.rollback()


# ---------------------------------------------------------------------------
# Streamlit page (AppTest)
# ---------------------------------------------------------------------------


def _run_app(pg_handle: PgHandle, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Run streamlit_app under AppTest with PGDATA_DIR at the test cluster."""
    monkeypatch.setenv("PGDATA_DIR", str(pg_handle.data_dir))
    return AppTest.from_file(_APP, default_timeout=180).run()


def _goto(at: AppTest, page: str) -> AppTest:
    """Switch the sidebar page and rerun the script."""
    at.sidebar.radio[0].set_value(page).run()
    return at


def test_page_feedback_button_writes_db(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clicking «Интересно» on the card writes feedback='interesting'."""
    connection = app_pg.get_conn()
    try:
        cluster_id = _seed_cluster(connection, "Кластер кнопка", "боль кнопочная")
    finally:
        connection.close()

    at = _run_app(app_pg, monkeypatch)
    assert not at.exception
    at = _goto(at, "Кластеры")
    assert not at.exception

    buttons = {b.label: b for b in at.button}
    assert {"Интересно", "Скрыть", "Это не одна боль"} <= set(buttons)
    buttons["Интересно"].click().run()
    assert not at.exception

    connection = app_pg.get_conn()
    try:
        row = connection.execute(
            "SELECT feedback FROM cluster WHERE id = %s", (cluster_id,)
        ).fetchone()
    finally:
        connection.close()
    assert row is not None and row[0] == "interesting"
    # The page reflects the label after the rerun.
    captions = " ".join(el.value for el in at.caption)
    assert "интересно" in captions


def test_page_split_button_toggles_flag(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clicking «Это не одна боль» writes split_flag=true."""
    connection = app_pg.get_conn()
    try:
        cluster_id = _seed_cluster(connection, "Кластер не одна", "боли две на самом деле")
    finally:
        connection.close()

    at = _run_app(app_pg, monkeypatch)
    assert not at.exception
    at = _goto(at, "Кластеры")
    assert not at.exception

    buttons = {b.label: b for b in at.button}
    buttons["Это не одна боль"].click().run()
    assert not at.exception

    connection = app_pg.get_conn()
    try:
        row = connection.execute(
            "SELECT split_flag FROM cluster WHERE id = %s", (cluster_id,)
        ).fetchone()
    finally:
        connection.close()
    assert row is not None and row[0] is True


def test_page_hidden_toggle_hides_and_restores(
    app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DoD end-to-end: «Скрыть» on the page hides the cluster by default;
    the «Показывать скрытые» toggle brings it back."""
    connection = app_pg.get_conn()
    try:
        cluster_id = _seed_cluster(connection, "Кластер скрыть", "боль на скрытие")
    finally:
        connection.close()

    at = _run_app(app_pg, monkeypatch)
    assert not at.exception
    at = _goto(at, "Кластеры")
    assert not at.exception

    buttons = {b.label: b for b in at.button}
    buttons["Скрыть"].click().run()
    assert not at.exception

    # Hidden by default: the cluster is gone from the table (empty state).
    infos = " ".join(el.value for el in at.info)
    tables = " ".join(
        cell.get("content", "") if isinstance(cell, dict) else str(cell)
        for cell in [el.value for el in at.dataframe]
    )
    assert "Кластер скрыть" not in tables
    assert "Нет кластеров" in infos or "Кластер скрыть" not in "".join(
        el.value for el in at.markdown
    )

    # Flip «Показывать скрытые»: the cluster is visible again, marked hidden.
    at.toggle[0].set_value(True).run()
    assert not at.exception
    tables = str([el.value for el in at.dataframe])
    assert "Кластер скрыть" in tables
    captions = " ".join(el.value for el in at.caption)
    assert "скрыт" in captions

    # And the label really is 'hidden' in the database.
    connection = app_pg.get_conn()
    try:
        row = connection.execute(
            "SELECT feedback FROM cluster WHERE id = %s", (cluster_id,)
        ).fetchone()
    finally:
        connection.close()
    assert row is not None and row[0] == "hidden"
