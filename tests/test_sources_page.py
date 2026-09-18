"""Sources page tests: registry read layer, toggle write, Streamlit page (T316).

One module-scoped embedded postgres for the read/write-layer assertions
(same pattern as test_cluster_card.py / test_run_health.py), one fresh
database per AppTest run. Covers the DoD chain: per-source counters on
seeded data, ``set_source_enabled`` flips the DB column both ways, the
page renders the registry with toggles and honours a disabled source.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

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
    list_enabled_sources,
    upsert_prompt_version,
)
from idea_finder.web.sources_view import SourceRow, list_sources, set_source_enabled

ROOT = Path(__file__).resolve().parents[1]

_APP = str(ROOT / "idea_finder" / "web" / "streamlit_app.py")


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """Embedded postgres for the read/write-layer tests."""
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


@pytest.fixture(scope="module")
def app_pg_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """Shared pg server for dashboard script tests."""
    data_dir = tmp_path_factory.mktemp("pgsourcesapp") / "pg"
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


def _post(
    conn: Connection,
    url: str,
    source_name: str,
    *,
    fetch_status: str = "fetched",
    created_at: datetime | None = None,
) -> str:
    """Insert one raw post; return its id."""
    post = RawPost(
        source_id=ensure_source(conn, source_name),
        url_canon=canonical_url(url),
        url=url,
        title="Post",
        text="Приложение постоянно падает при открытии профиля",
        published_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
        kind="complaint",
    )
    inserted = insert_raw_post(conn, post, fetch_status=fetch_status)
    assert inserted is not None
    if created_at is not None:
        # raw_post.created_at is a DB now() default; pin it for deterministic
        # last_ok assertions.
        with conn.transaction():
            conn.execute(
                "UPDATE raw_post SET created_at = %s WHERE id = %s", (created_at, inserted)
            )
    return inserted


def _pain(conn: Connection, post_id: str, body: str) -> None:
    """Insert one accepted pain for ``post_id``."""
    prompt_id = upsert_prompt_version(conn, "extract_pains", 1, "body", "file")
    insert_pain(
        conn,
        Pain(source_post_id=post_id, body=body, audience="частные пользователи", quote=body),
        None,
        prompt_id,
    )


@pytest.fixture(scope="module")
def seeded(conn: Connection) -> None:
    """Shared registry dataset: three sources with distinct health."""

    _post(conn, "https://fl.ru/p/1", "fl_ru", created_at=datetime(2026, 9, 10, tzinfo=UTC))
    ok_post = _post(
        conn, "https://fl.ru/p/2", "fl_ru", created_at=datetime(2026, 9, 12, tzinfo=UTC)
    )
    failed = _post(
        conn,
        "https://fl.ru/p/3",
        "fl_ru",
        fetch_status="failed",
        created_at=datetime(2026, 9, 13, tzinfo=UTC),
    )
    _pain(conn, ok_post, "не работает поиск по заказам")
    _pain(conn, failed, "брошенная заявка")

    gplay_post = _post(
        conn, "https://play/p/1", "gplay", created_at=datetime(2026, 9, 11, tzinfo=UTC)
    )
    _pain(conn, gplay_post, "приложение падает")

    # habr: enabled but no posts at all — zeros row.
    ensure_source(conn, "habr")


def test_list_sources_registry_rows(conn: Connection, seeded: None) -> None:
    """Registry lists every source with counters; disabled stays visible."""
    rows = {row.name: row for row in list_sources(conn)}
    assert sorted(rows) == ["fl_ru", "gplay", "habr"]
    assert all(isinstance(row, SourceRow) and row.enabled for row in rows.values())

    fl = rows["fl_ru"]
    # last_ok skips the failed post (2026-09-13): freshest ok is 2026-09-12.
    assert fl.last_ok == datetime(2026, 9, 12, tzinfo=UTC)
    assert fl.failed_count == 1
    assert fl.post_count == 3
    assert fl.pain_count == 2

    gplay = rows["gplay"]
    assert gplay.last_ok == datetime(2026, 9, 11, tzinfo=UTC)
    assert gplay.failed_count == 0
    assert gplay.post_count == 1
    assert gplay.pain_count == 1

    habr = rows["habr"]
    assert habr.last_ok is None
    assert habr.post_count == 0
    assert habr.pain_count == 0
    assert habr.failed_count == 0


def test_set_source_enabled_disables_and_restores(conn: Connection, seeded: None) -> None:
    """The toggle helper flips source.enabled in both directions."""
    assert set_source_enabled(conn, "gplay", False) is None
    row = next(row for row in list_sources(conn) if row.name == "gplay")
    assert row.enabled is False
    # The collect-stage view (repo.list_enabled_sources) must drop it.
    assert "gplay" not in [name for name, _rate in list_enabled_sources(conn)]

    set_source_enabled(conn, "gplay", True)
    row = next(row for row in list_sources(conn) if row.name == "gplay")
    assert row.enabled is True
    assert "gplay" in [name for name, _rate in list_enabled_sources(conn)]


def test_set_source_enabled_unknown_name_noop(conn: Connection, seeded: None) -> None:
    """Unknown source name: UPDATE matches nothing, no error, no new rows."""
    count_before = conn.execute("SELECT count(*) FROM source").fetchone()
    set_source_enabled(conn, "no-such-source", True)
    count_after = conn.execute("SELECT count(*) FROM source").fetchone()
    assert count_before == count_after


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


def test_sources_page_empty_state(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh database: the page renders the no-sources empty state."""
    at = _run_app(app_pg, monkeypatch)
    assert not at.exception
    at = _goto(at, "Источники")
    assert not at.exception
    texts = [el.value for el in at.info]
    assert any("Источники не настроены" in text for text in texts)
    assert len(at.toggle) == 0


def test_sources_page_renders_registry(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seeded registry: one toggle per source, metrics on the page."""
    connection = app_pg.get_conn()
    try:
        ok_post = _post(
            connection,
            "https://fl.ru/p/1",
            "fl_ru",
            created_at=datetime(2026, 9, 12, 10, tzinfo=UTC),
        )
        _pain(connection, ok_post, "не работает поиск по заказам")
        _post(
            connection,
            "https://play/p/1",
            "gplay",
            fetch_status="failed",
            created_at=datetime(2026, 9, 12, 11, tzinfo=UTC),
        )
    finally:
        connection.close()

    at = _run_app(app_pg, monkeypatch)
    assert not at.exception
    at = _goto(at, "Источники")
    assert not at.exception

    # One toggle per source, all on by default.
    toggles = {t.label: t for t in at.toggle}
    assert sorted(toggles) == ["fl_ru", "gplay"]
    assert all(t.value for t in toggles.values())

    # Per-source metrics line renders on the page.
    markdown_texts = [el.value for el in at.markdown]
    fl_line = next(text for text in markdown_texts if "**fl_ru**" in text)
    assert "включён" in fl_line
    assert "ошибок: 0" in fl_line
    assert "постов: 1" in fl_line
    assert "болей: 1" in fl_line
    gplay_line = next(text for text in markdown_texts if "**gplay**" in text)
    assert "ошибок: 1" in gplay_line


def test_sources_page_toggle_updates_db(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flipping a toggle on the page writes enabled=false to the database."""
    connection = app_pg.get_conn()
    try:
        _post(connection, "https://fl.ru/p/1", "fl_ru")
    finally:
        connection.close()

    at = _run_app(app_pg, monkeypatch)
    assert not at.exception
    at = _goto(at, "Источники")
    assert not at.exception

    at.toggle[0].set_value(False).run()
    assert not at.exception

    connection = app_pg.get_conn()
    try:
        row = connection.execute("SELECT enabled FROM source WHERE name = 'fl_ru'").fetchone()
    finally:
        connection.close()
    assert row is not None and row[0] is False

    # The page reflects the disabled state after the rerun.
    line = next(text for text in (el.value for el in at.markdown) if "**fl_ru**" in text)
    assert "выключен" in line
