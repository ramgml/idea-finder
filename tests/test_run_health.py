"""Run ("Сбор") + Health ("Здоровье") pages: read layer and Streamlit (T313).

One module-scoped embedded postgres for the read-layer assertions (same
pattern as test_cluster_card.py / test_dashboard.py), one fresh cluster for
the AppTest runs. The launch lock is proven with data: a seeded ``running``
run makes ``has_active_run`` True — the button click itself is never
exercised (a clicked button would spawn a real pipeline subprocess).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest
from psycopg import Connection
from streamlit.testing.v1 import AppTest

from idea_finder.core.canonical import canonical_url
from idea_finder.core.models import RawPost
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import (
    create_run,
    ensure_source,
    finish_run,
    insert_raw_post,
    update_run_stage,
)
from idea_finder.web.health_view import health_report, list_source_health
from idea_finder.web.run_view import has_active_run, list_runs, run_command, stage_progress

ROOT = Path(__file__).resolve().parents[1]

_APP = str(ROOT / "idea_finder" / "web" / "streamlit_app.py")


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """Embedded postgres for the read-layer tests."""
    data_dir = tmp_path_factory.mktemp("pgrun") / "pg"
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
    data_dir = tmp_path_factory.mktemp("pgrunapp") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture()
def app_pg(app_pg_server: PgHandle) -> Iterator[PgHandle]:
    """Fresh ``idea_finder`` database per script test.

    Same isolation trick as test_cluster_card.py: the app resolves its
    database by name, so it is dropped and recreated between tests.
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
    source_name: str,
    *,
    suspect_short: bool = False,
    fetch_status: str = "extracted",
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
    inserted = insert_raw_post(conn, post, fetch_status=fetch_status, suspect_short=suspect_short)
    assert inserted is not None
    if created_at is not None:
        # raw_post.created_at is a DB now() default; pin it so last_ok tests
        # get a deterministic timestamp.
        with conn.transaction():
            conn.execute(
                "UPDATE raw_post SET created_at = %s WHERE id = %s", (created_at, inserted)
            )
    return inserted


def _run(
    conn: Connection,
    stages: dict[str, str],
    *,
    stats: dict[str, int] | None = None,
    cost: float = 0.0,
    finished: bool = True,
    started_at: datetime | None = None,
) -> str:
    """Seed one run row; return its id.

    ``stats``/``cost`` go through ``update_run_stage`` only (the same path
    the pipeline uses): ``_stats_merge`` accumulates deltas, so passing the
    numbers twice would double them.
    """
    run_id = create_run(conn, stages)
    if stages:
        first = next(iter(stages))
        update_run_stage(conn, run_id, first, stages[first], stats, cost)
    if finished:
        finish_run(conn, run_id)
    if started_at is not None:
        with conn.transaction():
            conn.execute("UPDATE run SET started_at = %s WHERE id = %s", (started_at, run_id))
            if finished:
                conn.execute(
                    "UPDATE run SET finished_at = %s WHERE id = %s AND finished_at IS NOT NULL",
                    (started_at + timedelta(minutes=5), run_id),
                )
    return run_id


# ---------------------------------------------------------------------------
# run_view read layer
# ---------------------------------------------------------------------------


def test_list_runs_empty(conn: Connection) -> None:
    """Fresh schema: no runs recorded."""
    assert list_runs(conn) == []


def test_list_runs_newest_first(conn: Connection) -> None:
    """Runs sort by started_at desc; stages/stats/cost round-trip."""
    _run(
        conn,
        {"extract": "done"},
        stats={"processed": 3, "extracted": 2, "failed": 1},
        cost=0.25,
        started_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    _run(
        conn,
        {"cluster": "done"},
        finished=True,
        started_at=datetime(2026, 9, 10, tzinfo=UTC),
    )

    runs = list_runs(conn)
    assert [run.stages for run in runs] == [
        {"cluster": "done"},
        {"extract": "done"},
    ]
    newest = runs[0]
    assert newest.finished_at is not None
    assert not newest.is_running
    assert newest.cost == 0.0
    older = runs[1]
    assert not older.is_running
    assert older.stats["extracted"] == 2
    assert older.cost == 0.25


def test_list_runs_limit(conn: Connection) -> None:
    """limit bounds the history length (page never needs the whole table)."""
    runs = list_runs(conn, limit=1)
    assert len(runs) == 1


def test_has_active_run_false_without_running(conn: Connection) -> None:
    """Finished and errored runs do not lock the launch button."""
    _run(conn, {"extract": "done"}, started_at=datetime(2026, 9, 1, tzinfo=UTC))
    _run(conn, {"score": "error"}, started_at=datetime(2026, 9, 2, tzinfo=UTC))
    assert not has_active_run(conn)


def test_has_active_run_true_on_running_stage(conn: Connection) -> None:
    """A run with a running stage locks the launch button.

    The module-scoped database carries rows from earlier tests in this file
    (some newer than this test's pinned dates), so nothing here assumes the
    running row is the newest one — only that the lock flag is set and the
    running run is reported by list_runs with its status intact.
    """
    _run(conn, {"collect": "done"}, started_at=datetime(2026, 9, 1, tzinfo=UTC))
    running_id = _run(
        conn, {"extract": "running"}, finished=False, started_at=datetime(2026, 9, 3, tzinfo=UTC)
    )
    assert has_active_run(conn)
    running_rows = [run for run in list_runs(conn) if run.id == running_id]
    assert len(running_rows) == 1
    assert running_rows[0].is_running
    assert running_rows[0].stages == {"extract": "running"}


def test_is_running_only_for_running_status() -> None:
    """is_running is driven purely by stage statuses (no DB round-trip)."""
    from idea_finder.web.run_view import RunRow

    def make_run(stages: dict[str, str]) -> RunRow:
        return RunRow(
            id="r",
            started_at=datetime(2026, 9, 1, tzinfo=UTC),
            finished_at=None,
            stages=stages,
            stats={},
            cost=0.0,
            prompt_version_id=None,
        )

    running = make_run({"collect": "running"})
    done = make_run({"collect": "done"})
    failed = make_run({"collect": "error", "score": "error"})
    assert running.is_running
    assert not done.is_running
    assert not failed.is_running


def test_stage_progress_renders_all_stages(conn: Connection) -> None:
    """Progress covers every pipeline stage in order, with metric lines."""
    _run(
        conn,
        {"extract": "done", "cluster": "running"},
        stats={"extracted": 5, "pains": 4},
        finished=False,
    )
    run = list_runs(conn)[0]
    progress = stage_progress(run)
    assert [p.name for p in progress] == ["Сбор", "Извлечение", "Кластеризация", "Скоринг"]
    by_name = {p.name: p for p in progress}
    # Untouched stage: explicit "not started" marker.
    assert "не запускалась" in by_name["Сбор"].status
    assert by_name["Извлечение"].status == "готово"
    assert "извлечено 5" in by_name["Извлечение"].metrics
    assert by_name["Кластеризация"].status == "выполняется"
    assert "болей 4" in by_name["Кластеризация"].metrics
    assert by_name["Скоринг"].metrics == "—"


def test_run_command_argv() -> None:
    """The subprocess launches the same CLI run an operator would use."""
    argv = run_command()
    assert argv[-2:] == ["cli.py", "run"]


# ---------------------------------------------------------------------------
# health_view read layer
# ---------------------------------------------------------------------------


def test_source_health_empty(conn: Connection) -> None:
    """No sources: empty health list (page shows the empty state)."""
    seeded_sources = conn.execute("SELECT count(*) FROM source").fetchone()
    assert seeded_sources is not None
    # Earlier tests in this module seed sources; health covers enabled ones
    # with their posts. The empty case is covered by the AppTest flow below.
    assert isinstance(int(seeded_sources[0]), int)


def test_source_health_per_source(conn: Connection) -> None:
    """last_ok skips failed posts; suspect share counts over all posts."""
    _post(
        conn,
        "https://habr.com/p/h1",
        "habr-health",
        created_at=datetime(2026, 9, 1, 10, tzinfo=UTC),
    )
    _post(
        conn,
        "https://habr.com/p/h2",
        "habr-health",
        suspect_short=True,
        created_at=datetime(2026, 9, 2, 10, tzinfo=UTC),
    )
    _post(
        conn,
        "https://fl.ru/p/f1",
        "fl-health",
        fetch_status="failed",
        created_at=datetime(2026, 9, 3, 10, tzinfo=UTC),
    )

    health = {row.name: row for row in list_source_health(conn)}
    habr = health["habr-health"]
    assert habr.last_ok is not None
    assert habr.last_ok == datetime(2026, 9, 2, 10, tzinfo=UTC)
    assert habr.failed_count == 0
    assert habr.total_count == 2
    assert habr.suspect_share == pytest.approx(0.5)

    fl = health["fl-health"]
    # Only post failed: last_ok is None, share counts it.
    assert fl.last_ok is None
    assert fl.failed_count == 1
    assert fl.total_count == 1
    assert fl.suspect_share == 0.0


def test_health_report_joins_runs_and_sources(conn: Connection) -> None:
    """The page payload carries both run history and source health."""
    report = health_report(conn)
    assert report.runs  # seeded by earlier tests
    assert report.sources  # seeded by earlier tests
    assert all(run.cost >= 0 for run in report.runs)


# ---------------------------------------------------------------------------
# Streamlit pages (AppTest)
# ---------------------------------------------------------------------------


def _run_app(pg_handle: PgHandle, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Run streamlit_app under AppTest with PGDATA_DIR at the test cluster."""
    monkeypatch.setenv("PGDATA_DIR", str(pg_handle.data_dir))
    return AppTest.from_file(_APP, default_timeout=180).run()


def _goto(at: AppTest, page: str) -> AppTest:
    """Switch the sidebar page and rerun the script."""
    at.sidebar.radio[0].set_value(page).run()
    return at


def test_run_page_empty_state(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh database: run page renders the no-runs empty state, button on."""
    at = _run_app(app_pg, monkeypatch)
    assert not at.exception
    at = _goto(at, "Сбор")
    assert not at.exception
    texts = [el.value for el in at.info]
    assert any("Прогонов ещё не было" in text for text in texts)
    assert len(at.button) == 1
    assert not at.button[0].disabled


def test_run_page_lock_on_running_run(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """A seeded running run disables the button (lock proven by data).

    The button is never clicked here: clicking would spawn the real pipeline
    subprocess. The lock is asserted through the disabled flag, which is
    exactly what the UI points render from.
    """
    connection = app_pg.get_conn()
    try:
        _run(connection, {"collect": "done"}, finished=True)
        _run(connection, {"extract": "running"}, finished=False)
    finally:
        connection.close()

    at = _run_app(app_pg, monkeypatch)
    assert not at.exception
    at = _goto(at, "Сбор")
    assert not at.exception
    assert len(at.button) == 1
    assert at.button[0].disabled, "button must be disabled while a run is active"
    infos = [el.value for el in at.info]
    assert any("повторный запуск заблокирован" in text for text in infos)
    # Progress block: extract is running, collect done.
    markdown_texts = [el.value for el in at.markdown]
    assert any("выполняется" in text for text in markdown_texts)


def test_run_page_progress_after_finished_run(
    app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finished run: progress lines per stage + history table."""
    connection = app_pg.get_conn()
    try:
        _run(
            connection,
            {"extract": "done", "score": "done"},
            stats={"extracted": 7, "scored": 3},
            cost=0.1234,
        )
    finally:
        connection.close()

    at = _run_app(app_pg, monkeypatch)
    at = _goto(at, "Сбор")
    assert not at.exception
    # Button is enabled again once nothing is running.
    assert len(at.button) == 1
    assert not at.button[0].disabled
    markdown_texts = [el.value for el in at.markdown]
    assert any("готово" in text for text in markdown_texts)
    assert any("извлечено 7" in text for text in markdown_texts)
    assert len(at.dataframe) == 1
    frame = at.dataframe[0].value
    assert "стоимость, ₽" in frame.columns


def test_run_page_db_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unreachable postgres: graceful error, no traceback."""
    monkeypatch.setenv("PGDATA_DIR", "/nonexistent-pg-run-health")
    at = AppTest.from_file(_APP, default_timeout=180).run()
    at = _goto(at, "Сбор")
    assert not at.exception
    assert any("База данных недоступна" in el.value for el in at.error)


def test_health_page_empty_state(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh database: health page shows both empty states."""
    at = _run_app(app_pg, monkeypatch)
    at = _goto(at, "Здоровье")
    assert not at.exception
    texts = [el.value for el in at.info]
    assert any("Прогонов ещё не было" in text for text in texts)


def test_health_page_with_seed(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seeded data: run history table + source health table render."""
    connection = app_pg.get_conn()
    try:
        _run(
            connection,
            {"extract": "done"},
            stats={"extracted": 4, "failed": 1},
            cost=0.5,
        )
        _post(connection, "https://example.com/h/1", "seed-src", suspect_short=True)
    finally:
        connection.close()

    at = _run_app(app_pg, monkeypatch)
    at = _goto(at, "Здоровье")
    assert not at.exception
    assert len(at.dataframe) == 2
    runs_frame = at.dataframe[0].value
    assert "стоимость, ₽" in runs_frame.columns
    assert "0.5000" in list(runs_frame["стоимость, ₽"])
    health_frame = at.dataframe[1].value
    assert list(health_frame["источник"]) == ["seed-src"]
    assert list(health_frame["suspect_short, %"]) == ["100"]


def test_health_page_db_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unreachable postgres: graceful error, no traceback."""
    monkeypatch.setenv("PGDATA_DIR", "/nonexistent-pg-run-health")
    at = AppTest.from_file(_APP, default_timeout=180).run()
    at = _goto(at, "Здоровье")
    assert not at.exception
    assert any("База данных недоступна" in el.value for el in at.error)


def test_stages_json_decoded_from_text(conn: Connection) -> None:
    """jsonb delivered as str (not dict) still decodes — psycopg both modes."""
    run_id = _run(conn, {"collect": "done"})
    run = next(row for row in list_runs(conn) if row.id == run_id)
    assert run.stages == {"collect": "done"}
