"""Prompts page tests: versioning layer and Streamlit page (task 314).

One module-scoped embedded postgres for the read/write-layer assertions
(same pattern as test_cluster_card.py), one fresh database per AppTest run.
Covers the DoD chain: edit -> v2 with source='ui', rollback to v1 -> new
version with v1's body, active version = max(version), diff markers, page
render without exceptions.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from psycopg import Connection
from streamlit.testing.v1 import AppTest

from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import upsert_prompt_version
from idea_finder.web.prompts_view import (
    get_active_version,
    get_version,
    list_prompt_names,
    list_versions,
    rollback_to_version,
    save_new_version,
    seed_prompts_from_files,
    unified_diff,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

_APP = str(ROOT / "idea_finder" / "web" / "streamlit_app.py")


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """Embedded postgres for the versioning-layer tests."""
    data_dir = tmp_path_factory.mktemp("pgprompts") / "pg"
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
    data_dir = tmp_path_factory.mktemp("pgpromptsapp") / "pg"
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


def _v1(conn: Connection, name: str, body: str) -> None:
    """Seed v1 of a prompt the way the pipeline registers file prompts."""
    upsert_prompt_version(conn, name, 1, body, "file")


def test_seed_registers_both_packaged_prompts_once(conn: Connection) -> None:
    """First seed inserts file v1 for every packaged prompt, repeat is no-op."""
    seeded = seed_prompts_from_files(conn)
    assert seeded == ["extract_pains", "score"]
    v1 = get_version(conn, "extract_pains", 1)
    assert v1 is not None
    assert v1.source == "file"
    assert "{{ post_text }}" in v1.body
    # Second call: everything already present, nothing inserted.
    assert seed_prompts_from_files(conn) == []
    # An existing name is never re-seeded, even with a single UI row.
    save_new_version(conn, "extract_pains", "edited body")
    assert seed_prompts_from_files(conn) == []


def test_save_new_version_creates_v2_with_ui_source(conn: Connection) -> None:
    """Saving an edit appends max(version)+1 with source='ui' (DoD: edit -> v2)."""
    _v1(conn, "save-name", "first body")
    saved = save_new_version(conn, "save-name", "second body")
    assert saved.version == 2
    assert saved.source == "ui"
    assert saved.body == "second body"
    # v1 is immutable: same id, same body.
    v1 = get_version(conn, "save-name", 1)
    assert v1 is not None
    assert v1.body == "first body"
    assert v1.source == "file"


def test_rollback_creates_new_version_with_old_body(conn: Connection) -> None:
    """Rollback to v1 appends a fresh UI version carrying v1's body (DoD)."""
    _v1(conn, "rollback-name", "original")
    save_new_version(conn, "rollback-name", "broken edit")
    saved = rollback_to_version(conn, "rollback-name", 1)
    assert saved.version == 3
    assert saved.source == "ui"
    assert saved.body == "original"
    # History kept: v2 still exists untouched.
    v2 = get_version(conn, "rollback-name", 2)
    assert v2 is not None
    assert v2.body == "broken edit"


def test_rollback_unknown_version_raises(conn: Connection) -> None:
    """Rollback to a version that does not exist is an error, not a write."""
    _v1(conn, "rollback-miss", "original")
    with pytest.raises(ValueError, match="no version 9"):
        rollback_to_version(conn, "rollback-miss", 9)
    assert [row.version for row in list_versions(conn, "rollback-miss")] == [1]


def test_active_version_is_max_version(conn: Connection) -> None:
    """Active = highest version, whatever its source (DoD: active = last)."""
    _v1(conn, "active-name", "one")
    assert get_active_version(conn, "active-name") is not None
    saved = save_new_version(conn, "active-name", "two")
    active = get_active_version(conn, "active-name")
    assert active is not None
    assert active.id == saved.id
    assert active.version == 2
    # After a rollback the max-version row is active again (v3 here).
    rolled = rollback_to_version(conn, "active-name", 1)
    active = get_active_version(conn, "active-name")
    assert active is not None
    assert active.id == rolled.id
    assert active.version == 3
    assert active.body == "one"


def test_active_version_unknown_name(conn: Connection) -> None:
    """Unknown name: None (page shows the empty state, not a traceback)."""
    assert get_active_version(conn, "missing-name") is None


def test_list_names_and_versions_sorted(conn: Connection) -> None:
    """Names are alphabetical; versions ascending v1..vN."""
    _v1(conn, "b-list", "b1")
    _v1(conn, "a-list", "a1")
    save_new_version(conn, "b-list", "b2")
    names = list_prompt_names(conn)
    # Whole-table assertion is impossible on a shared module-scoped database
    # (earlier tests inserted rows), so assert the ordering property itself.
    assert names == sorted(names)
    assert [row.version for row in list_versions(conn, "b-list")] == [1, 2]


def test_unified_diff_marks_changes(conn: Connection) -> None:
    """Diff between neighbouring versions carries the change markers."""
    _v1(conn, "diff-name", "line one\nline two\n")
    v1 = get_version(conn, "diff-name", 1)
    assert v1 is not None
    v2 = save_new_version(conn, "diff-name", "line one\nline TWO changed\n")
    diff = unified_diff(v1, v2)
    assert "--- v1" in diff
    assert "+++ v2" in diff
    assert "-line two" in diff
    assert "+line TWO changed" in diff
    # Identical bodies -> empty diff.
    v3 = save_new_version(conn, "diff-name", "line one\nline TWO changed\n")
    assert unified_diff(v2, v3) == ""


def test_dashboard_prompts_page_renders(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Page flow: seeding, version radio, body view — no exceptions."""
    connection = app_pg.get_conn()
    try:
        _v1(connection, "extract_pains", "seeded body v1")
    finally:
        connection.close()

    monkeypatch.setenv("PGDATA_DIR", str(app_pg.data_dir))
    at = AppTest.from_file(_APP, default_timeout=180)
    at.run()  # populate the element tree before touching the radio
    at.radio[0].set_value("Промпты").run()
    assert not at.exception

    # The prompt selectbox lists the seeded name (sidebar radio + selectbox).
    boxes = [box for box in at.selectbox if "extract_pains" in box.options]
    assert len(boxes) == 1

    # The seeded body is rendered somewhere on the page.
    code_texts = [el.value for el in at.code]
    assert any("seeded body v1" in text for text in code_texts)

    # The save form is present with its caption above the editor.
    assert any(el.value for el in at.text_area)
    assert at.subheader  # version text / diff / save sections rendered


def test_dashboard_prompts_page_empty_db(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh database: the page seeds prompts and shows the file v1 bodies."""
    monkeypatch.setenv("PGDATA_DIR", str(app_pg.data_dir))
    at = AppTest.from_file(_APP, default_timeout=180)
    at.run()  # populate the element tree before touching the radio
    at.radio[0].set_value("Промпты").run()
    assert not at.exception

    # Seeding ran: both packaged prompts exist with their shipped text.
    connection = app_pg.get_conn()
    try:
        v1 = get_version(connection, "score", 1)
    finally:
        connection.close()
    assert v1 is not None
    assert v1.source == "file"
    assert "{{ cluster_summary }}" in v1.body
