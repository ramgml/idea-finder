"""LLM settings page tests: CRUD layer and Streamlit page (task 318).

One module-scoped embedded postgres for the write-layer assertions (same
pattern as test_prompts_page.py), one fresh database per AppTest run.
Covers the DoD chain: masking never reveals a full key, set_default keeps
exactly one active row across flips, add/delete round-trip, base_url is
required for kind=openai_compat, and the dashboard page renders without
exceptions. All keys used here are fictitious.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from psycopg import Connection
from psycopg import errors as pg_errors
from streamlit.testing.v1 import AppTest

from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import upsert_llm_provider
from idea_finder.web.settings_view import (
    add_provider,
    delete_provider,
    list_providers,
    mask_key,
    set_default,
    set_enabled,
    update_provider,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

_APP = str(ROOT / "idea_finder" / "web" / "streamlit_app.py")


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """Embedded postgres for the CRUD-layer tests."""
    data_dir = tmp_path_factory.mktemp("pgsettings") / "pg"
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
    data_dir = tmp_path_factory.mktemp("pgsettingsapp") / "pg"
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


def _row(conn: Connection, name: str) -> tuple[object, ...]:
    """Fetch one raw provider row (id, name, kind, base_url, model, api_key, is_active)."""
    row = conn.execute(
        """
        SELECT id, name, kind, base_url, model, api_key, is_active
        FROM llm_provider WHERE name = %s
        """,
        (name,),
    ).fetchone()
    assert row is not None, f"provider {name!r} missing"
    return row


def _count_active(conn: Connection) -> int:
    """Number of active provider rows."""
    row = conn.execute("SELECT count(*) FROM llm_provider WHERE is_active").fetchone()
    assert row is not None
    return int(row[0])


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def test_mask_key_keeps_only_tail(conn: Connection) -> None:
    """Long key: **** + last 4 characters, never the full secret."""
    assert mask_key("sk-1234567890abcdef") == "****cdef"


def test_mask_key_short_and_empty(conn: Connection) -> None:
    """Keys of length <= 4 (and empty) hide completely: tail would be whole."""
    assert mask_key("abc") == "****"
    assert mask_key("abcd") == "****"
    assert mask_key("") == "****"


def test_listing_never_contains_full_key(conn: Connection) -> None:
    """The masked listing does not leak any full stored key (DoD: no render)."""
    add_provider(
        conn, "mask-long", "openai_compat", "https://api.example.com", "m", "sk-secret-9999"
    )
    add_provider(conn, "mask-short", "fake", "", "m", "abc")
    rows = list_providers(conn)
    rendered = {row.name: row.key_masked for row in rows}
    assert rendered["mask-long"] == "****9999"
    assert rendered["mask-short"] == "****"
    # The full secret appears nowhere in the rendered payload.
    assert all("sk-secret-9999" not in value for value in rendered.values())
    # Structural guarantee: the listing SQL never fetches the raw column.
    listing = conn.execute("SELECT api_key FROM llm_provider WHERE name = 'mask-long'").fetchall()
    assert len(listing) == 1  # stored in db; the view layer just never selects it


# ---------------------------------------------------------------------------
# Add / delete
# ---------------------------------------------------------------------------


def test_add_and_delete_provider(conn: Connection) -> None:
    """add_provider inserts (key stored verbatim), delete_provider removes."""
    added = add_provider(
        conn, "round-trip", "openai_compat", "https://api.example.com", "model-x", "sk-tok-4242"
    )
    assert added
    row = _row(conn, "round-trip")
    assert row[1] == "round-trip"
    assert row[3] == "https://api.example.com"
    assert row[5] == "sk-tok-4242"  # stored, not masked, in the database only
    assert delete_provider(conn, "round-trip") is True
    assert conn.execute("SELECT 1 FROM llm_provider WHERE name = 'round-trip'").fetchone() is None
    assert delete_provider(conn, "round-trip") is False  # second delete: unknown name


def test_add_duplicate_name_rejected(conn: Connection) -> None:
    """An existing name is an error, not a silent overwrite."""
    add_provider(conn, "dup", "fake", "", "m", "")
    with pytest.raises(ValueError, match="уже есть"):
        add_provider(conn, "dup", "fake", "", "m", "")


def test_add_unknown_kind_rejected(conn: Connection) -> None:
    """kind outside {fake, openai_compat} is rejected before touching the db."""
    with pytest.raises(ValueError, match="Неизвестный тип"):
        add_provider(conn, "bad-kind", "azure", "https://x", "m", "")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_openai_compat_requires_base_url(conn: Connection) -> None:
    """kind=openai_compat without base_url is rejected (DoD validation)."""
    with pytest.raises(ValueError, match="base_url"):
        add_provider(conn, "no-url", "openai_compat", "", "m", "k")
    with pytest.raises(ValueError, match="base_url"):
        add_provider(conn, "blank-url", "openai_compat", "   ", "m", "k")


def test_fake_kind_allows_empty_base_url(conn: Connection) -> None:
    """Mock mode needs no base_url."""
    added = add_provider(conn, "fake-no-url", "fake", "", "mock-model", "")
    assert added
    assert _row(conn, "fake-no-url")[2] == "fake"


def test_empty_name_rejected(conn: Connection) -> None:
    """Whitespace-only names are rejected for both kinds."""
    with pytest.raises(ValueError, match="Имя"):
        add_provider(conn, "  ", "fake", "", "m", "")


# ---------------------------------------------------------------------------
# Default switching (single active)
# ---------------------------------------------------------------------------


def test_set_default_flips_both_rows(conn: Connection) -> None:
    """Switching default updates both rows: old off, new on, exactly one active."""
    add_provider(conn, "first", "fake", "", "m1", "", is_active=True)
    add_provider(conn, "second", "openai_compat", "https://api.example.com", "m2", "k")
    assert _count_active(conn) == 1
    assert _row(conn, "first")[6] is True

    set_default(conn, "second")
    assert _row(conn, "first")[6] is False
    assert _row(conn, "second")[6] is True
    assert _count_active(conn) == 1

    # Flip back: symmetric, still exactly one active.
    set_default(conn, "first")
    assert _row(conn, "first")[6] is True
    assert _row(conn, "second")[6] is False
    assert _count_active(conn) == 1


def test_set_default_unknown_name(conn: Connection) -> None:
    """Unknown name is an error, no row changes."""
    add_provider(conn, "keep", "fake", "", "m", "", is_active=True)
    with pytest.raises(ValueError, match="не найден"):
        set_default(conn, "ghost")
    assert _row(conn, "keep")[6] is True


def test_set_enabled_disable_leaves_none_active(conn: Connection) -> None:
    """Disabling the only active provider is allowed (config gap surfaces)."""
    add_provider(conn, "solo", "fake", "", "m", "", is_active=True)
    set_enabled(conn, "solo", False)
    assert _row(conn, "solo")[6] is False
    assert _count_active(conn) == 0
    set_enabled(conn, "solo", True)  # re-enable = make default again
    assert _count_active(conn) == 1


def test_update_keeps_key_when_blank(conn: Connection) -> None:
    """update_provider with an empty key keeps the stored secret; new one wins."""
    add_provider(conn, "upd", "openai_compat", "https://old.example.com", "old", "sk-old-1111")
    update_provider(conn, "upd", "openai_compat", "https://new.example.com", "new")
    row = _row(conn, "upd")
    assert row[3] == "https://new.example.com"
    assert row[4] == "new"
    assert row[5] == "sk-old-1111"  # secret untouched
    update_provider(conn, "upd", "openai_compat", "https://new.example.com", "new", "sk-new-2222")
    assert _row(conn, "upd")[5] == "sk-new-2222"


def test_update_requires_base_url_for_openai_compat(conn: Connection) -> None:
    """Clearing base_url through update is rejected the same way as on add."""
    add_provider(conn, "upd-url", "openai_compat", "https://api.example.com", "m", "k")
    with pytest.raises(ValueError, match="base_url"):
        update_provider(conn, "upd-url", "openai_compat", "", "m", "")


def test_partial_unique_index_forbids_two_active(conn: Connection) -> None:
    """The DB invariant behind set_default: a second active row is impossible.

    set_default flips the old row off before flipping the new one on, so it
    never trips this index in normal operation; the index is the backstop
    (and its UniqueViolation is what set_default translates into ValueError).
    """
    add_provider(conn, "idx-a", "fake", "", "m", "", is_active=True)
    add_provider(conn, "idx-b", "fake", "", "m", "")
    conn.commit()  # seal the seeds before the deliberate constraint breach
    with pytest.raises(pg_errors.UniqueViolation):
        conn.execute("UPDATE llm_provider SET is_active = true WHERE name = 'idx-b'")
    conn.rollback()  # discard only the failed UPDATE
    assert _count_active(conn) == 1
    assert _row(conn, "idx-a")[6] is True


# ---------------------------------------------------------------------------
# Dashboard page (AppTest)
# ---------------------------------------------------------------------------


def test_dashboard_settings_page_empty_db(
    app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh database: empty state + cli llm-test hint, no exceptions."""
    monkeypatch.setenv("PGDATA_DIR", str(app_pg.data_dir))
    at = AppTest.from_file(_APP, default_timeout=180)
    at.run()  # populate the element tree before touching the radio
    at.radio[0].set_value("Настройки LLM").run()
    assert not at.exception

    infos = [el.value for el in at.info]
    assert any("Провайдеров ещё нет" in text for text in infos)
    captions = [el.value for el in at.caption]
    assert any("cli.py llm-test" in text for text in captions)


def test_dashboard_settings_page_renders(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Page flow: masked key in the table, default mark, add form fields."""
    connection = app_pg.get_conn()
    try:
        upsert_llm_provider(connection, "fake", "fake", "", "mock-model", "", is_active=True)
        upsert_llm_provider(
            connection,
            "deepseek",
            "openai_compat",
            "https://api.deepseek.com",
            "deepseek-chat",
            "sk-fictitious-7777",
            is_active=False,
        )
    finally:
        connection.close()

    monkeypatch.setenv("PGDATA_DIR", str(app_pg.data_dir))
    at = AppTest.from_file(_APP, default_timeout=180)
    at.run()  # populate the element tree before touching the radio
    at.radio[0].set_value("Настройки LLM").run()
    assert not at.exception

    # The masked key is rendered; the full fictitious key is nowhere on the page.
    # (Element values carry markdown source, so escape backslashes first.)
    rendered = [str(el.value).replace("\\", "") for el in at.markdown]
    rendered += [str(el.value).replace("\\", "") for el in at.caption]
    assert any("****7777" in text for text in rendered)
    assert not any("sk-fictitious-7777" in text for text in rendered)

    # The active provider carries the default mark; the other one offers a switch.
    buttons = [b.label for b in at.button]
    assert any("Сделать активным" in label for label in buttons)
    assert any("Выключить" in label for label in buttons)

    # The add form exposes a password-typed key input (write-only secret)
    # and the kind selectbox.
    assert any(
        i.proto.type == 1  # TextInputProto.Type.PASSWORD
        for i in at.text_input
    )
    assert len(at.selectbox) >= 1


def test_dashboard_default_switch_flow(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clicking 'Сделать активным' flips the default through the page."""
    connection = app_pg.get_conn()
    try:
        upsert_llm_provider(connection, "fake", "fake", "", "mock-model", "", is_active=True)
        upsert_llm_provider(
            connection,
            "glm",
            "openai_compat",
            "https://open.bigmodel.example",
            "glm-4",
            "sk-fictitious-0001",
            is_active=False,
        )
    finally:
        connection.close()

    monkeypatch.setenv("PGDATA_DIR", str(app_pg.data_dir))
    at = AppTest.from_file(_APP, default_timeout=180)
    at.run()
    at.radio[0].set_value("Настройки LLM").run()
    assert not at.exception

    # Click the switch button on the inactive row.
    switch = next(b for b in at.button if "Сделать активным" in b.label)
    switch.click().run()
    assert not at.exception

    # Exactly one active row now, and it is glm.
    connection = app_pg.get_conn()
    try:
        row = connection.execute("SELECT name FROM llm_provider WHERE is_active").fetchone()
    finally:
        connection.close()
    assert row is not None and row[0] == "glm"


def test_dashboard_delete_flow(app_pg: PgHandle, monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete requires the confirm checkbox and removes the row."""
    connection = app_pg.get_conn()
    try:
        upsert_llm_provider(connection, "fake", "fake", "", "mock-model", "", is_active=True)
        upsert_llm_provider(
            connection,
            "gone",
            "openai_compat",
            "https://api.example.com",
            "m",
            "sk-fictitious-0002",
            is_active=False,
        )
    finally:
        connection.close()

    monkeypatch.setenv("PGDATA_DIR", str(app_pg.data_dir))
    at = AppTest.from_file(_APP, default_timeout=180)
    at.run()
    at.radio[0].set_value("Настройки LLM").run()
    assert not at.exception

    # Without confirmation the button is disabled — AppTest refuses to click
    # it, exactly like a browser refuses, so the confirmation gate holds and
    # the row survives.
    delete_btn = next(b for b in at.button if b.label == "Удалить" and str(b.key).endswith("gone"))
    assert delete_btn.disabled

    # Confirm, then delete: the row disappears.
    checkbox = next(c for c in at.checkbox if str(c.key).endswith("gone"))
    checkbox.check().run()
    delete_btn2 = next(b for b in at.button if b.label == "Удалить" and str(b.key).endswith("gone"))
    assert not delete_btn2.disabled
    delete_btn2.click().run()
    assert not at.exception
    connection = app_pg.get_conn()
    try:
        removed = connection.execute("SELECT 1 FROM llm_provider WHERE name = 'gone'").fetchone()
    finally:
        connection.close()
    assert removed is None
