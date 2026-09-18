"""Sources ("Источники") page backend: registry table + enable toggle (T316).

Schema note (task 316 constraint): ``db/repo.py`` is shared with other
concurrent streams, so this module keeps the sources-page queries and the
one UPDATE helper the page needs here instead of editing repo.py.
TODO(repo): move :func:`list_sources` and :func:`set_source_enabled` into
``db/repo.py`` once the concurrent streams merge; keep the signatures and
SQL as-is.

The toggle is a web-layer write by design (same pattern as
``prompts_view.save_new_version``): it only flips ``source.enabled`` and
never touches adapter data. Known gap: the collect stage itself is still a
skeleton in ``core/pipeline.py`` (B-flow task), so toggling today changes
only the data the future collect will read — ``repo.list_enabled_sources``
already filters ``WHERE enabled`` and ``health_view`` reads the same flag,
so no pipeline change is needed when collect gets wired.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Final, LiteralString

import streamlit as st
from psycopg import Connection

from idea_finder.db.bootstrap_pgserver import DbError, ensure_pgserver
from idea_finder.db.migrate import apply_migrations

logger = logging.getLogger(__name__)

#: Registry table: every source with its per-source health and counters.
#: All counters go through count(<joined pk>): a source with no posts joins
#: one all-NULL row, and count(*) would wrongly report 1 post for it.
_SOURCES_SQL: Final[LiteralString] = """
    SELECT s.name,
           s.enabled,
           max(rp.created_at) FILTER (WHERE rp.fetch_status <> 'failed') AS last_ok,
           count(*) FILTER (WHERE rp.fetch_status = 'failed')            AS failed_count,
           count(rp.id)                                                  AS post_count,
           count(p.id)                                                   AS pain_count
    FROM source s
    LEFT JOIN raw_post rp ON rp.source_id = s.id
    LEFT JOIN pain p ON p.raw_post_id = rp.id
    GROUP BY s.name, s.enabled
    ORDER BY s.name
"""


@dataclass(frozen=True, slots=True)
class SourceRow:
    """One ``source`` registry row with per-source counters for the table."""

    name: str
    #: Collected-stage switch (migration 003): False hides the source from
    #: ``repo.list_enabled_sources``, the future collect-stage input.
    enabled: bool
    #: Last post collected without a fetch failure (None = never succeeded).
    last_ok: datetime | None
    #: Posts whose fetch ended in the terminal ``failed`` state.
    failed_count: int
    #: All posts collected from the source.
    post_count: int
    #: Pains extracted from this source's posts.
    pain_count: int


def list_sources(conn: Connection) -> list[SourceRow]:
    """Return every source with health metrics, ordered by name.

    Unlike :mod:`idea_finder.web.health_view` (which lists only enabled
    sources), this is the registry view: disabled sources stay visible so
    the operator can re-enable them.
    """
    rows = conn.execute(_SOURCES_SQL).fetchall()
    return [
        SourceRow(
            name=str(name),
            enabled=bool(enabled),
            last_ok=last_ok,
            failed_count=int(failed_count),
            post_count=int(post_count),
            pain_count=int(pain_count),
        )
        for name, enabled, last_ok, failed_count, post_count, pain_count in rows
    ]


def set_source_enabled(conn: Connection, name: str, enabled: bool) -> None:
    """Flip ``source.enabled`` for ``name``; no-op when the name is unknown.

    Web-layer write (see module docstring): the only UPDATE on the page.
    The caller owns commit/rollback, same as ``save_new_version``.
    """
    with conn.transaction():
        conn.execute("UPDATE source SET enabled = %s WHERE name = %s", (enabled, name))
    logger.info("source %s: enabled -> %s", name, enabled)


#: Empty-state text when the registry has no rows at all.
_EMPTY_REGISTRY_TEXT: Final[str] = (
    "Источники не настроены — реестр заполняется адаптерами при первом запуске."
)

#: Russian caption under the toggle column.
_TOGGLE_CAPTION: Final[str] = (
    "Выключенный источник пропускается при сборе (collect читает только "
    "включённые); собранные ранее посты остаются в базе."
)

#: Database-down message, same contract as the other pages in streamlit_app.
_DB_DOWN_TEXT: Final[str] = (
    "База данных недоступна: встроенный Postgres не поднят. Выполните "
    "`uv run python cli.py status` (инициализирует кластер), затем "
    "перезагрузите страницу."
)


def render_sources_page() -> None:
    """Sources page: registry table with per-row enable toggles."""
    st.title("Источники")
    try:
        handle = ensure_pgserver()  # PGDATA_DIR is read per call
        conn = handle.get_conn()
        apply_migrations(conn)
    except (DbError, OSError) as e:
        logger.warning("dashboard: database unavailable: %s", e)
        st.error(_DB_DOWN_TEXT)
        return

    # The page owns its script-run connection: psycopg opens an implicit
    # transaction on first statement and nothing else commits it, so without
    # an explicit commit the toggle UPDATE below is rolled back when the
    # script run ends (same contract as prompts_view).
    conn.commit()

    sources = list_sources(conn)
    if not sources:
        st.info(_EMPTY_REGISTRY_TEXT)
        return

    st.caption(_TOGGLE_CAPTION)
    for row in sources:
        col_table, col_toggle = st.columns([4, 1])
        with col_table:
            st.markdown(_source_summary_md(row))
        with col_toggle:
            key = f"source-enabled-{row.name}"
            if key not in st.session_state:
                st.session_state[key] = row.enabled
            new_value = st.toggle(row.name, key=key)
        if new_value != row.enabled:
            set_source_enabled(conn, row.name, new_value)
            conn.commit()
            st.toast(f"Источник {row.name}: {'включён' if new_value else 'выключен'}.")
            st.rerun()


def _source_summary_md(row: SourceRow) -> str:
    """One table row as markdown: health + counters on a single line."""
    last_ok = (
        row.last_ok.strftime("%d.%m.%Y %H:%M") if row.last_ok is not None else "нет успешных постов"
    )
    status = "включён" if row.enabled else "выключен"
    return (
        f"**{row.name}** · {status} · последний успех: {last_ok} · "
        f"ошибок: {row.failed_count} · постов: {row.post_count} · болей: {row.pain_count}"
    )
