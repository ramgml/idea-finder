"""Streamlit dashboard for idea-finder: read-only views over the pipeline DB.

Task 311 skeleton: sidebar navigation, the Clusters page (title/score/size/
kinds/sources/updated, sorted by score, empty state on a fresh database) and
placeholders for later F-stream pages. Reads go through
:mod:`idea_finder.web.clusters_view` (read-only SELECTs; ``db/repo.py`` has
no cluster-list read function yet — see the TODO note in clusters_view).
The dashboard never writes to the pipeline.

Run: ``uv run streamlit run idea_finder/web/streamlit_app.py``
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd
import streamlit as st
from psycopg import Connection

from idea_finder.db.bootstrap_pgserver import DbError, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.web.clusters_view import ClusterRow, list_clusters

logger = logging.getLogger(__name__)

#: Sidebar page titles, in navigation order.
_PAGE_TITLES: tuple[str, ...] = (
    "Кластеры",
    "Источники",
    "Промпты",
    "Настройки LLM",
    "Здоровье",
)

_EMPTY_STATE_TEXT = (
    "Нет кластеров — запустите пайплайн: `uv run python cli.py run` "
    "(collect → extract → cluster → score)."
)

_DB_DOWN_TEXT = (
    "База данных недоступна: встроенный Postgres не поднят. Выполните "
    "`uv run python cli.py status` (инициализирует кластер), затем "
    "перезагрузите страницу."
)

#: Russian labels for PostKind values shown in the kinds cell.
_KIND_LABELS: dict[str, str] = {
    "demand": "заказ",
    "complaint": "жалоба",
    "discussion": "обсуждение",
}


def _connect() -> Connection | None:
    """Open the app connection: embedded postgres up, schema ensured.

    A fresh cluster (no ``data/pg`` yet) is bootstrapped and migrated right
    here — same entry path as the CLI — so ``streamlit run`` on a new
    checkout shows the empty state instead of an error screen. Returns None
    when the embedded postgres cannot be provided; the page then renders a
    graceful 'database unavailable' screen instead of a traceback. The
    connection lives for one script run: a long-lived one would break on
    cluster restarts between reruns.
    """
    try:
        handle = st.cache_resource(lambda: ensure_pgserver())()
        conn = handle.get_conn()
        apply_migrations(conn)
    except (DbError, OSError) as e:
        logger.warning("dashboard: database unavailable: %s", e)
        return None
    return conn


def _format_updated(created_at: datetime) -> str:
    """Render the updated-at cell as a coarse human age (naive-UTC math)."""
    created_naive = created_at.replace(tzinfo=None)
    age = datetime.now(UTC).replace(tzinfo=None) - created_naive
    seconds = max(int(age.total_seconds()), 0)
    if seconds < 60:
        return "только что"
    if seconds < 3600:
        return f"{seconds // 60} мин назад"
    if seconds < 86400:
        return f"{seconds // 3600} ч назад"
    return created_naive.strftime("%d.%m.%Y")


def _format_kinds(kinds: dict[str, int]) -> str:
    """Render kind counters as ``заказ 2, жалоба 1`` ('—' when empty)."""
    rendered = ", ".join(
        f"{_KIND_LABELS.get(kind, kind)} {count}" for kind, count in kinds.items() if count
    )
    return rendered or "—"


def _clusters_dataframe(rows: list[ClusterRow]) -> pd.DataFrame:
    """Build the clusters table with the required column captions."""
    data: dict[str, list[object]] = {
        "title": [row.title for row in rows],
        "score": [row.score if row.score is not None else None for row in rows],
        "size": [row.size for row in rows],
        "kinds": [_format_kinds(row.kinds) for row in rows],
        "sources": [", ".join(row.sources) if row.sources else "—" for row in rows],
        "обновлено": [_format_updated(row.created_at) for row in rows],
    }
    return pd.DataFrame(data, columns=("title", "score", "size", "kinds", "sources", "обновлено"))


def _render_clusters_page() -> None:
    """Clusters page: score-sorted table, empty state, graceful DB-down."""
    st.title("Кластеры")
    conn = _connect()
    if conn is None:
        st.error(_DB_DOWN_TEXT)
        return
    rows = list_clusters(conn)
    if not rows:
        st.info(_EMPTY_STATE_TEXT)
        return
    st.dataframe(
        _clusters_dataframe(rows),
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn(format="%.2f"),
            "size": st.column_config.NumberColumn(),
        },
    )
    st.caption(
        "Сортировка по скору (лучшие сверху); кластеры без оценки — в конце. "
        "«обновлено» — по created_at кластера."
    )


def _render_placeholder_page(title: str) -> None:
    """Placeholder for pages delivered by later F-stream tasks."""
    st.title(title)
    st.info(f"Страница «{title}» будет реализована в следующих задачах потока F.")


def main() -> None:
    """App skeleton: sidebar navigation over the page registry."""
    st.set_page_config(page_title="idea-finder", page_icon=":bulb:", layout="wide")
    st.sidebar.title("idea-finder")
    page = st.sidebar.radio("Навигация", _PAGE_TITLES)
    if page == "Кластеры":
        _render_clusters_page()
    else:
        _render_placeholder_page(page)


if __name__ == "__main__":
    main()
