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
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from psycopg import Connection

from idea_finder.db.bootstrap_pgserver import DbError, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.web.clusters_view import (
    MAX_SCORE,
    ClusterFilters,
    ClusterRow,
    get_cluster_detail,
    list_clusters,
    list_clusters_filtered,
    list_source_names,
)
from idea_finder.web.health_view import SourceHealthRow, health_report
from idea_finder.web.prompts_view import render_prompts_page
from idea_finder.web.run_view import RunRow, has_active_run, list_runs, run_command, stage_progress
from idea_finder.web.sources_view import render_sources_page

logger = logging.getLogger(__name__)

#: Sidebar page titles, in navigation order.
_PAGE_TITLES: tuple[str, ...] = (
    "Кластеры",
    "Сбор",
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
        handle = ensure_pgserver()  # PGDATA_DIR is read per call
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
    """Clusters page: filters, score-sorted table, card, empty states."""
    st.title("Кластеры")
    conn = _connect()
    if conn is None:
        st.error(_DB_DOWN_TEXT)
        return
    rows = list_clusters(conn)
    if not rows:
        st.info(_EMPTY_STATE_TEXT)
        return
    filters = _render_filters(conn)
    visible = list_clusters_filtered(conn, filters) if filters != ClusterFilters() else rows
    _render_clusters_table(visible)
    st.caption(
        "Сортировка по скору (лучшие сверху); кластеры без оценки — в конце. "
        "«обновлено» — по created_at кластера."
    )
    if visible:
        _render_cluster_card(conn, visible)


def _render_filters(conn: Connection) -> ClusterFilters:
    """Filter row: kind, sources, min score, date range (combined with AND)."""
    with st.container(horizontal=True):
        kinds = st.multiselect(
            "Тип сигнала",
            ("demand", "complaint", "discussion"),
            format_func=lambda kind: _KIND_LABELS.get(str(kind), str(kind)),
        )
        source_names = list_source_names(conn)
        sources = st.multiselect("Источники", source_names)
        min_score = st.slider("Минимальный скор", 0.0, float(MAX_SCORE), 0.0, step=0.5)
        from_date = st.date_input("Создан с", value=None)
        to_date = st.date_input("Создан по", value=None)
    return ClusterFilters(
        kinds=frozenset(kinds) if kinds else None,
        sources=frozenset(sources) if sources else None,
        min_score=min_score if min_score > 0.0 else None,
        from_date=from_date,
        to_date=to_date,
    )


def _render_clusters_table(rows: list[ClusterRow]) -> None:
    """The filtered clusters table."""
    st.dataframe(
        _clusters_dataframe(rows),
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn(format="%.2f"),
            "size": st.column_config.NumberColumn(),
        },
    )


def _render_cluster_card(conn: Connection, rows: list[ClusterRow]) -> None:
    """Master-detail card: pick a cluster, see rationale, quotes, pains."""
    labels = {
        f"{row.title} — скор {row.score:.2f}"
        if row.score is not None
        else f"{row.title} — без оценки": row.id
        for row in rows
    }
    chosen = st.selectbox("Карточка кластера", list(labels), index=0)
    detail = get_cluster_detail(conn, labels[chosen])
    if detail is None:
        st.warning("Кластер не найден.")
        return
    header_cols = st.columns(4)
    header_cols[0].metric("Скор", "—" if detail.score is None else f"{detail.score:.2f}")
    header_cols[1].metric("Размер", str(detail.size))
    header_cols[2].metric("Типы", _format_kinds(detail.kinds))
    header_cols[3].metric("Источники", str(len(detail.sources)))
    if detail.rationale_md:
        st.subheader("Обоснование")
        st.markdown(detail.rationale_md)
    if detail.quotes:
        st.subheader("Цитаты")
        for quote in detail.quotes:
            st.markdown(f"> {quote}")
    st.subheader(f"Боли ({len(detail.pains)})")
    for pain in detail.pains:
        date_suffix = f" · {pain.published_at:%d.%m.%Y}" if pain.published_at is not None else ""
        with st.expander(
            f"{_KIND_LABELS.get(pain.kind, pain.kind)} · {pain.source_name} · "
            f"{pain.post_title}{date_suffix}"
        ):
            st.markdown(pain.body)
            st.caption(f"Аудитория: {pain.audience}")
            st.markdown(f"> {pain.quote}")
            st.link_button("Открыть пост", pain.post_url)


#: Repo root: cwd for the run subprocess (resolves cli.py + PGDATA_DIR).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _render_run_page() -> None:
    """Run ("Сбор") page: launch button + stage progress from the run table."""
    st.title("Сбор")
    conn = _connect()
    if conn is None:
        st.error(_DB_DOWN_TEXT)
        return
    busy = has_active_run(conn)
    runs = list_runs(conn, limit=5)
    current = runs[0] if runs and runs[0].is_running else None
    if current is not None:
        st.info("Идёт прогон сборa — повторный запуск заблокирован, пока он не завершится.")
    button_disabled = busy
    if st.button(
        "Прогнать сбор",
        disabled=button_disabled,
        help=(
            "Запуск недоступен, пока выполняется другой прогон."
            if button_disabled
            else "Полный прогон: сбор → извлечение → кластеризация → скоринг."
        ),
        type="primary" if not button_disabled else "secondary",
    ):
        subprocess.Popen(
            run_command(),
            cwd=_REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        st.success("Прогон запущен в фоне. Обновите страницу, чтобы увидеть прогресс.")
        st.rerun()
    st.subheader("Прогресс последнего прогона")
    if current is not None:
        _render_run_progress(current)
    elif runs:
        _render_run_progress(runs[0])
    else:
        st.info(
            "Прогонов ещё не было — нажмите «Прогнать сбор» или выполните "
            "`uv run python cli.py run`."
        )
    if runs:
        st.subheader("Последние прогоны")
        st.dataframe(_runs_dataframe(runs), hide_index=True)


def _render_run_progress(run: RunRow) -> None:
    """Stage progress block for one run: status + counters per stage."""
    caption = f"Прогон от {_format_updated(run.started_at)}"
    if run.finished_at is not None:
        caption += f", завершён за {(run.finished_at - run.started_at).total_seconds():.0f} с"
    st.caption(caption)
    for progress in stage_progress(run):
        cols = st.columns([2, 2, 6])
        cols[0].write(f"**{progress.name}**")
        cols[1].write(progress.status)
        cols[2].write(progress.metrics)


def _runs_dataframe(runs: list[RunRow]) -> pd.DataFrame:
    """History table: one row per run, newest first."""
    data: dict[str, list[object]] = {
        "запущен": [_format_updated(run.started_at) for run in runs],
        "завершён": [
            _format_updated(run.finished_at) if run.finished_at is not None else "—" for run in runs
        ],
        "стадии": [
            ", ".join(f"{stage}: {status}" for stage, status in run.stages.items()) or "—"
            for run in runs
        ],
        "стоимость, ₽": [f"{run.cost:.4f}" for run in runs],
    }
    return pd.DataFrame(data, columns=("запущен", "завершён", "стадии", "стоимость, ₽"))


#: Status rendering for source health rows.
_SOURCE_WARN = "нет успешных постов"


def _health_dataframe(sources: list[SourceHealthRow]) -> pd.DataFrame:
    """Source health table: last success, failures, suspect-short share."""
    data: dict[str, list[object]] = {
        "источник": [row.name for row in sources],
        "последний успех": [
            _format_updated(row.last_ok) if row.last_ok is not None else _SOURCE_WARN
            for row in sources
        ],
        "ошибок": [str(row.failed_count) for row in sources],
        "всего постов": [str(row.total_count) for row in sources],
        "suspect_short, %": [f"{row.suspect_share * 100:.0f}" for row in sources],
    }
    return pd.DataFrame(
        data,
        columns=("источник", "последний успех", "ошибок", "всего постов", "suspect_short, %"),
    )


def _render_health_page() -> None:
    """Health ("Здоровье") page: run history, cost, source health."""
    st.title("Здоровье")
    conn = _connect()
    if conn is None:
        st.error(_DB_DOWN_TEXT)
        return
    report = health_report(conn)
    st.subheader("История прогонов")
    if not report.runs:
        st.info("Прогонов ещё не было — запустите сбор на странице «Сбор».")
    else:
        st.dataframe(_runs_dataframe(report.runs), hide_index=True)
        total_cost = sum(run.cost for run in report.runs)
        st.caption(f"Суммарная стоимость последних {len(report.runs)} прогонов: {total_cost:.4f} ₽")
    st.subheader("Здоровье источников")
    if not report.sources:
        st.info("Источники не настроены.")
        return
    st.dataframe(_health_dataframe(report.sources), hide_index=True)
    st.caption(
        "suspect_short — доля постов с подозрительно коротким текстом "
        "(кандидаты на повторную экстракцию)."
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
    elif page == "Сбор":
        _render_run_page()
    elif page == "Здоровье":
        _render_health_page()
    elif page == "Промпты":
        render_prompts_page()
    elif page == "Источники":
        render_sources_page()
    else:
        _render_placeholder_page(page)


if __name__ == "__main__":
    main()
