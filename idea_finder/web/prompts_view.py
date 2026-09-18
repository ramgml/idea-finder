"""Prompts page backend: versions, diffs, rollback, seeding (task 314).

Schema ownership note (task 314 constraint): ``db/repo.py`` is shared with
the extract/score streams, so only :func:`idea_finder.db.repo.upsert_prompt_version`
is reused (the existing immutable-(name, version) insert). Everything else —
reads, the UI write helpers, and first-start seeding from ``llm/prompts/*.md``
— lives in this web module.

TODO(repo): move the read/write functions below into ``db/repo.py`` once the
parallel streams merge; keep the signatures and SQL as-is.

Active version semantics (what the pipeline relies on): ``core/pipeline.py``
pins ``(name, PROMPT_VERSION)`` via ``upsert_prompt_version`` write-once, so a
pipeline run always receives an explicit ``prompt_version_id``; nothing reads
"the active version" implicitly. For the page (and any future pipeline
pickup) the active version of a name is defined as ``max(version)``, which is
what :func:`get_active_version` returns.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from typing import Final, LiteralString, cast

import streamlit as st
from psycopg import Connection

from idea_finder.db.bootstrap_pgserver import DbError, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import upsert_prompt_version

logger = logging.getLogger(__name__)

#: Prompt names seeded from ``idea_finder/llm/prompts/*.md`` on first start,
#: mapped to the loader used when the name is missing from the database.
_SEED_PROMPTS: Final[tuple[str, ...]] = ("extract_pains", "score")

#: Insert helper: same immutability contract as ``repo.upsert_prompt_version``
#: (bodies are immutable once shipped to a run), ``source`` is always 'ui'
#: for edits and rollbacks made from the dashboard.
_SOURCE_UI: Final[str] = "ui"

#: Russian labels for prompt_version.source values shown in the UI.
_SOURCE_LABELS: Final[dict[str, str]] = {"file": "из файла", "ui": "редактор"}

#: Database-down message, same contract as the Clusters page in streamlit_app.
_DB_DOWN_TEXT: Final[str] = (
    "База данных недоступна: встроенный Postgres не поднят. Выполните "
    "`uv run python cli.py status` (инициализирует кластер), затем "
    "перезагрузите страницу."
)


@dataclass(frozen=True, slots=True)
class PromptVersion:
    """One immutable prompt_version row as shown in the version list."""

    id: str
    name: str
    version: int
    body: str
    source: str
    created_at: datetime


_LIST_VERSIONS_SQL: Final[LiteralString] = """
    SELECT id, name, version, body, source, created_at
    FROM prompt_version
    WHERE name = %s
    ORDER BY version ASC
"""

#: Row shape of the two six-column SELECTs above/below, for ty.
_VersionRow = tuple[str, str, int, str, str, datetime]

_LIST_NAMES_SQL: Final[LiteralString] = """
    SELECT DISTINCT name
    FROM prompt_version
"""

_ACTIVE_VERSION_SQL: Final[LiteralString] = """
    SELECT id, name, version, body, source, created_at
    FROM prompt_version
    WHERE name = %s
    ORDER BY version DESC
    LIMIT 1
"""


def list_prompt_names(conn: Connection) -> list[str]:
    """Return every prompt name stored in the database, alphabetically.

    Sorted in Python, not ``ORDER BY``: the embedded cluster's ru_RU collation
    ignores punctuation, so ``'a-list'`` would land after ``'active-name'``.
    """
    rows = cast("list[tuple[str]]", conn.execute(_LIST_NAMES_SQL).fetchall())
    return sorted(str(name) for (name,) in rows)


def list_versions(conn: Connection, name: str) -> list[PromptVersion]:
    """Return all versions of one prompt name, oldest first (v1, v2, ...)."""
    rows = cast("list[_VersionRow]", conn.execute(_LIST_VERSIONS_SQL, (name,)).fetchall())
    return [_row_to_version(row) for row in rows]


def get_active_version(conn: Connection, name: str) -> PromptVersion | None:
    """Return the active (max version) row for a name, or None if unknown."""
    row = cast("_VersionRow | None", conn.execute(_ACTIVE_VERSION_SQL, (name,)).fetchone())
    return None if row is None else _row_to_version(row)


def get_version(conn: Connection, name: str, version: int) -> PromptVersion | None:
    """Return one exact ``(name, version)`` row, or None if unknown."""
    for row in list_versions(conn, name):
        if row.version == version:
            return row
    return None


def _row_to_version(row: _VersionRow) -> PromptVersion:
    """Shape one SELECT row into a :class:`PromptVersion`."""
    version_id, name, version, body, source, created_at = row
    return PromptVersion(
        id=version_id,
        name=name,
        version=version,
        body=body,
        source=source,
        created_at=created_at,
    )


def save_new_version(conn: Connection, name: str, body: str) -> PromptVersion:
    """Insert ``body`` as the next version of ``name`` with source='ui'.

    The next version number is ``max(version) + 1`` (1 for the first row of
    the name). Re-entrant per name only within one transaction: the page is
    single-user (embedded postgres, one owner), so no extra locking.
    """
    current = get_active_version(conn, name)
    next_version = 1 if current is None else current.version + 1
    upsert_prompt_version(conn, name, next_version, body, _SOURCE_UI)
    stored = get_version(conn, name, next_version)
    assert stored is not None  # upsert returned an id; the row exists
    return stored


def rollback_to_version(conn: Connection, name: str, version: int) -> PromptVersion:
    """Create a new UI version whose body is the content of ``version``.

    Rollback never rewrites history: the target row stays untouched and a
    fresh row (``max(version) + 1``, source='ui') carries the old content
    forward, so runs already pinned to earlier versions stay reproducible.
    """
    target = get_version(conn, name, version)
    if target is None:
        msg = f"prompt {name!r} has no version {version}"
        raise ValueError(msg)
    return save_new_version(conn, name, target.body)


def unified_diff(left: PromptVersion, right: PromptVersion) -> str:
    """Unified diff between two versions of the same prompt.

    Rendered with v(N-1) on the left and v(N) on the right; empty string
    when the bodies are identical.
    """
    return "".join(
        difflib.unified_diff(
            left.body.splitlines(keepends=True),
            right.body.splitlines(keepends=True),
            fromfile=f"v{left.version}",
            tofile=f"v{right.version}",
        )
    )


def _seed_template(name: str) -> str:
    """Return the packaged prompt template text for one seed name."""
    return (
        resources.files("idea_finder.llm")
        .joinpath(f"prompts/{name}.md")
        .read_text(encoding="utf-8")
    )


def seed_prompts_from_files(conn: Connection) -> list[str]:
    """Register v1 of each packaged prompt if its name is missing entirely.

    Idempotent by name, not by (name, version): a name that already has any
    row (e.g. a UI v2) is never re-seeded — the pipeline pins the file
    version itself on each run via ``upsert_prompt_version``, this only
    guarantees that the page lists both shipped prompts on a fresh database.
    Returns the names inserted now (empty list on a repeat call).
    """
    seeded: list[str] = []
    existing = set(list_prompt_names(conn))
    for name in _SEED_PROMPTS:
        if name in existing:
            continue
        upsert_prompt_version(conn, name, 1, _seed_template(name), "file")
        seeded.append(name)
    return seeded


def _version_caption(row: PromptVersion) -> str:
    """Radio label: ``v2 · редактор · 18.09.2026 12:00``."""
    stamp = row.created_at.strftime("%d.%m.%Y %H:%M")
    return f"v{row.version} · {_SOURCE_LABELS.get(row.source, row.source)} · {stamp}"


def render_prompts_page() -> None:
    """Prompts page: versions of one prompt, diff, rollback, save-as-new."""
    st.title("Промпты")
    try:
        handle = ensure_pgserver()  # PGDATA_DIR is read per call
        conn = handle.get_conn()
        apply_migrations(conn)
    except (DbError, OSError) as e:
        logger.warning("dashboard: database unavailable: %s", e)
        st.error(_DB_DOWN_TEXT)
        return

    seeded_now = seed_prompts_from_files(conn)
    # The page owns its script-run connection: psycopg opens an implicit
    # transaction on first statement and nothing else commits it, so without
    # an explicit commit the seeding (and every button write below) is rolled
    # back when the script run ends.
    conn.commit()
    if seeded_now:
        st.toast(f"Зарегистрированы промпты из файлов: {', '.join(seeded_now)}")

    names = list_prompt_names(conn)
    if not names:
        st.info("В базе нет ни одной версии промптов — не удалось выполнить сидирование.")
        return

    name = st.sidebar.selectbox("Промпт", names, key="prompts-name")
    versions = list_versions(conn, name)
    if not versions:  # unreachable: names come from prompt_version rows
        st.warning(f"У промпта {name!r} нет версий.")
        return

    active = versions[-1]
    st.caption(
        f"Активная версия: v{active.version} ({_SOURCE_LABELS.get(active.source, active.source)})"
        f" · всего версий: {len(versions)}"
    )

    left, right = st.columns([1, 3])
    with left:
        chosen_version = st.radio(
            "Версия",
            [row.version for row in versions],
            index=len(versions) - 1,
            format_func=lambda v: _version_caption(next(r for r in versions if r.version == v)),
        )
    version = get_version(conn, name, chosen_version)
    if version is None:  # unreachable: the radio lists rows we just read
        st.warning(f"Версия v{chosen_version} не найдена.")
        return

    with right:
        _render_version_view(conn, name, versions, version)


def _render_version_view(
    conn: Connection, name: str, versions: list[PromptVersion], version: PromptVersion
) -> None:
    """Version detail: text, diff against the previous one, actions."""
    st.subheader(f"Текст v{version.version}")
    st.code(version.body, language="markdown")

    previous = get_version(conn, name, version.version - 1)
    if previous is not None:
        st.subheader(f"Дифф v{previous.version} → v{version.version}")
        diff = unified_diff(previous, version)
        if diff:
            st.code(diff, language="diff")
        else:
            st.caption("Тексты идентичны.")

    st.subheader("Сохранить правку")
    edited = st.text_area(
        "Текст новой версии",
        value=version.body,
        height=320,
        key=f"prompts-edit-{version.version}",
    )
    if st.button(
        f"Сохранить как новую версию (v{len(versions) + 1})",
        disabled=edited == version.body,
    ):
        saved = save_new_version(conn, name, edited)
        conn.commit()
        st.toast(f"Сохранена v{saved.version} ({saved.source}).")
        st.rerun()

    earlier = [row.version for row in versions if row.version != version.version]
    if earlier and st.button(f"Откатить к v{version.version}"):
        saved = rollback_to_version(conn, name, version.version)
        conn.commit()
        st.toast(f"Откат: v{saved.version} повторяет содержимое v{version.version}.")
        st.rerun()
