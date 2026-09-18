"""LLM settings page backend: provider CRUD with write-only keys (task 318).

Schema ownership note (task 318 constraint): ``db/repo.py`` is shared with
the parallel streams, so the insert path reuses
:func:`idea_finder.db.repo.upsert_llm_provider` as-is. Everything else — the
masked listing read, enable/default/delete helpers, and masking — lives in
this web module, following the prompts_view precedent.

TODO(repo): move the read/write functions below into ``db/repo.py`` once the
parallel streams merge; keep the signatures and SQL as-is.

Secrets contract: the full ``api_key`` never leaves the database through
this module. The listing SELECT masks it server-side (the CASE mirrors
:func:`mask_key`), the add/update form takes the key through a password
field only, an empty key on update keeps the stored one, and nothing here
logs or renders the raw value. Smoke-testing a provider is done from the
terminal (``uv run python cli.py llm-test``), never from the page.

Exactly-one-active guarantee: the partial unique index
``uq_llm_provider_single_active`` makes a second active row impossible.
:func:`set_default` flips the previous active row off in the same
transaction (same shape as ``repo.upsert_llm_provider``); a
``UniqueViolation`` from a race is translated into a :class:`ValueError`
the page can show as a readable error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, LiteralString, cast

import streamlit as st
from psycopg import Connection
from psycopg import errors as pg_errors

from idea_finder.db.bootstrap_pgserver import DbError, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import upsert_llm_provider

logger = logging.getLogger(__name__)

#: Provider kinds allowed by the ``llm_provider.kind`` CHECK constraint.
_KINDS: Final[tuple[str, ...]] = ("fake", "openai_compat")

#: Russian labels for provider kinds shown in the list and the add form.
_KIND_LABELS: Final[dict[str, str]] = {
    "fake": "fake (мок-режим)",
    "openai_compat": "OpenAI-совместимый",
}

#: Database-down message, same contract as the other pages.
_DB_DOWN_TEXT: Final[str] = (
    "База данных недоступна: встроенный Postgres не поднят. Выполните "
    "`uv run python cli.py status` (инициализирует кластер), затем "
    "перезагрузите страницу."
)

#: Empty-state text: also explains how mock mode is set up.
_EMPTY_TEXT: Final[str] = (
    "Провайдеров ещё нет. Для мок-режима добавьте провайдер kind=fake и "
    "сделайте его активным; реальный провайдер — kind=openai_compat с "
    "base_url, моделью и API-ключом."
)

#: Terminal-only smoke hint: the page itself never calls the LLM.
_LLM_TEST_HINT: Final[str] = (
    "Проверка активного провайдера — из терминала: "
    "`uv run python cli.py llm-test` (страница не вызывает LLM)."
)


@dataclass(frozen=True, slots=True)
class ProviderRow:
    """One ``llm_provider`` row as shown in the list — masked key only."""

    name: str
    kind: str
    base_url: str
    model: str
    key_masked: str
    is_active: bool


#: Listing SELECT: the full api_key is never fetched — the CASE mirrors
#: :func:`mask_key` server-side, so a leak through this module is
#: structurally impossible.
_LIST_PROVIDERS_SQL: Final[LiteralString] = """
    SELECT name, kind, base_url, model,
           CASE WHEN length(api_key) > 4 THEN '****' || right(api_key, 4)
                ELSE '****'
           END AS key_masked,
           is_active
    FROM llm_provider
    ORDER BY is_active DESC, name
"""

#: Row shape of :data:`_LIST_PROVIDERS_SQL`, for ty.
_ProviderRow = tuple[str, str, str, str, str, bool]


def mask_key(api_key: str) -> str:
    """Mask an API key to ``****`` + its last 4 characters.

    Keys of four characters or fewer (and empty ones) are hidden entirely:
    the tail of a short key would be the whole key.
    """
    if len(api_key) > 4:
        return f"****{api_key[-4:]}"
    return "****"


def _validate_provider(name: str, kind: str, base_url: str) -> None:
    """Shared input validation for add and update; raises ``ValueError``."""
    if not name.strip():
        msg = "Имя провайдера не может быть пустым."
        raise ValueError(msg)
    if kind not in _KINDS:
        msg = f"Неизвестный тип провайдера: {kind!r} (допустимо: fake, openai_compat)."
        raise ValueError(msg)
    if kind == "openai_compat" and not base_url.strip():
        msg = "Для kind=openai_compat обязателен base_url."
        raise ValueError(msg)


def list_providers(conn: Connection) -> list[ProviderRow]:
    """Return every provider, active first, with the key masked server-side."""
    rows = cast("list[_ProviderRow]", conn.execute(_LIST_PROVIDERS_SQL).fetchall())
    return [
        ProviderRow(
            name=name,
            kind=kind,
            base_url=base_url,
            model=model,
            key_masked=key_masked,
            is_active=is_active,
        )
        for name, kind, base_url, model, key_masked, is_active in rows
    ]


def add_provider(
    conn: Connection,
    name: str,
    kind: str,
    base_url: str,
    model: str,
    api_key: str,
    *,
    is_active: bool = False,
) -> str:
    """Insert a new provider; an existing name is an error, not an overwrite.

    Returns the row id. ``api_key`` is stored only in the database; the
    caller passes it from the password field and nothing logs it.
    """
    clean = name.strip()
    _validate_provider(clean, kind, base_url)
    existing = conn.execute("SELECT 1 FROM llm_provider WHERE name = %s", (clean,)).fetchone()
    if existing is not None:
        msg = f"Провайдер с именем {clean!r} уже есть — обновите его через форму."
        raise ValueError(msg)
    return upsert_llm_provider(
        conn,
        clean,
        kind,
        base_url.strip(),
        model.strip(),
        api_key,
        is_active=is_active,
    )


def update_provider(
    conn: Connection,
    name: str,
    kind: str,
    base_url: str,
    model: str,
    api_key: str = "",
) -> str:
    """Update an existing provider; an empty ``api_key`` keeps the stored one.

    The stored secret is never read back into Python: the CASE in the SQL
    keeps the old value when the replacement is empty.
    """
    clean = name.strip()
    _validate_provider(clean, kind, base_url)
    with conn.transaction():
        row = conn.execute(
            """
            UPDATE llm_provider SET
                kind = %s,
                base_url = %s,
                model = %s,
                api_key = CASE WHEN %s = '' THEN api_key ELSE %s END
            WHERE name = %s
            RETURNING id
            """,
            (kind, base_url.strip(), model.strip(), api_key, api_key, clean),
        ).fetchone()
    if row is None:
        msg = f"Провайдер {clean!r} не найден."
        raise ValueError(msg)
    return str(row[0])


def delete_provider(conn: Connection, name: str) -> bool:
    """Delete one provider by name; ``False`` when the name is unknown."""
    cursor = conn.execute("DELETE FROM llm_provider WHERE name = %s", (name,))
    return cursor.rowcount > 0


def set_default(conn: Connection, name: str) -> None:
    """Make ``name`` the single active provider (atomically flips the old one).

    The previous active row is switched off in the same transaction, so the
    partial unique index ``uq_llm_provider_single_active`` never trips in
    normal operation; a race-induced ``UniqueViolation`` surfaces as a
    readable :class:`ValueError`.
    """
    try:
        with conn.transaction():
            deactivated = conn.execute(
                "UPDATE llm_provider SET is_active = false WHERE is_active AND name <> %s",
                (name,),
            ).rowcount
            activated = conn.execute(
                "UPDATE llm_provider SET is_active = true WHERE name = %s",
                (name,),
            ).rowcount
            # Inside the transaction on purpose: an unknown name rolls the
            # deactivation back, so a failed switch changes nothing.
            if activated == 0:
                msg = f"Провайдер {name!r} не найден."
                raise ValueError(msg)
    except pg_errors.UniqueViolation as e:
        msg = "Не удалось сделать провайдер активным: уже есть активный провайдер."
        raise ValueError(msg) from e
    logger.info(
        "llm settings: default provider set to %s (deactivated %d)",
        name,
        deactivated,
    )


def set_enabled(conn: Connection, name: str, enabled: bool) -> None:
    """Turn one provider's ``is_active`` on or off without touching others.

    Enabling through this helper means "make it the single default" (the
    partial unique index forbids two); disabling may legitimately leave no
    active provider — ``cli llm-test`` and the pipeline then report the
    missing configuration instead of silently guessing.
    """
    if enabled:
        set_default(conn, name)
        return
    cursor = conn.execute("UPDATE llm_provider SET is_active = false WHERE name = %s", (name,))
    if cursor.rowcount == 0:
        msg = f"Провайдер {name!r} не найден."
        raise ValueError(msg)


def render_settings_page() -> None:
    """LLM settings page: provider list, default switch, add/update form."""
    st.title("Настройки LLM")
    try:
        handle = ensure_pgserver()  # PGDATA_DIR is read per call
        conn = handle.get_conn()
        apply_migrations(conn)
    except (DbError, OSError) as e:
        logger.warning("dashboard: database unavailable: %s", e)
        st.error(_DB_DOWN_TEXT)
        return

    providers = list_providers(conn)
    st.caption(_LLM_TEST_HINT)
    if not providers:
        st.info(_EMPTY_TEXT)
    else:
        st.subheader("Провайдеры")
        _render_provider_list(conn, providers)

    _render_add_form(conn)


def _render_provider_list(conn: Connection, providers: list[ProviderRow]) -> None:
    """One interactive row per provider: default switch, toggle, delete."""
    for provider in providers:
        cols = st.columns([2, 3, 2, 1.5, 2, 2.5])
        cols[0].markdown(f"**{provider.name}**")
        cols[0].caption(_KIND_LABELS.get(provider.kind, provider.kind))
        cols[1].write(provider.base_url if provider.base_url else "—")
        cols[2].write(provider.model if provider.model else "—")
        # Markdown eats a bare **** (parses it as emphasis), so escape:
        # the mask must show literally, e.g. `****` or `****4242`.
        cols[3].write(provider.key_masked.replace("*", "\\*"))
        if provider.is_active:
            cols[4].markdown(":heavy_check_mark: **по умолчанию**")
            if cols[5].button("Выключить", key=f"llm-off-{provider.name}"):
                try:
                    set_enabled(conn, provider.name, False)
                    conn.commit()
                except ValueError as e:
                    cols[5].error(str(e))
                else:
                    st.toast(f"Провайдер {provider.name!r} выключен.")
                    st.rerun()
        else:
            cols[4].write("выключен")
            if cols[5].button("Сделать активным", key=f"llm-act-{provider.name}"):
                try:
                    set_default(conn, provider.name)
                    conn.commit()
                except ValueError as e:
                    cols[5].error(str(e))
                else:
                    st.toast(f"Провайдер {provider.name!r} теперь по умолчанию.")
                    st.rerun()
        confirmed = cols[5].checkbox("подтвердить", key=f"llm-delok-{provider.name}")
        if cols[5].button("Удалить", key=f"llm-del-{provider.name}", disabled=not confirmed):
            delete_provider(conn, provider.name)
            conn.commit()
            st.toast(f"Провайдер {provider.name!r} удалён.")
            st.rerun()


def _render_add_form(conn: Connection) -> None:
    """Add/update form: the API key field is write-only (password type)."""
    st.subheader("Добавить или обновить провайдера")
    name = st.text_input("Имя", key="llm-add-name")
    kind = st.selectbox(
        "Тип", _KINDS, key="llm-add-kind", format_func=lambda k: _KIND_LABELS.get(k, k)
    )
    base_url = st.text_input("Base URL", key="llm-add-url")
    model = st.text_input("Модель", key="llm-add-model")
    st.caption("Ключ хранится только в базе и не отображается; пусто — не менять при обновлении.")
    api_key = st.text_input("API-ключ", key="llm-add-key", type="password")
    activate_now = st.checkbox("Сделать активным сразу", key="llm-add-active")

    if not st.button("Сохранить провайдера", key="llm-add-save"):
        return
    clean = name.strip()
    try:
        if clean and _name_exists(conn, clean):
            update_provider(conn, clean, kind, base_url, model, api_key)
            if activate_now:
                set_default(conn, clean)
            conn.commit()
            st.toast(f"Провайдер {clean!r} обновлён.")
        else:
            add_provider(conn, clean, kind, base_url, model, api_key, is_active=activate_now)
            conn.commit()
            st.toast(f"Провайдер {clean!r} добавлен.")
    except ValueError as e:
        st.error(str(e))
        return
    # The secret must not outlive the submission in session state.
    st.session_state.pop("llm-add-key", None)
    st.rerun()


def _name_exists(conn: Connection, name: str) -> bool:
    """Whether a provider row with this name is already stored."""
    row = conn.execute("SELECT 1 FROM llm_provider WHERE name = %s", (name,)).fetchone()
    return row is not None
