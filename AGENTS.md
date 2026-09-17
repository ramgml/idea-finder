# AGENTS.md — idea-finder: правила для AI-агентов

> Краткий guide для агентов, работающих с кодовой базой idea-finder. Концепция и архитектурные решения — `context/CONTEXT.md` (источник истины по дизайну). Очередь работ — проект idea-finder (#15) в локальном Orenda (`http://127.0.0.1:2137`), агентский конфиг — `.orenda/agent.yaml` (gitignored).

## Что такое idea-finder?

Система поиска бизнес-идей: собирает боли пользователей с российских площадок
(freelance-биржи, Habr, отзывы), извлекает их через LLM, кластеризует, оценивает
по рубрике рынка РФ и показывает в Streamlit-дашборде. Локальный однопользовательский
инструмент, запуск по запросу (без планировщика). Python 3.12+.

## Stack

| Layer | Tech |
|-------|------|
| Язык | Python 3.12+, пакетный менеджер `uv` |
| Хранилище | PostgreSQL embedded (`pgserver`, unix-сокет, каталог `data/pg`) + pgvector |
| Эмбеддинги | `sentence-transformers` multilingual-e5-small (CPU) |
| LLM | OpenAI-совместимый API (DeepSeek/GLM), реестр провайдеров в БД, `FakeLlmClient` для мок-режима |
| Fetch | httpx (async, aiolimiter) → trafilatura (fallback selectolax) → Playwright (только JS-домены) |
| Дашборд | Streamlit (только чтение + subprocess запуска прогона) |
| Качество | ruff (линт) + ty (типы, strict в `core/` и `db/`) + pytest |

## Directory map

```
idea-finder/
├── idea_finder/
│   ├── sources/          # адаптеры источников: base.py (Protocol), fl_ru.py, habr.py, gplay.py
│   ├── fetch/            # httpx_fetcher.py, extract.py, browser.py (playwright/captcha)
│   ├── llm/              # client.py (factory + FakeLlmClient), extract_pains.py, score.py, prompts/*.md
│   ├── core/             # pipeline.py (стадии), models.py, canonical.py, embed.py
│   ├── db/               # bootstrap_pgserver.py, schema.sql, repo.py (ВСЕ записи через него)
│   └── web/              # streamlit_app.py (страницы: Кластеры, Источники, Промпты, Настройки LLM, Здоровье)
├── cli.py                # collect|extract|cluster|score|run|status|captcha|llm-test
├── fixtures/             # регресс-датасет постов + expected_pains.json
├── context/CONTEXT.md    # концепция, решения, system design
├── data/                 # runtime (gitignored): pg/, browser_profiles/
├── .env                  # PGDATA_DIR; LLM-ключи — в БД, не здесь
└── pyproject.toml
```

## Команды

```bash
uv sync                                  # установка зависимостей
uv run ruff check . && uv run ty check . # линт + типы (гейты)
uv run pytest                            # тесты
uv run python cli.py run                 # полный прогон: collect→extract→cluster→score
uv run python cli.py status              # счётчики таблиц
uv run python cli.py llm-test            # смоук активного LLM-провайдера
uv run python cli.py captcha <domain>    # headed-браузер для ручной капчи
uv run streamlit run idea_finder/web/streamlit_app.py   # дашборд
```

## Coding rules

### Python
1. **Типизация**: аннотации везде; `ty` strict обязателен в `core/` и `db/`; `any` запрещён (используй `object` + narrowing).
2. **Ошибки**: свои исключения по подсистемам (`FetchError`, `LlmError`); `raise ... from e`; никаких голых `except:` — минимум `except Exception as e` с логом.
3. **Асинхронность**: fetch-слой — async httpx; остальной пайплайн — синхронный. Не смешивать без причины.
4. **Записи в БД** — только через `db/repo.py` (идемпотентность, unique-констрейнты в одном месте). Стадии не знают друг о друге — только таблицы.
5. **Идемпотентность стадий**: повторный запуск стадии не должен менять результат (canonical URL + unique + ON CONFLICT).
6. **LLM**: обращения только через `LlmClient` factory (по активному провайдеру из БД); промпты — файлы `llm/prompts/*.md` (Jinja2), версия промпта пишется в `run`/`score`.
7. **Анти-галлюцинация**: `quote` из извлечённой боли обязана строкой входить в исходный текст, иначе запись бракуется (счётчик в stats прогона).
8. **Логи**: `logging`, структурированные сообщения, никаких `print` вне CLI-вывода.
9. **Комментарии и докстринги**: русский в постановках/доках, код и комментарии в коде — английский.

### Database
1. Схема: `source, raw_post, pain, cluster, score, run, prompt_version, llm_provider` — миграции `NNN_*.sql`, sequential, never edit shipped.
2. IDs — UUIDv7; timestamps — UTC ISO 8601.
3. Вектора — `vector` колонки + ivfflat-индекс (pgvector).
4. `api_key` провайдеров — только в БД, в UI маскируется (последние 4 символа).

### Sources (адаптеры)
1. Новый источник = новый файл-адаптер по `SourceAdapter` Protocol: `name`, `fetch_new(since) -> list[RawPost]`. Пайплайн не меняется.
2. `RawPost.kind` обязателен: `demand | complaint | discussion`.
3. Рейт-лимиты per-domain (default 1 rps), реалистичный UA, robots.txt уважать.
4. Антибот-домены (vc.ru, Дзен, Ozon) напрямую не парсить.

## Definition of Done — бинарный

1. **Каждый пункт DoD проверен исполнением**: вывод команды/теста/смоука можно процитировать. «Implemented» — не доказательство.
2. **Гейты зелёные**: `ruff check` + `ty check` + `pytest`. ty strict не ослаблять.
3. **Частичное — сообщай частичным**: какие пункты не прошли и почему. DoD-чекбокс ставится только когда всё зелёное.
4. **No silent scope reduction**: заблокированное/неясное — surface (в отчёте, `orenda agent propose`), не выбрасывать молча.
5. **No stubs**: никакого `TODO: implement`, полей, которые никто не пишет, фейковых fallback'ов. Отложенный шов = отдельная задача в Orenda.

## Workflow

1. Задача — из Orenda проект #15 (`orenda agent next --peek` — read-only; `next` клеймИТ).
2. `orenda agent claim T<N>` → `orenda agent context T<N>` (постановка, DoD, UPDATE-пометки).
3. Worktree per task: `git worktree add .worktrees/task-<N>-<slug> -b task-<N>-<slug>`. Главный checkout — read-only.
4. Коммиты: `task(<N>): short description`; маленькие и частые.
5. Гейты (из корня worktree): `uv run ruff check . && uv run ty check . && uv run pytest`.
6. Готово → доклад владельцу/PM с доказательствами; мержит владелец. После мержа — `git worktree remove` + `prune`.
7. Приёмка дашборда: `hub start review-t<N>` / `streamlit run` на свободном порту 21400–21499; потушить после приёмки.

### Секреты
- LLM-ключи: в БД (`llm_provider`), заводятся через дашборд «Настройки LLM».
- `.env` (только `PGDATA_DIR` и будущая локальная конфигурация) и `.orenda/` — gitignored, 600.
- Ключ в коде/коммите/логе — инцидент: ротация + доклад.

### Что НЕ делать
- ❌ Не писать в БД мимо `db/repo.py`.
- ❌ Не вызывать LLM мимо `LlmClient` factory (в т.ч. хардкодить base_url).
- ❌ Не ослаблять ty strict и не гасить предупреждения инлайн-комментариями.
- ❌ Не трогать чужие worktree и не делать `git checkout/reset/clean/restore` в чужих чекаутах.
- ❌ `--no-verify` на хуках/гейтах не использовать.
- ❌ Не запускать сбор по антибот-доменам напрямую; не поднимать TCP у Postgres (только unix-сокет).
- ❌ Не коммитить `data/`, `.env`, `.orenda/`, `.omp/`.
- ❌ Не выдавать частичную работу за готовую.
