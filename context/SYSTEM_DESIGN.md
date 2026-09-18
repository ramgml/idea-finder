# System design idea-finder

> Полные обоснования решений — context/CONTEXT.md. Этот файл — рабочая справка по архитектуре для воркеров.

## Стиль

Модульный монолит, batch-ETL, ports & adapters на стыках изменчивости.
Локальный однопользовательский инструмент. Нет: Celery/очереди, REST между компонентами, Docker, микросервисы.

## Схема

```
                    ┌────────────────────────────────┐
  Streamlit (web/) ──▶   PostgreSQL (pgserver)  ◀── CLI (cli.py)
  read + кнопка        ▲    raw_post, pain,       run collect|extract|
  «Прогнать сбор»      │    cluster, score,       cluster|score|run|status
       │               │    run (статусы)              
       ▼               │                              
  subprocess ──▶ core/pipeline.py (оркестратор стадий)
                  collect → extract → cluster → score
                     │          │        │        │
                 sources/     llm/    e5+pgvector  llm/
                 adapters    extract              score
                     │
                 fetch/ (httpx→trafilatura→playwright)
```

## Ключевые инварианты

1. **Стадии — идемпотентные функции над состоянием БД.** collect→extract→cluster→score;
   каждая читает свои входные таблицы, пишет выходные; прогресс в таблице `run`
   (оттуда дашборд рисует статус кнопки). Падение на середине → перезапуск стадии безопасен.
2. **Ports & adapters только на стыках изменчивости:**
   - `SourceAdapter` (Protocol): `name`, `fetch_new(since) -> list[RawPost]` — новый источник = новый файл;
   - `Fetcher`: `fetch(url) -> text` — можно подменить на Firecrawl-адаптер при необходимости;
   - `LlmClient`: factory по активному провайдеру из БД; `FakeLlmClient` — провайдер kind=fake (мок-режим).
3. **Две точки входа:** CLI (стадии по отдельности + полный прогон) и Streamlit
   (чтение + subprocess-запуск прогона). Streamlit никогда не пишет в пайплайн напрямую.
4. **Все записи в БД — через `db/repo.py`** (одно место для идемпотентности/unique-констрейнтов).
   Стадии не знают друг о друге — только таблицы.
5. **ty strict в `core/` и `db/`.**

## Структура репо

```
idea_finder/
  sources/     # base.py (Protocol SourceAdapter), fl_ru.py, habr.py, gplay.py
  fetch/       # httpx_fetcher.py, extract.py (trafilatura→selectolax), browser.py (playwright, captcha)
  llm/         # client.py (factory + FakeLlmClient), extract_pains.py, score.py, prompts/*.md
  core/        # pipeline.py (стадии), models.py (dataclasses), canonical.py, embed.py
  db/          # bootstrap_pgserver.py, schema.sql, repo.py
  web/         # streamlit_app.py
  cli.py       # argparse: collect|extract|cluster|score|run|status|captcha|llm-test
  fixtures/    # регресс-датасет для промптов
```

## Поток данных (пайплайн)

```
Источники → RawPost{source_id, url_canon, title, text, published_at, kind}
  → LLM-извлечение боли (FakeLlmClient в мок-режиме) → Pain{body, audience, quote}
  -> эмбеддинг pain.body (e5-small, CPU) -> кластеризация (косинус, порог 0.88, G1-калибровка; известное ограничение centroid-drift на малых корпусах — см. D3)
  → Cluster{size, kind_mix} → скоринг по рубрике (context/SCORING.md) → Score{total, rationale, quotes}
```

## Таблицы БД

`source, raw_post, pain, cluster, score, run, prompt_version, llm_provider` —
детали в `db/schema.sql` (задача T297); миграции sequential, никогда не править существующие.

## Промпты

- Файлы `llm/prompts/*.md` (Jinja2) — базовые версии в git.
- Таблица `prompt_version(name, version, body, source=file|ui)` — активные версии, история, откат.
- `run`/`score` хранят `prompt_version_id` — воспроизводимость и сравнение версий.
- UI-страница «Промпты»: редактирование = новая версия, диффы, «Прогнать регрессию» на фикстурах.
