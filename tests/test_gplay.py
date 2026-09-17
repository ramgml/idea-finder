"""Tests for the Google Play adapter (task T304).

Layers:

* Unit layer (the scraper is patched at the adapter's ``_fetch_reviews``
  seam, no network): the 1-3 star complaint filter, kind naming, permalink
  shape and uniqueness, full-text passthrough, the ``since`` filter,
  config-file loading, and per-app failure isolation.
* E2E layer (module-scoped embedded postgres): a 60-review fixture
  (10 apps x 6 complaints) runs through the adapter and
  ``insert_raw_post`` — first run inserts all 60, the re-run inserts 0
  (the DoD acceptance criterion: repeat adds only new reviews).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from psycopg import Connection

from idea_finder.core.canonical import canonical_url
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import ensure_source, insert_raw_post, table_counts
from idea_finder.fetch.httpx_fetcher import FetchError
from idea_finder.sources.base import SourceAdapter, seed_sources
from idea_finder.sources.gplay import (
    DEFAULT_APPS_CONFIG,
    GPLAY_SOURCE_NAME,
    GPlayAdapter,
    GPlayConfigError,
    _review_url,
)

#: First fixture review moment; reviews are spaced a day apart.
_FIRST_AT_UTC: datetime = datetime(2026, 9, 1, tzinfo=UTC)

#: Complaint bodies must read like real RU pains (quote-safe, non-trivial).
_COMPLAINTS: tuple[str, ...] = (
    "Приложение вылетает при попытке оплатить картой, приходится перезапускать",
    "После обновления перестали приходить уведомления о сообщениях",
    "Постоянно виснет на экране загрузки, помогает только перезагрузка телефона",
    "Не могу войти в аккаунт: вечная ошибка авторизации по номеру телефона",
    "Такси приходит через полчаса вместо пяти минут, водители отменяют заказ",
    "Списали деньги дважды за одну подписку, поддержка не отвечает неделю",
)


def _review(n: int, *, score: int, content: str, at: datetime) -> dict[str, object]:
    """Build one scraper-shaped review dict (keys as the library returns)."""
    return {
        "reviewId": f"review-{n:04d}",
        "userName": f"Пользователь {n}",
        "content": content,
        "score": score,
        "thumbsUpCount": n,
        "at": at,
    }


def _complaint_at(day: int) -> datetime:
    """Fixture review date: a day within September 2026."""
    return datetime(2026, 9, day, 12, 0, tzinfo=UTC)


def _six_complaints(start_index: int) -> list[dict[str, object]]:
    """Six 1-3 star reviews with unique ids and texts."""
    reviews: list[dict[str, object]] = []
    for offset in range(6):
        n = start_index + offset
        reviews.append(
            _review(
                n,
                score=n % 3 + 1,  # cycles 1, 2, 3, 1, 2, 3
                content=_COMPLAINTS[offset],
                at=_complaint_at(offset + 1),
            )
        )
    return reviews


def _config(tmp_path: Path, app_ids: list[str]) -> Path:
    """Write an app-id config file and return its path."""
    path = tmp_path / "gplay_apps.json"
    path.write_text(json.dumps({"app_ids": app_ids}), encoding="utf-8")
    return path


def _patched_adapter(
    monkeypatch: pytest.MonkeyPatch,
    reviews_by_app: dict[str, list[dict[str, object]]],
    config_path: Path,
) -> GPlayAdapter:
    """Adapter whose scraper seam returns the fixture reviews per app."""
    adapter = GPlayAdapter(apps_path=config_path)

    def fake_reviews(app_id: str) -> list[dict[str, object]]:
        if app_id not in reviews_by_app:
            raise RuntimeError(f"unexpected app {app_id}")
        return reviews_by_app[app_id]

    monkeypatch.setattr(adapter, "_fetch_reviews", fake_reviews)
    return adapter


async def test_adapter_satisfies_protocol() -> None:
    """The adapter is a SourceAdapter with the registry name gplay."""
    adapter = GPlayAdapter()
    assert isinstance(adapter, SourceAdapter)
    assert adapter.name == GPLAY_SOURCE_NAME


async def test_star_filter_keeps_only_one_to_three_as_complaints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """1-3 star reviews become complaints; 4-5 star ones are dropped."""
    reviews = [
        _review(1, score=1, content="Одна звезда: всё сломалось", at=_complaint_at(1)),
        _review(2, score=2, content="Две звезды: работает через раз", at=_complaint_at(2)),
        _review(3, score=3, content="Три звезды: много багов", at=_complaint_at(3)),
        _review(4, score=4, content="Четыре звезды: хорошо", at=_complaint_at(4)),
        _review(5, score=5, content="Пять звезд: отлично", at=_complaint_at(5)),
    ]
    adapter = _patched_adapter(monkeypatch, {"app.one": reviews}, _config(tmp_path, ["app.one"]))
    posts = await adapter.fetch_new(None)
    assert [post.text for post in posts] == [
        "Одна звезда: всё сломалось",
        "Две звезды: работает через раз",
        "Три звезды: много багов",
    ]
    assert {post.kind for post in posts} == {"complaint"}


async def test_starless_and_textless_reviews_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A star-only rating (no content) and a missing reviewId are skipped."""
    reviews: list[dict[str, object]] = [
        _review(1, score=1, content="   ", at=_complaint_at(1)),
        {"reviewId": "review-0002", "score": 2, "at": _complaint_at(2)},
        _review(3, score=3, content="Баг с уведомлениями", at=_complaint_at(3)),
    ]
    adapter = _patched_adapter(monkeypatch, {"app.one": reviews}, _config(tmp_path, ["app.one"]))
    posts = await adapter.fetch_new(None)
    assert [post.text for post in posts] == ["Баг с уведомлениями"]


async def test_urls_are_unique_and_carry_the_review_fragment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every post URL is the app page plus a per-review fragment, unique."""
    reviews = _six_complaints(start_index=1)
    adapter = _patched_adapter(monkeypatch, {"app.one": reviews}, _config(tmp_path, ["app.one"]))
    posts = await adapter.fetch_new(None)
    assert len({post.url_canon for post in posts}) == len(posts)
    expected = _review_url("app.one", "review-0001")
    assert (
        posts[0].url
        == expected
        == "https://play.google.com/store/apps/details?id=app.one#review=review-0001"
    )
    assert posts[0].url_canon == canonical_url(expected)


async def test_published_at_comes_from_review_at(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The scraper's ``at`` datetime lands in ``published_at`` unchanged."""
    reviews = _six_complaints(start_index=1)
    adapter = _patched_adapter(monkeypatch, {"app.one": reviews}, _config(tmp_path, ["app.one"]))
    posts = await adapter.fetch_new(None)
    assert posts[0].published_at == _complaint_at(1)
    assert posts[0].published_at is not None and posts[0].published_at.tzinfo is not None


async def test_since_keeps_reviews_at_or_after_cutoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reviews at/after ``since`` survive; older and undated ones drop."""
    reviews = [
        _review(1, score=1, content="Старая боль", at=datetime(2026, 8, 30, tzinfo=UTC)),
        _review(2, score=2, content="Свежая боль", at=datetime(2026, 9, 5, tzinfo=UTC)),
    ]
    adapter = _patched_adapter(monkeypatch, {"app.one": reviews}, _config(tmp_path, ["app.one"]))
    posts = await adapter.fetch_new(since=datetime(2026, 9, 1, tzinfo=UTC))
    assert [post.text for post in posts] == ["Свежая боль"]


async def test_naive_since_is_treated_as_utc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A naive ``since`` must not raise on comparison; UTC is assumed."""
    reviews = [_review(1, score=1, content="Боль", at=_complaint_at(2))]
    adapter = _patched_adapter(monkeypatch, {"app.one": reviews}, _config(tmp_path, ["app.one"]))
    posts = await adapter.fetch_new(since=datetime(2026, 9, 2))  # noqa: DTZ001 -- naive by design
    assert len(posts) == 1


async def test_dead_app_does_not_sink_the_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One failing app logs a warning; the other app's reviews survive."""
    reviews_by_app = {"app.good": _six_complaints(start_index=1)}
    adapter = GPlayAdapter(apps_path=_config(tmp_path, ["app.bad", "app.good"]))

    def fake_reviews(app_id: str) -> list[dict[str, object]]:
        if app_id == "app.bad":
            raise RuntimeError("play is down")
        return reviews_by_app[app_id]

    monkeypatch.setattr(adapter, "_fetch_reviews", fake_reviews)
    with caplog.at_level("WARNING"):
        posts = await adapter.fetch_new(None)
    assert len(posts) == 6
    assert "app.bad" in caplog.text


async def test_network_failure_degrades_to_empty_batch_and_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every app timing out degrades to [] with warnings, no exception.

    Task T334 DoD: ``collect`` must survive an unreachable Google Play —
    the adapter returns an empty harvest, logs one warning per app, and
    bumps ``network_errors`` once per failed app.
    """
    adapter = GPlayAdapter(apps_path=_config(tmp_path, ["app.one", "app.two"]))

    def timeout_reviews(app_id: str) -> list[dict[str, object]]:
        raise TimeoutError(f"read timed out for {app_id}")

    monkeypatch.setattr(adapter, "_fetch_reviews", timeout_reviews)
    with caplog.at_level("WARNING"):
        posts = await adapter.fetch_new(None)
    assert posts == []
    assert adapter.network_errors == 2
    assert "TimeoutError" in caplog.text
    assert "app.one" in caplog.text and "app.two" in caplog.text


async def test_one_network_failure_keeps_healthy_apps_and_counts_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A single network-failing app is skipped and counted; the rest survive."""
    reviews_by_app = {"app.good": _six_complaints(start_index=1)}
    adapter = GPlayAdapter(apps_path=_config(tmp_path, ["app.down", "app.good"]))

    def flaky_reviews(app_id: str) -> list[dict[str, object]]:
        if app_id == "app.down":
            raise ConnectionError("name resolution failed")
        return reviews_by_app[app_id]

    monkeypatch.setattr(adapter, "_fetch_reviews", flaky_reviews)
    posts = await adapter.fetch_new(None)
    assert [post.text for post in posts] == list(_COMPLAINTS)
    assert adapter.network_errors == 1


async def test_config_error_from_seam_still_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GPlayConfigError crossing the seam propagates instead of degrading."""
    adapter = GPlayAdapter(apps_path=_config(tmp_path, ["app.broken"]))

    def broken_reviews(app_id: str) -> list[dict[str, object]]:
        raise GPlayConfigError(f"unreachable config seam for {app_id}")

    monkeypatch.setattr(adapter, "_fetch_reviews", broken_reviews)
    with pytest.raises(GPlayConfigError):
        await adapter.fetch_new(None)
    assert adapter.network_errors == 0


async def test_missing_config_raises_gplay_config_error(tmp_path: Path) -> None:
    """A missing config file is GPlayConfigError (a FetchError subtype)."""
    adapter = GPlayAdapter(apps_path=tmp_path / "absent.json")
    with pytest.raises(GPlayConfigError) as excinfo:
        await adapter.fetch_new(None)
    assert isinstance(excinfo.value, FetchError)


async def test_unparsable_config_raises_gplay_config_error(tmp_path: Path) -> None:
    """Broken JSON in the config is GPlayConfigError, not a JSON crash."""
    path = tmp_path / "gplay_apps.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(GPlayConfigError):
        await GPlayAdapter(apps_path=path).fetch_new(None)


async def test_config_without_app_ids_raises(tmp_path: Path) -> None:
    """A config whose app_ids list is missing or empty is rejected."""
    for payload in ({}, {"app_ids": []}, {"app_ids": "not-a-list"}):
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(GPlayConfigError):
            await GPlayAdapter(apps_path=path).fetch_new(None)


def test_default_config_ships_a_real_app_list() -> None:
    """The bundled config holds >=6 RU-relevant app ids."""
    payload = json.loads(Path(DEFAULT_APPS_CONFIG).read_text(encoding="utf-8"))
    assert len(payload["app_ids"]) >= 6
    assert all(isinstance(app_id, str) and app_id for app_id in payload["app_ids"])


# ------------------------------------------------------------ e2e pg layer


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """One embedded postgres cluster for the e2e leg."""
    data_dir = tmp_path_factory.mktemp("pg-gplay") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture(scope="module")
def conn(pg: PgHandle) -> Iterator[Connection]:
    """Connection to the migrated application database."""
    with pg.get_conn() as connection:
        apply_migrations(connection)
        yield connection


def _ten_app_reviews() -> dict[str, list[dict[str, object]]]:
    """DoD fixture: 10 apps x 6 complaint reviews = 60 reviews total."""
    return {f"ru.test.app{n}": _six_complaints(start_index=n * 100) for n in range(1, 11)}


async def test_dod_first_run_inserts_sixty_second_run_zero(
    monkeypatch: pytest.MonkeyPatch,
    conn: Connection,
    tmp_path: Path,
) -> None:
    """DoD acceptance: >50 complaints first run, 0 new on the re-run.

    The adapter re-delivers the same reviews on the second ``fetch_new``
    (Google Play has no read cursor); the url_canon dedup in
    ``insert_raw_post`` must swallow the whole second pass.
    """
    reviews_by_app = _ten_app_reviews()
    adapter = _patched_adapter(
        monkeypatch,
        reviews_by_app,
        _config(tmp_path, list(reviews_by_app)),
    )
    seed_sources(conn)
    source_id = ensure_source(conn, GPLAY_SOURCE_NAME)

    posts = await adapter.fetch_new(None)
    assert len(posts) == 60  # 10 apps x 6 complaints, all >50 combined
    first = [insert_raw_post(conn, replace(post, source_id=source_id)) for post in posts]
    inserted_first = sum(1 for row_id in first if row_id is not None)
    assert inserted_first == 60
    counts = table_counts(conn)
    assert counts["raw_post"] == 60

    posts_again = await adapter.fetch_new(None)
    assert len(posts_again) == 60  # adapter itself re-delivers; repo dedups
    second = [insert_raw_post(conn, replace(post, source_id=source_id)) for post in posts_again]
    assert all(row_id is None for row_id in second)
    counts_after = table_counts(conn)
    assert counts_after["raw_post"] == 60  # 0 new, 0 duplicates


async def test_since_run_after_first_run_yields_only_new(
    monkeypatch: pytest.MonkeyPatch,
    conn: Connection,
    tmp_path: Path,
) -> None:
    """The incremental leg: a since-backed run adds only genuinely new rows."""
    reviews_by_app = _ten_app_reviews()
    adapter = _patched_adapter(
        monkeypatch,
        reviews_by_app,
        _config(tmp_path, list(reviews_by_app)),
    )
    source_id = ensure_source(conn, GPLAY_SOURCE_NAME)

    posts = await adapter.fetch_new(None)
    for post in posts:
        insert_raw_post(conn, replace(post, source_id=source_id))

    fresh_review = _review(
        9901,
        score=2,
        content="Новая боль: приложение съедает батарею за два часа",
        at=datetime(2026, 9, 20, tzinfo=UTC),
    )
    reviews_by_app["ru.test.app1"].append(fresh_review)

    incremental = await adapter.fetch_new(
        since=datetime(2026, 9, 20, tzinfo=UTC) - timedelta(days=1)
    )
    inserted = insert_raw_post(conn, replace(incremental[0], source_id=source_id))
    assert inserted is not None
    assert table_counts(conn)["raw_post"] == 61
