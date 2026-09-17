"""Tests for the FL.ru adapter (task T302): RSS parsing, full-text
extraction through the fetch layer, since-filtering, partial-failure
tolerance, and e2e idempotency against embedded Postgres.

All network interaction goes through httpx.MockTransport injected into
HttpFetcher — no real requests are made. The live smoke lives in
``idea_finder.sources.fl_ru.__main__`` and is run manually.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from xml.sax.saxutils import escape as xml_escape

import httpx
import pytest
from psycopg import Connection

from idea_finder.core.models import RawPost
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import ensure_source, insert_raw_post, table_counts
from idea_finder.fetch.httpx_fetcher import FetchError, HttpFetcher
from idea_finder.sources.base import SourceAdapter, seed_sources
from idea_finder.sources.fl_ru import FL_RU_RSS_URL, FlRuAdapter

Handler = Callable[[httpx.Request], httpx.Response]

#: Publication moment of item 0; items are spaced 10 minutes apart.
_FIRST_PUB_UTC: datetime = datetime(2026, 9, 10, 7, 0, tzinfo=UTC)

# RFC 822 needs English day/month names; strftime would be locale-dependent.
_RFC822_DAYS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_RFC822_MONTHS: tuple[str, ...] = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

#: A word present only in the fixture page body, never in the RSS teaser.
_FULL_TEXT_MARKER: str = "портфолио"


def _rfc822(dt: datetime) -> str:
    """Render a UTC datetime as an RFC 822 date string."""
    return (
        f"{_RFC822_DAYS[dt.weekday()]}, {dt.day:02d} "
        f"{_RFC822_MONTHS[dt.month - 1]} {dt.year} {dt:%H:%M:%S} +0000"
    )


def _item(i: int) -> dict[str, str]:
    """Build one realistic feed item (Cyrillic title, project-page link)."""
    return {
        "title": f"Требуется доработать сайт на {i} страниц",
        "link": f"https://www.fl.ru/projects/52300{i}/razrabotka-{i}.html",
        "pubDate": _rfc822(_FIRST_PUB_UTC + timedelta(minutes=10 * i)),
        "description": f"Краткое описание заказа номер {i}: нужен подрядчик.",
    }


def _items(n: int) -> list[dict[str, str]]:
    """Build ``n`` feed items."""
    return [_item(i) for i in range(n)]


def _feed_xml(items: list[dict[str, str]], *, drop_pub_date_at: int | None = None) -> str:
    """Render items as an RSS 2.0 document, optionally omitting one pubDate.

    Field values are XML-escaped exactly as a real feed would escape them
    (link ampersands become ``&amp;``); the adapter must unescape them.
    """
    chunks: list[str] = []
    for i, item in enumerate(items):
        pub = "" if i == drop_pub_date_at else f"<pubDate>{item['pubDate']}</pubDate>"
        chunks.append(
            f"<item><title>{xml_escape(item['title'])}</title>"
            f"<link>{xml_escape(item['link'])}</link>{pub}"
            f"<description>{xml_escape(item['description'])}</description></item>"
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<rss version="2.0"><channel><title>FL.ru: все проекты</title>'
        + "".join(chunks)
        + "</channel></rss>"
    )


def _page_html(item: dict[str, str]) -> str:
    """Render the item's project page: teaser plus extra full-text sentences."""
    full_text = (
        f"{item['description']} Полное описание проекта: требуется опытный "
        f"подрядчик, бюджет обсуждается, срок две недели. Дополнительные "
        f"детали для заказа «{item['title']}» будут сообщены после отклика; "
        f"ожидаем {_FULL_TEXT_MARKER} и оценку стоимости работ."
    )
    return (
        "<html><head><title>"
        + item["title"]
        + "</title></head><body><article><p>"
        + full_text
        + "</p></article></body></html>"
    )


def _mock_handler(
    items: list[dict[str, str]],
    *,
    broken_links: frozenset[str] = frozenset(),
    feed_status: int = 200,
    feed_body: str | None = None,
    drop_pub_date_at: int | None = None,
) -> Handler:
    """MockTransport handler serving the feed and every item's page."""
    feed = feed_body if feed_body is not None else _feed_xml(items, drop_pub_date_at=drop_pub_date_at)
    pages = {str(httpx.URL(item["link"])): _page_html(item) for item in items}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == FL_RU_RSS_URL:
            return httpx.Response(feed_status, text=feed)
        if url.endswith("/robots.txt"):
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if url in broken_links:
            return httpx.Response(500, text="server error")
        return httpx.Response(200, text=pages[url])

    return handler


def _mock_fetcher(handler: Handler) -> HttpFetcher:
    """Fetcher on the mock transport; the rate limit is lifted for speed."""
    return HttpFetcher(
        transport=httpx.MockTransport(handler),
        rate_limits={"www.fl.ru": 1000.0},
    )


async def _collect(
    items: list[dict[str, str]],
    since: datetime | None = None,
    *,
    broken_links: frozenset[str] = frozenset(),
    feed_status: int = 200,
    feed_body: str | None = None,
    drop_pub_date_at: int | None = None,
) -> list[RawPost]:
    """Run the adapter once against mock fixtures and return its posts."""
    handler = _mock_handler(
        items,
        broken_links=broken_links,
        feed_status=feed_status,
        feed_body=feed_body,
        drop_pub_date_at=drop_pub_date_at,
    )
    async with _mock_fetcher(handler) as fetcher:
        return await FlRuAdapter(fetcher).fetch_new(since)


async def test_adapter_satisfies_protocol() -> None:
    """The adapter is a SourceAdapter with the registry name fl_ru."""
    async with _mock_fetcher(_mock_handler(_items(1))) as fetcher:
        adapter = FlRuAdapter(fetcher)
    assert isinstance(adapter, SourceAdapter)
    assert adapter.name == "fl_ru"


async def test_fetch_new_builds_demand_posts_with_full_text() -> None:
    """Feed items become demand posts whose text is the fetched page body."""
    items = _items(4)
    posts = await _collect(items)
    assert len(posts) == 4
    assert all(post.kind == "demand" for post in posts)
    assert all(post.source_id == "fl_ru" for post in posts)
    first = posts[0]
    item = items[0]
    assert first.title == item["title"]
    assert first.url == item["link"]
    assert first.published_at == parsedate_to_datetime(item["pubDate"])
    # The text is the page body (teaser + extra sentences), not the RSS cut.
    assert first.text.startswith(item["description"])
    assert len(first.text) > len(item["description"])
    assert _FULL_TEXT_MARKER in first.text


async def test_url_canon_is_canonical() -> None:
    """url_canon lowercases the host and strips tracking parameters."""
    messy = "HTTPS://WWW.FL.RU/projects/523/razrabotka.html?utm_source=rss&id=7"
    item = {
        "title": "Мусорный URL",
        "link": messy,
        "pubDate": _rfc822(_FIRST_PUB_UTC),
        "description": "Краткое описание: нужен подрядчик.",
    }
    (post,) = await _collect([item])
    assert post.url == messy
    assert post.url_canon == "https://www.fl.ru/projects/523/razrabotka.html?id=7"


async def test_fetch_new_filters_by_since_inclusively() -> None:
    """Posts at or after ``since`` are kept; older ones are dropped."""
    items = _items(4)
    cutoff = parsedate_to_datetime(items[1]["pubDate"])
    posts = await _collect(items, since=cutoff)
    assert [post.url for post in posts] == [items[1]["link"], items[2]["link"], items[3]["link"]]


async def test_naive_since_is_treated_as_utc() -> None:
    """A naive ``since`` must not crash the comparison; UTC is assumed."""
    items = _items(4)
    cutoff = parsedate_to_datetime(items[2]["pubDate"]).replace(tzinfo=None)
    posts = await _collect(items, since=cutoff)
    assert [post.url for post in posts] == [items[2]["link"], items[3]["link"]]


async def test_item_without_pubdate_stays_without_since_only() -> None:
    """An undated item passes ``since=None`` but is filtered by any since."""
    items = _items(3)
    any_cutoff = _FIRST_PUB_UTC - timedelta(days=1)
    all_posts = await _collect(items, drop_pub_date_at=1)
    assert [post.url for post in all_posts] == [item["link"] for item in items]
    assert all_posts[1].published_at is None
    dated_posts = await _collect(items, since=any_cutoff, drop_pub_date_at=1)
    assert [post.url for post in dated_posts] == [items[0]["link"], items[2]["link"]]


async def test_broken_item_page_is_skipped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """One failing page logs a warning and the rest of the feed survives."""
    items = _items(4)
    with caplog.at_level(logging.WARNING, logger="idea_finder.sources.fl_ru"):
        posts = await _collect(items, broken_links=frozenset({items[2]["link"]}))
    assert [post.url for post in posts] == [items[i]["link"] for i in (0, 1, 3)]
    assert "skipping item" in caplog.text
    assert items[2]["link"] in caplog.text


async def test_feed_fetch_failure_raises_fetch_error() -> None:
    """A failed feed fetch surfaces FetchError instead of returning [] ."""
    with pytest.raises(FetchError):
        await _collect(_items(2), feed_status=500)


async def test_unparsable_feed_raises_fetch_error() -> None:
    """Non-XML feed bodies are a FetchError, not an XML crash."""
    with pytest.raises(FetchError):
        await _collect(_items(2), feed_body="<not-xml")


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """One embedded postgres cluster for the e2e leg."""
    data_dir = tmp_path_factory.mktemp("pgflru") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture(scope="module")
def conn(pg: PgHandle) -> Iterator[Connection]:
    """Connection to the migrated application database."""
    with pg.get_conn() as connection:
        apply_migrations(connection)
        yield connection


async def test_e2e_first_run_inserts_more_than_ten_second_run_inserts_zero(
    conn: Connection,
) -> None:
    """DoD acceptance: first run inserts >10 posts, rerun inserts zero new.

    The same 12-item feed is fetched twice; url_canon dedup in
    insert_raw_post must swallow the whole second pass (every insert
    returns None, the row count does not grow).
    """
    seed_sources(conn)
    source_id = ensure_source(conn, "fl_ru")
    items = _items(12)

    async with _mock_fetcher(_mock_handler(items)) as fetcher:
        posts = await FlRuAdapter(fetcher).fetch_new(None)
    db_posts = [replace(post, source_id=source_id) for post in posts]

    first_ids = [insert_raw_post(conn, post) for post in db_posts]
    inserted_first = sum(1 for row_id in first_ids if row_id is not None)
    assert inserted_first > 10

    async with _mock_fetcher(_mock_handler(items)) as fetcher:
        reposts = await FlRuAdapter(fetcher).fetch_new(None)
    second_ids = [insert_raw_post(conn, replace(post, source_id=source_id)) for post in reposts]
    inserted_second = sum(1 for row_id in second_ids if row_id is not None)
    assert inserted_second == 0

    counts = table_counts(conn)
    assert counts["raw_post"] == inserted_first
