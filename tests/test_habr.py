"""Tests for the Habr RSS adapter (task T303).

Layers:

* Unit layer (httpx.MockTransport via the fetcher's ``transport`` seam, no
  network): feed parsing (kind, canonical URLs, full text, pubDate), the
  ``since`` filter, partial failure isolation (dead feed, dead article,
  broken XML), and duplicate-article collapsing across feeds.
* E2E layer (module-scoped embedded postgres): a 12-item fixture feed runs
  through the adapter and ``insert_raw_post`` — first run inserts 12 rows,
  second run inserts 0 (the DoD acceptance criterion), no duplicates.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime

import httpx
import pytest
from psycopg import Connection

from idea_finder.core.models import RawPost
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import ensure_source, insert_raw_post, table_counts
from idea_finder.sources.habr import HABR_FEEDS, HabrAdapter

ARTICLE_TMPL = "<html><body><article><h1>{title}</h1><p>{body}</p></article></body></html>"
FEED_TMPL = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>{title}</title>
    <link>https://habr.com/ru/</link>
    <description>test feed</description>
    {items}
  </channel>
</rss>
"""
ITEM_TMPL = """    <item>
      <title><![CDATA[{title}]]></title>
      <link>{link}</link>
      <description><![CDATA[{teaser}]]></description>
      <pubDate>{pub_date}</pubDate>
    </item>"""


def _xml_escape(url: str) -> str:
    """Escape ``&`` the way a real RSS generator does inside a link tag."""
    return url.replace("&", "&amp;")


def _feed(*items: str) -> str:
    return FEED_TMPL.format(title="Test habr feed", items="\n".join(items))


def _item(n: int, *, pub_date: str, link: str | None = None, title: str | None = None) -> str:
    return ITEM_TMPL.format(
        title=title or f"Как мы выбирали инструмент №{n} и пожалели",
        link=_xml_escape(
            link or f"https://habr.com/ru/articles/{100000 + n}/?utm_source=habrahabr&utm_medium=rss"
        ),
        teaser=f"<p>Тизер поста №{n}</p>",
        pub_date=pub_date,
    )


# RFC 822 dates (what Habr actually sends), spaced a day apart.
DATES = [
    "Thu, 10 Sep 2026 10:00:00 GMT",
    "Fri, 11 Sep 2026 11:00:00 GMT",
    "Sat, 12 Sep 2026 12:00:00 GMT",
    "Sun, 13 Sep 2026 13:00:00 GMT",
    "Mon, 14 Sep 2026 14:00:00 GMT",
]


def _five_item_feed() -> str:
    """Feed with 5 valid items: cyrillic titles, tracking-param links."""
    return _feed(
        *(_item(n, pub_date=DATES[n - 1]) for n in range(1, 6)),
    )

def _mock_transport(articles: Mapping[str, str], feeds: Mapping[str, str]) -> httpx.MockTransport:
    """Transport serving feed XML and article HTML by exact URL.

    Unlisted URLs get 404: a test that sees one asked for something it did
    not stub has a bug in its expectations.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in feeds:
            return httpx.Response(200, text=feeds[url])
        if url in articles:
            return httpx.Response(200, text=articles[url])
        return httpx.Response(404, text="not stubbed")

    return httpx.MockTransport(handler)


async def _fetch_posts(
    feeds: Mapping[str, str],
    articles: Mapping[str, str],
    since: datetime | None = None,
    feed_urls: tuple[str, ...] = HABR_FEEDS,
) -> list[RawPost]:
    """Run the adapter against mocks over the given feed URLs.

    The rate limit is raised for tests: per-domain politeness belongs to
    the fetch layer's own suite; waiting 1s per mock request here only
    slows the run down.
    """
    adapter = HabrAdapter(
        rate_limit_rps=100.0,
        feeds=feed_urls,
        transport=_mock_transport(articles, feeds),
    )
    return await adapter.fetch_new(since)


# ---------------------------------------------------------------- unit layer


async def test_fetch_new_returns_discussion_posts_with_full_text() -> None:
    """Every post is kind=discussion and carries the fetched article body."""
    feeds = {HABR_FEEDS[0]: _five_item_feed()}
    articles = {
        f"https://habr.com/ru/articles/{100000 + n}": ARTICLE_TMPL.format(
            title=f"Инструмент №{n}", body=f"Полный текст боли номер {n}. " * 20
        )
        for n in range(1, 6)
    }
    posts = await _fetch_posts(feeds, articles)
    assert len(posts) == 5
    assert all(p.kind == "discussion" for p in posts)
    assert all("Полный текст боли" in p.text for p in posts)
    # The teaser from RSS must NOT be what lands in .text.
    assert all("Тизер поста" not in p.text for p in posts)


async def test_links_are_canonicalized_and_deduplicated() -> None:
    """utm_* links collapse to canonical URLs; url keeps the original."""
    feeds = {HABR_FEEDS[0]: _five_item_feed()}
    articles = {
        f"https://habr.com/ru/articles/{100000 + n}": ARTICLE_TMPL.format(
            title="t", body="body " * 50
        )
        for n in range(1, 6)
    }
    posts = await _fetch_posts(feeds, articles)
    for p in posts:
        assert "utm_" not in p.url_canon
        assert p.url_canon == p.url.split("?")[0].rstrip("/")
        assert p.url != p.url_canon  # original keeps tracking params
    canon_set = {p.url_canon for p in posts}
    assert len(canon_set) == 5


async def test_same_article_in_two_feeds_yields_one_post() -> None:
    """Cross-feed duplicate: the same article delivered twice collapses."""
    item = _item(1, pub_date=DATES[0])
    feed = _feed(item, _item(2, pub_date=DATES[1]))
    feeds = {HABR_FEEDS[0]: feed, HABR_FEEDS[1]: feed}
    articles = {
        f"https://habr.com/ru/articles/{100000 + n}": ARTICLE_TMPL.format(title="t", body="b " * 50)
        for n in (1, 2)
    }
    posts = await _fetch_posts(feeds, articles)
    assert len(posts) == 2  # 3 feed slots, one article shared -> 2 unique


async def test_since_filters_old_and_undated_posts() -> None:
    """Posts older than ``since`` are skipped; undated ones warn-and-skip."""
    old = _item(1, pub_date="Mon, 01 Sep 2025 00:00:00 GMT")
    fresh = _item(2, pub_date="Wed, 16 Sep 2026 09:00:00 GMT")
    no_date = ITEM_TMPL.format(
        title="Пост без даты",
        link="https://habr.com/ru/articles/100999",
        teaser="<p>x</p>",
        pub_date="",
    ).replace("<pubDate></pubDate>", "")
    feeds = {HABR_FEEDS[0]: _feed(old, fresh, no_date)}
    articles = {
        "https://habr.com/ru/articles/100002": ARTICLE_TMPL.format(title="t", body="b " * 50)
    }
    since = datetime(2026, 9, 15, tzinfo=UTC)
    posts = await _fetch_posts(feeds, articles, since=since)
    assert [p.url_canon for p in posts] == ["https://habr.com/ru/articles/100002"]


async def test_since_none_returns_all_posts() -> None:
    """First run (since=None) returns every item including old ones."""
    feeds = {HABR_FEEDS[0]: _five_item_feed()}
    articles = {
        f"https://habr.com/ru/articles/{100000 + n}": ARTICLE_TMPL.format(title="t", body="b " * 50)
        for n in range(1, 6)
    }
    posts = await _fetch_posts(feeds, articles)
    assert len(posts) == 5


async def test_dead_feed_does_not_kill_the_batch() -> None:
    """One feed 500s, the other still delivers its items."""
    feeds = {HABR_FEEDS[1]: _feed(_item(1, pub_date=DATES[0]))}
    articles = {
        "https://habr.com/ru/articles/100001": ARTICLE_TMPL.format(title="t", body="b " * 50)
    }
    posts = await _fetch_posts(feeds, articles)
    assert len(posts) == 1


async def test_broken_xml_feed_is_skipped() -> None:
    """A feed serving garbage XML is skipped with a warning, not raised."""
    feeds = {HABR_FEEDS[0]: "this is << not xml at all"}
    feeds2 = {HABR_FEEDS[0]: feeds[HABR_FEEDS[0]], HABR_FEEDS[1]: _feed(_item(3, pub_date=DATES[0]))}
    articles = {
        "https://habr.com/ru/articles/100003": ARTICLE_TMPL.format(title="t", body="b " * 50)
    }
    posts = await _fetch_posts(feeds2, articles)
    assert len(posts) == 1


async def test_dead_article_is_skipped_others_survive() -> None:
    """One article 404s: a warning, and the remaining posts still land."""
    feeds = {HABR_FEEDS[0]: _five_item_feed()}
    articles = {
        f"https://habr.com/ru/articles/{100000 + n}": ARTICLE_TMPL.format(title="t", body="b " * 50)
        for n in range(1, 5)  # article 5 missing -> 404
    }
    posts = await _fetch_posts(feeds, articles)
    assert len(posts) == 4
    assert all("100005" not in p.url_canon for p in posts)


async def test_item_without_link_is_skipped() -> None:
    """An item with an empty link is dropped at parse time."""
    broken = ITEM_TMPL.format(
        title="Без ссылки", link="", teaser="<p>x</p>", pub_date=DATES[0]
    )
    feeds = {HABR_FEEDS[0]: _feed(broken, _item(1, pub_date=DATES[1]))}
    articles = {
        "https://habr.com/ru/articles/100001": ARTICLE_TMPL.format(title="t", body="b " * 50)
    }
    posts = await _fetch_posts(feeds, articles)
    assert len(posts) == 1


async def test_unparsable_pubdate_degrades_to_none() -> None:
    """A broken pubDate does not crash parsing; the post only passes when since is None."""
    item = _item(1, pub_date="not a date at all")
    feeds = {HABR_FEEDS[0]: _feed(item)}
    articles = {
        "https://habr.com/ru/articles/100001": ARTICLE_TMPL.format(title="t", body="b " * 50)
    }
    posts_all = await _fetch_posts(feeds, articles, since=None)
    assert len(posts_all) == 1
    assert posts_all[0].published_at is None
    posts_since = await _fetch_posts(
        feeds, articles, since=datetime(2020, 1, 1, tzinfo=UTC)
    )
    assert posts_since == []


# ------------------------------------------------------------ e2e pg layer


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """One embedded postgres cluster for the whole module."""
    data_dir = tmp_path_factory.mktemp("pg-habr") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture(scope="module")
def conn(pg: PgHandle) -> Iterator[Connection]:
    """Connection to the migrated application database."""
    with pg.get_conn() as connection:
        apply_migrations(connection)
        yield connection


def _twelve_item_feed() -> str:
    """DoD acceptance fixture: 12 items, cyrillic, RFC 822 dates."""
    return _feed(*(_item(n, pub_date=DATES[(n - 1) % 5]) for n in range(1, 13)))


async def test_dod_first_run_inserts_twelve_second_run_zero(
    conn: Connection,
) -> None:
    """DoD: >10 posts first run, 0 new + 0 duplicates on the re-run."""
    feeds = {HABR_FEEDS[0]: _twelve_item_feed()}
    articles = {
        f"https://habr.com/ru/articles/{100000 + n}": ARTICLE_TMPL.format(
            title=f"Боль №{n}", body=f"Развёрнутый текст боли {n}. " * 15
        )
        for n in range(1, 13)
    }
    adapter = HabrAdapter(
        rate_limit_rps=100.0,
        feeds=(HABR_FEEDS[0],),
        transport=_mock_transport(articles, feeds),
    )
    posts = await adapter.fetch_new(None)
    assert len(posts) == 12
    source_id = ensure_source(conn, "habr")
    first = [insert_raw_post(conn, _with_source(p, source_id)) for p in posts]
    assert all(r is not None for r in first)
    counts = table_counts(conn)
    assert counts["raw_post"] == 12

    posts_again = await adapter.fetch_new(None)
    assert len(posts_again) == 12  # adapter itself re-delivers; repo dedups
    second = [insert_raw_post(conn, _with_source(p, source_id)) for p in posts_again]
    assert all(r is None for r in second)
    counts_after = table_counts(conn)
    assert counts_after["raw_post"] == 12  # 0 new, 0 duplicates


def _with_source(post: RawPost, source_id: str) -> RawPost:
    """Swap the registry name for the database UUID."""
    return RawPost(
        source_id=source_id,
        url_canon=post.url_canon,
        url=post.url,
        title=post.title,
        text=post.text,
        published_at=post.published_at,
        kind=post.kind,
    )
