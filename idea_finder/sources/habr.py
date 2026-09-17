"""Habr source adapter (task T303): RSS feeds -> discussion RawPosts.

Feed choice (DoD goal: discussion posts about business/tooling pains):

* ``/ru/rss/articles/?fl=ru`` — the all-articles firehose: widest coverage,
  includes company-blog tooling stories ("how we chose X", "why Y broke").
* ``/ru/rss/flows/management/articles/?fl=ru`` — the Management flow:
  processes, team tooling and product/business failures discussed
  first-hand — the highest pain density of the live feeds.
* ``/ru/rss/best/daily/?fl=ru`` — best-of feed: a quality filter; heavily
  discussed posts usually contain concrete pain narratives.

Hub-level feeds (``/ru/rss/hubs/<alias>/...``) were probed live (2026-09)
and return 404 for every business/startup alias, so flow-level feeds are
used instead.

Full text: each item's link is fetched with :class:`HttpFetcher` and run
through :func:`extract_main_text` — the collect stage (T306) expects the
adapter to deliver the resolved full body, not the RSS teaser description.

The RSS ``link`` carries ``utm_*`` tracking params that robots.txt disallows
(``Disallow: /*?*utm_``); articles are fetched via the canonical URL
(tracking params stripped by :func:`canonical_url`), which both satisfies
robots.txt and matches the deduplication key.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

import httpx

from idea_finder.core.canonical import canonical_url
from idea_finder.core.models import RawPost
from idea_finder.db.bootstrap_pgserver import ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import ensure_source, insert_raw_post
from idea_finder.fetch.extract import extract_main_text
from idea_finder.fetch.httpx_fetcher import FetchError, HttpFetcher

logger = logging.getLogger(__name__)

__all__ = ["HABR_FEEDS", "HABR_SOURCE_NAME", "HabrAdapter"]

#: Registry key matching ``source.name`` (see ``idea_finder.sources.base``).
HABR_SOURCE_NAME: Final[str] = "habr"

#: The domain every feed and article URL lives on (rate-limit bucket).
_HABR_DOMAIN: Final[str] = "habr.com"

#: Live RSS feeds harvested by this adapter (choice documented in the
#: module docstring).
HABR_FEEDS: Final[tuple[str, ...]] = (
    "https://habr.com/ru/rss/articles/?fl=ru",
    "https://habr.com/ru/rss/flows/management/articles/?fl=ru",
    "https://habr.com/ru/rss/best/daily/?fl=ru",
)


@dataclass(frozen=True, slots=True)
class _FeedItem:
    """One parsed RSS ``<item>``: everything needed before the full fetch."""

    title: str
    link: str
    pub_date: datetime | None


class HabrAdapter:
    """Habr RSS adapter implementing the :class:`SourceAdapter` protocol.

    Create one adapter per pipeline run::

        adapter = HabrAdapter(rate_limit_rps=1.0)
        posts = await adapter.fetch_new(since=None)

    ``rate_limit_rps`` comes from ``source.rate_limit_rps`` (habr: 1.0) and
    is enforced per-domain by the fetch layer. ``feeds`` and ``transport``
    are injection seams for tests (e.g. httpx.MockTransport + a narrower
    feed list); production leaves both at their defaults.
    """

    def __init__(
        self,
        *,
        rate_limit_rps: float = 1.0,
        feeds: tuple[str, ...] = HABR_FEEDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = HABR_SOURCE_NAME
        self._feeds = feeds
        self._fetcher = HttpFetcher(
            rate_limits={_HABR_DOMAIN: rate_limit_rps},
            transport=transport,
        )

    async def fetch_new(self, since: datetime | None) -> list[RawPost]:
        """Return discussion posts newer than ``since`` (all when None).

        Every returned post carries the full article text fetched from the
        canonical article URL. ``RawPost.source_id`` is filled with the
        registry name; the collect stage remaps it to the database UUID via
        ``dataclasses.replace`` before ``insert_raw_post``.

        Failures are isolated: a broken feed or a failing article is logged
        as a warning and skipped, never fatal for the rest of the batch.
        """
        if since is not None and since.tzinfo is None:
            # Repo convention stores UTC ISO timestamps; a naive `since` is
            # treated as UTC so the aware/naive comparison cannot raise.
            since = since.replace(tzinfo=UTC)
        posts: dict[str, RawPost] = {}
        seen_links: set[str] = set()
        async with self._fetcher:
            for feed_url in self._feeds:
                for item in await self._feed_items(feed_url):
                    if item.link in seen_links:
                        continue  # same article delivered by two feeds
                    seen_links.add(item.link)
                    if since is not None:
                        if item.pub_date is None:
                            logger.warning(
                                "habr: item without parsable pubDate skipped under since filter: %s",
                                item.link,
                            )
                            continue
                        if item.pub_date <= since:
                            continue
                    post = await self._to_raw_post(item)
                    if post is not None:
                        posts.setdefault(post.url_canon, post)
        logger.info("habr: fetched %d new posts (since=%s)", len(posts), since)
        return list(posts.values())

    async def _feed_items(self, feed_url: str) -> list[_FeedItem]:
        """Fetch and parse one RSS feed into items; a broken feed yields [].

        A single dead feed must not sink the batch: the error is logged and
        the remaining feeds are still harvested.
        """
        try:
            xml_text = await self._fetcher.fetch(feed_url)
        except FetchError as e:
            logger.warning("habr: feed fetch failed url=%s: %s", feed_url, e)
            return []
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError as e:
            logger.warning("habr: feed is not valid XML url=%s: %s", feed_url, e)
            return []
        items: list[_FeedItem] = []
        for node in root.findall("./channel/item"):
            item = _parse_item(node)
            if item is not None:
                items.append(item)
        return items

    async def _to_raw_post(self, item: _FeedItem) -> RawPost | None:
        """Build a RawPost with the full article text; None on any failure.

        The article is fetched from the canonical URL: tracking params that
        robots.txt disallows are already stripped, and the canonical form is
        the dedup key the repo inserts on.
        """
        try:
            url_canon = canonical_url(item.link)
        except ValueError as e:
            logger.warning("habr: bad article link %r: %s", item.link, e)
            return None
        try:
            html = await self._fetcher.fetch(url_canon)
        except FetchError as e:
            logger.warning("habr: article fetch failed url=%s: %s", url_canon, e)
            return None
        text, suspect_short = extract_main_text(html)
        if not text:
            logger.warning("habr: empty article text, post skipped: %s", url_canon)
            return None
        if suspect_short:
            # Kept, but flagged in the log: the collect stage decides what
            # to do with short bodies via its own stats.
            logger.warning("habr: suspect short text (%d chars): %s", len(text), url_canon)
        return RawPost(
            source_id=self.name,
            url_canon=url_canon,
            url=item.link,
            title=item.title,
            text=text,
            published_at=item.pub_date,
            kind="discussion",
        )


def _parse_item(node: ElementTree.Element) -> _FeedItem | None:
    """Extract title/link/pubDate from one ``<item>`` node.

    Items missing a title or a link are useless regardless of the fetch
    outcome: they are skipped with a warning. An unparsable pubDate degrades
    to ``None`` (the post then only passes when ``since`` is None).
    """
    title = (node.findtext("title") or "").strip()
    link = (node.findtext("link") or "").strip()
    if not title or not link:
        logger.warning("habr: RSS item without title/link skipped: %r", link or title)
        return None
    pub_date: datetime | None = None
    raw_date = (node.findtext("pubDate") or "").strip()
    if raw_date:
        try:
            pub_date = parsedate_to_datetime(raw_date)
        except (TypeError, ValueError):
            logger.warning("habr: unparsable pubDate %r for %s", raw_date, link)
    return _FeedItem(title=title, link=link, pub_date=pub_date)


async def _smoke() -> None:
    """Live smoke (task T303 acceptance): first 12 posts, then re-run.

    Prints ``len(text)  url  title`` for the first 12 posts of a live feed
    run, then re-runs the fetch against a throwaway embedded-postgres copy
    to prove idempotency: the second insert must add zero rows.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    adapter = HabrAdapter()
    posts = await adapter.fetch_new(None)
    print(f"live fetch: {len(posts)} posts")
    for post in posts[:12]:
        print(f"{len(post.text):>6}  {post.url_canon}  {post.title[:70]}")
    if not posts:
        return
    pg_dir = Path(tempfile.mkdtemp(prefix="habr-smoke-")) / "pg"
    handle = ensure_pgserver(pg_dir)
    try:
        with handle.get_conn() as conn:
            apply_migrations(conn)
            source_id = ensure_source(conn, HABR_SOURCE_NAME)
            first = [insert_raw_post(conn, replace(p, source_id=source_id)) for p in posts]
            print(f"first run: inserted {sum(1 for r in first if r is not None)} raw_post rows")
            again = await adapter.fetch_new(None)
            second = [insert_raw_post(conn, replace(p, source_id=source_id)) for p in again]
            print(
                f"second run: fetched {len(again)} posts, "
                f"inserted {sum(1 for r in second if r is not None)} new rows"
            )
    finally:
        handle.stop()


if __name__ == "__main__":
    asyncio.run(_smoke())
