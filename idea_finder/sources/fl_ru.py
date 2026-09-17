"""FL.ru source adapter: RSS of project orders -> RawPost(kind="demand").

The FL.ru all-projects feed (https://www.fl.ru/rss/all.xml) lists fresh
orders as RSS 2.0 items. Item ``description`` is a short teaser only, so
the adapter fetches every item's page through the fetch layer
(:class:`HttpFetcher`) and extracts the full text with
:func:`extract_main_text` — the adapter owns all source-specific fetching
details and hands fully-fetched posts to the collect stage (decision from
the B3/B4 contract).

Idempotency: the feed repeats fresh items between runs, but ``url_canon``
duplicates are absorbed by ``db/repo.insert_raw_post`` (it returns ``None``
for known URLs), so re-running collect adds nothing. ``fetch_new(since)``
additionally filters ``published_at >= since`` when ``since`` is given.

Per-item failures (one broken page, one flaky network call) are logged as
warnings and the item is skipped — partial success is acceptable; a total
feed failure raises :class:`FetchError` to the caller.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Final
from xml.etree.ElementTree import ParseError as XmlParseError

from idea_finder.core.canonical import canonical_url
from idea_finder.core.models import RawPost
from idea_finder.fetch.extract import extract_main_text
from idea_finder.fetch.httpx_fetcher import FetchError, HttpFetcher

logger = logging.getLogger(__name__)

__all__ = ["FL_RU_RSS_URL", "FlRuAdapter"]

#: All-projects RSS feed of FL.ru orders (category filter omitted).
FL_RU_RSS_URL: Final[str] = "https://www.fl.ru/rss/all.xml"

#: RSS item child elements the adapter consumes.
_ITEM_TAGS: Final[tuple[str, ...]] = ("title", "link", "pubDate", "description")


class FlRuAdapter:
    """Collect FL.ru orders as ``demand`` posts via the all-projects RSS.

    Create one adapter per pipeline run and share the ``fetcher`` with the
    other adapters of that run (per-domain rate limits live in the
    fetcher)::

        async with HttpFetcher() as fetcher:
            posts = await FlRuAdapter(fetcher).fetch_new(None)

    ``fetcher`` is injectable so tests can pass an :class:`HttpFetcher`
    wired to an ``httpx.MockTransport``; production leaves the default.
    """

    name: str = "fl_ru"

    def __init__(self, fetcher: HttpFetcher | None = None) -> None:
        self._fetcher = fetcher if fetcher is not None else HttpFetcher()

    def _since_utc(self, since: datetime | None) -> datetime | None:
        """Normalize ``since`` to aware UTC so comparisons cannot mix zones."""
        if since is None or since.tzinfo is not None:
            return since
        return since.replace(tzinfo=UTC)

    async def fetch_new(self, since: datetime | None) -> list[RawPost]:
        """Return feed items as posts, newer than ``since`` when given.

        ``since`` is ``None`` on the first run: all feed items are returned.
        Items whose page fetch or text extraction fails are skipped with a
        warning (partial success); a failed feed fetch raises FetchError.
        """
        cutoff = self._since_utc(since)
        xml_text = await self._fetcher.fetch(FL_RU_RSS_URL)
        items = _parse_feed(xml_text)
        posts: list[RawPost] = []
        for item in items:
            published_at = _parse_pub_date(item.get("pubDate", ""))
            if cutoff is not None and (published_at is None or published_at < cutoff):
                continue
            try:
                html = await self._fetcher.fetch(item["link"])
                text, _suspect_short = extract_main_text(html)
            except FetchError as e:
                logger.warning("fl_ru: skipping item url=%s: %s", item["link"], e)
                continue
            if not text:
                logger.warning("fl_ru: no text extracted for url=%s, skipping", item["link"])
                continue
            posts.append(
                RawPost(
                    source_id=self.name,
                    url=item["link"],
                    url_canon=canonical_url(item["link"]),
                    title=item.get("title", ""),
                    text=text,
                    published_at=published_at,
                    kind="demand",
                )
            )
        logger.info("fl_ru: fetched %d posts (feed items: %d)", len(posts), len(items))
        return posts


def _parse_feed(xml_text: str) -> list[dict[str, str]]:
    """Parse RSS 2.0 XML into a list of item dicts.

    Namespace-agnostic: the feed has no namespace, but parsing via
    ``iter()`` keeps it robust to one being added. Each dict carries the
    consumed child-element texts; the live feed double-escapes entities
    (e.g. ``&amp;#8381;`` in titles), so values are unescaped twice.
    Items missing ``link`` are dropped.
    """
    try:
        root = ET.fromstring(xml_text)
    except XmlParseError as e:
        raise FetchError(f"fl_ru: cannot parse RSS feed {FL_RU_RSS_URL}") from e
    items: list[dict[str, str]] = []
    for item_el in root.iter("item"):
        values: dict[str, str] = {}
        for tag in _ITEM_TAGS:
            child = item_el.find(tag)
            if child is not None and child.text:
                values[tag] = unescape(child.text.strip())
        link = values.get("link")
        if not link:
            logger.warning("fl_ru: RSS item without link skipped")
            continue
        items.append(values)
    return items


def _parse_pub_date(raw: str) -> datetime | None:
    """Parse an RFC 822 pubDate into an aware UTC datetime.

    Returns ``None`` for an empty or malformed value: the item then only
    survives ``since=None`` runs.
    """
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        logger.warning("fl_ru: unparsable pubDate %r", raw)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    async def _smoke() -> int:
        """Live-feed smoke: print posts, then prove DB idempotency on a tmp cluster."""
        async with HttpFetcher() as fetcher:
            posts = await FlRuAdapter(fetcher).fetch_new(None)
        print(f"fetched {len(posts)} posts from live feed")
        for post in posts[:12]:
            print(
                f"{post.kind}\tlen(text)={len(post.text)}\t{post.published_at}\t"
                f"{post.url_canon}\t{post.title[:80]}"
            )
        if not posts:
            return 1
        # Idempotency leg: throwaway embedded cluster, insert twice.
        from idea_finder.db.bootstrap_pgserver import pgserver_session
        from idea_finder.db.migrate import apply_migrations
        from idea_finder.db.repo import ensure_source, insert_raw_post, table_counts

        with pgserver_session() as handle, handle.get_conn() as conn:
            apply_migrations(conn)
            source_id = ensure_source(conn, "fl_ru")
            # raw_post.source_id is a uuid FK: the collect stage maps the
            # adapter's registry name onto the source row id before insert.
            db_posts = [replace(post, source_id=source_id) for post in posts]
            first_ids = [insert_raw_post(conn, post) for post in db_posts]
            inserted_first = sum(1 for pid in first_ids if pid is not None)
            second_ids = [insert_raw_post(conn, post) for post in db_posts]
            inserted_second = sum(1 for pid in second_ids if pid is not None)
            counts = table_counts(conn)
            print(
                f"db: first insert -> {inserted_first} rows, "
                f"second insert -> {inserted_second} new, "
                f"raw_post total = {counts['raw_post']}"
            )
        return 0

    sys.exit(asyncio.run(_smoke()))
