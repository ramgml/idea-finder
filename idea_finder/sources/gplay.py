"""Google Play source adapter (task T304): RU app reviews -> complaint RawPosts.

Reviews of the configured apps (``configs/gplay_apps.json``) are harvested
with the sync ``google-play-scraper`` library. Every scraper call is a
blocking HTTP exchange, so it is wrapped in :func:`asyncio.to_thread` — the
adapter itself is ``async`` per the :class:`SourceAdapter` protocol while
the scraper stays untouched.

Complaint threshold: only reviews rated **1-3 stars** are collected. 4-5
star reviews praise the app and carry no pain signal for idea discovery.
Reviews without text content (star-only ratings) are skipped: the pipeline
needs at least a sentence to extract a pain from.

Permalink decision: the scraper's ``reviewId`` is a GUID that Google Play
does not expose as a standalone review page, so no reliable
``...&reviewId=<id>`` permalink exists. Each post URL is therefore the
app details page (``https://play.google.com/store/apps/details?id=<app_id>``)
and the review GUID itself is appended as a fragment
(``#review=<reviewId>``). The fragment survives :func:`canonical_url` and
makes ``url_canon`` unique per review, which is the dedup key the repo
inserts on.

Idempotency is two-legged: ``fetch_new(since)`` filters ``at >= since``
and ``db/repo.insert_raw_post`` absorbs ``url_canon`` duplicates on the
database side.

Configuration errors raise :class:`GPlayConfigError` (a subclass of
:class:`FetchError` so generic fetch handling still catches it): a missing
or unparsable config file, or a list without any app id, is a setup
problem, not a network problem. Per-app scraper failures are isolated:
they are logged as warnings and the remaining apps are still harvested —
one dead app id must not sink the batch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from google_play_scraper import Sort, reviews

from idea_finder.core.canonical import canonical_url
from idea_finder.core.models import RawPost
from idea_finder.fetch.httpx_fetcher import FetchError

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_APPS_CONFIG", "GPLAY_SOURCE_NAME", "GPlayAdapter", "GPlayConfigError"]

#: Registry key matching ``source.name`` (see ``idea_finder.sources.base``).
GPLAY_SOURCE_NAME: Final[str] = "gplay"

#: Default app list, relative to the repository root.
DEFAULT_APPS_CONFIG: Final[str] = "configs/gplay_apps.json"

#: Reviews at or below this star rating are complaints; above it, praise.
#: 1-3 stars signal real user pain; 4-5 stars do not (task T304 decision).
_MAX_COMPLAINT_STARS: Final[int] = 3

#: Reviews fetched per app per run. 200 is the scraper's own page size, so
#: this costs one request per app and still yields >50 complaints even
#: after the 1-3 star filter drops the positive share.
_REVIEWS_PER_APP: Final[int] = 200

#: Review language and storefront of interest (task T304: RU market).
_LANG: Final[str] = "ru"
_COUNTRY: Final[str] = "ru"


class GPlayConfigError(FetchError):
    """The Google Play app config is missing, unparsable, or empty."""


class GPlayAdapter:
    """Google Play reviews adapter implementing :class:`SourceAdapter`.

    Create one adapter per pipeline run::

        adapter = GPlayAdapter()
        posts = await adapter.fetch_new(since=None)

    ``apps_path`` points at the JSON app list (``{"app_ids": [...]}``);
    tests pass a tmp file there. ``_fetch_reviews`` is the single seam
    around the sync scraper: tests monkeypatch it instead of the library.
    """

    name: str = GPLAY_SOURCE_NAME

    def __init__(
        self,
        *,
        apps_path: str | Path = DEFAULT_APPS_CONFIG,
        count_per_app: int = _REVIEWS_PER_APP,
    ) -> None:
        self._apps_path = Path(apps_path)
        self._count_per_app = count_per_app

    def _load_app_ids(self) -> list[str]:
        """Read the app-id list from the JSON config.

        Raises:
            GPlayConfigError: If the file is missing, is not valid JSON,
                has no ``app_ids`` list, or the list is empty.
        """
        try:
            payload = json.loads(self._apps_path.read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            msg = f"gplay: apps config not found: {self._apps_path}"
            raise GPlayConfigError(msg) from e
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            msg = f"gplay: apps config unreadable: {self._apps_path}: {e}"
            raise GPlayConfigError(msg) from e
        if not isinstance(payload, dict) or not isinstance(payload.get("app_ids"), list):
            msg = f"gplay: apps config must hold an 'app_ids' list: {self._apps_path}"
            raise GPlayConfigError(msg)
        app_ids = [str(app_id) for app_id in payload["app_ids"] if str(app_id).strip()]
        if not app_ids:
            msg = f"gplay: apps config lists no app ids: {self._apps_path}"
            raise GPlayConfigError(msg)
        return app_ids

    def _fetch_reviews(self, app_id: str) -> list[dict[str, object]]:
        """Run one sync scraper call; the patch seam for tests.

        Returns the newest RU reviews of ``app_id`` (no score filter: the
        complaint threshold is applied here, not in the API, so tests can
        pin the filter behavior).
        """
        result, _token = reviews(
            app_id,
            lang=_LANG,
            country=_COUNTRY,
            sort=Sort.NEWEST,
            count=self._count_per_app,
        )
        return list(result)

    def _since_utc(self, since: datetime | None) -> datetime | None:
        """Normalize ``since`` to aware UTC so comparisons cannot mix zones."""
        if since is None or since.tzinfo is not None:
            return since
        return since.replace(tzinfo=UTC)

    async def fetch_new(self, since: datetime | None) -> list[RawPost]:
        """Return 1-3 star reviews as complaint posts newer than ``since``.

        ``since`` is ``None`` on the first run: every harvested complaint
        is returned. Each scraper call runs in a worker thread; a failing
        app is logged and skipped, the rest of the batch survives.
        """
        cutoff = self._since_utc(since)
        app_ids = self._load_app_ids()
        posts: list[RawPost] = []
        for app_id in app_ids:
            try:
                raw_reviews = await asyncio.to_thread(self._fetch_reviews, app_id)
            except Exception:
                # Any scraper error is app-local (bad id, network flap):
                # one dead app must not sink the rest of the batch.
                logger.exception("gplay: app %s failed, skipped", app_id)
                continue
            kept = self._to_raw_posts(app_id, raw_reviews, cutoff)
            posts.extend(kept)
        logger.info(
            "gplay: fetched %d complaint posts from %d apps (since=%s)",
            len(posts),
            len(app_ids),
            since,
        )
        return posts

    def _to_raw_posts(
        self,
        app_id: str,
        raw_reviews: list[dict[str, object]],
        cutoff: datetime | None,
    ) -> list[RawPost]:
        """Convert one app's reviews to complaint posts under the filters.

        Drops non-complaint stars, empty bodies, and reviews older than
        ``cutoff``; de-duplicates within the batch by ``url_canon`` (the
        scraper can repeat a review across pages).
        """
        posts: dict[str, RawPost] = {}
        for review in raw_reviews:
            score = review.get("score")
            if not isinstance(score, int) or not 1 <= score <= _MAX_COMPLAINT_STARS:
                continue
            content = review.get("content")
            if not isinstance(content, str) or not content.strip():
                continue  # star-only rating: nothing to extract a pain from
            review_id = str(review.get("reviewId", ""))
            if not review_id:
                continue
            published_at = review.get("at")
            if cutoff is not None:
                if not isinstance(published_at, datetime):
                    continue  # undated review cannot be placed against since
                at_utc = published_at if published_at.tzinfo else published_at.replace(tzinfo=UTC)
                if at_utc < cutoff:
                    continue
            url = _review_url(app_id, review_id)
            posts.setdefault(
                canonical_url(url),
                RawPost(
                    source_id=self.name,
                    url=url,
                    url_canon=canonical_url(url),
                    title=str(review.get("userName", "")),
                    text=content,
                    published_at=published_at if isinstance(published_at, datetime) else None,
                    kind="complaint",
                ),
            )
        return list(posts.values())


def _review_url(app_id: str, review_id: str) -> str:
    """Build the review URL: app details page plus a review fragment.

    See the module docstring for why the GUID lives in the fragment.
    """
    return f"https://play.google.com/store/apps/details?id={app_id}#review={review_id}"


if __name__ == "__main__":
    import dataclasses
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    async def _smoke() -> int:
        """Live smoke: fetch RU complaints, then prove DB idempotency.

        Google Play is unreachable from this network (T299 precedent:
        timeouts / HTTP 451); in that case the error is surfaced honestly
        and the e2e proof remains the fixture-based pytest leg.
        """
        adapter = GPlayAdapter()
        try:
            posts = await adapter.fetch_new(None)
        except GPlayConfigError as e:
            print(f"CONFIG ERROR: {e}")
            return 2
        except Exception as e:  # noqa: BLE001 -- smoke surfaces the failure, then exits.
            print(f"LIVE FETCH FAILED: {type(e).__name__}: {e}")
            return 1
        print(f"fetched {len(posts)} complaint posts from live Play")
        for post in posts[:12]:
            print(
                f"{post.kind}\tstars-text={len(post.text)}\t{post.published_at}\t"
                f"{post.url_canon[:100]}\t{post.title[:40]}"
            )
        if not posts:
            return 1
        # Idempotency leg: throwaway embedded cluster, insert twice.
        from idea_finder.db.bootstrap_pgserver import pgserver_session
        from idea_finder.db.migrate import apply_migrations
        from idea_finder.db.repo import ensure_source, insert_raw_post, table_counts

        with pgserver_session() as handle, handle.get_conn() as conn:
            apply_migrations(conn)
            source_id = ensure_source(conn, GPLAY_SOURCE_NAME)
            db_posts = [dataclasses.replace(post, source_id=source_id) for post in posts]
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
