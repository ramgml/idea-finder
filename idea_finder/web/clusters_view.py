"""Read-only dashboard queries for the Clusters page (tasks 311/312).

Schema ownership note (task 311 constraint): ``db/repo.py`` is shared with
the extract/score streams, so the read queries the web layer needs live here
instead of repo.py. TODO(repo): move all functions below into ``db/repo.py``
once the parallel streams merge; keep the signatures and SQL as-is.

Everything here is read-only SELECTs; the dashboard never writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Literal, LiteralString

from psycopg import Connection, sql

#: Score rows are optional per cluster (score stage may not have run yet).
#: Sort missing scores last: NULLS LAST keeps unassessed clusters below any
#: scored one while preserving score-descending order for the rest.
_LIST_CLUSTERS_SQL: Final[LiteralString] = """
    SELECT cluster.id,
           cluster.title,
           cluster.size,
           cluster.kind_mix,
           score.total,
           cluster.created_at,
           cluster.feedback,
           cluster.split_flag
    FROM cluster
    LEFT JOIN score ON score.cluster_id = cluster.id
    WHERE ({hide_clause})
    ORDER BY score.total DESC NULLS LAST, cluster.size DESC, cluster.created_at DESC
"""

#: Filtered ordering: parameterized WHERE comes before ORDER BY.
_CLUSTERS_WITH_FILTERS_SQL: Final[LiteralString] = """
    SELECT cluster.id,
           cluster.title,
           cluster.size,
           cluster.kind_mix,
           score.total,
           cluster.created_at,
           cluster.feedback,
           cluster.split_flag
    FROM cluster
    LEFT JOIN score ON score.cluster_id = cluster.id
    LEFT JOIN pain ON pain.cluster_id = cluster.id
    LEFT JOIN raw_post ON raw_post.id = pain.raw_post_id
    LEFT JOIN source ON source.id = raw_post.source_id
    WHERE ({hide_clause}) AND ({conditions})
    GROUP BY cluster.id, score.total, cluster.feedback, cluster.split_flag
    ORDER BY score.total DESC NULLS LAST, cluster.size DESC, cluster.created_at DESC
"""

#: «Скрытые не видны без фильтра» (T315): the default listing drops
#: feedback='hidden' clusters; include_hidden brings them back.
_HIDE_DEFAULT_SQL: Final[LiteralString] = "cluster.feedback IS DISTINCT FROM 'hidden'"
_HIDE_NONE_SQL: Final[LiteralString] = "TRUE"

#: Rubric score 0-10.
MAX_SCORE: Final[int] = 10


@dataclass(frozen=True, slots=True)
class ClusterRow:
    """One dashboard row: a cluster joined with its optional score."""

    id: str
    title: str
    size: int
    #: Kind counters from ``cluster.kind_mix`` (e.g. ``{"demand": 4}``).
    kinds: dict[str, int]
    #: Rubric score 0-10, or None while the score stage has not rated the cluster.
    score: float | None
    #: Cluster creation time (clusters are recomputed per run; no updated_at).
    created_at: datetime
    #: Distinct source names behind the cluster's pains, alphabetically sorted.
    sources: tuple[str, ...]
    #: Owner feedback label (T315): 'interesting' | 'hidden' | None.
    feedback: Literal["interesting", "hidden"] | None = None
    #: «Это не одна боль» calibration flag (T315).
    split_flag: bool = False


def _build_rows(
    conn: Connection,
    rows: list[tuple[str, str, int, object, float | None, datetime, str | None, bool]],
) -> list[ClusterRow]:
    """Assemble ClusterRow list (kind_mix parse + per-cluster sources)."""
    base: list[
        tuple[str, str, int, dict[str, int], float | None, datetime, str | None, bool]
    ] = []
    for cluster_id, title, size, kind_mix, total, created_at, feedback, split_flag in rows:
        raw_mix = kind_mix if isinstance(kind_mix, dict) else json.loads(str(kind_mix))
        kinds = {str(kind): int(count) for kind, count in raw_mix.items()}
        base.append(
            (
                str(cluster_id),
                str(title),
                int(size),
                kinds,
                None if total is None else float(total),
                created_at,
                None if feedback is None else str(feedback),
                bool(split_flag),
            )
        )
    clusters: list[ClusterRow] = []
    for (
        cluster_id,
        title,
        size,
        kinds,
        score_value,
        created_at,
        feedback,
        split_flag,
    ) in base:
        source_rows = conn.execute(
            """
            SELECT DISTINCT source.name
            FROM pain
            JOIN raw_post ON raw_post.id = pain.raw_post_id
            JOIN source ON source.id = raw_post.source_id
            WHERE pain.cluster_id = %s
            ORDER BY source.name
            """,
            (cluster_id,),
        ).fetchall()
        feedback_typed: Literal["interesting", "hidden"] | None = (
            feedback if feedback in ("interesting", "hidden") else None
        )
        clusters.append(
            ClusterRow(
                id=cluster_id,
                title=title,
                size=size,
                kinds=kinds,
                score=score_value,
                created_at=created_at,
                sources=tuple(str(name) for (name,) in source_rows),
                feedback=feedback_typed,
                split_flag=split_flag,
            )
        )
    return clusters


def list_clusters(conn: Connection, *, include_hidden: bool = False) -> list[ClusterRow]:
    """Return every cluster with its score, sorted by score (best first).

    Ties fall back to cluster size, then creation time, so the page order is
    deterministic across reruns. Clusters without a score sort after scored
    ones (``score`` is None). Hidden clusters (``feedback = 'hidden'``) are
    dropped unless ``include_hidden`` is set (T315: «скрытые не видны без
    фильтра»).
    """
    hide = _HIDE_NONE_SQL if include_hidden else _HIDE_DEFAULT_SQL
    query = sql.SQL(_LIST_CLUSTERS_SQL).format(hide_clause=sql.SQL(hide))
    rows = conn.execute(query).fetchall()
    return _build_rows(conn, rows)


def count_clusters(conn: Connection) -> int:
    """Return the number of cluster rows (empty-state check for the page)."""
    row = conn.execute("SELECT count(*) FROM cluster").fetchone()
    return int(row[0]) if row is not None else 0


@dataclass(frozen=True, slots=True)
class ClusterFilters:
    """Combined cluster-list filters (task 312).

    All fields optional: None/empty means «not filtered». ``kinds`` matches
    against ``cluster.kind_mix`` (a cluster passes when at least one of its
    kinds is in the set). ``min_score`` is an inclusive floor on
    ``score.total``; clusters without a score fail it. ``from``/``to`` bound
    ``cluster.created_at`` inclusively.
    """

    kinds: frozenset[str] | None = None
    sources: frozenset[str] | None = None
    min_score: float | None = None
    from_date: date | None = None
    to_date: date | None = None


def _filter_conditions(filters: ClusterFilters) -> list[sql.Composed | sql.SQL]:
    """Render WHERE fragments for the set filters (AND across them)."""
    conditions: list[sql.Composed | sql.SQL] = []
    if filters.kinds:
        kinds = sorted(filters.kinds)
        conditions.append(
            sql.SQL("cluster.kind_mix ?| array[{}]").format(
                sql.SQL(", ").join(sql.Literal(kind) for kind in kinds)
            )
        )
    if filters.sources:
        names = sorted(filters.sources)
        conditions.append(
            sql.SQL("source.name = ANY({})").format(
                sql.SQL("ARRAY[{}]").format(sql.SQL(", ").join(sql.Literal(name) for name in names))
            )
        )
    if filters.min_score is not None:
        conditions.append(
            sql.SQL("score.total >= {}").format(sql.Literal(float(filters.min_score)))
        )
    if filters.from_date is not None:
        conditions.append(
            sql.SQL("cluster.created_at >= date {}").format(
                sql.Literal(filters.from_date.isoformat())
            )
        )
    if filters.to_date is not None:
        # Inclusive upper bound: created_at carries a time component.
        conditions.append(
            sql.SQL("cluster.created_at < date {} + interval '1 day'").format(
                sql.Literal(filters.to_date.isoformat())
            )
        )
    return conditions


def list_clusters_filtered(
    conn: Connection,
    filters: ClusterFilters,
    *,
    include_hidden: bool = False,
) -> list[ClusterRow]:
    """Filtered :func:`list_clusters`; same ordering, combined filters AND."""
    conditions = _filter_conditions(filters)
    if not conditions:
        return list_clusters(conn, include_hidden=include_hidden)
    where = sql.SQL(" AND ").join(
        sql.SQL("(") + condition + sql.SQL(")") for condition in conditions
    )
    hide = _HIDE_NONE_SQL if include_hidden else _HIDE_DEFAULT_SQL
    query = sql.SQL(_CLUSTERS_WITH_FILTERS_SQL).format(
        hide_clause=sql.SQL(hide), conditions=where
    )
    rows = conn.execute(query).fetchall()
    return _build_rows(conn, rows)


def list_source_names(conn: Connection) -> list[str]:
    """Distinct source names that have pains attached (filter dropdown)."""
    rows = conn.execute(
        """
        SELECT DISTINCT source.name
        FROM source
        JOIN raw_post ON raw_post.source_id = source.id
        JOIN pain ON pain.raw_post_id = raw_post.id
        ORDER BY source.name
        """
    ).fetchall()
    return [str(name) for (name,) in rows]


@dataclass(frozen=True, slots=True)
class PainDetail:
    """One pain inside the cluster card, with its originating post/source."""

    body: str
    audience: str
    quote: str
    source_name: str
    post_title: str
    post_url: str
    kind: str
    published_at: datetime | None


@dataclass(frozen=True, slots=True)
class ClusterDetail:
    """Everything the cluster card renders (task 312)."""

    id: str
    title: str
    size: int
    kinds: dict[str, int]
    score: float | None
    rationale_md: str | None
    quotes: tuple[str, ...]
    created_at: datetime
    pains: tuple[PainDetail, ...]
    sources: tuple[str, ...]


def get_cluster_detail(conn: Connection, cluster_id: str) -> ClusterDetail | None:
    """Return the full card payload for one cluster, or None if unknown.

    Pains come with their raw post and source so the card can link back;
    ordering by pain id (UUIDv7) keeps the list stable across reruns.
    """
    head = conn.execute(
        """
        SELECT cluster.id,
               cluster.title,
               cluster.size,
               cluster.kind_mix,
               score.total,
               score.rationale_md,
               score.quotes_json,
               cluster.created_at
        FROM cluster
        LEFT JOIN score ON score.cluster_id = cluster.id
        WHERE cluster.id = %s
        """,
        (cluster_id,),
    ).fetchone()
    if head is None:
        return None
    (
        cluster_key,
        title,
        size,
        kind_mix,
        total,
        rationale_md,
        quotes,
        created_at,
    ) = head
    raw_mix = kind_mix if isinstance(kind_mix, dict) else json.loads(str(kind_mix))
    pain_rows = conn.execute(
        """
        SELECT pain.body,
               pain.audience,
               pain.quote,
               source.name,
               raw_post.title,
               raw_post.url,
               raw_post.kind,
               raw_post.published_at
        FROM pain
        JOIN raw_post ON raw_post.id = pain.raw_post_id
        JOIN source ON source.id = raw_post.source_id
        WHERE pain.cluster_id = %s
        ORDER BY pain.id
        """,
        (cluster_id,),
    ).fetchall()
    pains = tuple(
        PainDetail(
            body=str(body),
            audience=str(audience),
            quote=str(quote),
            source_name=str(source_name),
            post_title=str(post_title),
            post_url=str(post_url),
            kind=str(kind),
            published_at=published_at,
        )
        for body, audience, quote, source_name, post_title, post_url, kind, published_at in pain_rows
    )
    return ClusterDetail(
        id=str(cluster_key),
        title=str(title),
        size=int(size),
        kinds={str(k): int(v) for k, v in raw_mix.items()},
        score=None if total is None else float(total),
        rationale_md=None if rationale_md is None else str(rationale_md),
        quotes=tuple(str(q) for q in quotes) if quotes is not None else (),
        created_at=created_at,
        pains=pains,
        sources=tuple(sorted({pain.source_name for pain in pains})),
    )


#: Wordstat answers shown in the cluster card (task T321).
@dataclass(frozen=True, slots=True)
class WordstatCard:
    """One cached Wordstat check as the card renders it."""

    phrase: str
    frequency: int
    checked_at: datetime


def list_wordstat_card(conn: Connection, cluster_id: str) -> tuple[WordstatCard, ...]:
    """Return the cluster's cached Wordstat checks, newest first.

    Read-only; an empty tuple means the validate stage has not checked
    this cluster (the card shows a quiet placeholder, not an error).
    """
    rows = conn.execute(
        """
        SELECT phrase, frequency, checked_at
        FROM wordstat_query
        WHERE cluster_id = %s::uuid
        ORDER BY checked_at DESC, phrase
        """,
        (cluster_id,),
    ).fetchall()
    return tuple(
        WordstatCard(phrase=str(phrase), frequency=int(frequency), checked_at=checked_at)
        for phrase, frequency, checked_at in rows
    )
