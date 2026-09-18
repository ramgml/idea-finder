"""Read-only dashboard queries for the Clusters page (task 311).

Schema ownership note (task 311 constraint): ``db/repo.py`` is shared with
the parallel extract-stage stream, so this module keeps the two read queries
the web layer needs here instead of editing repo.py. TODO(repo): move both
functions into ``db/repo.py`` once the extract stream merges; keep the
signatures and SQL as-is.

Everything here is read-only SELECTs; the dashboard never writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Final, LiteralString

from psycopg import Connection

#: Score rows are optional per cluster (score stage may not have run yet).
#: Sort missing scores last: NULLS LAST keeps unassessed clusters below any
#: scored one while preserving score-descending order for the rest.
_LIST_CLUSTERS_SQL: Final[LiteralString] = """
    SELECT cluster.id,
           cluster.title,
           cluster.size,
           cluster.kind_mix,
           score.total,
           cluster.created_at
    FROM cluster
    LEFT JOIN score ON score.cluster_id = cluster.id
    ORDER BY score.total DESC NULLS LAST, cluster.size DESC, cluster.created_at DESC
"""


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


def list_clusters(conn: Connection) -> list[ClusterRow]:
    """Return every cluster with its score, sorted by score (best first).

    Ties fall back to cluster size, then creation time, so the page order is
    deterministic across reruns. Clusters without a score sort after scored
    ones (``score`` is None).
    """
    rows = conn.execute(_LIST_CLUSTERS_SQL).fetchall()
    base: list[tuple[str, str, int, dict[str, int], float | None, datetime]] = []
    for cluster_id, title, size, kind_mix, total, created_at in rows:
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
            )
        )
    clusters: list[ClusterRow] = []
    for cluster_id, title, size, kinds, score_value, created_at in base:
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
        clusters.append(
            ClusterRow(
                id=cluster_id,
                title=title,
                size=size,
                kinds=kinds,
                score=score_value,
                created_at=created_at,
                sources=tuple(str(name) for (name,) in source_rows),
            )
        )
    return clusters


def count_clusters(conn: Connection) -> int:
    """Return the number of cluster rows (empty-state check for the page)."""
    row = conn.execute("SELECT count(*) FROM cluster").fetchone()
    return int(row[0]) if row is not None else 0
