"""Read-only dashboard queries for the Health ("Здоровье") page (task 313).

Schema ownership note (task 313 constraint): ``db/repo.py`` is shared with
other concurrent streams, so this module keeps the health-page read queries
the web layer needs here instead of editing repo.py. TODO(repo): move these
functions into ``db/repo.py`` once the concurrent streams merge; keep the
signatures and SQL as-is.

Everything here is read-only SELECTs; the dashboard never writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final, LiteralString

from psycopg import Connection

from idea_finder.web.run_view import RunRow, list_runs

#: How many recent runs the history table shows.
_HISTORY_LIMIT: Final[int] = 20


_SOURCE_HEALTH_SQL: Final[LiteralString] = """
    SELECT s.name,
           max(rp.created_at) FILTER (WHERE rp.fetch_status <> 'failed') AS last_ok,
           count(*) FILTER (WHERE rp.fetch_status = 'failed')            AS failed_count,
           count(*)                                                      AS total_count,
           count(*) FILTER (WHERE rp.suspect_short)                      AS suspect_short_count
    FROM source s
    LEFT JOIN raw_post rp ON rp.source_id = s.id
    WHERE s.enabled
    GROUP BY s.name
    ORDER BY s.name
"""


@dataclass(frozen=True, slots=True)
class SourceHealthRow:
    """Per-source extraction health over collected raw posts."""

    name: str
    #: Last post collected without a fetch failure (None = never succeeded).
    last_ok: datetime | None
    #: Posts whose extraction answer was hopeless (fetch_status='failed').
    failed_count: int
    #: All posts collected from the source.
    total_count: int
    #: Posts flagged with a suspiciously short body (suspect_short).
    suspect_short_count: int

    @property
    def suspect_share(self) -> float:
        """Share of suspect-short posts, 0.0-1.0 (0.0 when nothing collected)."""
        if self.total_count == 0:
            return 0.0
        return self.suspect_short_count / self.total_count


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Everything the Health page renders: runs history + source health."""

    runs: list[RunRow]
    sources: list[SourceHealthRow]


def list_source_health(conn: Connection) -> list[SourceHealthRow]:
    """Return per-source health for every enabled source, ordered by name.

    ``last_ok`` is the freshest created_at among posts that did not end in
    the terminal ``failed`` state; a source that never produced a usable
    post reports None and the page shows a warning marker.
    """
    rows = conn.execute(_SOURCE_HEALTH_SQL).fetchall()
    return [
        SourceHealthRow(
            name=str(name),
            last_ok=last_ok,
            failed_count=int(failed_count),
            total_count=int(total_count),
            suspect_short_count=int(suspect_short_count),
        )
        for name, last_ok, failed_count, total_count, suspect_short_count in rows
    ]


def health_report(conn: Connection) -> HealthReport:
    """Aggregate the Health page payload: recent runs + source health."""
    return HealthReport(runs=list_runs(conn, _HISTORY_LIMIT), sources=list_source_health(conn))
