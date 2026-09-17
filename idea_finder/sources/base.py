"""Source adapter contract: how to plug a new pain source into the pipeline.

To add a source, write three things:

1. A new adapter file in ``idea_finder/sources/`` (e.g. ``fl_ru.py``).
2. In it, a class implementing the :class:`SourceAdapter` protocol below.
3. A registry entry (see ``idea_finder.sources``; the seeded names live in
   ``seed_sources``).

The pipeline never changes: stages consume ``RawPost`` rows, not adapters.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol, runtime_checkable

from psycopg import Connection

from idea_finder.core.models import RawPost

logger = logging.getLogger(__name__)

__all__ = ["SEED_SOURCES", "SourceAdapter", "seed_sources"]

#: Registry of sources seeded into the ``source`` table. ``rate_limit_rps``
#: is the per-domain politeness cap for the fetch layer (default 1 rps,
#: AGENTS.md sources rule 3). A new adapter adds its ``(name, rate)`` pair
#: here, then ``seed_sources`` puts it in the database idempotently.
SEED_SOURCES: tuple[tuple[str, float], ...] = (
    ("fl_ru", 1.0),
    ("habr", 1.0),
    ("gplay", 1.0),
)


@runtime_checkable
class SourceAdapter(Protocol):
    """Contract every source adapter must satisfy.

    ``name`` is the stable registry key matching a ``source.name`` row;
    ``fetch_new`` returns posts newer than ``since`` (or all known posts when
    ``since`` is None, e.g. on the first run).

    Adapters are async because the fetch layer under them is async
    (httpx + aiolimiter). The pipeline's full run is currently synchronous;
    the B4 collect stage will bridge with ``asyncio.run`` when wiring
    adapters in, so adapter implementations must define ``fetch_new``
    as ``async def``.
    """

    name: str

    async def fetch_new(self, since: datetime | None) -> list[RawPost]:
        """Return posts newer than ``since`` (all known posts when None)."""
        ...


def seed_sources(conn: Connection) -> None:
    """Insert :data:`SEED_SOURCES` into the ``source`` table, idempotently.

    Existing rows keep their ``enabled``/``rate_limit_rps`` values: repeated
    calls must not silently re-enable a source an operator disabled or
    override a tuned rate. Only missing names are inserted.
    """
    with conn.transaction():
        for name, rate in SEED_SOURCES:
            conn.execute(
                """
                INSERT INTO source (name, rate_limit_rps) VALUES (%s, %s)
                ON CONFLICT (name) DO NOTHING
                """,
                (name, rate),
            )
    logger.info("seeded sources: %s", ", ".join(name for name, _ in SEED_SOURCES))
