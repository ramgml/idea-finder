"""Pipeline stages of idea-finder.

The lifecycle of one idea-finder run is the ordered stage sequence
collect → extract → cluster → score. Each stage is a synchronous function
that takes an open :class:`psycopg.Connection` as its first argument,
manages its own transaction (open / commit, or roll back on error), and
returns a ``dict[str, int]`` of counters. Restarting the pipeline is safe:
every stage is idempotent by contract (canonical URL + unique constraints),
and a stage that fails only rolls back its own transaction.

The bodies of the stages land with tasks B/C/D (collect: source adapters;
extract: LLM extraction; cluster: embeddings + grouping; score: market
rubric). Until then each function is a declared skeleton: it logs that its
body is not implemented yet and returns ``{"rows": 0}``.
"""

from __future__ import annotations

import logging

from psycopg import Connection

LOGGER = logging.getLogger(__name__)

type StageStats = dict[str, int]


def _skeleton_stage(conn: Connection, name: str) -> StageStats:
    """Shared skeleton body: own transaction, announce, zero counters."""
    with conn.transaction():
        LOGGER.info("stage %s not implemented", name)
    return {"rows": 0}


def run_collect(conn: Connection) -> StageStats:
    """Fetch new posts from all enabled sources into ``raw_post``.

    Implemented in task B-flow (source adapters).
    """
    return _skeleton_stage(conn, "collect")


def run_extract(conn: Connection) -> StageStats:
    """Extract pains from collected posts via the active LLM provider.

    Implemented in task C-flow (LLM extraction).
    """
    return _skeleton_stage(conn, "extract")


def run_cluster(conn: Connection) -> StageStats:
    """Embed pains and group them into clusters.

    Implemented in task D-flow (embeddings + clustering).
    """
    return _skeleton_stage(conn, "cluster")


def run_score(conn: Connection) -> StageStats:
    """Score clusters against the Russian-market rubric.

    Implemented in task E-flow (scoring rubric).
    """
    return _skeleton_stage(conn, "score")
