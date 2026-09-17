"""Extract stage: pains from collected posts via the active LLM provider.

Contract (T307a, design frozen after 5 failed worker attempts):

* the stage creates and owns its ``run`` row (create_run / update_run_stage /
  finish_run) so CLI ``run`` composes stages without run plumbing;
* per-batch ``update_run_stage`` pins ``prompt_version_id`` write-once
  (COALESCE in repo), making the run reproducible against the shipped prompt;
* pending posts come from :func:`repo.list_posts_pending_extract` (no pain yet
  AND fetch_status new/fetched); a processed post is marked ``fetched``
  (even with zero accepted pains), a hopeless answer (unparseable JSON) is
  marked ``failed`` — brick-once, never retry, so repeated runs converge;
* LLM/programmatic errors are surfaced as ``llm_errors`` counters without
  status labels: the post stays pending and the next run retries it;
* costs accumulate as float rubles in ``run.cost`` (estimate_cost) and as
  integer micro-units in ``stats["cost_micro"]`` for the int-only stats JSON.

Usage (smoke)::

    python -m idea_finder.core.pipeline extract [data_dir]
"""

from __future__ import annotations

import logging

from psycopg import Connection

from idea_finder.core.models import Pain
from idea_finder.db import repo
from idea_finder.llm.client import LlmError, build_llm_client, estimate_cost
from idea_finder.llm.extract import (
    PROMPT_NAME,
    PROMPT_VERSION,
    ExtractStats,
    InvalidResponseError,
    extract_stats_to_dict,
    load_extract_template,
    parse_extract_response,
    render_extract_prompt,
)

LOGGER = logging.getLogger(__name__)

type StageStats = dict[str, int]


def _to_micro(cost: float) -> int:
    """Convert a float ruble cost to integer micro-units (1e-6 precision)."""
    return round(cost * 1_000_000)


def run_extract(conn: Connection) -> StageStats:
    """Extract pains from pending posts via the active LLM provider.

    Returns the stage counters: ``processed`` (posts given to the LLM),
    ``extracted`` (pains accepted and stored), ``failed`` (posts bricked
    with hopeless answers), ``llm_errors``, plus the brick breakdown from
    :class:`ExtractStats` and ``cost_micro``.
    """
    template = load_extract_template()
    prompt_version_id = repo.upsert_prompt_version(
        conn, PROMPT_NAME, PROMPT_VERSION, template, "file"
    )
    run_id = repo.create_run(conn, {"extract": "running"})
    client = build_llm_client(conn)
    provider = repo.get_active_llm_provider(conn)

    stats: ExtractStats = ExtractStats()
    counters = {"processed": 0, "extracted": 0, "failed": 0, "llm_errors": 0}
    cost_total = 0.0

    try:
        while True:
            batch = repo.list_posts_pending_extract(conn, limit=50)
            if not batch:
                break
            for post_id, body in batch:
                counters["processed"] += 1
                try:
                    completion = client.complete(render_extract_prompt(body))
                    answer = completion.text
                except LlmError:
                    counters["llm_errors"] += 1
                    # No status label: the post stays pending, next run retries.
                    continue
                try:
                    pains, answer_stats = parse_extract_response(answer, body)
                except InvalidResponseError:
                    stats.invalid_json += 1
                    counters["failed"] += 1
                    repo.set_raw_post_fetch_status(conn, post_id, "failed")
                    continue
                stats.add(answer_stats)
                _store_pains(conn, post_id, pains, prompt_version_id)
                counters["extracted"] += len(pains)
                # Terminal pin: processed with any outcome; never re-sent.
                repo.set_raw_post_fetch_status(conn, post_id, "extracted")

                cost_total += estimate_cost(
                    completion,
                    provider.price_per_mtok if provider else None,
                )
            repo.update_run_stage(
                conn, run_id, "extract", "running",
                stats_delta={**counters, **extract_stats_to_dict(stats),
                             "cost_micro": _to_micro(cost_total)},
                prompt_version_id=prompt_version_id,
            )
    finally:
        status = "done" if counters["llm_errors"] == 0 else "error"
        repo.update_run_stage(
            conn, run_id, "extract", status,
            stats_delta={**counters, **extract_stats_to_dict(stats),
                         "cost_micro": _to_micro(cost_total)},
            prompt_version_id=prompt_version_id,
        )
        repo.finish_run(conn, run_id)
    return {**counters, **extract_stats_to_dict(stats), "cost_micro": _to_micro(cost_total)}


def _store_pains(conn: Connection, post_id: str, pains: list[Pain],
                 prompt_version_id: str) -> None:
    """Persist accepted pains bound to their source post."""
    for pain in pains:
        repo.insert_pain(
            conn,
            Pain(source_post_id=post_id, body=pain.body,
                 audience=pain.audience, quote=pain.quote),
            None,
            prompt_version_id,
        )


def run_collect(conn: Connection) -> StageStats:
    """Fetch new posts from all enabled sources (task B-flow, pending)."""
    raise NotImplementedError


def run_cluster(conn: Connection) -> StageStats:
    """Embed pains and group them into clusters (task D-flow, pending)."""
    raise NotImplementedError


def run_score(conn: Connection) -> StageStats:
    """Score clusters against the Russian-market rubric (task E-flow)."""
    raise NotImplementedError
