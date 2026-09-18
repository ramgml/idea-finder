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
from typing import Final

from psycopg import Connection

from idea_finder.core.embed import embed_pains
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

#: Cosine similarity above which a pain joins an existing cluster (the
#: value comes from context/SYSTEM_DESIGN.md; moved to config once the
#: project grows a configuration surface).
CLUSTER_THRESHOLD: Final = 0.82

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
    if not repo.list_posts_pending_extract(conn, limit=1):
        # Nothing to do: no run row, no provider lookup, zero counters. A
        # fresh cluster without an active LLM provider stays a no-op here.
        return {"processed": 0, "extracted": 0, "failed": 0, "llm_errors": 0,
                **extract_stats_to_dict(ExtractStats()), "cost_micro": 0}
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
    seen: set[str] = set()  # per-run guard: LlmError posts stay pending,
    # but one run must not re-send the same post to the provider forever.
    try:
        while True:
            batch = [(pid, b) for pid, b in repo.list_posts_pending_extract(
                conn, limit=50) if pid not in seen]
            if not batch:
                break
            for post_id, body in batch:
                seen.add(post_id)
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


def run_cluster(conn: Connection) -> StageStats:
    """Embed pains and group them into clusters (T309, D2).

    Greedy centroid clustering: pains arrive in deterministic order
    (``created_at, id``); a pain joins the first cluster whose centroid is
    within :data:`CLUSTER_THRESHOLD` cosine similarity, otherwise it opens
    a new cluster. Centroids are the L2-renormalized mean of member
    vectors, so the decision depends only on the partition, and a rerun
    over unchanged input reproduces the exact same grouping.

    Repeated runs are idempotent by full recompute: previous assignments
    are cleared (via :func:`repo.reset_cluster_assignment`) and every
    cluster row is rebuilt, so no orphans survive and ``count(*)`` of
    clusters matches the partition. Embeddings are backfilled first via
    :func:`idea_finder.core.embed.embed_pains` (no-op when current).

    Returns:
        ``{"pains": N, "clusters_new": X, "clusters_merged": Y,
        "singletons": Z, "embedded": E}`` — all int, written to the run's
        stats via the usual per-pass ``update_run_stage`` discipline.
    """
    embedded_counts = embed_pains(conn)
    pains = repo.list_cluster_input_pains(conn)
    if not pains:
        # Nothing to group: no run row, empty stats (mirrors run_extract's
        # empty-database no-op so a fresh cluster stays quiet).
        LOGGER.info("run_cluster: no embedded pains, nothing to do")
        return {"pains": 0, "clusters_new": 0, "clusters_merged": 0,
                "singletons": 0, "embedded": int(embedded_counts.get("embedded", 0))}

    repo.reset_cluster_assignment(conn)

    run_id = repo.create_run(conn, {"cluster": "running"})
    counters = {"pains": 0, "clusters_new": 0, "clusters_merged": 0,
                "singletons": 0, "embedded": 0}
    try:
        # Partition state: cluster id -> running centroid (L2-normalized)
        # and member pains. Recomputed greedily in deterministic order.
        centroids: dict[str, list[float]] = {}
        members: dict[str, list[repo.ClusterInputPain]] = {}
        for pain in pains:
            target = _nearest_cluster(centroids, pain.embedding, CLUSTER_THRESHOLD)
            if target is None:
                title = _cluster_title(pain)
                cluster_id = repo.upsert_cluster(conn, title, size=1,
                                                 kind_mix={pain.kind: 1})
                centroids[cluster_id] = _normalized(pain.embedding)
                members[cluster_id] = [pain]
                counters["clusters_new"] += 1
            else:
                cluster_id = target
                members[cluster_id].append(pain)
                counters["clusters_merged"] += 1
                centroid = _normalized(
                    _mean_vector([m.embedding for m in members[cluster_id]])
                )
                centroids[cluster_id] = centroid
            repo.assign_pain_cluster(conn, pain.id, cluster_id)
            counters["pains"] += 1

        for cluster_id, member_pains in members.items():
            kind_mix: dict[str, int] = {}
            for member in member_pains:
                kind_mix[member.kind] = kind_mix.get(member.kind, 0) + 1
            repo.update_cluster_stats(conn, cluster_id,
                                      len(member_pains), kind_mix)
        counters["singletons"] = sum(
            1 for group in members.values() if len(group) == 1
        )
        counters["embedded"] = int(embedded_counts.get("embedded", 0))
        repo.update_run_stage(conn, run_id, "cluster", "running",
                              stats_delta=dict(counters))
    finally:
        status = "done" if counters["pains"] == len(pains) else "error"
        # The deltas were already written once above; _stats_merge
        # accumulates, so the final update contributes a zero delta and
        # only flips the stage status.
        zero_delta = dict.fromkeys(counters, 0)
        repo.update_run_stage(conn, run_id, "cluster", status,
                              stats_delta=zero_delta)
        repo.finish_run(conn, run_id)
    return dict(counters)


def _nearest_cluster(centroids: dict[str, list[float]], vector: list[float],
                     threshold: float) -> str | None:
    """Return the first cluster id whose centroid is within ``threshold``.

    Cosine similarity is plain dot product here: embeddings and centroids
    are L2-normalized by construction.
    """
    best_id: str | None = None
    best_similarity = threshold
    for cluster_id, centroid in centroids.items():
        similarity = sum(a * b for a, b in zip(vector, centroid, strict=True))
        if similarity >= best_similarity:
            best_similarity = similarity
            best_id = cluster_id
    return best_id


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Component-wise mean of equal-length vectors."""
    length = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(length)]


def _normalized(vector: list[float]) -> list[float]:
    """Scale a vector to unit length (zero vector is returned as-is)."""
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _cluster_title(pain: repo.ClusterInputPain) -> str:
    """Placeholder cluster title; the D-flow UI names clusters later."""
    return f"cluster {pain.kind}"


def run_score(conn: Connection) -> StageStats:
    """Score clusters against the Russian-market rubric.

    Implemented in task E-flow (market rubric).
    """
    return _skeleton_stage(conn, "score")
