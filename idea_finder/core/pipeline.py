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

import asyncio
import logging
import socket
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import replace
from datetime import timedelta
from typing import Final

import aiolimiter
from psycopg import Connection

from idea_finder.core.embed import embed_pains
from idea_finder.core.models import Pain, RawPost, Score
from idea_finder.db import repo
from idea_finder.db.repo import list_clusters_stale_for_wordstat
from idea_finder.fetch.httpx_fetcher import HttpFetcher
from idea_finder.llm.client import LlmClient, LlmError, build_llm_client, estimate_cost
from idea_finder.llm.extract import (
    PROMPT_NAME,
    PROMPT_VERSION,
    ExtractStats,
    FakeExtractLlmClient,
    InvalidResponseError,
    extract_stats_to_dict,
    load_extract_template,
    parse_extract_response,
    render_extract_prompt,
)
from idea_finder.llm.score import (
    SCORE_PROMPT_NAME,
    SCORE_PROMPT_VERSION,
    ClusterSummary,
    FakeScoreLlmClient,
    InvalidScoreResponseError,
    llm_score_to_storage,
    load_score_template,
    parse_score_response,
    render_score_prompt,
)
from idea_finder.llm.wordstat_phrases import (
    WORDSTAT_PROMPT_NAME,
    WORDSTAT_PROMPT_VERSION,
    FakeWordstatLlmClient,
    InvalidPhrasesResponseError,
    load_wordstat_template,
    parse_phrases_response,
    render_wordstat_prompt,
)
from idea_finder.sources.base import SourceAdapter
from idea_finder.sources.fl_ru import FlRuAdapter
from idea_finder.sources.gplay import GPlayAdapter
from idea_finder.sources.habr import HabrAdapter
from idea_finder.wordstat.client import (
    WordstatApiError,
    WordstatClient,
    build_wordstat_client,
)

LOGGER = logging.getLogger(__name__)

#: Cosine similarity above which a pain joins an existing cluster
#: (first-member similarity, threshold 0.88, G1-calibrated, re-verified on
#: the 41-pain fix corpus in T342 — see :func:`run_cluster` for why the
#: centroid is frozen at the first member; configuration surface pending).
CLUSTER_THRESHOLD: Final = 0.88

#: A phrase whose Wordstat frequency reaches this many monthly searches
#: marks the cluster's demand as search-confirmed; the signal goes into
#: the score rationale. The config surface (dashboard/tables) comes later
#: — a module constant keeps v1 behavior reviewable.
WORDSTAT_DEMAND_THRESHOLD: Final = 100

#: Wordstat API politeness cap: 10 requests/second (task T321). Enforced
#: per query burst through ``aiolimiter``, mirroring the fetch layer's
#: per-domain limiting composition.
WORDSTAT_RATE_RPS: Final[float] = 10.0

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
        return {
            "processed": 0,
            "extracted": 0,
            "failed": 0,
            "llm_errors": 0,
            **extract_stats_to_dict(ExtractStats()),
            "cost_micro": 0,
        }
    template = load_extract_template()
    prompt_version_id = repo.upsert_prompt_version(
        conn, PROMPT_NAME, PROMPT_VERSION, template, "file"
    )
    run_id = repo.create_run(conn, {"extract": "running"})
    provider = repo.get_active_llm_provider(conn)
    if provider is not None and provider.kind == "fake":
        # Mock mode (CONTEXT.md: is_default on fake = full pipeline run
        # without code or a key): the factory's FakeLlmClient answers with
        # raw pain texts, but the extract stage needs extract-JSON, so the
        # stage swaps in the format-aware fake (same pattern as the score
        # stage's FakeScoreLlmClient). Real providers come from the
        # factory unchanged.
        client = FakeExtractLlmClient(model=provider.model or "fake", provider_name=provider.name)
    else:
        client = build_llm_client(conn)

    stats: ExtractStats = ExtractStats()
    counters = {"processed": 0, "extracted": 0, "failed": 0, "llm_errors": 0}
    cost_total = 0.0
    seen: set[str] = set()  # per-run guard: LlmError posts stay pending,
    # but one run must not re-send the same post to the provider forever.
    try:
        while True:
            batch = [
                (pid, b)
                for pid, b in repo.list_posts_pending_extract(conn, limit=50)
                if pid not in seen
            ]
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
                conn,
                run_id,
                "extract",
                "running",
                stats_delta={
                    **counters,
                    **extract_stats_to_dict(stats),
                    "cost_micro": _to_micro(cost_total),
                },
                prompt_version_id=prompt_version_id,
            )
        # Deltas were written with each batch; the final update only flips
        # the stage status with a zero delta (_stats_merge accumulates, so
        # re-sending the cumulative totals here would double the run row).
    finally:
        status = "done" if counters["llm_errors"] == 0 else "error"
        repo.update_run_stage(
            conn,
            run_id,
            "extract",
            status,
            stats_delta=None,
            prompt_version_id=prompt_version_id,
        )
        repo.finish_run(conn, run_id)
    return {**counters, **extract_stats_to_dict(stats), "cost_micro": _to_micro(cost_total)}


def _store_pains(conn: Connection, post_id: str, pains: list[Pain], prompt_version_id: str) -> None:
    """Persist accepted pains bound to their source post."""
    for pain in pains:
        repo.insert_pain(
            conn,
            Pain(source_post_id=post_id, body=pain.body, audience=pain.audience, quote=pain.quote),
            None,
            prompt_version_id,
        )


#: Registry mapping ``source.name`` rows to adapter factories. The factory
#: takes the row's ``rate_limit_rps`` so each adapter tunes its limiter
#: from the operator-tuned value (default 1 rps, migration 003). Tests
#: monkeypatch this dict — real network is never touched in the suite.
_ADAPTERS: Final[dict[str, Callable[[float], SourceAdapter]]] = {
    "fl_ru": lambda rate: FlRuAdapter(),
    "habr": lambda rate: HabrAdapter(rate_limit_rps=rate),
    "gplay": lambda rate: GPlayAdapter(),
}


#: Per-source wall-clock budget for one ``fetch_new`` call (seconds). The
#: stage must stay responsive even when a scraper library ignores socket
#: timeouts, so the coroutine is cancelled at the stage boundary.
SOURCE_FETCH_TIMEOUT_S: Final[float] = 300.0

#: Process-wide socket fallback for libs that call ``urlopen`` without a
#: timeout (google-play-scraper): against a half-open TLS endpoint the
#: worker thread would block forever and ``asyncio.run`` — which joins the
#: default executor on exit — would hang the stage after cancellation.
#: Scoped set/restore around each source; explicit timeouts win elsewhere.
FETCH_SOCKET_TIMEOUT_S: Final[float] = 30.0


def _open_fetcher(
    adapter: SourceAdapter,
) -> AbstractAsyncContextManager[HttpFetcher | None]:
    """Return the adapter's fetcher as an async context manager.

    Fetcher-backed adapters (fl_ru, habr) build their ``HttpFetcher`` in
    ``__init__`` but open the HTTP client only inside ``__aenter__``;
    ``fetch_new`` outside the context fails on the first request with
    "used outside 'async with'". Adapters without a fetcher (gplay owns
    its sync client) get a no-op wrapper.
    """
    fetcher = getattr(adapter, "_fetcher", None)
    if isinstance(fetcher, HttpFetcher):
        return fetcher
    return nullcontext(None)


async def _fetch_source(adapter: SourceAdapter) -> list[RawPost]:
    """One ``fetch_new`` with the adapter's fetcher context opened."""
    async with _open_fetcher(adapter):
        return list(await adapter.fetch_new(None))


async def _fetch_source_bounded(adapter: SourceAdapter) -> list[RawPost]:
    """``_fetch_source`` under the stage's wall-clock budget.

    ``asyncio.wait_for`` cancels the coroutine on expiry. The loop is
    closed without ``shutdown_default_executor`` (the caller runs this on
    its own loop via :func:`_run_bounded`), so a worker thread a scraper
    parked work in cannot block the stage: its sockets are bounded by
    :data:`FETCH_SOCKET_TIMEOUT_S` instead.
    """
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(FETCH_SOCKET_TIMEOUT_S)
    try:
        return await asyncio.wait_for(_fetch_source(adapter), timeout=SOURCE_FETCH_TIMEOUT_S)
    finally:
        socket.setdefaulttimeout(previous)


def _run_bounded(adapter: SourceAdapter) -> list[RawPost]:
    """Sync bridge: one bounded fetch on a fresh, explicitly-closed loop.

    ``asyncio.run`` would join the loop's default executor at shutdown —
    with a wedged worker thread that join has no timeout, so the stage
    would hang past its own budget. A bare ``run_until_complete`` + close
    returns control at the budget; leftover worker threads die with the
    process (bounded by :data:`FETCH_SOCKET_TIMEOUT_S`).
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_fetch_source_bounded(adapter))
    finally:
        loop.close()


def run_collect(conn: Connection) -> StageStats:
    """Fetch new posts from all enabled sources into ``raw_post`` (T335, B5).

    Orchestrates the source adapters (T301-T304, library layer): enabled
    rows from ``source`` (operator-controlled via the dashboard), each
    served by its registry adapter, async ``fetch_new`` bridged to sync at
    the stage boundary. Records land through
    :func:`repo.insert_raw_post` — its ``url_canon`` UNIQUE + DO NOTHING
    makes the stage idempotent (a rerun with no new feed items collects
    nothing).

    One failing source never fails the stage (isolation pattern from the
    gplay adapter): each source runs under
    :data:`SOURCE_FETCH_TIMEOUT_S` and the error is counted in
    ``warnings``, the remaining sources still deliver. A ``source`` row
    whose name has no registered adapter counts as a warning too.
    Records only ever enter via ``repo``.

    Returns ``{"collected": N, "skipped": D, "warnings": W}`` — new rows,
    URL duplicates, and failed/unknown sources respectively.
    """
    enabled = repo.list_enabled_sources(conn)
    run_id = repo.create_run(conn, {"collect": "running"})
    counters = {"collected": 0, "skipped": 0, "warnings": 0}
    status = "done"
    try:
        for name, rate in enabled:
            adapter_factory = _ADAPTERS.get(name)
            if adapter_factory is None:
                # A source row without a registered adapter is an
                # operator/registry mismatch, not a network flap.
                LOGGER.warning("collect: no adapter registered for %s", name)
                counters["warnings"] += 1
                continue
            source_id = repo.ensure_source(conn, name)
            try:
                posts = _run_bounded(adapter_factory(rate))
            except Exception as e:  # noqa: BLE001 - per-source isolation
                LOGGER.warning(
                    "collect: source %s failed, skipped: %s: %s",
                    name,
                    type(e).__name__,
                    e,
                )
                counters["warnings"] += 1
                continue
            # raw_post.source_id is a uuid FK: the adapter's registry name
            # maps onto the source row id before insert.
            for post in posts:
                db_post = replace(post, source_id=source_id)
                if repo.insert_raw_post(conn, db_post) is None:
                    counters["skipped"] += 1
                else:
                    counters["collected"] += 1
        repo.update_run_stage(conn, run_id, "collect", status, stats_delta=dict(counters))
    except Exception:
        status = "error"
        repo.update_run_stage(conn, run_id, "collect", status, stats_delta=dict(counters))
        raise
    finally:
        # The deltas were written once above; this update only flips the
        # stage status (_stats_merge accumulates) and closes the run even
        # on a crash mid-loop — a run row never hangs "running". The
        # status variable preserves "error" from the except branch: the
        # unconditional finally must never overwrite it with "done"
        # (same discipline as run_cluster/run_score).
        repo.update_run_stage(
            conn, run_id, "collect", status, stats_delta=dict.fromkeys(counters, 0)
        )
        repo.finish_run(conn, run_id)
    return dict(counters)


def run_cluster(conn: Connection) -> StageStats:
    """Embed pains and group them into clusters (T309, T342/D3).

    First-member greedy clustering: pains arrive in deterministic order
    (``created_at, id``); a pain joins the first cluster whose centroid is
    within :data:`CLUSTER_THRESHOLD` cosine similarity, otherwise it opens
    a new cluster. The centroid of a cluster is the vector of its FIRST
    member and is never updated afterwards — this is a deliberate fix of
    the D3 centroid-drift failure mode: an L2-normalized mean of many
    short-phrase embeddings drifts toward the corpus average (cosine
    0.89-0.94 to any new pain), so under centroid update a single
    attractor cluster absorbed everything. Freezing the centroid keeps
    every decision against one fixed reference vector, so the partition
    is order-stable and a rerun over unchanged input reproduces the exact
    same grouping.

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
        return {
            "pains": 0,
            "clusters_new": 0,
            "clusters_merged": 0,
            "singletons": 0,
            "embedded": int(embedded_counts.get("embedded", 0)),
        }

    repo.reset_cluster_assignment(conn)

    run_id = repo.create_run(conn, {"cluster": "running"})
    counters = {"pains": 0, "clusters_new": 0, "clusters_merged": 0, "singletons": 0, "embedded": 0}
    try:
        # Partition state: cluster id -> frozen first-member centroid
        # (L2-normalized) and member pains. Greedy assignment in
        # deterministic order; centroids never change once created.
        centroids: dict[str, list[float]] = {}
        members: dict[str, list[repo.ClusterInputPain]] = {}
        for pain in pains:
            target = _nearest_cluster(centroids, pain.embedding, CLUSTER_THRESHOLD)
            cluster_id: str
            if target is None:
                title = _cluster_title(pain)
                cluster_id = repo.upsert_cluster(conn, title, size=1, kind_mix={pain.kind: 1})
                centroids[cluster_id] = _normalized(pain.embedding)
                members[cluster_id] = [pain]
                counters["clusters_new"] += 1
            else:
                cluster_id = target
                members[cluster_id].append(pain)
                counters["clusters_merged"] += 1
            repo.assign_pain_cluster(conn, pain.id, cluster_id)
            counters["pains"] += 1

        for cluster_id, member_pains in members.items():
            kind_mix: dict[str, int] = {}
            for member in member_pains:
                kind_mix[member.kind] = kind_mix.get(member.kind, 0) + 1
            repo.update_cluster_stats(conn, cluster_id, len(member_pains), kind_mix)
        counters["singletons"] = sum(1 for group in members.values() if len(group) == 1)
        counters["embedded"] = int(embedded_counts.get("embedded", 0))
        repo.update_run_stage(conn, run_id, "cluster", "running", stats_delta=dict(counters))
    finally:
        status = "done" if counters["pains"] == len(pains) else "error"
        # The deltas were already written once above; _stats_merge
        # accumulates, so the final update contributes a zero delta and
        # only flips the stage status.
        zero_delta = dict.fromkeys(counters, 0)
        repo.update_run_stage(conn, run_id, "cluster", status, stats_delta=zero_delta)
        repo.finish_run(conn, run_id)
    return dict(counters)


def _nearest_cluster(
    centroids: dict[str, list[float]], vector: list[float], threshold: float
) -> str | None:
    """Return the first cluster id whose centroid is within ``threshold``.

    Cosine similarity is plain dot product here: embeddings and centroids
    are L2-normalized by construction, and each centroid is the frozen
    first-member vector (T342) — it does not move as members join.
    """
    best_id: str | None = None
    best_similarity = threshold
    for cluster_id, centroid in centroids.items():
        similarity = sum(a * b for a, b in zip(vector, centroid, strict=True))
        if similarity >= best_similarity:
            best_similarity = similarity
            best_id = cluster_id
    return best_id


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
    """Score unscored clusters against the Russian-market rubric (T310, E1).

    Each cluster without a score gets the v1 rubric prompt
    (:func:`~idea_finder.llm.score.render_score_prompt`, 7 weighted
    criteria, demand-boost on monetization) answered by the active LLM
    provider; the validated answer lands in ``score`` via
    :func:`repo.insert_score` (one row per cluster, replaced on re-score).

    Idempotent by selection: clusters already carrying a score are skipped,
    so a rerun over unchanged state processes nothing. Validation rejects
    (never raises) bad model output: unparseable JSON, out-of-range score,
    empty rationale, hallucinated quotes — the rejected counter tracks them
    and the cluster stays unscored for a later run.

    The distribution sanity gate (context/SCORING.md rule 3) sets
    ``sanity_flag`` to 1 when at least 3 clusters were scored into a
    suspiciously narrow band (spread < 10 points) — a degenerate rubric
    the system-health page should surface.

    Returns:
        ``{"clusters": N, "scored": S, "rejected": R, "sanity_flag": F}``
        run's stats; empty clusters -> zero stats and no
        run row (mirrors the other stages' no-op).
    """
    clusters = repo.list_clusters_without_score(conn)
    if not clusters:
        LOGGER.info("run_score: no unscored clusters, nothing to do")
        return {"clusters": 0, "scored": 0, "rejected": 0, "sanity_flag": 0}

    repo.upsert_prompt_version(
        conn,
        SCORE_PROMPT_NAME,
        SCORE_PROMPT_VERSION,
        load_score_template(),
        "file",
    )
    run_id = repo.create_run(conn, {"score": "running"})
    provider = repo.get_active_llm_provider(conn)
    if provider is not None and provider.kind == "fake":
        # Owner UPDATE 2026-09-17: scoring runs on the fake provider —
        # its fixture-derived raw-pain answers are extract-shaped, not
        # score JSON, so the stage swaps in the score-format fake.
        client = FakeScoreLlmClient(model=provider.model or "fake", provider_name=provider.name)
    else:
        client = build_llm_client(conn)

    counters = {"clusters": len(clusters), "scored": 0, "rejected": 0, "sanity_flag": 0}
    totals: list[float] = []
    try:
        for cluster in clusters:
            prompt = render_score_prompt(
                ClusterSummary(
                    size=cluster.size,
                    kinds=cluster.kind_mix,
                    sources=cluster.sources,
                    first_seen=cluster.first_seen,
                    last_seen=cluster.last_seen,
                    bodies=cluster.bodies,
                )
            )
            try:
                answer = parse_score_response(client.complete(prompt).text, cluster.bodies)
            except InvalidScoreResponseError:
                # Rejected output: the cluster stays unscored, the run
                # moves to the next one (SCORING.md rule 2: no rationale /
                # no quotes -> no score).
                counters["rejected"] += 1
                continue
            repo.insert_score(
                conn,
                Score(
                    cluster_id=cluster.id,
                    # LLM rubric is 0-100 (SCORING.md); storage column is
                    # 0-10 (migration 001) — convert at write time.
                    total=llm_score_to_storage(answer.total),
                    rationale_md=answer.rationale_md,
                    quotes=answer.quotes,
                ),
            )
            totals.append(answer.total)
            counters["scored"] += 1
        if len(totals) >= 3 and max(totals) - min(totals) < 10:
            # Degenerate rubric: everything scored into one narrow band
            # (SCORING.md rule 3) — surface it in run.stats.
            counters["sanity_flag"] = 1
            LOGGER.warning(
                "run_score: score distribution is suspiciously narrow "
                "(%d scores within %.1f points)",
                len(totals),
                max(totals) - min(totals),
            )
        repo.update_run_stage(conn, run_id, "score", "running", stats_delta=dict(counters))
    finally:
        status = "done" if counters["scored"] + counters["rejected"] == len(clusters) else "error"
        # Deltas were written once above; the final update only flips the
        # stage status (_stats_merge accumulates).
        repo.update_run_stage(conn, run_id, "score", status, stats_delta=dict.fromkeys(counters, 0))
        repo.finish_run(conn, run_id)
    return dict(counters)


#: Cluster-summary renderer shared with the score stage: the wordstat
#: prompt receives the same human-readable facts block.
def _wordstat_cluster_summary(
    cluster: repo.ClusterScoreInput | object,
    bodies: list[str],
) -> str:
    """Render a compact facts block for the phrase-generation prompt."""
    lines = ["Боли кластера:"]
    lines.extend(f"- {body}" for body in bodies)
    return "\n".join(lines)


def run_validate(
    conn: Connection,
    *,
    wordstat_client: WordstatClient | None = None,
    llm: LlmClient | None = None,
    revalidate_after: timedelta | None = None,
) -> StageStats:
    """Validate search demand per cluster through Yandex Wordstat (T321).

    For every cluster without a fresh ``wordstat_query`` cache (missing or
    older than ``revalidate_after``, default ~a week via
    :data:`repo.WORDSTAT_REVALIDATE_AFTER`):

    1. the active LLM provider generates 3-5 search phrases for the pain
       (through the :class:`LlmClient` port; mock mode = fake provider
       answered by ``FakeWordstatLlmClient``, same pattern as extract/score);
    2. each phrase goes to the Wordstat client (the port hides fake vs
       real Yandex API), throttled to :data:`WORDSTAT_RATE_RPS` via
       ``aiolimiter``;
    3. the frequencies land in ``wordstat_query`` via the repo UPSERT —
       idempotent per (cluster, phrase), so a rerun over fresh state is a
       no-op and a crashed run resumes safely (same discipline as
       run_collect: one failing cluster never fails the stage, and the
       run row always closes; an ``error`` status is never overwritten by
       ``done``).

    Retries stay convergent: a phrase whose Wordstat call failed after the
    client's internal retry schedule is counted (``api_errors``) and the
    cluster keeps its partial cache — the next validate run re-picks the
    cluster (cache is incomplete but stale/absent rows keep it in the
    selection) and re-queries only what the LLM regenerates.

    A cluster whose best phrase reaches :data:`WORDSTAT_DEMAND_THRESHOLD`
    has its search demand confirmed; the stage appends the signal line to
    the cluster's score rationale (rubric v1 untouched — the signal only
    enriches the existing text, T321 owner constraint).

    Returns ``{"clusters": N, "phrases": P, "validated": V,
    "rejected": R, "api_errors": A, "demand_confirmed": D,
    "rationale_updated": U}`` written to the run stats per batch.
    """
    if revalidate_after is not None:
        clusters = list_clusters_stale_for_wordstat(
            conn, revalidate_after=revalidate_after
        )
    else:
        clusters = list_clusters_stale_for_wordstat(conn)
    if not clusters:
        LOGGER.info("run_validate: no stale clusters, nothing to do")
        return {
            "clusters": 0,
            "phrases": 0,
            "validated": 0,
            "rejected": 0,
            "api_errors": 0,
            "demand_confirmed": 0,
            "rationale_updated": 0,
        }

    repo.upsert_prompt_version(
        conn,
        WORDSTAT_PROMPT_NAME,
        WORDSTAT_PROMPT_VERSION,
        load_wordstat_template(),
        "file",
    )
    run_id = repo.create_run(conn, {"validate": "running"})
    provider = repo.get_active_llm_provider(conn)
    if llm is not None:
        client: LlmClient = llm
    elif provider is not None and provider.kind == "fake":
        # Mock mode: the fake provider answers pain texts, not phrase
        # JSON — swap in the stage-format fake (extract/score pattern).
        client = FakeWordstatLlmClient(model=provider.model or "fake", provider_name=provider.name)
    else:
        client = build_llm_client(conn)
    wordstat: WordstatClient = (
        wordstat_client if wordstat_client is not None else build_wordstat_client(conn)
    )

    counters = {
        "clusters": len(clusters),
        "phrases": 0,
        "validated": 0,
        "rejected": 0,
        "api_errors": 0,
        "demand_confirmed": 0,
        "rationale_updated": 0,
    }
    status = "done"
    try:
        limiter = aiolimiter.AsyncLimiter(WORDSTAT_RATE_RPS, 1.0)

        def _bounded_frequency(phrase: str) -> int:
            """One rate-limited Wordstat query through the sync bridge."""

            async def _one() -> int:
                async with limiter:
                    return wordstat.frequency(phrase)

            return asyncio.run(_one())

        for cluster in clusters:
            prompt = render_wordstat_prompt(_wordstat_cluster_summary(cluster, cluster.bodies))
            try:
                answer = parse_phrases_response(client.complete(prompt).text)
            except InvalidPhrasesResponseError:
                counters["rejected"] += 1
                continue
            except LlmError:
                counters["rejected"] += 1
                continue
            counters["phrases"] += len(answer.phrases)
            cluster_confirmed = False
            for phrase in answer.phrases:
                try:
                    frequency = _bounded_frequency(phrase)
                except WordstatApiError as e:
                    LOGGER.warning(
                        "validate: wordstat query failed for %r: %s: %s",
                        phrase,
                        type(e).__name__,
                        e,
                    )
                    counters["api_errors"] += 1
                    continue
                repo.upsert_wordstat_query(conn, cluster.id, phrase, frequency)
                counters["validated"] += 1
                if frequency >= WORDSTAT_DEMAND_THRESHOLD:
                    cluster_confirmed = True
            if cluster_confirmed:
                counters["demand_confirmed"] += 1
                if _append_wordstat_signal(conn, cluster.id):
                    counters["rationale_updated"] += 1
            repo.update_run_stage(conn, run_id, "validate", "running", stats_delta=dict(counters))
    except Exception:
        status = "error"
        raise
    finally:
        # Deltas were written per batch; the final update only flips the
        # stage status (_stats_merge accumulates). Never overwrite an
        # "error" status with "done" (run_collect/run_score discipline).
        repo.update_run_stage(conn, run_id, "validate", status, stats_delta=dict.fromkeys(counters, 0))
        repo.finish_run(conn, run_id)
    return dict(counters)


def _append_wordstat_signal(conn: Connection, cluster_id: str) -> bool:
    """Append the confirmed-demand signal to the cluster's score rationale.

    Rubric v1 weights stay untouched (owner constraint): the signal is a
    rationale-only annotation. Returns False when the cluster has no score
    row yet (the next run_score will then see only cached wordstat rows —
    the signal lands after the next scoring pass) or the signal is already
    there (idempotent: a revalidated cluster must not stack duplicates).
    """
    signal = (
        "\n\n**Поисковый спрос подтверждён** (Яндекс Wordstat: частотность "
        f"≥ {WORDSTAT_DEMAND_THRESHOLD}/мес по ключевой фразе)."
    )
    with conn.transaction():
        row = conn.execute(
            """
            SELECT s.rationale_md
            FROM score s
            WHERE s.cluster_id = %s::uuid
            FOR UPDATE
            """,
            (cluster_id,),
        ).fetchone()
        if row is None:
            return False
        rationale = str(row[0])
        if "Поисковый спрос подтверждён" in rationale:
            return False
        conn.execute(
            "UPDATE score SET rationale_md = %s WHERE cluster_id = %s::uuid",
            (rationale + signal, cluster_id),
        )
    return True
