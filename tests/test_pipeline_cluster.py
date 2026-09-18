"""Cluster stage tests: greedy centroid grouping, threshold, idempotency.

One embedded postgres cluster per module (same pattern as
``test_pipeline_extract.py``). The real e5 model is never loaded: the
module monkeypatches ``idea_finder.core.embed._EMBEDDER.embed_texts`` with
a deterministic body-to-vector map, so the stage's production code path
embed backfill included - runs on synthetic unit vectors.

Test map (T309 DoD):
a. 21 pains in 3 groups with mixed kinds -> correct partition, sizes,
   kind mixes and consistent stats (the manual 20+ pains acceptance);
b. second run reproduces the exact same partition, no orphan clusters;
c. threshold: cosine ~0.9 merges, ~0.5 does not;
d. empty database -> zero stats, no run row;
e. a run row is written with stages_json.cluster == 'done'.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from psycopg import Connection

from idea_finder.core.canonical import canonical_url
from idea_finder.core.models import Pain, PostKind, RawPost
from idea_finder.core.pipeline import CLUSTER_THRESHOLD, run_cluster
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import (
    ensure_source,
    insert_pain,
    insert_raw_post,
    table_counts,
)


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """One embedded postgres cluster for the whole module."""
    data_dir = tmp_path_factory.mktemp("pgcluster") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture()
def conn(pg: PgHandle) -> Iterator[Connection]:
    """Fresh migrated database for every test."""
    with pg.get_conn() as connection:
        apply_migrations(connection)
        yield connection
        connection.execute(
            "TRUNCATE source, prompt_version, run, llm_provider, pain,"
            " raw_post, cluster CASCADE"
        )


# --- deterministic test vectors -------------------------------------------
# pain.embedding is vector(384) (e5-small dim, migration 001), so synthetic
# vectors must match that dimensionality. A group's basis vector is 1.0 in
# one of the first 3 coordinates; the noise tilt spreads members of the same
# group slightly (cosine stays ~0.999) while distinct groups stay orthogonal.

DIM = 384


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


def _group_vector(group: int, noise: int) -> list[float]:
    """Unit vector near basis direction ``group`` with a tiny noise tilt."""
    base = [0.0] * DIM
    base[group] = 1.0
    base[(group + 1) % DIM] += 0.04 * (noise + 1)
    return _unit(base)


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


@pytest.fixture()
def fake_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[float]]:
    """Map pain bodies to deterministic vectors; patch the embed stage seam."""
    import idea_finder.core.embed as embed_module

    vectors: dict[str, list[float]] = {}

    def _embed_texts(texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            if text not in vectors:
                vectors[text] = _unit(
                    _group_vector(hash(text) % 3, len(vectors))
                )
            result.append(vectors[text])
        return result

    monkeypatch.setattr(embed_module._EMBEDDER, "embed_texts", _embed_texts)
    return vectors


# --- fixtures helpers ------------------------------------------------------

_source_counter = {"n": 0}


def _add_post_with_pains(
    conn: Connection,
    bodies: list[str],
    kind: PostKind,
    *,
    n: int,
    prompt_version_id: str,
) -> list[str]:
    """Insert one post with a body per pain; return the pain ids."""
    source_id = ensure_source(conn, "cluster-test")
    url = f"https://example.com/cluster/{n}"
    post_id = insert_raw_post(
        conn,
        RawPost(
            source_id=source_id,
            url_canon=canonical_url(url),
            url=url,
            title=f"post {n}",
            text=" ".join(bodies),
            published_at=datetime.now(tz=UTC),
            kind=kind,
        ),
    )
    assert post_id is not None
    pain_ids: list[str] = []
    for body in bodies:
        pain_ids.append(
            insert_pain(
                conn,
                Pain(source_post_id=post_id, body=body, audience="частники",
                     quote=body[:20]),
                None,
                prompt_version_id,
            )
        )
    return pain_ids


KINDS = ("demand", "complaint", "discussion")


@pytest.fixture()
def prompt_version_id(conn: Connection) -> str:
    """One registered extract prompt version for pain FK in this test."""
    from idea_finder.db.repo import upsert_prompt_version

    return upsert_prompt_version(conn, "extract_pains", 1, "test body", "file")


def _seed_groups(
    conn: Connection,
    per_group: int = 7,
    prompt_version_id: str = "",
    vectors: dict[str, list[float]] | None = None,
) -> list[str]:
    """Seed 3 groups x per_group pains with rotating kinds; return ids.

    When ``vectors`` (the fake_embedder map) is given, each body is pinned
    to its group's basis direction up front, so the partition is known
    before running the stage (hash-based assignment would be unbalanced).
    """
    ids: list[str] = []
    n = 0
    for group in range(3):
        for i in range(per_group):
            body = f"боль номер {len(ids)} группы {group}"
            if vectors is not None:
                vectors[body] = _group_vector(group, i)
            ids.extend(
                _add_post_with_pains(conn, [body], KINDS[len(ids) % 3],
                                     n=n, prompt_version_id=prompt_version_id)
            )
            n += 1
    return ids


# --- tests -----------------------------------------------------------------


def test_cluster_groups_pains_and_reports_stats(
    conn: Connection,
    fake_embedder: dict[str, list[float]],
    prompt_version_id: str,
) -> None:
    """21 pains in 3 groups -> 3 clusters, sizes/mixes/stats consistent (a)."""
    pain_ids = _seed_groups(conn, per_group=7, prompt_version_id=prompt_version_id,
                            vectors=fake_embedder)
    assert len(pain_ids) == 21

    stats = run_cluster(conn)

    assert stats["pains"] == 21
    assert stats["clusters_new"] == 3
    assert stats["clusters_merged"] == 18
    assert stats["singletons"] == 0
    counts = table_counts(conn)
    assert counts["cluster"] == 3
    rows = conn.execute(
        "SELECT c.id, c.size, c.kind_mix FROM cluster c ORDER BY c.id"
    ).fetchall()
    sizes = sorted(int(row[1]) for row in rows)
    assert sizes == [7, 7, 7]
    total_assigned = conn.execute(
        "SELECT count(*) FROM pain WHERE cluster_id IS NOT NULL"
    ).fetchone()
    assert total_assigned is not None and total_assigned[0] == 21
    for _, _, kind_mix in rows:
        raw_mix: object = kind_mix
        mix = raw_mix if isinstance(raw_mix, dict) else json.loads(str(raw_mix))
        assert set(mix) <= set(KINDS)
        assert sum(mix.values()) == 7


def test_cluster_rerun_is_stable(
    conn: Connection,
    fake_embedder: dict[str, list[float]],
    prompt_version_id: str,
) -> None:
    """A rerun reproduces the same partition with no orphan clusters (b)."""
    _seed_groups(conn, per_group=7, prompt_version_id=prompt_version_id,
                 vectors=fake_embedder)
    first = run_cluster(conn)

    partition_first = _partition(conn)
    second = run_cluster(conn)

    # "embedded" legitimately differs on the rerun: the backfill is a no-op
    # once every pain has a vector. The partition must be identical.
    first_partition_stats = {k: v for k, v in first.items() if k != "embedded"}
    second_partition_stats = {k: v for k, v in second.items() if k != "embedded"}
    assert second_partition_stats == first_partition_stats
    assert second["embedded"] == 0
    assert _partition(conn) == partition_first
    counts = table_counts(conn)
    assert counts["cluster"] == first["clusters_new"]


def _partition(conn: Connection) -> set[frozenset[str]]:
    """Grouping of pain ids by cluster, as a set of frozensets."""
    rows = conn.execute(
        "SELECT pain.id::text, pain.cluster_id::text FROM pain"
    ).fetchall()
    groups: dict[str, set[str]] = {}
    for pain_id, cluster_id in rows:
        assert cluster_id is not None
        groups.setdefault(cluster_id, set()).add(pain_id)
    return {frozenset(members) for members in groups.values()}


def test_cluster_threshold_decides_merge(
    conn: Connection,
    fake_embedder: dict[str, list[float]],
    prompt_version_id: str,
) -> None:
    """Cosine ~0.9 merges into one cluster, ~0.5 stays apart (c)."""
    close_pair: list[list[float]] = [
        _unit([1.0 if i == 0 else 0.05 if i == 1 else 0.0 for i in range(DIM)]),
        _unit([1.0 if i == 0 else 0.08 if i == 1 else 0.0 for i in range(DIM)]),
    ]
    far_pair: list[list[float]] = [
        _unit([1.0 if i == 2 else 0.0 for i in range(DIM)]),
        _unit([1.0 if i == DIM - 1 else 0.0 for i in range(DIM)]),
    ]
    assert _cosine(*[_unit(v) for v in close_pair]) >= CLUSTER_THRESHOLD
    assert _cosine(*[_unit(v) for v in far_pair]) < CLUSTER_THRESHOLD
    for i, vector in enumerate([*close_pair, *far_pair]):
        body = f"explicit vector pain {i}"
        fake_embedder[body] = _unit(vector)
        _add_post_with_pains(conn, [body], "demand", n=100 + i,
                             prompt_version_id=prompt_version_id)

    stats = run_cluster(conn)

    assert stats["pains"] == 4
    assert stats["clusters_new"] == 3
    assert stats["singletons"] == 2
    counts = table_counts(conn)
    assert counts["cluster"] == 3


def test_cluster_on_empty_database_is_noop(conn: Connection) -> None:
    """Empty database -> zero stats and no run row (d)."""
    stats = run_cluster(conn)

    assert stats == {"pains": 0, "clusters_new": 0, "clusters_merged": 0,
                     "singletons": 0, "embedded": 0}
    counts = table_counts(conn)
    assert counts["run"] == 0
    assert counts["cluster"] == 0


def test_cluster_writes_done_run_row(
    conn: Connection,
    fake_embedder: dict[str, list[float]],
    prompt_version_id: str,
) -> None:
    """A successful pass records a run with stages_json.cluster done (e)."""
    _seed_groups(conn, per_group=2, prompt_version_id=prompt_version_id,
                 vectors=fake_embedder)

    run_cluster(conn)

    runs = conn.execute(
        "SELECT stages_json, stats_json FROM run"
    ).fetchall()
    assert len(runs) == 1
    stages = runs[0][0] if isinstance(runs[0][0], dict) else json.loads(str(runs[0][0]))
    assert stages["cluster"] == "done"
    run_stats: dict[str, Any] = (
        runs[0][1] if isinstance(runs[0][1], dict) else json.loads(str(runs[0][1]))
    )
    assert run_stats["pains"] == 6
