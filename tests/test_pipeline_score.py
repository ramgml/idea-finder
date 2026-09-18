"""Score stage tests: rubric prompt, validation, idempotency, sanity gate.

One embedded postgres cluster per module (same pattern as the other
pipeline test modules). The score stage runs against a stub client
injected through the ``build_llm_client`` factory seam — the production
code path is unchanged, only the LLM answer is deterministic. The stub
answers with valid JSON whose quotes are taken verbatim from the cluster
bodies rendered into the prompt, so the grounding validator accepts them.

Clusters are created directly (upsert_cluster + pains via insert_pain),
NOT by running the cluster stage — the score stage must not depend on how
the partition was produced.

Test map (T310 DoD):
a. 3 unscored clusters -> every cluster gets score + rationale + quotes,
   score in [0, 10] storage scale (LLM 0-100 / 10), prompt_version gains
   (score, 1);
b. rerun: scored=0, the score table does not grow (selection skips scored);
c. garbage answer -> rejected counter, no score row for that cluster;
d. narrow score band -> sanity_flag=1 in stats;
e. empty database -> zero stats, no run row.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from psycopg import Connection

from idea_finder.core.canonical import canonical_url
from idea_finder.core.models import Pain, PostKind, RawPost
from idea_finder.core.pipeline import run_score
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import (
    assign_pain_cluster,
    ensure_source,
    insert_pain,
    insert_raw_post,
    table_counts,
    upsert_cluster,
)
from idea_finder.llm.client import Completion


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """One embedded postgres cluster for the whole module."""
    data_dir = tmp_path_factory.mktemp("pgscore") / "pg"
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
            " raw_post, cluster, score CASCADE"
        )


@pytest.fixture()
def prompt_version_id(conn: Connection) -> str:
    """One registered extract prompt version for the pain FK."""
    from idea_finder.db.repo import upsert_prompt_version

    return upsert_prompt_version(conn, "extract_pains", 1, "test body", "file")


def _add_pain(conn: Connection, body: str, kind: PostKind, *, n: int,
              prompt_version_id: str) -> str:
    """Insert one post (kind applies) carrying a single pain; return pain id."""
    source_id = ensure_source(conn, "score-test")
    url = f"https://example.com/score/{n}"
    post_id = insert_raw_post(
        conn,
        RawPost(
            source_id=source_id,
            url_canon=canonical_url(url),
            url=url,
            title=f"post {n}",
            text=body,
            published_at=datetime.now(tz=UTC),
            kind=kind,  # type: ignore[arg-type]
        ),
    )
    assert post_id is not None
    pain_id = insert_pain(
        conn,
        Pain(source_post_id=post_id, body=body, audience="частники",
             quote=body[:20]),
        None,
        prompt_version_id,
    )
    return pain_id


def _seed_cluster(conn: Connection, bodies: list[str], *, n: int,
                  prompt_version_id: str) -> str:
    """Create one cluster with one post per body; return the cluster id."""
    kind_cycle = ("demand", "complaint", "discussion")
    cluster_id = upsert_cluster(conn, f"cluster {n}", len(bodies),
                                {kind_cycle[i % 3]: 1 for i in range(len(bodies))})
    for i, body in enumerate(bodies):
        pain_id = _add_pain(conn, body, kind_cycle[(n + i) % 3],
                            n=n * 10 + i, prompt_version_id=prompt_version_id)
        assign_pain_cluster(conn, pain_id, cluster_id)
    return cluster_id


class StaticScoreClient:
    """Stub LLM client answering a fixed (possibly invalid) JSON body."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def complete(self, prompt: str) -> Completion:
        return Completion(
            text=self._payload, prompt_tokens=1, completion_tokens=1,
            model="stub", provider_name="stub",
        )


def _valid_answer(prompt: str) -> str:
    """Build a valid score answer grounded in the prompt's pain bodies.

    LLM scale is 0-100 (SCORING.md); storage converts to 0-10, so a 60
    lands as 6.0 (checked in the conversion test below).
    """
    bodies = [
        line[2:].strip() for line in prompt.splitlines()
        if line.startswith("- ") and len(line) > 4
    ]
    return json.dumps(
        {
            "score": 60,
            "rationale_md": "Кластер подтверждён постами; спрос устойчивый.",
            "quotes": [bodies[0][:30]] if bodies else ["боль"],
        },
        ensure_ascii=False,
    )


def _use_client(conn: Connection, client: object,
                monkeypatch: pytest.MonkeyPatch) -> None:
    """Point build_llm_client at the stub for this test."""
    monkeypatch.setattr(
        "idea_finder.core.pipeline.build_llm_client", lambda _conn: client
    )


def test_score_all_clusters_get_scores(
    conn: Connection, prompt_version_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 unscored clusters -> scores + rationales + version pin (a)."""
    for i in range(3):
        _seed_cluster(conn, [f"не работает оплата в приложении {i}"],
                      n=i, prompt_version_id=prompt_version_id)
    _use_client(conn, StaticScoreClient("OK"), monkeypatch)
    # The stub must answer per-prompt: wrap to ground quotes in bodies.
    class GroundedClient:
        def complete(self, prompt: str) -> Completion:
            return StaticScoreClient(_valid_answer(prompt)).complete(prompt)

    _use_client(conn, GroundedClient(), monkeypatch)

    stats = run_score(conn)

    assert stats["clusters"] == 3 and stats["scored"] == 3
    assert stats["rejected"] == 0
    counts = table_counts(conn)
    assert counts["score"] == 3
    rows = conn.execute(
        "SELECT total, rationale_md, quotes_json FROM score"
    ).fetchall()
    for total, rationale, quotes in rows:
        assert 0 <= float(total) <= 10  # storage scale: 60 LLM -> 6.0
        assert str(rationale).strip()
        assert quotes  # psycopg decodes jsonb to a non-empty list
    versions = conn.execute(
        "SELECT name, version FROM prompt_version ORDER BY name"
    ).fetchall()
    assert ("score", 1) in [(str(name), int(version)) for name, version in versions]


def test_score_rerun_is_noop(
    conn: Connection, prompt_version_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rerun scores nothing: selection skips already-scored clusters (b)."""
    for i in range(2):
        _seed_cluster(conn, [f"тормозит отчёт в 1С {i}"],
                      n=i, prompt_version_id=prompt_version_id)

    class GroundedClient:
        def complete(self, prompt: str) -> Completion:
            return StaticScoreClient(_valid_answer(prompt)).complete(prompt)

    _use_client(conn, GroundedClient(), monkeypatch)
    run_score(conn)
    counts_first = table_counts(conn)

    stats = run_score(conn)

    assert stats == {"clusters": 0, "scored": 0, "rejected": 0,
                     "sanity_flag": 0}
    assert table_counts(conn)["score"] == counts_first["score"]


def test_score_rejects_garbage_answers(
    conn: Connection, prompt_version_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unparseable answer -> rejected, cluster stays unscored (c)."""
    _seed_cluster(conn, ["биллинг списывает дважды"],
                  n=0, prompt_version_id=prompt_version_id)
    _use_client(conn, StaticScoreClient("не JSON вообще"), monkeypatch)

    stats = run_score(conn)

    assert stats["clusters"] == 1 and stats["rejected"] == 1
    assert stats["scored"] == 0
    counts = table_counts(conn)
    assert counts["score"] == 0


def test_score_sanity_flag_on_narrow_band(
    conn: Connection, prompt_version_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3+ scores within <10 points -> sanity_flag=1 in stats (d)."""
    for i in range(3):
        _seed_cluster(conn, [f"медленная доставка заказов {i}"],
                      n=i, prompt_version_id=prompt_version_id)

    class NarrowClient:
        def complete(self, prompt: str) -> Completion:
            answer = json.loads(_valid_answer(prompt))
            answer["score"] = 60  # constant LLM score: spread 0 < 10
            return StaticScoreClient(json.dumps(answer, ensure_ascii=False)) \
                .complete(prompt)

    _use_client(conn, NarrowClient(), monkeypatch)

    stats = run_score(conn)

    assert stats["scored"] == 3
    assert stats["sanity_flag"] == 1


def test_score_on_empty_database_is_noop(conn: Connection) -> None:
    """Empty database -> zero stats and no run row (e)."""
    stats = run_score(conn)

    assert stats == {"clusters": 0, "scored": 0, "rejected": 0,
                     "sanity_flag": 0}
    counts = table_counts(conn)
    assert counts["run"] == 0 and counts["score"] == 0


def test_llm_score_to_storage_boundaries() -> None:
    """0->0, 100->10, 55->5.5: the two documented scales (T310 decision)."""
    from idea_finder.llm.score import llm_score_to_storage

    assert llm_score_to_storage(0) == 0.0
    assert llm_score_to_storage(100) == 10.0
    assert llm_score_to_storage(55) == 5.5
