"""Embedding stage tests: e5-small vectors, idempotent backfill, repo write.

A single module-scoped embedded postgres cluster serves the whole module;
pains are synthetic strings (no fixture dataset), inserted with
``embedding=None`` so the embed stage has work to do. The model test uses
the real ``multilingual-e5-small`` (first run downloads ~120 MB into the
local HF cache) — that is the DoD acceptance, so it stays in the normal
test run.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from psycopg import Connection

from idea_finder.core.canonical import canonical_url
from idea_finder.core.embed import (
    DEFAULT_BATCH_SIZE,
    MODEL_ID,
    Embedder,
    EmbedError,
    embed_pains,
    embed_texts,
)
from idea_finder.core.models import Pain, RawPost
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import (
    ensure_source,
    insert_pain,
    insert_raw_post,
    upsert_prompt_version,
)

_N_PAINS = 10

# Bodies of the synthetic pains; distinct enough that e5 vectors differ.
_BODIES = [
    "No affordable plumber available on weekends in my district",
    "Courier service lost my parcel and support ignores my claims",
    "Mobile banking app crashes when I try to pay utilities",
    "Cannot find a cheap coworking space with stable internet",
    "Food delivery always arrives cold and late in the evening",
    "Taxi drivers cancel rides to the airport at rush hour",
    "Online grocery store has no delivery slot for same day",
    "Language tutors are too expensive for group lessons",
    "Fitness club membership cannot be frozen during vacation",
    "Home internet provider raises prices without notice",
]


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """One embedded postgres cluster for the whole module."""
    data_dir = tmp_path_factory.mktemp("pgembed") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture(scope="module")
def conn(pg: PgHandle) -> Iterator[Connection]:
    """Connection to the migrated application database with 10 pains."""
    with pg.get_conn() as connection:
        apply_migrations(connection)
        source_id = ensure_source(connection, "test-embed")
        pv_id = upsert_prompt_version(connection, "extract_pains", 1, "PROMPT", "file")
        for i, body in enumerate(_BODIES):
            url = f"https://example.com/embed-test/{i}"
            post_id = insert_raw_post(
                connection,
                RawPost(
                    source_id=source_id,
                    url_canon=canonical_url(url),
                    url=url,
                    title=f"post {i}",
                    text=body,
                    published_at=datetime.now(tz=UTC),
                    kind="demand",
                ),
            )
            assert post_id is not None
            insert_pain(
                connection,
                Pain(
                    source_post_id=post_id,
                    body=body,
                    audience="users",
                    quote="",
                ),
                None,
                pv_id,
            )
        yield connection


def test_embed_texts_returns_normalized_384_dim_vectors() -> None:
    """Real model: 384-dim vectors, L2 norm ~1 (normalize_embeddings=True)."""
    vectors = embed_texts(["водопроводчик не работает по выходным"] * 2)
    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) == 384
        norm = math.sqrt(sum(value * value for value in vector))
        assert norm == pytest.approx(1.0, abs=1e-4)


def test_embed_texts_rejects_empty_input() -> None:
    """Empty input list is an EmbedError, not a silent empty result."""
    with pytest.raises(EmbedError, match="non-empty"):
        embed_texts([])


def test_embedder_query_prefix_differs_from_passage() -> None:
    """e5 prefixes matter: same text under query/passage gives other vector."""
    embedder = Embedder()
    passage = embedder.embed_texts(["доставка еды приезжает холодной"])[0]
    query = embedder.embed_query("доставка еды")
    assert passage != query


def test_embed_pains_backfills_all_ten(conn: Connection) -> None:
    """First run embeds all 10 pains; vectors are 384-dim unit vectors."""
    stats = embed_pains(conn, batch_size=3)
    assert stats == {"embedded": 10, "skipped": 0}
    rows = conn.execute(
        """
        SELECT count(*), count(embedding), min(vector_dims(embedding)),
               max(vector_dims(embedding))
        FROM pain
        """
    ).fetchone()
    assert rows is not None
    total, embedded, min_dim, max_dim = (int(value) for value in rows)
    assert (total, embedded, min_dim, max_dim) == (10, 10, 384, 384)
    norms = conn.execute(
        """
        SELECT DISTINCT round(sqrt(vector_norm(embedding))::numeric, 4)
        FROM pain
        """
    ).fetchall()
    assert all(float(norm[0]) == pytest.approx(1.0, abs=1e-3) for norm in norms)


def test_embed_pains_rerun_is_noop(conn: Connection) -> None:
    """Rerun embeds nothing: everything is already embedded (DoD)."""
    stats = embed_pains(conn)
    assert stats == {"embedded": 0, "skipped": _N_PAINS}


def test_embed_pains_embeds_only_new_pain(conn: Connection) -> None:
    """A pain inserted later is the only one embedded on the next run."""
    source_id = ensure_source(conn, "test-embed")
    pv_id = upsert_prompt_version(conn, "extract_pains", 1, "PROMPT", "file")
    url = "https://example.com/embed-test/new"
    post_id = insert_raw_post(
        conn,
        RawPost(
            source_id=source_id,
            url_canon=canonical_url(url),
            url=url,
            title="late post",
            text="landlord raised rent again",
            published_at=datetime.now(tz=UTC),
            kind="complaint",
        ),
    )
    assert post_id is not None
    insert_pain(
        conn,
        Pain(
            source_post_id=post_id,
            body="Landlord raised the rent again without warning",
            audience="renters",
            quote="",
        ),
        None,  # embedding: filled by embed_pains
        pv_id,
    )
    stats = embed_pains(conn, batch_size=DEFAULT_BATCH_SIZE)
    assert stats == {"embedded": 1, "skipped": _N_PAINS}
    row = conn.execute(
        "SELECT embedding IS NOT NULL FROM pain WHERE body = %s",
        ("Landlord raised the rent again without warning",),
    ).fetchone()
    assert row is not None and row[0] is True


def test_model_id_points_at_canonical_e5_repo() -> None:
    """The module pins the canonical e5 repo (guard against silent swap)."""
    assert MODEL_ID == "intfloat/multilingual-e5-small"
