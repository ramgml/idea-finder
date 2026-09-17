"""Extract stage tests: run ownership, idempotency, brick-once, LLM errors.

One embedded postgres cluster per module. The stage runs against the
``FakeExtractLlmClient`` answer format via a stub client injected through
monkeypatching ``build_llm_client`` (the factory seam): the stage code path
is identical to production, only the LLM answer is deterministic.
Happy-path texts are real fixture posts: the fake answers with pains whose
quotes occur in the text, so synthetic strings would extract nothing.

Test map (contract a-f):
a. happy path: pains land bound to the run's pinned prompt_version;
b. second run processes nothing (idempotency through terminal fetch_status);
c. unparseable answer -> post marked 'failed', third run does NOT retry it;
d. post with zero accepted pains -> terminal 'extracted', not retried;
e. migrations apply cleanly including 005 (run.prompt_version_id + status);
f. LlmError -> llm_errors counter, post stays pending, next run retries.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from importlib import resources

import pytest
from psycopg import Connection

from idea_finder.core.canonical import canonical_url
from idea_finder.core.models import RawPost
from idea_finder.core.pipeline import run_extract
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import apply_migrations
from idea_finder.db.repo import (
    ensure_source,
    get_run,
    insert_raw_post,
    list_posts_pending_extract,
    table_counts,
)
from idea_finder.llm.client import (
    Completion,
    LlmError,
)
from idea_finder.llm.extract import FakeExtractLlmClient


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """One embedded postgres cluster for the whole module."""
    data_dir = tmp_path_factory.mktemp("pgextract") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture()
def conn(pg: PgHandle) -> Iterator[Connection]:
    """Fresh migrated database for every test (isolation between runs)."""
    with pg.get_conn() as connection:
        apply_migrations(connection)
        yield connection
        connection.execute("TRUNCATE source, prompt_version, run, llm_provider CASCADE")


def _add_post(conn: Connection, text: str, *, n: int = 0) -> str:
    """Insert one pending demand post; return its id."""
    source_id = ensure_source(conn, "extract-test")
    url = f"https://example.com/extract/{n}/{hash(text) & 0xffff}"
    post_id = insert_raw_post(
        conn,
        RawPost(
            source_id=source_id,
            url_canon=canonical_url(url),
            url=url,
            title=f"post {n}",
            text=text,
            published_at=datetime.now(tz=UTC),
            kind="demand",
        ),
    )
    assert post_id is not None
    return post_id


def _use_client(conn: Connection, client: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point build_llm_client at the given stub for this test."""
    monkeypatch.setattr(
        "idea_finder.core.pipeline.build_llm_client", lambda _conn: client
    )


def test_extract_happy_path_pins_prompt_version(
    conn: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepted pains land in pain, and run pins the prompt version (a)."""
    fixture_text = json.loads(
        resources.files("idea_finder")
        .joinpath("fixtures/posts/fl_ru.json")
        .read_text(encoding="utf-8")
    )[0]["text"]
    _add_post(conn, fixture_text)
    _use_client(conn, FakeExtractLlmClient(), monkeypatch)

    stats = run_extract(conn)

    assert stats["processed"] == 1
    counts = table_counts(conn)
    assert counts["pain"] >= 1
    # Exactly one run, and it records which prompt version produced the pains.
    run_rows = conn.execute("SELECT id FROM run").fetchall()
    assert len(run_rows) == 1
    run = get_run(conn, str(run_rows[0][0]))
    assert run is not None and run.prompt_version_id is not None
    pv = conn.execute(
        "SELECT name, version FROM prompt_version WHERE id = %s",
        (run.prompt_version_id,),
    ).fetchone()
    assert pv is not None and pv[0] == "extract_pains" and pv[1] == 1


def test_extract_rerun_is_noop(conn: Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    """Second run processes zero posts: fetched posts are not re-sent (b)."""
    text = "Курьерская служба потеряла посылку и поддержка игнорирует мои претензии"
    _add_post(conn, text)
    _use_client(conn, FakeExtractLlmClient(), monkeypatch)

    first = run_extract(conn)
    assert first["processed"] == 1

    second = run_extract(conn)
    assert second["processed"] == 0
    assert second["extracted"] == 0


def test_extract_unparseable_answer_bricks_post(
    conn: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hopeless answer -> failed status; later runs never retry it (c)."""

    class GarbageClient:
        def complete(self, prompt: str) -> Completion:
            return Completion(
                text="это вообще не json", prompt_tokens=1, completion_tokens=1,
                model="stub", provider_name="stub",
            )

    _add_post(conn, "Таксисты отменяют поездки в аэропорт в час пик")
    _use_client(conn, GarbageClient(), monkeypatch)

    first = run_extract(conn)
    assert first["failed"] == 1
    status = conn.execute(
        "SELECT fetch_status FROM raw_post"
    ).fetchone()
    assert status is not None and status[0] == "failed"

    # The brick sticks: a run after the failure must not re-send the post.
    third = run_extract(conn)
    assert third["processed"] == 0
    assert third["failed"] == 0


def test_extract_zero_pains_marks_extracted(
    conn: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A processed post with zero accepted pains is terminal, not retried (d)."""

    class EmptyClient:
        def complete(self, prompt: str) -> Completion:
            return Completion(
                text=json.dumps({"pains": []}), prompt_tokens=1,
                completion_tokens=1, model="stub", provider_name="stub",
            )

    _add_post(conn, "Всё в этом городе работает идеально, жалоб нет")
    _use_client(conn, EmptyClient(), monkeypatch)

    stats = run_extract(conn)
    assert stats["processed"] == 1 and stats["extracted"] == 0
    assert stats["failed"] == 0
    status = conn.execute("SELECT fetch_status FROM raw_post").fetchone()
    assert status is not None and status[0] == "extracted"
    # The terminal pin means the next run does NOT re-send the post.
    assert len(list_posts_pending_extract(conn)) == 0


def test_migrations_include_run_prompt_version(pg: PgHandle) -> None:
    """Migrations up to 005 are applied on the module cluster (e)."""
    with pg.get_conn() as conn:
        applied = apply_migrations(conn)  # [] when earlier tests migrated already
        assert applied == [] or applied == [1, 2, 3, 4, 5]
        versions = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [int(v[0]) for v in versions] == [1, 2, 3, 4, 5]
        row = conn.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'run' AND column_name = 'prompt_version_id'
            """
        ).fetchone()
        assert row is not None


def test_extract_llm_error_leaves_post_pending(
    conn: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LlmError counts llm_errors, keeps the post pending for retry (f)."""

    class FailingClient:
        def complete(self, prompt: str) -> Completion:
            raise LlmError("provider down")

    _add_post(conn, "Мобильный банк падает при попытке оплатить ЖКХ")
    _use_client(conn, FailingClient(), monkeypatch)

    stats = run_extract(conn)
    assert stats["llm_errors"] == 1 and stats["processed"] == 1
    pending = list_posts_pending_extract(conn)
    assert len(pending) == 1  # still pending

    # Recovery: a working client on the next run processes the same post.
    _use_client(conn, FakeExtractLlmClient(), monkeypatch)
    retry = run_extract(conn)
    assert retry["processed"] == 1 and retry["extracted"] >= 1
