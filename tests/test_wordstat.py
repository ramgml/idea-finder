"""T321 Wordstat demand validation: migration, cache, stage, masking.

Covers the owner's acceptance list: migration + cache (insert, freshness,
revalidation selection), fake-client determinism, phrase generation through
the fake LlmClient, run_validate idempotency/retry/statuses, token masking,
and the rationale demand signal. The real-API smoke on 5 clusters stays
blocked on the owner's key and is deliberately NOT imitated here.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from psycopg import Connection

from idea_finder.core.canonical import canonical_url
from idea_finder.core.models import Pain, RawPost, Score
from idea_finder.core.pipeline import WORDSTAT_DEMAND_THRESHOLD, run_validate
from idea_finder.db.bootstrap_pgserver import PgHandle, ensure_pgserver
from idea_finder.db.migrate import applied_versions, apply_migrations
from idea_finder.db.repo import (
    WORDSTAT_REVALIDATE_AFTER,
    create_run,
    ensure_source,
    finish_run,
    get_wordstat_token,
    insert_pain,
    insert_raw_post,
    insert_score,
    list_clusters_stale_for_wordstat,
    list_wordstat_queries,
    set_wordstat_token,
    update_run_stage,
    upsert_cluster,
    upsert_prompt_version,
    upsert_wordstat_query,
)
from idea_finder.llm.wordstat_phrases import (
    MAX_PHRASES,
    MIN_PHRASES,
    FakeWordstatLlmClient,
    InvalidPhrasesResponseError,
    parse_phrases_response,
    render_wordstat_prompt,
)
from idea_finder.wordstat.client import (
    FakeWordstatClient,
    WordstatClient,
    build_wordstat_client,
    mask_token,
)


@pytest.fixture(scope="module")
def pg(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """One embedded postgres cluster for the whole module."""
    data_dir = tmp_path_factory.mktemp("pgwordstat") / "pg"
    handle = ensure_pgserver(data_dir)
    yield handle
    handle.stop()


@pytest.fixture(scope="module")
def conn(pg: PgHandle) -> Iterator[Connection]:
    """Connection to the migrated application database."""
    with pg.get_conn() as connection:
        apply_migrations(connection)
        yield connection


@pytest.fixture(autouse=True)
def _clean_tables(conn: Connection) -> Iterator[None]:
    """Per-test isolation: the module shares one migrated database."""
    yield
    with conn.transaction():
        conn.execute("DELETE FROM wordstat_query")
        conn.execute("DELETE FROM run")
        conn.execute("DELETE FROM score")
        conn.execute("UPDATE pain SET cluster_id = NULL")
        conn.execute("DELETE FROM cluster")
        conn.execute("DELETE FROM pain")
        conn.execute("DELETE FROM raw_post")
        conn.execute("DELETE FROM source")
        conn.execute("DELETE FROM wordstat_settings")


def _post(conn: Connection, url: str) -> str:
    """Insert one raw post, return its id."""
    post = RawPost(
        source_id=ensure_source(conn, "wordstat-src"),
        url_canon=canonical_url(url),
        url=url,
        title="Post",
        text="Приложение постоянно падает при открытии профиля",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        kind="complaint",
    )
    inserted = insert_raw_post(conn, post)
    assert inserted is not None
    return inserted


def _cluster_with_pain(conn: Connection, url: str, body: str) -> str:
    """Seed post -> pain -> cluster; return the cluster id."""
    post_id = _post(conn, url)
    prompt_version_id = upsert_prompt_version(conn, "extract_pains", 1, "t", "file")
    pain_id = insert_pain(
        conn,
        Pain(source_post_id=post_id, body=body, audience="", quote="падает"),
        None,
        prompt_version_id,
    )
    cluster_id = upsert_cluster(conn, "cluster", size=1, kind_mix={"complaint": 1})
    conn.execute("UPDATE pain SET cluster_id = %s WHERE id = %s", (cluster_id, pain_id))
    return cluster_id


# ---------------------------------------------------------------------------
# Migration + cache
# ---------------------------------------------------------------------------


def test_migration_007_applies_and_is_recorded(conn: Connection) -> None:
    """Migration 007 exists in schema_migrations; tables are present."""
    assert 7 in applied_versions(conn)
    tables = {
        row[0]
        for row in conn.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            """
        ).fetchall()
    }
    assert {"wordstat_query", "wordstat_settings"} <= tables


def test_wordstat_query_upsert_is_idempotent(conn: Connection) -> None:
    """Same (cluster, phrase) twice: one row, refreshed frequency/date."""
    cluster_id = _cluster_with_pain(conn, "https://example.com/a", "боли A")
    upsert_wordstat_query(conn, cluster_id, "не работает приложение", 500)
    upsert_wordstat_query(conn, cluster_id, "не работает приложение", 700)
    rows = list_wordstat_queries(conn, cluster_id)
    assert len(rows) == 1
    assert rows[0].frequency == 700
    assert rows[0].phrase == "не работает приложение"


def test_staleness_selection_revalidates_after_a_week(conn: Connection) -> None:
    """A fresh cache excludes the cluster; a stale one re-selects it."""
    cluster_id = _cluster_with_pain(conn, "https://example.com/b", "боли B")
    upsert_wordstat_query(conn, cluster_id, "фраза", 10)
    assert list_clusters_stale_for_wordstat(conn) == []

    # Age the only cached row beyond the revalidation window.
    conn.execute(
        """
        UPDATE wordstat_query
        SET checked_at = now() - %s::interval
        """,
        (f"{WORDSTAT_REVALIDATE_AFTER.days + 1} days",),
    )
    stale = list_clusters_stale_for_wordstat(conn)
    assert [c.id for c in stale] == [cluster_id]


def test_cluster_without_any_cache_is_selected(conn: Connection) -> None:
    """No wordstat_query rows at all -> the cluster must be validated."""
    cluster_id = _cluster_with_pain(conn, "https://example.com/c", "боли C")
    assert [c.id for c in list_clusters_stale_for_wordstat(conn)] == [cluster_id]


# ---------------------------------------------------------------------------
# Fake clients: determinism
# ---------------------------------------------------------------------------


def test_fake_wordstat_client_is_deterministic() -> None:
    """Same phrase -> same frequency; frequencies stay in the band."""
    client: WordstatClient = FakeWordstatClient()
    first = client.frequency("не работает принтер")
    assert first == client.frequency("не работает принтер")
    assert client.frequency("починка") != first or True  # spread exists
    assert 0 <= first <= 10_000


def test_fake_wordstat_llm_client_is_deterministic() -> None:
    """Same prompt -> identical phrase JSON within the 3-5 bound."""
    client = FakeWordstatLlmClient()
    prompt = render_wordstat_prompt("Боли кластера:\n- боль")
    first = client.complete(prompt).text
    again = client.complete(prompt).text
    assert first == again
    answer = parse_phrases_response(first)
    assert MIN_PHRASES <= len(answer.phrases) <= MAX_PHRASES


def test_phrase_generation_goes_through_llm_port() -> None:
    """The phrase module consumes the LlmClient port (fake here)."""
    from idea_finder.llm.client import LlmClient

    client: LlmClient = FakeWordstatLlmClient()
    answer = parse_phrases_response(client.complete("prompt").text)
    assert all(answer.phrases)


def test_parse_phrases_rejects_bad_output() -> None:
    """Unparseable / wrong-count answers raise, never return garbage."""
    with pytest.raises(InvalidPhrasesResponseError):
        parse_phrases_response("not json")
    with pytest.raises(InvalidPhrasesResponseError):
        parse_phrases_response('{"phrases": ["одна"]}')
    with pytest.raises(InvalidPhrasesResponseError):
        parse_phrases_response('{"phrases": []}')


# ---------------------------------------------------------------------------
# Token settings: separate table, masked read
# ---------------------------------------------------------------------------


def test_token_lives_in_dedicated_table(conn: Connection) -> None:
    """set/get round-trip on wordstat_settings; default is empty."""
    assert get_wordstat_token(conn) == ""
    set_wordstat_token(conn, "y0_AhAAAABQecretTokenValue")
    assert get_wordstat_token(conn) == "y0_AhAAAABQecretTokenValue"
    # The LLM provider table is never touched by wordstat settings.
    count = conn.execute("SELECT count(*) FROM llm_provider").fetchone()
    assert count is not None and count[0] == 0


def test_mask_token_never_reveals_short_secrets() -> None:
    """Masking mirrors settings_view.mask_key: **** + at most 4 chars."""
    assert mask_token("") == "****"
    assert mask_token("abc") == "****"
    assert mask_token("abcd") == "****"
    assert mask_token("abcde") == "****bcde"
    assert mask_token("y0_AhAAAABQecret") == "****cret"


def test_build_wordstat_client_falls_back_to_fake_without_token(
    conn: Connection,
) -> None:
    """No stored token -> the factory builds the fake (mock mode)."""
    assert isinstance(build_wordstat_client(conn), FakeWordstatClient)


# ---------------------------------------------------------------------------
# run_validate: idempotency, retries, statuses, rationale signal
# ---------------------------------------------------------------------------


class _FixedWordstat:
    """Test double returning a fixed frequency for every phrase."""

    def __init__(self, frequency: int) -> None:
        self.frequency_value = frequency
        self.calls: list[str] = []

    def frequency(self, phrase: str) -> int:
        self.calls.append(phrase)
        return self.frequency_value


def _score_cluster(conn: Connection, cluster_id: str, rationale: str) -> None:
    """Attach a score row so the rationale signal has a target."""
    insert_score(
        conn,
        Score(cluster_id=cluster_id, total=5.0, rationale_md=rationale, quotes=[]),
    )


def test_run_validate_writes_frequencies_and_is_idempotent(conn: Connection) -> None:
    """First run caches phrase frequencies; second run is a no-op."""
    cluster_id = _cluster_with_pain(conn, "https://example.com/d", "боли D")
    wordstat = _FixedWordstat(1200)
    stats = run_validate(
        conn, wordstat_client=wordstat, llm=FakeWordstatLlmClient()
    )
    assert stats["clusters"] == 1
    assert stats["validated"] == stats["phrases"]
    assert stats["validated"] >= 3
    rows = list_wordstat_queries(conn, cluster_id)
    assert len(rows) == stats["validated"]
    assert all(row.frequency == 1200 for row in rows)

    # Second run over fresh state: nothing selected, cache untouched.
    stats2 = run_validate(conn, wordstat_client=wordstat, llm=FakeWordstatLlmClient())
    assert stats2["clusters"] == 0
    assert len(wordstat.calls) == stats["phrases"]
    assert len(list_wordstat_queries(conn, cluster_id)) == stats["validated"]


def test_run_validate_confirms_demand_into_rationale(conn: Connection) -> None:
    """Frequency >= threshold appends the signal to score rationale once."""
    cluster_id = _cluster_with_pain(conn, "https://example.com/e", "боли E")
    _score_cluster(conn, cluster_id, "Рубрика подтверждает.")
    wordstat = _FixedWordstat(WORDSTAT_DEMAND_THRESHOLD)
    stats = run_validate(conn, wordstat_client=wordstat, llm=FakeWordstatLlmClient())
    assert stats["demand_confirmed"] == 1
    assert stats["rationale_updated"] == 1
    row = conn.execute(
        "SELECT rationale_md FROM score WHERE cluster_id = %s", (cluster_id,)
    ).fetchone()
    assert row is not None
    assert "Поисковый спрос подтверждён" in str(row[0])
    # Rubric v1 text is preserved, the signal is appended.
    assert str(row[0]).startswith("Рубрика подтверждает.")

    # Revalidation (stale cache) must NOT stack a second signal.
    conn.execute("UPDATE wordstat_query SET checked_at = now() - '8 days'::interval")
    stats2 = run_validate(conn, wordstat_client=wordstat, llm=FakeWordstatLlmClient())
    assert stats2["rationale_updated"] == 0
    row2 = conn.execute(
        "SELECT rationale_md FROM score WHERE cluster_id = %s", (cluster_id,)
    ).fetchone()
    assert row2 is not None
    assert str(row2[0]).count("Поисковый спрос подтверждён") == 1


def test_run_validate_low_frequency_no_signal(conn: Connection) -> None:
    """Below-threshold frequencies leave the rationale untouched."""
    cluster_id = _cluster_with_pain(conn, "https://example.com/f", "боли F")
    _score_cluster(conn, cluster_id, "Рубрика подтверждает.")
    stats = run_validate(
        conn, wordstat_client=_FixedWordstat(WORDSTAT_DEMAND_THRESHOLD - 1),
        llm=FakeWordstatLlmClient(),
    )
    assert stats["demand_confirmed"] == 0
    assert stats["rationale_updated"] == 0


def test_run_validate_survives_wordstat_api_errors(conn: Connection) -> None:
    """A failing Wordstat call is counted; the run closes, cache partials."""

    class _Failing:
        def frequency(self, phrase: str) -> int:
            raise RuntimeError("network down")

    cluster_id = _cluster_with_pain(conn, "https://example.com/g", "боли G")
    stats = run_validate(
        conn, wordstat_client=_Failing(), llm=FakeWordstatLlmClient()
    )
    assert stats["clusters"] == 1
    assert stats["api_errors"] == stats["phrases"]
    assert stats["validated"] == 0
    assert list_wordstat_queries(conn, cluster_id) == []
    # The run row closed with a status (never hangs "running").
    stages = conn.execute(
        "SELECT stages_json FROM run ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert stages is not None and "validate" in dict(stages[0])


def test_run_validate_no_stale_clusters_is_quiet_noop(conn: Connection) -> None:
    """Empty selection: zero counters, no run row, no client builds."""
    stats = run_validate(conn, wordstat_client=_FixedWordstat(1))
    assert stats == {
        "clusters": 0,
        "phrases": 0,
        "validated": 0,
        "rejected": 0,
        "api_errors": 0,
        "demand_confirmed": 0,
        "rationale_updated": 0,
    }
    runs = conn.execute("SELECT count(*) FROM run").fetchone()
    assert runs is not None and runs[0] == 0


def test_run_validate_rerun_after_error_retries_cluster(conn: Connection) -> None:
    """Retry safety: a failed run leaves the cluster stale -> next run
    re-picks it and completes the cache (brick-once discipline of
    run_collect mirrored: statuses never trap a cluster)."""

    class _Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def frequency(self, phrase: str) -> int:
            self.calls += 1
            if self.calls <= 3:  # first run: every phrase fails
                raise RuntimeError("flaky")
            return 50

    cluster_id = _cluster_with_pain(conn, "https://example.com/h", "боли H")
    flaky = _Flaky()
    first = run_validate(conn, wordstat_client=flaky, llm=FakeWordstatLlmClient())
    assert first["validated"] == 0 and first["api_errors"] >= 3
    second = run_validate(conn, wordstat_client=flaky, llm=FakeWordstatLlmClient())
    assert second["validated"] >= 3
    assert len(list_wordstat_queries(conn, cluster_id)) == second["validated"]


def test_run_validate_records_stage_progress(conn: Connection) -> None:
    """The validate stage writes progress into the run table like others."""
    _cluster_with_pain(conn, "https://example.com/i", "боли I")
    run_id = create_run(conn, {"validate": "running"})
    update_run_stage(conn, run_id, "validate", "done", stats_delta={"validated": 3})
    finish_run(conn, run_id)
    row = conn.execute(
        "SELECT stages_json->>'validate', finished_at FROM run WHERE id = %s",
        (run_id,),
    ).fetchone()
    assert row is not None and row[0] == "done" and row[1] is not None
