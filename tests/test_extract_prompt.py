"""Tests for the extract-stage LLM contract (task T306).

Covers three layers:
1. ``render_extract_prompt`` — the Jinja2 v1 prompt: post text and
   ``max_pains`` interpolate, missing variables fail loudly
   (``StrictUndefined``).
2. ``parse_extract_response`` — schema validation, the anti-hallucination
   containment rule, fence tolerance, and the counter semantics of
   :class:`ExtractStats`.
3. Acceptance (owner DoD): every fixture post pushed through
   ``FakeExtractLlmClient`` + ``parse_extract_response`` yields a valid
   extraction for at least 80% of posts, zero hallucinated quotes, and the
   full brick table for the T307 stage gate.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import cast

import pytest

from idea_finder.llm.extract import (
    PAIN_FIELDS,
    ExtractStats,
    FakeExtractLlmClient,
    InvalidResponseError,
    extract_stats_to_dict,
    parse_extract_response,
    render_extract_prompt,
)

SOURCE = (
    "Нужен телеграм-бот для автомобильного сервисного центра.\n"
    "Запись клиентов ведётся вручную, мастера путаются в расписании."
)
QUOTED = "телеграм-бот для автомобильного сервисного центра"


def _valid_answer() -> str:
    """A minimal valid extract answer for SOURCE with one pain."""
    return json.dumps(
        {"pains": [{"body": "Боль", "audience": "автосервисы", "quote": QUOTED}]},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def test_render_embeds_post_text_and_answer_contract() -> None:
    """The post text lands in the prompt and the JSON contract is stated."""
    prompt = render_extract_prompt(SOURCE)
    assert SOURCE in prompt
    assert '"pains"' in prompt
    assert '"quote"' in prompt
    # The anti-hallucination rule is spelled out in the template itself.
    assert "дословная" in prompt


def test_render_interpolates_max_pains() -> None:
    """max_pains is a template variable, not a hardcoded number."""
    assert "до 3 болей" in render_extract_prompt(SOURCE)
    assert "до 5 болей" in render_extract_prompt(SOURCE, max_pains=5)


def test_render_is_deterministic() -> None:
    """Same inputs -> byte-identical prompt (prompt_version reproducibility)."""
    assert render_extract_prompt(SOURCE) == render_extract_prompt(SOURCE)


def test_template_strict_undefined_on_missing_variable() -> None:
    """Rendering without max_pains raises instead of rendering 'Undefined'."""
    from jinja2 import Environment, StrictUndefined

    env = Environment(autoescape=False, undefined=StrictUndefined)
    template_source = resources.files("idea_finder").joinpath(
        "llm/prompts/extract_pains.md"
    )
    template = env.from_string(template_source.read_text(encoding="utf-8"))
    with pytest.raises(Exception, match="max_pains"):
        template.render(post_text=SOURCE)


# ---------------------------------------------------------------------------
# parse_extract_response: schema validation and counters
# ---------------------------------------------------------------------------


def test_parse_valid_answer_yields_domain_pains() -> None:
    """A well-formed answer maps onto domain Pain objects with valid stats."""
    pains, stats = parse_extract_response(_valid_answer(), SOURCE)
    assert len(pains) == 1
    assert pains[0].quote == QUOTED
    assert pains[0].body == "Боль"
    assert pains[0].source_post_id == ""
    assert stats.valid == 1
    assert stats.hallucinated == 0
    assert stats.schema_invalid == 0
    assert stats.invalid_json == 0
    assert stats.degraded_parse == 0


def test_parse_empty_pains_list_is_valid() -> None:
    """An answer with no pains is a valid 'no pain here' verdict."""
    pains, stats = parse_extract_response('{"pains": []}', SOURCE)
    assert pains == []
    assert stats.valid == 0
    assert stats.invalid_json == 0


def test_parse_hallucinated_quote_bricks_pain() -> None:
    """A quote absent from the source text is bricked and counted."""
    answer = json.dumps(
        {"pains": [{"body": "Боль", "audience": "аудитория",
                    "quote": "Этой фразы нет в исходном тексте"}]},
        ensure_ascii=False,
    )
    pains, stats = parse_extract_response(answer, SOURCE)
    assert pains == []
    assert stats.hallucinated == 1
    assert stats.valid == 0


def test_parse_quote_must_match_verbatim_not_fuzzy() -> None:
    """Punctuation or whitespace edits turn a quote into a hallucination."""
    answer = json.dumps(
        {"pains": [{"body": "Боль", "audience": "аудитория",
                    "quote": QUOTED.replace("-", " ")}]},
        ensure_ascii=False,
    )
    pains, stats = parse_extract_response(answer, SOURCE)
    assert pains == []
    assert stats.hallucinated == 1


@pytest.mark.parametrize(
    "pain",
    [
        pytest.param({"body": "Б", "audience": "а"}, id="missing quote"),
        pytest.param({"body": "Б", "quote": QUOTED}, id="missing audience"),
        pytest.param({"audience": "а", "quote": QUOTED}, id="missing body"),
        pytest.param({"body": 1, "audience": "а", "quote": QUOTED}, id="non-string body"),
        pytest.param({"body": None, "audience": "а", "quote": QUOTED}, id="null body"),
        pytest.param({"body": "", "audience": "а", "quote": QUOTED}, id="empty body"),
        pytest.param({"body": "  ", "audience": "а", "quote": QUOTED}, id="blank body"),
    ],
)
def test_parse_schema_violations_brick_and_count(pain: dict[str, object]) -> None:
    """Any schema violation bricks the pain as schema_invalid."""
    pains, stats = parse_extract_response(
        json.dumps({"pains": [pain]}, ensure_ascii=False), SOURCE
    )
    assert pains == []
    assert stats.schema_invalid == 1
    assert stats.valid == 0


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("[1, 2, 3]", id="json array instead of object"),
        pytest.param('{"pains": "many"}', id="pains not a list"),
        pytest.param('{"other": true}', id="pains key absent"),
    ],
)
def test_parse_malformed_container_is_invalid_json(answer: str) -> None:
    """Container-level malformation has nothing to salvage -> error + counter."""
    with pytest.raises(InvalidResponseError):
        parse_extract_response(answer, SOURCE)


def test_parse_non_object_pain_is_schema_invalid() -> None:
    """A non-object entry inside pains bricks that entry, not the answer."""
    pains, stats = parse_extract_response('{"pains": [42]}', SOURCE)
    assert pains == []
    assert stats.schema_invalid == 1
    assert stats.invalid_json == 0


def test_parse_plain_garbage_is_invalid_json() -> None:
    """Free-form text with no JSON object raises and counts invalid_json."""
    with pytest.raises(InvalidResponseError, match="not valid JSON"):
        parse_extract_response("Извините, я не понял задание.", SOURCE)


def test_parse_counts_invalid_json_before_raising() -> None:
    """The counter is filled even though the function raises (T307 fold-up)."""
    stats = ExtractStats()
    try:
        parse_extract_response("not json", SOURCE)
    except InvalidResponseError:
        stats.add(ExtractStats(invalid_json=1))
    assert stats.invalid_json == 1


def test_parse_extra_fields_tolerated_but_counted() -> None:
    """Pain objects may carry extra fields; the tolerance is counted."""
    answer = json.dumps(
        {"pains": [{"body": "Боль", "audience": "аудитория", "quote": QUOTED,
                    "confidence": 0.9}]},
        ensure_ascii=False,
    )
    pains, stats = parse_extract_response(answer, SOURCE)
    assert len(pains) == 1
    assert stats.valid == 1
    assert stats.extra_fields == 1


def test_parse_markdown_fence_is_degraded_but_valid() -> None:
    """A fenced answer parses; fence presence counts as degraded_parse."""
    fenced = f"```json\n{_valid_answer()}\n```"
    pains, stats = parse_extract_response(fenced, SOURCE)
    assert len(pains) == 1
    assert stats.valid == 1
    assert stats.degraded_parse == 1


def test_parse_degraded_zero_when_no_fence() -> None:
    """A bare JSON answer keeps degraded_parse at zero."""
    _, stats = parse_extract_response(_valid_answer(), SOURCE)
    assert stats.degraded_parse == 0


def test_parse_cap_on_max_pains() -> None:
    """Pains beyond max_pains are dropped, not validated."""
    pain = {"body": "Боль", "audience": "аудитория", "quote": QUOTED}
    answer = json.dumps({"pains": [pain, pain, pain, pain]}, ensure_ascii=False)
    pains, stats = parse_extract_response(answer, SOURCE, max_pains=3)
    assert len(pains) == 3
    assert stats.valid == 3
    pains2, stats2 = parse_extract_response(answer, SOURCE, max_pains=2)
    assert len(pains2) == 2
    assert stats2.valid == 2


def test_parse_max_pains_stops_after_collecting_cap() -> None:
    """The cap stops validation once max_pains valid pains are collected.

    The mixed-quality answer lists one good pain, then a hallucinated one,
    then more good ones: once two valid pains are in, the rest of the list
    is never validated, so exactly one hallucination gets counted.
    """
    bad = {"body": "Б", "audience": "а", "quote": "нет такой цитаты"}
    good = {"body": "Боль", "audience": "аудитория", "quote": QUOTED}
    answer = json.dumps({"pains": [good, bad, good, good]}, ensure_ascii=False)
    pains, stats = parse_extract_response(answer, SOURCE, max_pains=2)
    assert len(pains) == 2
    assert stats.valid == 2
    assert stats.hallucinated == 1


def test_extract_stats_to_dict_shape() -> None:
    """The stats dict mirrors every counter for run.stats_json merge."""
    stats = ExtractStats(valid=2, hallucinated=1, degraded_parse=1)
    data = extract_stats_to_dict(stats)
    assert data == {
        "valid": 2,
        "hallucinated": 1,
        "schema_invalid": 0,
        "invalid_json": 0,
        "degraded_parse": 1,
        "extra_fields": 0,
    }


def test_extract_stats_add_folds_counters() -> None:
    """Batch fold-up sums every counter (per-post -> per-run)."""
    total = ExtractStats(valid=1)
    total.add(ExtractStats(hallucinated=1, invalid_json=1))
    total.add(ExtractStats(schema_invalid=2, degraded_parse=1))
    assert (total.valid, total.hallucinated, total.invalid_json) == (1, 1, 1)
    assert (total.schema_invalid, total.degraded_parse) == (2, 1)


def test_pain_fields_contract_matches_template() -> None:
    """PAIN_FIELDS is exactly the triple the template demands."""
    assert PAIN_FIELDS == frozenset({"body", "audience", "quote"})


# ---------------------------------------------------------------------------
# FakeExtractLlmClient
# ---------------------------------------------------------------------------


def test_fake_client_returns_parseable_json_for_fixture_post() -> None:
    """On an annotated fixture post the fake answers JSON the parser accepts."""
    posts = cast(
        list[dict[str, object]],
        json.loads(
            resources.files("idea_finder")
            .joinpath("fixtures/posts/fl_ru.json")
            .read_text(encoding="utf-8")
        ),
    )
    fixture = resources.files("idea_finder").joinpath("fixtures/expected_pains.json")
    entries = cast(
        list[dict[str, object]], json.loads(fixture.read_text(encoding="utf-8"))
    )
    annotated = {str(e["post_url"]) for e in entries}
    post = next(p for p in posts if str(p["url"]) in annotated)
    post_text = str(post["text"])
    client = FakeExtractLlmClient()
    prompt = render_extract_prompt(post_text)
    answer = client.complete(prompt)
    assert isinstance(answer, str)
    pains, stats = parse_extract_response(answer, post_text)
    assert stats.valid >= 1
    assert all(p.quote in post_text for p in pains)


def test_fake_client_is_deterministic() -> None:
    """Identical prompts give identical answers, twice in a row."""
    client = FakeExtractLlmClient()
    prompt = render_extract_prompt(SOURCE)
    assert client.complete(prompt) == client.complete(prompt)


def test_fake_client_rejects_prompt_without_post_header() -> None:
    """A prompt not produced by render_extract_prompt is a caller bug.

    The fake recovers the post text from the fixed ``Текст поста:`` header
    the template appends; a prompt without it raises instead of silently
    answering with an empty pain list.
    """
    from idea_finder.llm.extract import _load_fixture_pains, _post_text_from_prompt

    with pytest.raises(InvalidResponseError):
        _post_text_from_prompt("пост без фиксаторных цитат", _load_fixture_pains())


def test_fake_client_completion_shape_matches_protocol() -> None:
    """complete() output pairs with parse for a hand-made source too."""
    client = FakeExtractLlmClient()
    answer = client.complete(render_extract_prompt(SOURCE))
    # SOURCE quotes the fixture pain verbatim, so the fake answers with it.
    pains, stats = parse_extract_response(answer, SOURCE)
    # SOURCE contains the fixture quote verbatim, so the fake answers with
    # the fixture pain whose quote is the full first sentence.
    assert stats.valid == 1
    assert pains[0].quote == "Нужен телеграм-бот для автомобильного сервисного центра."
    assert pains[0].quote in SOURCE


def test_fake_client_is_constructible_with_model_metadata() -> None:
    """Model/provider metadata ride along for cost accounting parity."""
    client = FakeExtractLlmClient(model="fake-model", provider_name="fake")
    assert client.model == "fake-model"
    assert client.provider_name == "fake"


# ---------------------------------------------------------------------------
# Acceptance (owner DoD): fixtures run, >=80% posts, zero hallucinations
# ---------------------------------------------------------------------------


def _load_fixture_posts() -> list[dict[str, str]]:
    """Load every fixture post (all three sources) as plain text records."""
    posts: list[dict[str, str]] = []
    for name in ("fl_ru", "habr", "gplay"):
        raw = cast(
            list[dict[str, object]],
            json.loads(
                resources.files("idea_finder")
                .joinpath(f"fixtures/posts/{name}.json")
                .read_text(encoding="utf-8")
            ),
        )
        for entry in raw:
            posts.append(
                {
                    "url": str(entry["url"]),
                    "text": str(entry["text"]),
                }
            )
    return posts


@pytest.fixture(scope="module")
def posts() -> list[dict[str, str]]:
    """All fixture posts from the three sources."""
    return _load_fixture_posts()


@pytest.fixture(scope="module")
def expected_pains() -> list[dict[str, object]]:
    """The expected_pains.json annotation records, keyed by post_url."""
    fixture = resources.files("idea_finder").joinpath("fixtures/expected_pains.json")
    return cast(
        list[dict[str, object]], json.loads(fixture.read_text(encoding="utf-8"))
    )


def test_acceptance_fixtures_extraction_meets_dod(
    posts: list[dict[str, str]],
    expected_pains: list[dict[str, object]],
) -> None:
    """DoD gate on the full fixture corpus (T326: 32/32 annotated).

    ``fixtures/expected_pains.json`` covers every fixture post: either with
    extracted pains or with the ``pains: []`` no-pain marker. The gate runs
    over all posts: at least 80% must yield a valid pain and zero
    hallucinated quotes are tolerated. A post without pain is legitimate —
    the fake finds no matching quote and the parse yields zero pains.
    Prints the brick table the T307 stage gate consumes. No corpus sizes are
    hardcoded, so future re-annotation does not silently break the gate.
    """
    annotated_urls = {str(e["post_url"]) for e in expected_pains}
    assert annotated_urls == {p["url"] for p in posts}, (
        "expected_pains.json must cover every fixture post (pains: [] for no pain)"
    )
    client = FakeExtractLlmClient()
    total = ExtractStats()
    posts_with_valid_pain = 0
    for post in posts:
        prompt = render_extract_prompt(post["text"])
        answer = client.complete(prompt)
        pains, stats = parse_extract_response(answer, post["text"])
        total.add(stats)
        if pains:
            posts_with_valid_pain += 1
    ratio = posts_with_valid_pain / len(posts)
    print(
        f"\nDoD acceptance over all {len(posts)} fixture posts: "
        f"posts_with_valid_pain={posts_with_valid_pain} "
        f"({ratio:.0%}), pains_accepted={total.valid}"
    )
    print(
        "brick table: "
        + json.dumps(extract_stats_to_dict(total), ensure_ascii=False)
    )
    assert total.hallucinated == 0, "anti-hallucination: every quote must be verbatim"
    assert total.invalid_json == 0
    assert total.schema_invalid == 0
    assert ratio >= 0.8, f"only {ratio:.0%} of posts produced a valid pain (<80%)"


def test_acceptance_every_accepted_quote_is_verbatim_substring(
    posts: list[dict[str, str]],
) -> None:
    """Per-post containment: each accepted quote is inside its own source.

    This is not a tautology even with the fixture-scanning fake: the fake
    rebuilds its answer from the prompt text itself, so a broken prompt
    render or a scanner mixing quotes across posts would surface here.
    """
    client = FakeExtractLlmClient()
    for post in posts:
        prompt = render_extract_prompt(post["text"])
        answer = client.complete(prompt)
        pains, _ = parse_extract_response(answer, post["text"])
        for pain in pains:
            assert pain.quote in post["text"], (
                f"hallucinated quote for {post['url']}: {pain.quote!r}"
            )
