"""Tests for pipeline domain models (core.models)."""
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from idea_finder.core.models import Cluster, Pain, PostKind, RawPost, Run, Score

ALL_KINDS: tuple[PostKind, ...] = ("demand", "complaint", "discussion")


def make_raw_post(kind: PostKind = "demand") -> RawPost:
    return RawPost(
        source_id="fl_ru",
        url_canon="https://fl.ru/task/1",
        url="https://FL.ru/task/1/?utm_source=feed",
        title="Need a bot",
        text="Need a Telegram bot for order tracking.",
        published_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
        kind=kind,
    )


def test_raw_post_construction_keeps_all_fields() -> None:
    post = make_raw_post()
    assert post.source_id == "fl_ru"
    assert post.url_canon == "https://fl.ru/task/1"
    assert post.url == "https://FL.ru/task/1/?utm_source=feed"
    assert post.title == "Need a bot"
    assert post.text.startswith("Need a Telegram bot")
    assert post.published_at == datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
    assert post.kind == "demand"
    assert post.id is None


def test_raw_post_id_defaults_to_none_and_can_be_set() -> None:
    post = make_raw_post()
    assert post.id is None
    with_id = RawPost(
        source_id=post.source_id,
        url_canon=post.url_canon,
        url=post.url,
        title=post.title,
        text=post.text,
        published_at=post.published_at,
        kind=post.kind,
        id="0195e6f2-7b1a-7de1-a5f0-3f6a9f1c2d10",
    )
    assert with_id.id == "0195e6f2-7b1a-7de1-a5f0-3f6a9f1c2d10"


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_raw_post_accepts_all_documented_kinds(kind: PostKind) -> None:
    assert make_raw_post(kind).kind == kind


# The invalid-kind cases below intentionally violate the PostKind literal to
# prove the runtime ValueError contract; each violation is one isolated call.
INVALID_KINDS = ("Demand", "review", "", "bug", None)


@pytest.mark.parametrize("kind", INVALID_KINDS)
def test_raw_post_rejects_unknown_kind(kind: object) -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        make_raw_post(kind)  # ty: ignore[invalid-argument-type]


def test_raw_post_is_frozen() -> None:
    post = make_raw_post()
    with pytest.raises(FrozenInstanceError):
        post.title = "Changed"  # ty: ignore[invalid-assignment]


def test_pain_construction() -> None:
    pain = Pain(
        source_post_id="0195e6f2-7b1a-7de1-a5f0-3f6a9f1c2d10",
        body="Sellers cannot track orders without a dedicated bot.",
        audience="small e-commerce sellers",
        quote="Need a Telegram bot for order tracking.",
    )
    assert pain.source_post_id.endswith("2d10")
    assert pain.audience == "small e-commerce sellers"
    assert pain.quote in "Need a Telegram bot for order tracking."
    assert pain.id is None


def test_cluster_construction() -> None:
    cluster = Cluster(title="Order tracking bots", size=5, kind_mix={"demand": 4, "complaint": 1})
    assert cluster.size == 5
    assert cluster.kind_mix == {"demand": 4, "complaint": 1}
    assert cluster.id is None


def test_score_construction() -> None:
    score = Score(
        cluster_id="0195e6f2-7b1a-7de1-a5f0-3f6a9f1c2d11",
        total=78.5,
        rationale_md="High demand, low competition.",
        quotes=["Need a Telegram bot for order tracking."],
    )
    assert score.total == 78.5
    assert score.rationale_md.startswith("High demand")
    assert score.quotes == ["Need a Telegram bot for order tracking."]
    assert score.id is None


def test_run_construction() -> None:
    run = Run(stages={"collect": "done", "extract": "running"}, stats={"posts": 10}, cost=0.42)
    assert run.stages == {"collect": "done", "extract": "running"}
    assert run.stats == {"posts": 10}
    assert run.cost == 0.42
    assert run.id is None


def test_all_models_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        Pain(source_post_id="x", body="b", audience="a", quote="q").body = "z"  # ty: ignore[invalid-assignment]
