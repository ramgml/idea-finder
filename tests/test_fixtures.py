"""Contract tests for the regression dataset fixtures (fixtures/posts/*.json, fixtures/expected_pains.json).

The dataset is the fixed input for the prompt-regression harness: these tests pin the
dataset contract itself (loadability, coverage, uniqueness, verbatim quotes), so a
dataset change that would silently break prompt regression fails here first.
"""

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
POSTS_DIR = FIXTURES_DIR / "posts"

VALID_KINDS = ("demand", "complaint", "discussion")
VALID_SOURCES = ("fl_ru", "habr", "gplay")

MIN_TOTAL_POSTS = 25

POST_FIELDS = {"url", "text", "kind", "title", "source"}
JSONObj = dict[str, object]


def load_posts() -> list[JSONObj]:
    posts: list[JSONObj] = []
    for path in sorted(POSTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list), f"{path.name} must contain a JSON array"
        posts.extend(data)
    return posts


@pytest.fixture(scope="module")
def posts() -> list[JSONObj]:
    return load_posts()


@pytest.fixture(scope="module")
def posts_by_url(posts: list[JSONObj]) -> dict[str, JSONObj]:
    return {str(p["url"]): p for p in posts}


def test_dataset_meets_minimum_size(posts: list[JSONObj]) -> None:
    assert len(posts) >= MIN_TOTAL_POSTS


def test_all_kinds_present(posts: list[JSONObj]) -> None:
    kinds = {p["kind"] for p in posts}
    assert kinds == set(VALID_KINDS)


@pytest.mark.parametrize("kind", VALID_KINDS)
def test_kind_coverage(posts: list[JSONObj], kind: str) -> None:
    count = sum(1 for p in posts if p["kind"] == kind)
    assert count >= 5, f"kind={kind}: expected >=5 posts, got {count}"


def test_urls_are_unique(posts: list[JSONObj]) -> None:
    urls = [p["url"] for p in posts]
    assert len(urls) == len(set(urls)), "duplicate post urls in dataset"


def test_posts_have_only_documented_fields(posts: list[JSONObj]) -> None:
    for p in posts:
        extra = set(p) - POST_FIELDS
        assert not extra, f"{p['url']}: unexpected fields {extra}"


def test_post_fields_are_valid(posts: list[JSONObj]) -> None:
    for p in posts:
        assert p["kind"] in VALID_KINDS, p["url"]
        assert p["source"] in VALID_SOURCES, p["url"]
        assert isinstance(p["text"], str) and p["text"].strip(), p["url"]
        assert isinstance(p["url"], str) and p["url"].startswith("http"), p["url"]


def test_source_files_cover_their_named_sources() -> None:
    expected = {"fl_ru.json": "fl_ru", "habr.json": "habr", "gplay.json": "gplay"}
    for filename, source in expected.items():
        data = json.loads((POSTS_DIR / filename).read_text(encoding="utf-8"))
        assert data, f"{filename} is empty"
        assert all(p["source"] == source for p in data), f"{filename}: wrong source"


def test_russian_text_present(posts: list[JSONObj]) -> None:
    cyrillic = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
    for p in posts:
        assert any(ch.lower() in cyrillic for ch in str(p["text"])), p["url"]


def test_expected_pains_exists_and_valid() -> None:
    path = FIXTURES_DIR / "expected_pains.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) >= 8


def test_expected_pain_quotes_are_verbatim_substrings(
    posts_by_url: dict[str, JSONObj],
) -> None:
    data = json.loads((FIXTURES_DIR / "expected_pains.json").read_text(encoding="utf-8"))
    for entry in data:
        post = posts_by_url[entry["post_url"]]
        for pain in entry["pains"]:
            assert str(pain["quote"]) in str(post["text"]), (
                f"quote not verbatim for {entry['post_url']}: {pain['quote']!r}"
            )


def test_expected_pains_have_filled_fields() -> None:
    data = json.loads((FIXTURES_DIR / "expected_pains.json").read_text(encoding="utf-8"))
    for entry in data:
        assert entry["pains"], entry["post_url"]
        for pain in entry["pains"]:
            assert pain["body"].strip(), entry["post_url"]
            assert pain["audience"].strip(), entry["post_url"]
            assert pain["quote"].strip(), entry["post_url"]


def test_expected_pains_reference_known_posts(posts_by_url: dict[str, JSONObj]) -> None:
    data = json.loads((FIXTURES_DIR / "expected_pains.json").read_text(encoding="utf-8"))
    for entry in data:
        assert entry["post_url"] in posts_by_url, entry["post_url"]
