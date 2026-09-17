"""Tests for URL canonicalization (core.canonical.canonical_url)."""

import pytest

from idea_finder.core.canonical import canonical_url


def test_strips_utm_source() -> None:
    assert canonical_url("https://habr.com/ru/articles/1/?utm_source=telegram") == (
        "https://habr.com/ru/articles/1"
    )


def test_strips_each_tracking_parameter_individually() -> None:
    assert canonical_url("https://habr.com/p/?utm_campaign=x") == "https://habr.com/p"
    assert canonical_url("https://habr.com/p/?fbclid=abc") == "https://habr.com/p"
    assert canonical_url("https://habr.com/p/?gclid=def") == "https://habr.com/p"


def test_strips_all_utm_variants_in_mix_with_regular_params() -> None:
    url = (
        "https://example.com/post?id=7"
        "&utm_source=feed&utm_medium=social&utm_campaign=launch&utm_content=btn"
        "&fbclid=IwAR&gclid=Cj0K&flag=1"
    )
    assert canonical_url(url) == "https://example.com/post?id=7&flag=1"


def test_preserves_non_tracking_query() -> None:
    assert canonical_url("https://example.com/search?q=python&lang=ru") == (
        "https://example.com/search?q=python&lang=ru"
    )


def test_preserves_blank_valued_non_tracking_param() -> None:
    assert canonical_url("https://example.com/list?page=2&tag=") == (
        "https://example.com/list?page=2&tag="
    )


def test_strips_trailing_slash_but_keeps_root() -> None:
    assert canonical_url("https://example.com/blog/") == "https://example.com/blog"
    assert canonical_url("https://example.com/") == "https://example.com/"
    assert canonical_url("https://example.com") == "https://example.com"


def test_lowercases_scheme_and_host() -> None:
    assert canonical_url("HTTPS://Example.COM/Path") == "https://example.com/Path"


def test_preserves_http_scheme() -> None:
    assert canonical_url("http://Example.com/a/?utm_source=x") == "http://example.com/a"


def test_preserves_fragment() -> None:
    assert canonical_url("https://example.com/a#section") == "https://example.com/a#section"


@pytest.mark.parametrize("bad", ["", "   ", "not a url", "/relative/path", "example.com/page"])
def test_invalid_urls_raise_value_error(bad: str) -> None:
    with pytest.raises(ValueError, match="not an absolute URL"):
        canonical_url(bad)
