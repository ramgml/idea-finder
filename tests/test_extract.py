"""Tests for main-text extraction (task T301): trafilatura, selectolax
fallback, suspect_short flag."""

from __future__ import annotations

import pytest

from idea_finder.fetch.extract import SHORT_TEXT_THRESHOLD, extract_main_text

ARTICLE_HTML = """
<html>
  <head><title>Test article</title><style>body { color: red; }</style></head>
  <body>
    <nav><a href="/">Home</a><a href="/about">About</a></nav>
    <article>
      <h1>Freelance platforms in Russia</h1>
      <p>Customers on freelance platforms complain that it is hard to find a
      reliable contractor for small Home Assistant setup jobs. They describe
      wasting days on candidates who disappear mid-project, and having no way
      to verify past work before paying an advance.</p>
      <p>Another recurring pain is dispute resolution: when a contractor
      delivers broken work, the platform arbitration takes weeks and the
      customer rarely recovers the prepayment, so experienced customers start
      demanding escrow-like guarantees before they post a job at all.</p>
    </article>
    <script>window.tracker("never-leak-into-text");</script>
    <footer>© 2026 test footer</footer>
  </body>
</html>
"""

PLAIN_HTML = """
<html><body>
  <script>var x = "script-noise";</script>
  <style>.hidden { display: none; }</style>
  <div>Plain page without semantic article markup, just a couple of
  paragraphs inside nested divs that trafilatura may refuse to extract.</div>
  <div>Second block of text so the fallback extraction still has some body
  content to return for the caller to persist.</div>
</body></html>
"""


def test_trafilatura_extracts_article_body() -> None:
    """Article markup yields the main text without nav/footer chrome."""
    text, suspect = extract_main_text(ARTICLE_HTML)
    assert not suspect
    assert "reliable contractor" in text
    assert "escrow-like guarantees" in text
    assert "Home" != text.strip()  # nav alone never becomes the text
    assert "test footer" not in text


@pytest.mark.parametrize("noise", ["script-noise", "display: none"])
def test_scripts_and_styles_never_leak_into_text(noise: str) -> None:
    """Script bodies and CSS rules are excluded from the extracted text."""
    for html in (ARTICLE_HTML, PLAIN_HTML):
        text, _ = extract_main_text(html)
        assert noise not in text


def test_selectolax_fallback_covers_unstructured_html() -> None:
    """Without article semantics the selectolax path still returns body text."""
    text, _ = extract_main_text(PLAIN_HTML)
    assert "Plain page without semantic" in text
    assert "Second block" in text


def test_short_extraction_is_flagged() -> None:
    """Text under the threshold carries suspect_short=True."""
    text, suspect = extract_main_text("<html><body><p>Too short.</p></body></html>")
    assert suspect
    assert len(text) < SHORT_TEXT_THRESHOLD


def test_empty_extraction_is_flagged() -> None:
    """A page with no textual content at all is flagged, not crashing."""
    text, suspect = extract_main_text("<html><body></body></html>")
    assert text == ""
    assert suspect


def test_long_text_is_not_flagged() -> None:
    """A fallback extraction over the threshold clears the flag."""
    paragraph = (
        "<p>The marketplace arbitration process has one more structural "
        "problem: neither side can see the full decision history, so "
        "repeat offenders simply re-register under a new name and the "
        "customers who hired them before have no way to warn others.</p>"
    )
    html = f"<html><body>{paragraph * 2}</body></html>"
    text, suspect = extract_main_text(html)
    assert not suspect
    assert len(text) >= SHORT_TEXT_THRESHOLD
