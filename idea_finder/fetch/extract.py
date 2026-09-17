"""Main-text extraction: trafilatura first, selectolax fallback.

Pure functions, no network (task T301). A source adapter fetches HTML via
:mod:`idea_finder.fetch.httpx_fetcher` (or :mod:`idea_finder.fetch.browser`
for JS-heavy domains), then calls :func:`extract_main_text` to get the
article text. The boolean in the result flags suspiciously short
extractions so the collect stage can log or discard them instead of
feeding boilerplate into the LLM.
"""

from __future__ import annotations

from typing import Final

import trafilatura
from selectolax.parser import HTMLParser

__all__ = ["SHORT_TEXT_THRESHOLD", "extract_main_text"]

#: Extractions shorter than this many characters are flagged
#: ``suspect_short`` (assignment T301): likely cookie banners, paywalls,
#: or JS-only pages.
SHORT_TEXT_THRESHOLD: Final[int] = 200


def _selectolax_fallback(html: str) -> str:
    """Last-resort text grab: strip scripts/styles, return the body text.

    Used when trafilatura finds no main content (non-article pages,
    heavily templated layouts). The result keeps boilerplate, but it is
    better than losing the page entirely.
    """
    tree = HTMLParser(html)
    for node in tree.css("script, style, noscript"):
        node.decompose()
    body = tree.body or tree.root
    if body is None:
        return ""
    return body.text(separator=" ", strip=True)


def extract_main_text(html: str) -> tuple[str, bool]:
    """Extract readable main text from ``html``.

    Returns ``(text, suspect_short)`` where ``suspect_short`` is True when
    the extraction is empty or shorter than :data:`SHORT_TEXT_THRESHOLD`
    characters.
    """
    extracted = trafilatura.extract(html, favor_recall=True) or ""
    text = extracted.strip() or _selectolax_fallback(html).strip()
    return text, len(text) < SHORT_TEXT_THRESHOLD
