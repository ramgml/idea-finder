"""Canonical URL normalization used for raw post deduplication.

Canonical form: lowercase scheme and host, tracking query parameters
(``utm_*``, ``fbclid``, ``gclid``) stripped, no trailing slash on the
path (except the root ``/``). Everything else — the scheme itself, non
tracking query parameters and the fragment — is preserved as-is.
"""

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

__all__ = ["canonical_url"]

# Query parameters removed during canonicalization: campaign trackers
# (any ``utm_*``) plus the click-IDs of Facebook and Google.
_TRACKING_PREFIXES: tuple[str, ...] = ("utm_",)
_TRACKING_PARAMS: frozenset[str] = frozenset({"fbclid", "gclid"})


def canonical_url(url: str) -> str:
    """Return the canonical form of ``url``.

    Raises:
        ValueError: If ``url`` is empty, not a string-parseable URL, or
            has no scheme/network location (e.g. a relative path).
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        msg = f"not an absolute URL: {url!r}"
        raise ValueError(msg)

    scheme = parsed.scheme.lower()
    # The port must stay intact; lowercasing the whole netloc is safe
    # for the host part because userinfo and port contain no letters
    # that carry meaning (userinfo is case-sensitive, but it never
    # appears in the URLs this pipeline collects).
    netloc = parsed.netloc.lower()

    kept: list[tuple[str, str]] = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not (name.startswith(_TRACKING_PREFIXES) or name in _TRACKING_PARAMS)
    ]
    query = urlencode(kept)

    path = parsed.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    return urlunparse((scheme, netloc, path, parsed.params, query, parsed.fragment))
