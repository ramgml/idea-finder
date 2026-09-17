"""Extract-stage prompt rendering and LLM response validation.

The extract stage (core/pipeline.py, T307) renders the v1 prompt from
``idea_finder/llm/prompts/extract_pains.md``, sends it through an
:class:`~idea_finder.llm.client.LlmClient`, and validates the answer with
:func:`parse_extract_response`.

Validation never raises for bad model output: bricked pains and parse
failures are counted in :class:`ExtractStats` (the stage folds them into
``run.stats``), so one malformed answer does not kill a batch run. Only
programmatic errors (missing template, unreadable fixtures) raise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import cast

from jinja2 import Environment, StrictUndefined

from idea_finder.core.models import Pain

__all__ = [
    "PAIN_FIELDS",
    "ExtractStats",
    "ExtractedPain",
    "FakeExtractLlmClient",
    "InvalidResponseError",
    "PromptTemplateError",
    "extract_stats_to_dict",
    "parse_extract_response",
    "render_extract_prompt",
]

#: Name the extract prompt is registered under (prompt_version table).
PROMPT_NAME = "extract_pains"

#: Version of the prompt shipped in this module.
PROMPT_VERSION = 1

#: Fields every pain object in the model answer must carry.
PAIN_FIELDS: frozenset[str] = frozenset({"body", "audience", "quote"})

#: Cap on pains per post when neither the caller nor the prompt says otherwise.
_DEFAULT_MAX_PAINS = 3


class PromptTemplateError(Exception):
    """Raised when the extract prompt template is missing or unreadable."""


class InvalidResponseError(Exception):
    """Raised by :func:`parse_extract_response` on unparseable model output.

    The extract stage counts these (``invalid_json`` counter) and moves on;
    an exception is the deliberate signal that there is nothing to salvage
    in the answer at all — not even a degraded parse.
    """


@dataclass(frozen=True, slots=True)
class ExtractedPain:
    """One pain as the model stated it, before source-post binding.

    ``quote`` is still unvalidated here; :func:`parse_extract_response`
    promotes only containment-passing instances to domain :class:`Pain`.
    """

    body: str
    audience: str
    quote: str


@dataclass(slots=True)
class ExtractStats:
    """Brick counters for one extraction answer (or one batch, summed).

    ``valid`` counts pains accepted after schema + anti-hallucination
    checks; every other counter is a rejection reason. ``degraded_parse``
    counts answers that only parsed because the fence/tolerance path
    stripped a markdown wrapper. ``extra_fields`` counts pain objects that
    were valid apart from carrying fields beyond ``PAIN_FIELDS``.
    """

    valid: int = 0
    hallucinated: int = 0
    schema_invalid: int = 0
    invalid_json: int = 0
    degraded_parse: int = 0
    extra_fields: int = 0

    def add(self, other: ExtractStats) -> None:
        """Accumulate ``other`` into self (batch fold-up)."""
        self.valid += other.valid
        self.hallucinated += other.hallucinated
        self.schema_invalid += other.schema_invalid
        self.invalid_json += other.invalid_json
        self.degraded_parse += other.degraded_parse
        self.extra_fields += other.extra_fields


def extract_stats_to_dict(stats: ExtractStats) -> dict[str, int]:
    """Return the stats as a plain dict for ``run.stats_json`` merge."""
    return {
        "valid": stats.valid,
        "hallucinated": stats.hallucinated,
        "schema_invalid": stats.schema_invalid,
        "invalid_json": stats.invalid_json,
        "degraded_parse": stats.degraded_parse,
        "extra_fields": stats.extra_fields,
    }


def _load_template() -> str:
    """Return the extract prompt template shipped with the package."""
    try:
        return (
            resources.files("idea_finder.llm")
            .joinpath("prompts/extract_pains.md")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError) as e:
        msg = "extract prompt template idea_finder/llm/prompts/extract_pains.md is missing"
        raise PromptTemplateError(msg) from e


def render_extract_prompt(post_text: str, max_pains: int = _DEFAULT_MAX_PAINS) -> str:
    """Render the extract v1 prompt for one post.

    ``StrictUndefined`` makes a template referencing a variable the caller
    did not provide fail loudly instead of silently rendering an empty
    placeholder.
    """
    env = Environment(autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined)
    try:
        template = env.from_string(_load_template())
    except PromptTemplateError:
        raise
    except Exception as e:
        msg = "failed to compile extract prompt template"
        raise PromptTemplateError(msg) from e
    return template.render(post_text=post_text, max_pains=max_pains)


@dataclass(frozen=True, slots=True)
class _ParseOutcome:
    """Intermediate parse result before anti-hallucination filtering."""

    raw_pains: list[dict[str, object]]
    degraded_parse: bool


def _strip_code_fence(text: str) -> tuple[str, bool]:
    """Strip a markdown fence around JSON; report whether one was present."""
    stripped = text.strip()
    if not (stripped.startswith("```") and stripped.endswith("```")):
        return stripped, False
    lines = stripped.splitlines()
    # Drop the opening fence (with its info string) and the closing fence.
    inner = lines[1:-1] if len(lines) >= 3 else []
    return "\n".join(inner).strip(), True


def _extract_json_payload(text: str) -> _ParseOutcome:
    """Parse the model answer into raw pain dicts.

    Accepts a bare JSON object or one wrapped in a markdown fence; fence
    presence marks the parse as degraded. Raises ``InvalidResponseError``
    when no JSON object with a pains list can be recovered.
    """
    body, fenced = _strip_code_fence(text)
    try:
        decoded: object = json.loads(body)
    except json.JSONDecodeError as e:
        if not fenced:
            raise InvalidResponseError(f"model answer is not valid JSON: {e}") from e
        # Fence-stripping can join a prose prefix with the JSON; fall back
        # to the first {...} block.
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise InvalidResponseError(f"model answer contains no JSON object: {e}") from e
        try:
            decoded = json.loads(body[start : end + 1])
        except json.JSONDecodeError as e2:
            raise InvalidResponseError(f"model answer is not valid JSON: {e2}") from e2
    if not isinstance(decoded, dict):
        raise InvalidResponseError(f"model answer is not a JSON object: {type(decoded).__name__}")
    pains = decoded.get("pains")
    if not isinstance(pains, list):
        raise InvalidResponseError("model answer JSON has no pains list")
    return _ParseOutcome(
        raw_pains=[p for p in pains if isinstance(p, dict)],
        degraded_parse=fenced,
    )


def _validate_pain_shape(raw: dict[str, object]) -> tuple[ExtractedPain | None, bool]:
    """Check one raw pain dict against the schema.

    Returns ``(pain, extra)`` where ``pain`` is None on schema violation and
    ``extra`` flags fields beyond ``PAIN_FIELDS`` (tolerated but counted).
    """
    extra = not PAIN_FIELDS.issuperset(raw.keys())
    values: dict[str, str] = {}
    for name in ("body", "audience", "quote"):
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            return None, extra
        values[name] = value
    return ExtractedPain(**values), extra


def parse_extract_response(
    response_text: str,
    source_text: str,
    *,
    max_pains: int = _DEFAULT_MAX_PAINS,
) -> tuple[list[Pain], ExtractStats]:
    """Validate one model answer against ``source_text``.

    Anti-hallucination contract: a pain is accepted only if its ``quote``
    is a verbatim substring of ``source_text``. Schema violations and
    hallucinated quotes brick the pain (counted, never raised); unparseable
    answers raise :class:`InvalidResponseError` with ``invalid_json``
    counted in the returned stats. Pains beyond ``max_pains`` are dropped.
    """
    stats = ExtractStats()
    try:
        outcome = _extract_json_payload(response_text)
    except InvalidResponseError:
        stats.invalid_json += 1
        raise
    stats.degraded_parse += outcome.degraded_parse
    pains: list[Pain] = []
    for raw in outcome.raw_pains:
        extracted, extra = _validate_pain_shape(raw)
        if extracted is None:
            stats.schema_invalid += 1
            continue
        if extra:
            stats.extra_fields += 1
        if extracted.quote not in source_text:
            stats.hallucinated += 1
            continue
        stats.valid += 1
        pains.append(
            Pain(
                source_post_id="",  # bound to the post by the extract stage
                body=extracted.body,
                audience=extracted.audience,
                quote=extracted.quote,
            )
        )
        if len(pains) >= max_pains:
            break
    return pains, stats


class FakeExtractLlmClient:
    """Deterministic JSON-answer client for extract-stage acceptance runs.

    Implements the same ``LlmClient`` port as
    :class:`~idea_finder.llm.client.FakeLlmClient` but answers with a
    well-formed extract JSON instead of a raw pain body: it scans
    ``fixtures/expected_pains.json`` and returns every fixture pain whose
    ``quote`` occurs in the post text embedded in the prompt. Answers are
    built by the same containment rule the validator enforces, so the fake
    never hallucinates — yet the pairing (fake -> parse) still proves the
    pipeline, because a broken prompt render or parser bricks everything
    and the acceptance test fails.
    """

    def __init__(self, *, model: str = "fake", provider_name: str = "fake") -> None:
        self.model = model
        self.provider_name = provider_name
        self._pains = _load_fixture_pains()

    def complete(self, prompt: str) -> str:
        """Return the deterministic extract JSON answer for ``prompt``."""
        post_text = _post_text_from_prompt(prompt, self._pains)
        matched = [pain for quote, pain in self._pains if quote in post_text]
        return json.dumps({"pains": matched[:_DEFAULT_MAX_PAINS]}, ensure_ascii=False)


def _load_fixture_pains() -> list[tuple[str, dict[str, object]]]:
    """Load fixture pains as (quote, pain-dict) pairs for prompt scanning."""
    fixture = resources.files("idea_finder").joinpath("fixtures/expected_pains.json")
    try:
        text = fixture.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as e:
        msg = "fixtures/expected_pains.json is missing from the package"
        raise PromptTemplateError(msg) from e
    entries = cast(list[object], json.loads(text))
    pairs: list[tuple[str, dict[str, object]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_pains = entry.get("pains")
        if not isinstance(entry_pains, list):
            continue
        for pain in entry_pains:
            if not isinstance(pain, dict):
                continue
            quote = pain.get("quote")
            if isinstance(quote, str) and quote:
                pairs.append((quote, pain))
    return pairs


def _post_text_from_prompt(prompt: str, pairs: list[tuple[str, dict[str, object]]]) -> str:
    """Recover the post text from a rendered prompt by longest-quote anchor.

    The fake client does not re-render the template; instead it locates the
    longest fixture quote that occurs verbatim in the prompt (only the
    template would place fixture quotes there, joined with the post text)
    and returns everything from that anchor to the end. Raises
    ``InvalidResponseError`` when no fixture quote is found — a prompt the
    fixtures cannot answer is a caller bug, not an empty answer.
    """
    best_quote = ""
    best_index = -1
    for quote, _ in pairs:
        index = prompt.find(quote)
        if index >= 0 and len(quote) > len(best_quote):
            best_quote, best_index = quote, index
    if best_index < 0:
        raise InvalidResponseError("prompt does not contain any fixture quote to anchor on")
    return prompt[best_index:]


