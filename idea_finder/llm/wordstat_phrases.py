"""Wordstat-phrase prompt rendering and LLM answer validation (task T321).

The validate stage (core/pipeline.py ``run_validate``) renders the
wordstat-phrases prompt from ``idea_finder/llm/prompts/wordstat_phrases.md``,
sends it through an :class:`~idea_finder.llm.client.LlmClient` (mock mode =
the fake provider), and validates the answer with
:func:`parse_phrases_response`. The validated phrases are what the Wordstat
client is queried with; the cache lives in ``wordstat_query``.

Validation never raises for bad model output: unparseable answers are
rejected (the stage counts them and moves on); only programmatic errors
(missing template) raise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources

from jinja2 import Environment, StrictUndefined

from idea_finder.llm.client import Completion

__all__ = [
    "WORDSTAT_PROMPT_NAME",
    "WORDSTAT_PROMPT_VERSION",
    "FakeWordstatLlmClient",
    "InvalidPhrasesResponseError",
    "PromptTemplateError",
    "load_wordstat_template",
    "parse_phrases_response",
    "render_wordstat_prompt",
]

#: Name the wordstat-phrases prompt is registered under (prompt_version).
WORDSTAT_PROMPT_NAME = "wordstat_phrases"

#: Version of the prompt shipped in this module.
WORDSTAT_PROMPT_VERSION = 1

#: Phrase-count bounds the prompt asks for (3-5) and the parser enforces.
MIN_PHRASES = 3
MAX_PHRASES = 5

#: Cap on phrase length: a 2-4-word Russian query never needs more.
_MAX_PHRASE_LEN = 120


class PromptTemplateError(Exception):
    """Raised when the wordstat prompt template is missing or unreadable."""


class InvalidPhrasesResponseError(Exception):
    """Raised by :func:`parse_phrases_response` on unparseable model output.

    The validate stage counts these (``rejected`` counter) and moves on to
    the next cluster; the rejected cluster stays unvalidated until a later
    run.
    """


def _load_template() -> str:
    """Return the wordstat prompt template shipped with the package."""
    try:
        return (
            resources.files("idea_finder.llm")
            .joinpath("prompts/wordstat_phrases.md")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError) as e:
        msg = "wordstat prompt template idea_finder/llm/prompts/wordstat_phrases.md is missing"
        raise PromptTemplateError(msg) from e


def load_wordstat_template() -> str:
    """Return the raw wordstat prompt template (the versioned artifact).

    The validate stage registers this exact text under
    ``(WORDSTAT_PROMPT_NAME, WORDSTAT_PROMPT_VERSION)`` in the
    ``prompt_version`` table once per run, so runs stay reproducible
    against the shipped prompt source.
    """
    return _load_template()


def render_wordstat_prompt(cluster_summary: str) -> str:
    """Render the wordstat-phrases prompt for one cluster summary.

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
        msg = "failed to compile wordstat prompt template"
        raise PromptTemplateError(msg) from e
    return template.render(cluster_summary=cluster_summary)


@dataclass(frozen=True, slots=True)
class PhrasesAnswer:
    """A validated phrase list, ready for Wordstat queries."""

    phrases: tuple[str, ...]


def parse_phrases_response(answer: str) -> PhrasesAnswer:
    """Parse and validate a model phrases answer.

    Raises:
        InvalidPhrasesResponseError: On unparseable JSON, a missing/empty
            or non-string phrase list, wrong phrase count (the prompt asks
            for 3-5), blank phrases, or phrases that are not real search
            queries (overlong).
    """
    try:
        data: object = json.loads(answer)
    except json.JSONDecodeError as e:
        msg = "wordstat phrases answer is not valid JSON"
        raise InvalidPhrasesResponseError(msg) from e
    if not isinstance(data, dict):
        msg = "wordstat phrases answer must be a JSON object"
        raise InvalidPhrasesResponseError(msg)
    raw_phrases = data.get("phrases")
    if not isinstance(raw_phrases, list) or not raw_phrases:
        msg = "wordstat phrases answer has no phrases list"
        raise InvalidPhrasesResponseError(msg)
    phrases: list[str] = []
    for raw in raw_phrases:
        if not isinstance(raw, str) or not raw.strip():
            msg = "every phrase must be a non-empty string"
            raise InvalidPhrasesResponseError(msg)
        phrases.append(raw.strip())
    if not MIN_PHRASES <= len(phrases) <= MAX_PHRASES:
        msg = f"expected {MIN_PHRASES}-{MAX_PHRASES} phrases, got {len(phrases)}"
        raise InvalidPhrasesResponseError(msg)
    if any(len(phrase) > _MAX_PHRASE_LEN for phrase in phrases):
        msg = "phrase exceeds the maximum query length"
        raise InvalidPhrasesResponseError(msg)
    return PhrasesAnswer(phrases=tuple(dict.fromkeys(phrases)))


class FakeWordstatLlmClient:
    """Deterministic phrases-answer client for validate-stage mock runs.

    Same contract as the score stage's ``FakeScoreLlmClient`` (owner rule:
    mock mode = fake provider answers stage-shaped JSON). Derives 3
    phrases from a hash of the rendered prompt so different clusters get
    different but stable phrase sets; two identical prompts yield
    identical answers.
    """

    def __init__(self, *, model: str = "fake", provider_name: str = "fake") -> None:
        self.model = model
        self.provider_name = provider_name

    def complete(self, prompt: str) -> Completion:
        """Return the deterministic phrases JSON answer for ``prompt``."""
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        seeds = (digest[0] % 7, digest[1] % 11, digest[2] % 13)
        stems = (
            ("как исправить", "не работает", "почему сломался"),
            ("как убрать", "перестал работать", "что делать если сломался"),
            ("как настроить", "ошибка при запуске", "не открывается"),
        )
        chosen = stems[seeds[0] % len(stems)]
        phrases = [f"{stem} проблема {seed}" for stem, seed in zip(chosen, seeds, strict=True)]
        text = json.dumps({"phrases": phrases}, ensure_ascii=False)
        return Completion(
            text=text, prompt_tokens=0, completion_tokens=0,
            model=self.model, provider_name=self.provider_name,
        )
