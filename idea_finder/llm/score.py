"""Score-stage prompt rendering and LLM response validation.

The score stage (core/pipeline.py, T310) renders the v1 rubric prompt from
``idea_finder/llm/prompts/score.md``, sends it through an
:class:`~idea_finder.llm.client.LlmClient`, and validates the answer with
:func:`parse_score_response`. The rubric itself (weights, demand-boost,
grounding rules) lives in ``context/SCORING.md`` — this module enforces the
machine-checkable part: score in range, non-empty rationale, verbatim quotes.

Validation never raises for bad model output: rejections are counted by the
stage (``rejected`` counter), so one malformed answer does not kill a run.
Only programmatic errors (missing template) raise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

from jinja2 import Environment, StrictUndefined

from idea_finder.llm.client import Completion

__all__ = [
    "SCORE_PROMPT_NAME",
    "SCORE_PROMPT_VERSION",
    "ClusterSummary",
    "FakeScoreLlmClient",
    "InvalidScoreResponseError",
    "PromptTemplateError",
    "load_score_template",
    "parse_score_response",
    "render_score_prompt",
]

#: Name the score prompt is registered under (prompt_version table).
SCORE_PROMPT_NAME = "score"

#: Version of the prompt shipped in this module.
SCORE_PROMPT_VERSION = 1


class PromptTemplateError(Exception):
    """Raised when the score prompt template is missing or unreadable."""


class InvalidScoreResponseError(Exception):
    """Raised by :func:`parse_score_response` on unparseable model output.

    The score stage counts these (``rejected`` counter) and moves on to the
    next cluster; the rejected cluster stays without a score.
    """


def _load_template() -> str:
    """Return the score prompt template shipped with the package."""
    try:
        return (
            resources.files("idea_finder.llm")
            .joinpath("prompts/score.md")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError) as e:
        msg = "score prompt template idea_finder/llm/prompts/score.md is missing"
        raise PromptTemplateError(msg) from e


def load_score_template() -> str:
    """Return the raw score prompt template (the versioned artifact).

    The score stage registers this exact text under
    ``(SCORE_PROMPT_NAME, SCORE_PROMPT_VERSION)`` in the ``prompt_version``
    table once per run, so runs stay reproducible against the shipped
    rubric source.
    """
    return _load_template()


@dataclass(frozen=True, slots=True)
class ClusterSummary:
    """Facts about one cluster the prompt scores.

    ``bodies`` are the cluster's pain bodies (quote grounding checks run
    against them); ``kinds`` is the kind_mix; ``sources`` counts distinct
    source feeds; ``first_seen``/``last_seen`` bound the post ages.
    """

    size: int
    kinds: dict[str, int]
    sources: int
    first_seen: str
    last_seen: str
    bodies: list[str]

    def rendered(self) -> str:
        """Render the summary as the prompt's human-readable block."""
        lines = [
            f"Размер кластера: {self.size} постов",
            f"Типы постов (kind_mix): {json.dumps(self.kinds, ensure_ascii=False)}",
            f"Число разных источников: {self.sources}",
            f"Первый пост: {self.first_seen}",
            f"Последний пост: {self.last_seen}",
            "",
            "Боли кластера:",
        ]
        lines.extend(f"- {body}" for body in self.bodies)
        return "\n".join(lines)


def render_score_prompt(summary: ClusterSummary) -> str:
    """Render the score v1 prompt for one cluster summary.

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
        msg = "failed to compile score prompt template"
        raise PromptTemplateError(msg) from e
    return template.render(cluster_summary=summary.rendered())


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """A validated score response, ready for :class:`repo.insert_score`."""

    total: float
    rationale_md: str
    quotes: list[str]


def parse_score_response(answer: str, pain_bodies: list[str]) -> ScoreAnswer:
    """Parse and validate a model score answer against the cluster's pains.

    Raises:
        InvalidScoreResponseError: On unparseable JSON, an out-of-range
            score, an empty rationale, missing quotes, or a quote that is
            not a verbatim substring of any pain body (hallucinated
            evidence — same grounding rule as the extract stage).
    """
    try:
        data = json.loads(answer)
    except json.JSONDecodeError as e:
        msg = "score answer is not valid JSON"
        raise InvalidScoreResponseError(msg) from e
    if not isinstance(data, dict):
        msg = "score answer must be a JSON object"
        raise InvalidScoreResponseError(msg)

    total = data.get("score")
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        msg = "score must be a number"
        raise InvalidScoreResponseError(msg)
    if not 0 <= float(total) <= 100:
        msg = f"score out of range [0, 100]: {total}"
        raise InvalidScoreResponseError(msg)

    rationale = data.get("rationale_md")
    if not isinstance(rationale, str) or not rationale.strip():
        msg = "rationale_md must be a non-empty string"
        raise InvalidScoreResponseError(msg)

    quotes = data.get("quotes")
    if not isinstance(quotes, list) or not quotes:
        msg = "quotes must be a non-empty list"
        raise InvalidScoreResponseError(msg)
    cleaned: list[str] = []
    for quote in quotes:
        if not isinstance(quote, str):
            msg = "every quote must be a string"
            raise InvalidScoreResponseError(msg)
        if not any(quote in body for body in pain_bodies):
            msg = "quote is not a verbatim substring of any pain body"
            raise InvalidScoreResponseError(msg)
        cleaned.append(quote)

    return ScoreAnswer(
        total=float(total), rationale_md=rationale, quotes=cleaned
    )


def llm_score_to_storage(llm_score: float) -> float:
    """Convert the rubric scale (0-100, context/SCORING.md) to storage (0-10).

    Two scales exist by design: the LLM rubric answers 0-100 per
    SCORING.md, while the DB column ``score.total`` is CHECK-bound to
    0-10 (migrations/001_init.sql) and the dashboard renders 0-10
    (T311). The stage divides by 10 at write time; the sanity gate
    operates on the LLM scale before conversion.
    """
    return round(llm_score / 10, 2)


class FakeScoreLlmClient:
    """Deterministic JSON-answer client for score-stage acceptance runs.

    Same contract as :class:`~idea_finder.llm.extract.FakeExtractLlmClient`
    (per the owner's 2026-09-17 UPDATE: scoring runs on the fake provider
    until a real one is configured). Answers with a mid-range score, a
    template rationale, and the cluster's first pain body prefix as the
    quote — a verbatim substring, so the grounding validator accepts it.
    Two identical prompts yield identical answers.
    """

    def __init__(self, *, model: str = "fake", provider_name: str = "fake") -> None:
        self.model = model
        self.provider_name = provider_name

    def complete(self, prompt: str) -> Completion:
        """Return the deterministic score JSON answer for ``prompt``."""
        # Ground the quote in a pain body carried by the prompt itself:
        # the summary block renders bodies as "- <body>" lines.
        quote = "боль"
        for line in prompt.splitlines():
            candidate = line.removeprefix("- ").strip()
            if line.startswith("- ") and candidate:
                quote = candidate[:40]
                break
        text = json.dumps(
            {
                "score": 60,
                "rationale_md": (
                    "Кластер показывает устойчивый спрос: посты повторяются "
                    "в нескольких источниках, аудитория платёжеспособна. "
                    "Готовых решений мало, MVP под силу одному разработчику. "
                    "Основной риск — регуляторика, поэтому скор средний."
                ),
                "quotes": [quote],
            },
            ensure_ascii=False,
        )
        return Completion(
            text=text, prompt_tokens=0, completion_tokens=0,
            model=self.model, provider_name=self.provider_name,
        )
