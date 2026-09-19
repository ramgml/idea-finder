"""Read-only dashboard queries for the Run ("Сбор") page (task 313).

Schema ownership note (task 313 constraint): ``db/repo.py`` is shared with
other concurrent streams, so this module keeps the run-page read queries the
web layer needs here instead of editing repo.py. TODO(repo): move these
functions into ``db/repo.py`` once the concurrent streams merge; keep the
signatures and SQL as-is.

Everything here is read-only SELECTs; the dashboard never writes. The one
write-shaped action on the page — launching a pipeline run — goes through
``cli.py run`` as a subprocess, exactly like the manual CLI launch.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Final, LiteralString

from psycopg import Connection

#: Pipeline stages in execution order, as recorded in ``run.stages_json``.
#: The page renders them in this fixed order regardless of JSON key order.
STAGES: Final[tuple[str, ...]] = ("collect", "extract", "cluster", "score")

#: A run counts as active while any recorded stage sits in ``running``. The
#: UI blocks a second launch while such a run exists; the CLI itself finishes
#: its stage runs before the subprocess exits, so leftover ``running`` rows
#: mean a genuinely in-flight (or crashed) process.
_RUNNING_STATUSES: Final[frozenset[str]] = frozenset({"running"})


_LIST_RUNS_SQL: Final[LiteralString] = """
    SELECT id::text,
           started_at,
           finished_at,
           stages_json,
           stats_json,
           cost,
           prompt_version_id::text
    FROM run
    ORDER BY started_at DESC, id
    LIMIT %s
"""


@dataclass(frozen=True, slots=True)
class RunRow:
    """One pipeline execution as shown on the Run and Health pages."""

    id: str
    started_at: datetime
    #: None while the run is still in flight.
    finished_at: datetime | None
    #: Stage name -> status (e.g. ``{"collect": "done"}``).
    stages: dict[str, str]
    #: Per-run counters (e.g. ``{"extracted": 12, "cost_micro": 90}``).
    stats: dict[str, int]
    #: Accumulated LLM spend in rubles.
    cost: float
    #: Pinned prompt version (NULL for legacy runs).
    prompt_version_id: str | None

    @property
    def is_running(self) -> bool:
        """True while any recorded stage has a ``running`` status."""
        return any(status in _RUNNING_STATUSES for status in self.stages.values())


@dataclass(frozen=True, slots=True)
class StageProgress:
    """Render-ready progress for one pipeline stage within one run."""

    name: str
    #: Canonical status: ``running`` | ``done`` | ``error`` (or ``—`` unset).
    status: str
    #: Human-readable metric line, e.g. ``извлечено 12, брак 2``; ``—`` when
    #: the stage recorded no counters this run.
    metrics: str


def _load_json_mapping(raw: object) -> dict[str, str]:
    """Decode a jsonb cell that psycopg may deliver as dict or as str.

    Values are stringified: ``stages_json`` holds status strings, and
    ``stats_json`` values are converted to int by the caller on use.
    """
    decoded: object = raw if isinstance(raw, dict) else json.loads(str(raw))
    if not isinstance(decoded, dict):
        msg = f"run jsonb cell is not an object: {raw!r}"
        raise TypeError(msg)
    return {str(key): str(value) for key, value in decoded.items()}


def list_runs(conn: Connection, limit: int = 20) -> list[RunRow]:
    """Return the most recent runs, newest first.

    ``limit`` bounds the page: the history table and the health page never
    need more than the last few runs.
    """
    rows = conn.execute(_LIST_RUNS_SQL, (limit,)).fetchall()
    runs: list[RunRow] = []
    for run_id, started_at, finished_at, stages_raw, stats_raw, cost, prompt_version_id in rows:
        stages = _load_json_mapping(stages_raw)
        stats_raw_map = _load_json_mapping(stats_raw)
        stats = {key: int(value) for key, value in stats_raw_map.items()}
        runs.append(
            RunRow(
                id=str(run_id),
                started_at=started_at,
                finished_at=None if finished_at is None else finished_at,
                stages=stages,
                stats=stats,
                cost=float(cost),
                prompt_version_id=None if prompt_version_id is None else str(prompt_version_id),
            )
        )
    return runs


def has_active_run(conn: Connection) -> bool:
    """True while a run with a ``running`` stage exists.

    This is the launch-lock check: the Run page disables its button when
    this returns True, so a second subprocess cannot start alongside the
    first. Only the current pipeline statuses produce a lock; finished or
    errored runs never do.
    """
    row = conn.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM run
            WHERE stages_json @> '{"collect": "running"}'
               OR stages_json @> '{"extract": "running"}'
               OR stages_json @> '{"cluster": "running"}'
               OR stages_json @> '{"score": "running"}'
        )
        """
    ).fetchone()
    return bool(row[0]) if row is not None else False


#: Which run.stats counters describe each stage, in render order. Keys are
#: exact ``run.stats_json`` names written by the stages (pipeline.py).
#:
#: Collect (T341 P2-2): run_collect writes ``collected``/``skipped``/
#: ``warnings`` (the old "rows" key no longer exists since T335) — a
#: stale key here renders the Сбор stage with zero counters.
_STAGE_METRIC_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "collect": ("collected", "skipped", "warnings"),
    "extract": ("processed", "extracted", "failed", "llm_errors", "hallucinated"),
    "cluster": ("pains", "clusters_new", "clusters_merged", "singletons"),
    "score": ("clusters", "scored", "rejected", "sanity_flag"),
}

#: Russian labels for the counter keys above.
_METRIC_LABELS: Final[dict[str, str]] = {
    "collected": "собрано",
    "skipped": "дубликаты URL",
    "warnings": "источники с ошибками",
    "processed": "обработано",
    "extracted": "извлечено",
    "failed": "брак",
    "llm_errors": "ошибки LLM",
    "hallucinated": "галлюцинации",
    "pains": "болей",
    "clusters_new": "новых кластеров",
    "clusters_merged": "присоединено",
    "singletons": "синглтонов",
    "clusters": "кластеров",
    "scored": "оценено",
    "rejected": "отклонено",
    "sanity_flag": "узкий разброс",
}

_STAGE_LABELS: Final[dict[str, str]] = {
    "collect": "Сбор",
    "extract": "Извлечение",
    "cluster": "Кластеризация",
    "score": "Скоринг",
}

_STATUS_LABELS: Final[dict[str, str]] = {
    "running": "выполняется",
    "done": "готово",
    "error": "ошибка",
}

#: Placeholder for a stage with no status/counters in the current run.
_NOT_STARTED = "—"


def stage_progress(run: RunRow) -> list[StageProgress]:
    """Build per-stage progress lines for one run, in pipeline order.

    Metrics come from the run's ``stats_json`` (the stage counters the
    pipeline records); stages the run has not touched render as "не
    запускалась".
    """
    progress: list[StageProgress] = []
    for stage in STAGES:
        status = run.stages.get(stage)
        keys = _STAGE_METRIC_KEYS.get(stage, ())
        metrics = ", ".join(
            f"{_METRIC_LABELS[key]} {run.stats[key]}" for key in keys if key in run.stats
        )
        if status is None:
            rendered_status = f"{_NOT_STARTED} (не запускалась)"
        else:
            label = _STATUS_LABELS.get(status, status)
            rendered_status = label
        progress.append(
            StageProgress(
                name=_STAGE_LABELS.get(stage, stage),
                status=rendered_status,
                metrics=metrics or _NOT_STARTED,
            )
        )
    return progress


def run_command() -> list[str]:
    """Return the subprocess argv that launches a full pipeline run.

    Same entry path as the operator CLI (``uv run python cli.py run``):
    cwd is set to the repository root by the caller, so the child resolves
    the embedded postgres and the package exactly like a manual launch.
    """
    return [sys.executable, "cli.py", "run"]
