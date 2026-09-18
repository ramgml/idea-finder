"""End-to-end tests for cli.py stage commands against a fresh embedded postgres.

Each test runs the real CLI as a subprocess with ``PGDATA_DIR`` pointed at a
fresh cluster directory (module-scoped fixture): no migrations are applied,
which is exactly the "empty database" state the status/stage contract covers.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from idea_finder.db.bootstrap_pgserver import ensure_pgserver

STAGES = ("collect", "extract", "cluster", "score")


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "cli.py", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def pgdata_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Fresh cluster dir for every CLI subprocess; server stopped at exit.

    A fresh ``PGDATA_DIR`` guarantees the schema is not applied: no
    migrations have ever run against it.
    """
    data_dir = tmp_path_factory.mktemp("cli-pg") / "pg"
    patch = pytest.MonkeyPatch()
    patch.setenv("PGDATA_DIR", str(data_dir))
    yield data_dir
    ensure_pgserver(data_dir).stop()
    patch.undo()


@pytest.mark.parametrize("stage", STAGES)
def test_stage_command_prints_skeleton_stats(pgdata_dir: Path, stage: str) -> None:
    del pgdata_dir  # only ensures PGDATA_DIR is set for the subprocess
    result = run_cli(stage)
    assert result.returncode == 0, result.stderr
    if stage == "extract":
        # Real stage: empty database -> zero counters, no LLM provider needed.
        assert "extract: {'processed': 0," in result.stdout
    elif stage == "cluster":
        # Real stage: empty database -> zero counters, no embeddings needed.
        assert "cluster: {'pains': 0," in result.stdout
    elif stage == "score":
        # Real stage (T310): empty database -> zero counters, no run row,
        # no LLM provider needed.
        assert "score: {'clusters': 0, 'scored': 0," in result.stdout
    else:
        assert f"{stage}: {{'rows': 0}}" in result.stdout


def test_run_executes_all_stages_in_order(pgdata_dir: Path) -> None:
    del pgdata_dir
    result = run_cli("run")
    assert result.returncode == 0, result.stderr
    last4 = result.stdout.splitlines()[-4:]
    # extract, cluster and score are real now (own stats dicts); collect
    # is still a skeleton stage from the B-flow.
    assert last4[0] == "collect: {'rows': 0}"
    assert last4[1].startswith("extract: {'processed': 0,")
    assert last4[2].startswith("cluster: {'pains': 0,")
    assert last4[3].startswith("score: {'clusters': 0, 'scored': 0,")


def test_status_after_stages_reports_counters(pgdata_dir: Path) -> None:
    """Stage commands apply migrations idempotently, so status sees tables.

    Previously stages ran on a raw schema and ``status`` printed "schema
    not applied" on this fresh cluster; the extract stage needs
    prompt_version/run tables, so stage commands now migrate first.
    """
    del pgdata_dir
    result = run_cli("status")
    assert result.returncode == 0, result.stderr
    assert "schema not applied" not in result.stdout
    assert "source: 0" in result.stdout and "pain: 0" in result.stdout
