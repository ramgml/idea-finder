"""End-to-end test for ``cli.py llm-test`` against a fresh embedded postgres.

Runs the real CLI as a subprocess (module-scoped fresh ``PGDATA_DIR``), plants
a fake provider via a tiny in-process snippet executed with the same env, then
asserts the green exit and the report contents on stdout (acceptance DoD).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from idea_finder.db.bootstrap_pgserver import ensure_pgserver


def run_cli(*arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "cli.py", *arguments],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.fixture(scope="module")
def pgdata_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Fresh cluster dir for the llm-test subprocess; server stopped at exit."""
    data_dir = tmp_path_factory.mktemp("llmcli-pg") / "pg"
    patch = pytest.MonkeyPatch()
    patch.setenv("PGDATA_DIR", str(data_dir))
    yield data_dir
    ensure_pgserver(data_dir).stop()
    patch.undo()


def _run_cli_with_pgdata(pgdata_dir: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin", "PGDATA_DIR": str(pgdata_dir)}
    return run_cli(*arguments, env=env)


def test_llm_test_without_any_provider_fails_with_exit_1(pgdata_dir: Path) -> None:
    """No active provider -> LlmError -> logging.exception + exit 1."""
    result = _run_cli_with_pgdata(pgdata_dir, "llm-test")
    assert result.returncode == 1
    assert "LLM call failed" in result.stderr
    assert "no active LLM provider" in result.stderr


def test_llm_test_on_fake_provider_is_green(pgdata_dir: Path) -> None:
    """Fake provider active -> exit 0 with provider name and report (DoD)."""
    # The worktree root the CLI runs from; needed for the in-process setup.
    worktree = Path(__file__).resolve().parents[1]
    # Plant two providers (fake + openai_compat stub) with the fake active,
    # using the project's own venv python against the same PGDATA_DIR.
    setup_code = (
        "from idea_finder.db.bootstrap_pgserver import ensure_pgserver\n"
        "from idea_finder.db.migrate import apply_migrations\n"
        "from idea_finder.db.repo import upsert_llm_provider\n"
        "h = ensure_pgserver()\n"
        "c = h.get_conn()\n"
        "apply_migrations(c)\n"
        "upsert_llm_provider(c, 'real-stub', 'openai_compat',"
        " 'http://localhost:1', 'stub-model', 'k', is_active=False)\n"
        "upsert_llm_provider(c, 'mock', 'fake', '', 'mock-model', '', is_active=True)\n"
    )
    setup = subprocess.run(
        [sys.executable, "-c", setup_code],
        capture_output=True,
        text=True,
        check=False,
        cwd=worktree,
        env={"PATH": "/usr/bin:/bin", "PGDATA_DIR": str(pgdata_dir)},
    )
    assert setup.returncode == 0, setup.stderr
    result = _run_cli_with_pgdata(pgdata_dir, "llm-test")
    assert result.returncode == 0, result.stderr
    assert "provider: mock" in result.stdout
    assert "model: mock-model" in result.stdout
    assert "answer: " in result.stdout
    assert "tokens: prompt=0 completion=0" in result.stdout
    assert "cost: 0.0000" in result.stdout
