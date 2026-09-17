"""Tests for the project skeleton: package import and CLI surface."""

import subprocess
import sys

import pytest

from idea_finder import __version__

EXPECTED_COMMANDS = (
    "collect",
    "extract",
    "cluster",
    "score",
    "run",
    "status",
    "captcha",
    "llm-test",
)


def test_package_import_exposes_docstring() -> None:
    assert "0.0.0" == __version__


def test_subpackages_importable() -> None:
    for name in ("sources", "fetch", "llm", "core", "db", "web"):
        module = pytest.importorskip(f"idea_finder.{name}")
        assert module.__doc__


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "cli.py", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help_lists_all_subcommands() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    for name in EXPECTED_COMMANDS:
        assert name in result.stdout


def test_cli_status_responds() -> None:
    result = run_cli("status")
    assert result.returncode == 0
    assert "counters" in result.stdout


def test_cli_captcha_requires_domain_argument() -> None:
    result = run_cli("captcha")
    assert result.returncode != 0
    assert "domain" in result.stderr


def test_cli_rejects_unknown_command() -> None:
    result = run_cli("bogus")
    assert result.returncode != 0
