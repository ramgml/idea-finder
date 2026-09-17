"""Command-line entry point for the idea-finder pipeline.

Subcommands mirror the architecture (context/SYSTEM_DESIGN.md):
collect|extract|cluster|score|run|status|captcha|llm-test. Pipeline stages
are wired to :mod:`idea_finder.core.pipeline`; captcha and llm-test only
print their purpose until fetch/browser.py and the LLM client land.
"""

import argparse
import logging
import sys
from collections.abc import Callable

from psycopg import Connection

from idea_finder import __version__
from idea_finder.core import pipeline
from idea_finder.db.bootstrap_pgserver import DbError, ensure_pgserver

LOGGER = logging.getLogger(__name__)

CommandHelp = tuple[str, str]


# (name, one-line purpose) for every CLI subcommand, in help order.
COMMANDS: tuple[CommandHelp, ...] = (
    ("collect", "Fetch new posts from all enabled sources into raw_post."),
    ("extract", "Extract pains from collected posts via the active LLM provider."),
    ("cluster", "Embed pains and group them into clusters."),
    ("score", "Score clusters against the Russian-market rubric."),
    ("run", "Run the full pipeline: collect, extract, cluster, score."),
    ("status", "Show table counters and the state of the latest run."),
    ("captcha", "Open a headed browser for manual captcha solving on a domain."),
    ("llm-test", "Smoke-test the active LLM provider with a fixed prompt."),
)

_STAGES: dict[str, Callable[[Connection], pipeline.StageStats]] = {
    "collect": pipeline.run_collect,
    "extract": pipeline.run_extract,
    "cluster": pipeline.run_cluster,
    "score": pipeline.run_score,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="idea-finder pipeline control CLI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO).",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, purpose in COMMANDS:
        subparser = subparsers.add_parser(name, help=purpose, description=purpose)
        if name == "captcha":
            subparser.add_argument("domain", help="domain to open in the headed browser.")
        subparser.set_defaults(purpose=purpose)
    return parser


def _open_connection() -> Connection:
    """Ensure the embedded postgres is up and return a connection to it."""
    handle = ensure_pgserver()
    return handle.get_conn()


def _run_stage_command(stage_names: tuple[str, ...]) -> int:
    """Run the named stages in order on one connection, print each stats dict."""
    with _open_connection() as conn:
        for name in stage_names:
            stats = _STAGES[name](conn)
            print(f"{name}: {stats}")
    return 0


def _cmd_status(conn: Connection) -> int:
    """Print per-table counters over direct SQL (read-only status view)."""
    table_names = (
        "source",
        "raw_post",
        "pain",
        "cluster",
        "score",
        "run",
        "prompt_version",
        "llm_provider",
    )
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT c.relname
            FROM pg_class AS c
            JOIN pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relkind = 'r'
            ORDER BY c.relname
            """
        )
        existing = {row[0] for row in cursor.fetchall()}
    if not existing:
        print("status: database schema not applied")
        return 0
    for table in table_names:
        if table not in existing:
            print(f"{table}: (table absent)")
            continue
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {table}")
            count = cursor.fetchone()
        print(f"{table}: {count[0] if count is not None else 0}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the requested command, and return the exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    logging.getLogger().setLevel(args.log_level)
    if args.command is None:
        _build_parser().print_help()
        return 0
    LOGGER.debug("dispatching command=%s", args.command)
    try:
        if args.command in _STAGES:
            return _run_stage_command((args.command,))
        if args.command == "run":
            return _run_stage_command(("collect", "extract", "cluster", "score"))
        if args.command == "status":
            with _open_connection() as conn:
                return _cmd_status(conn)
    except DbError:
        LOGGER.exception("command=%s failed", args.command)
        return 1
    # captcha / llm-test: purpose-only until their subsystems land.
    print(args.purpose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
