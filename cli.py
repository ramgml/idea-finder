"""Command-line entry point for the idea-finder pipeline.

Subcommands mirror the architecture (context/SYSTEM_DESIGN.md):
collect|extract|cluster|score|run|status|captcha|llm-test. Stage implementations
land with tasks B/C/D; each command here documents its purpose until then.
"""

import argparse
import logging
import sys

from idea_finder import __version__

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
    print(args.purpose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
