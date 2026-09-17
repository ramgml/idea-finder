"""Sequential SQL migration runner for the embedded postgres cluster.

Applies ``NNN_*.sql`` files from this package's ``migrations/`` directory in
lexicographic order, tracking applied versions in ``schema_migrations``. Each
migration runs inside a single transaction: either it fully applies or the
database is left untouched. Re-running ``apply_migrations`` is a no-op for
already-applied versions, so calling it at every startup is safe.

Usage (smoke test)::

    python -m idea_finder.db.migrate [data_dir]
"""

from __future__ import annotations

import importlib.resources
import logging
import re
import sys
from pathlib import Path
from typing import LiteralString, cast

from psycopg import Connection

from idea_finder.db.bootstrap_pgserver import DbError, ensure_pgserver

logger = logging.getLogger(__name__)

#: Pattern for migration file names: ``001_init.sql`` -> version 1.
_MIGRATION_FILE_RE = re.compile(r"^(?P<version>\d{3})_[a-z0-9_]+\.sql$")


class MigrationError(DbError):
    """Raised when a migration file is invalid or fails to apply."""


def _migrations_dir() -> Path:
    """Return the filesystem directory holding the packaged migrations."""
    traversable = importlib.resources.files("idea_finder.db") / "migrations"
    path = Path(str(traversable))
    if not path.is_dir():
        msg = f"migrations directory is missing: {path}"
        raise MigrationError(msg)
    return path


def discover_migrations() -> list[tuple[int, str, Path]]:
    """Return ``(version, name, path)`` sorted by version, strictly sequential.

    Versions must start at 001 and increase by one without gaps — a missing
    or duplicate number is a packaging bug, not something to run past.
    """
    migrations: list[tuple[int, str, Path]] = []
    for entry in sorted(_migrations_dir().iterdir()):
        match = _MIGRATION_FILE_RE.match(entry.name)
        if match is None:
            msg = f"unexpected file in migrations directory: {entry.name}"
            raise MigrationError(msg)
        migrations.append((int(match.group("version")), entry.name, entry))
    if not migrations:
        msg = "no migration files found"
        raise MigrationError(msg)
    for index, (version, name, _path) in enumerate(migrations):
        if version != index + 1:
            msg = (
                f"migration versions must be sequential from 001 without gaps; "
                f"got {name} at position {index + 1}"
            )
            raise MigrationError(msg)
    return migrations


def applied_versions(conn: Connection) -> set[int]:
    """Return the set of migration versions recorded as applied."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    integer PRIMARY KEY,
            applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {int(row[0]) for row in rows}


def apply_migrations(conn: Connection) -> list[int]:
    """Apply all pending migrations in order; return versions applied now.

    Idempotent: versions already present in ``schema_migrations`` are skipped.
    Each file executes in its own transaction. Already-`CREATE TABLE IF NOT
    EXISTS`-style SQL stays re-runnable even if a partial run left objects.
    """
    done = applied_versions(conn)
    newly_applied: list[int] = []
    for version, name, path in discover_migrations():
        if version in done:
            logger.debug("migration %03d already applied, skipping", version)
            continue
        # Migration files are repo-controlled static content, so the
        # LiteralString guarantee is upheld by construction.
        sql_text = cast(LiteralString, path.read_text(encoding="utf-8"))
        try:
            with conn.transaction():
                conn.execute(sql_text)
                conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (version,),
                )
        except Exception as e:
            msg = f"migration {name} failed"
            raise MigrationError(msg) from e
        logger.info("applied migration %s", name)
        newly_applied.append(version)
    conn.commit()
    return newly_applied


def current_version(conn: Connection) -> int | None:
    """Return the highest applied migration version, or None if none applied."""
    versions = applied_versions(conn)
    return max(versions, key=int) if versions else None


def _run(data_dir: Path | None) -> int:
    """CLI entry: ensure server, apply migrations twice, print table list."""
    handle = ensure_pgserver(data_dir)
    try:
        with handle.get_conn() as conn:
            apply_migrations(conn)
            second = apply_migrations(conn)
            logger.info("second apply returned %s (must be empty)", second)
            if second:
                msg = f"re-applying migrations was not a no-op: {second}"
                raise MigrationError(msg)
            rows = conn.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' ORDER BY table_name
                """
            ).fetchall()
            for (table_name,) in rows:
                print(table_name)
            versions = conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
            print("applied versions:", [v[0] for v in versions])
        return 0
    except Exception as e:
        logger.error("migration run failed: %s", e)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(_run(directory))
