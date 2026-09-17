"""Bootstrap the embedded PostgreSQL server used by idea-finder.

The server is managed by ``pgserver`` and is reachable only through a
unix-domain socket: postgres is started with no TCP listeners, and the DSN
handed out by :class:`PgHandle` always names a socket directory via
``?host=<dir>``, never a host/port pair.

Usage from other subsystems (T297/T298)::

    from idea_finder.db.bootstrap_pgserver import ensure_pgserver, pgserver

    handle = ensure_pgserver()          # idempotent; reuses a live instance
    with handle.get_conn() as conn:     # psycopg connection
        ...
    with pgserver() as h:               # scoped variant, stops on exit
        ...
"""

from __future__ import annotations

import logging
import os
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Self

import pgserver
import psycopg
from psycopg import Connection, sql

logger = logging.getLogger(__name__)

#: Name of the application database created inside the cluster.
DB_NAME = "idea_finder"

#: Socket files live at ``<socket_dir>/.s.PGSQL.<port>``.
_SOCKET_NAME_TEMPLATE = ".s.PGSQL.{port}"

#: How long to wait for the socket to accept connections after a crash
#: recovery start before giving up (seconds).
_CONNECT_TIMEOUT = 30.0


class DbError(Exception):
    """Raised when the embedded PostgreSQL server cannot be provided."""


def _repo_root() -> Path:
    """Return the repository root (the directory containing ``idea_finder/``)."""
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    """Resolve the cluster directory: ``PGDATA_DIR`` env or ``<repo>/data/pg``."""
    env_dir = os.environ.get("PGDATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (_repo_root() / "data" / "pg").resolve()


def _wait_for_socket(socket_path: Path, timeout: float) -> None:
    """Block until the unix socket accepts connections or the timeout expires.

    After a crash (stale ``postmaster.pid``) postgres spends some time in WAL
    recovery before it listens again, so callers must wait here instead of
    failing on the first ``Connection refused``.
    """
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                sock.connect(str(socket_path))
            return
        except OSError as e:
            last_error = e
            time.sleep(0.2)
    msg = f"unix socket {socket_path} did not accept connections within {timeout}s"
    raise DbError(msg) from last_error


def _socket_dir_from_dsn(dsn: str) -> Path:
    """Extract the socket directory from a ``?host=<dir>`` query parameter."""
    marker = "?host="
    index = dsn.rfind(marker)
    if index == -1:
        msg = f"dsn does not carry a unix-socket host parameter: {dsn!r}"
        raise DbError(msg)
    return Path(dsn[index + len(marker) :])


def _validate_unix_dsn(dsn: str) -> None:
    """Assert the DSN targets a unix socket only (no TCP host/port).

    Enforces the project invariant that postgres is never exposed over TCP:
    the DSN must name a socket directory and must not contain a TCP host or
    an explicit port component.
    """
    if "host=localhost" in dsn or "host=127.0.0.1" in dsn or "host=::1" in dsn:
        msg = f"dsn must not name a TCP host: {dsn!r}"
        raise DbError(msg)
    # A TCP port in a DSN always appears as "host=<something>:<port>" or an
    # explicit "port=" parameter; pgserver's socket DSN has neither.
    if "port=" in dsn:
        msg = f"dsn must not carry a TCP port: {dsn!r}"
        raise DbError(msg)
    if "?host=" not in dsn:
        msg = f"dsn must carry a unix-socket host parameter: {dsn!r}"
        raise DbError(msg)


def _create_database(conn: psycopg.Connection, db_name: str) -> None:
    """Create ``db_name`` if it does not exist yet (idempotent)."""
    exists = conn.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
    ).fetchone()
    if exists is not None:
        return
    # Identifiers cannot be passed as query parameters; use sql.Identifier to
    # quote them safely. The surrounding statement is a fixed literal.
    conn.execute(
        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
    )
    logger.info("created database %s", db_name)


def _evict_stale_instance(pgdata: Path) -> None:
    """Drop pgserver's cached handle if its server process is gone.

    ``PostgresServer`` caches one instance per pgdata directory. After a
    ``kill -9`` of the postmaster the cached object still refers to the dead
    process, and ``get_server`` would hand it back instead of performing the
    crash-recovery start; evict it so the restart path runs.
    """
    cached = pgserver.PostgresServer._instances.get(pgdata)
    if cached is None:
        return
    pid = cached.get_pid()
    if pid is None:
        del pgserver.PostgresServer._instances[pgdata]
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        logger.info("postmaster %s is gone; evicting stale pgserver handle", pid)
        del pgserver.PostgresServer._instances[pgdata]
    except PermissionError:
        # Process exists but belongs to another user: leave it alone.
        return


class PgHandle:
    """Handle to the embedded postgres cluster in a data directory.

    Wraps :class:`pgserver.PostgresServer`, guaranteeing unix-socket-only
    connectivity and the presence of the application database.
    """

    def __init__(self, server: pgserver.PostgresServer, database: str) -> None:
        self._server = server
        self._database = database
        dsn = server.get_uri(database)
        _validate_unix_dsn(dsn)
        self._dsn = dsn

    @property
    def data_dir(self) -> Path:
        """Cluster directory (pgdata) of this handle."""
        return self._server.pgdata

    def dsn(self) -> str:
        """Return the unix-socket-only DSN for the application database."""
        return self._dsn

    def socket_path(self) -> Path:
        """Return the filesystem path of the postgres unix socket."""
        socket_dir = _socket_dir_from_dsn(self._dsn)
        port = self._server.get_postmaster_info().port or 5432
        return socket_dir / _SOCKET_NAME_TEMPLATE.format(port=port)

    def get_conn(self) -> Connection:
        """Open a new psycopg connection to the application database.

        The caller owns the connection and must close it (or use it as a
        context manager). Connections are only possible over the unix socket.
        """
        socket_path = self.socket_path()
        if not socket_path.exists():
            msg = f"postgres socket is missing: {socket_path}"
            raise DbError(msg)
        try:
            conn = psycopg.connect(self._dsn)
        except psycopg.Error as e:
            msg = f"failed to connect over unix socket {socket_path}"
            raise DbError(msg) from e
        # Belt and braces: refuse a connection that is not over a unix socket.
        # A unix-socket connection reports the socket directory as host;
        # anything else (TCP address) is a contract violation.
        if not conn.info.host.startswith("/"):
            conn.close()
            msg = f"connection did not use the unix socket: host={conn.info.host!r}"
            raise DbError(msg)
        return conn

    def stop(self) -> None:
        """Stop the postgres server if the process is still alive."""
        pid = self._server.get_pid()
        if pid is None:
            return
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        self._server.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()


def ensure_pgserver(data_dir: Path | None = None) -> PgHandle:
    """Start (or reuse) the embedded postgres cluster and return a handle.

    Idempotent: if the server in ``data_dir`` is already running it is reused
    as is. After a hard kill of the postmaster the next call performs a
    crash-recovery start (WAL replay) and returns a working handle.

    Args:
        data_dir: cluster directory; defaults to ``PGDATA_DIR`` env, then
            ``<repo>/data/pg``.

    Returns:
        A :class:`PgHandle` wired to the ``idea_finder`` database, creating
        that database first if the cluster is fresh.

    Raises:
        DbError: if the server cannot be started or verified.
    """
    pgdata = (data_dir if data_dir is not None else default_data_dir()).resolve()
    pgdata.parent.mkdir(parents=True, exist_ok=True)
    _evict_stale_instance(pgdata)

    # cleanup_mode=None: the server must outlive this process (CLI exits,
    # dashboard subprocesses attach later); shutdown is explicit via stop().
    try:
        server = pgserver.get_server(pgdata, cleanup_mode=None)
        server.ensure_postgres_running()
    except Exception as e:
        msg = f"failed to start embedded postgres in {pgdata}"
        raise DbError(msg) from e

    handle = PgHandle(server, DB_NAME)
    socket_path = handle.socket_path()
    _wait_for_socket(socket_path, _CONNECT_TIMEOUT)

    # CREATE DATABASE cannot run inside a transaction block and the
    # application database may not exist yet, so connect to the cluster's
    # maintenance database ("postgres") for the creation step.
    try:
        with psycopg.connect(server.get_uri("postgres"), autocommit=True) as conn:
            _create_database(conn, DB_NAME)
    except (psycopg.Error, DbError) as e:
        msg = f"failed to ensure database {DB_NAME!r} exists"
        raise DbError(msg) from e
    logger.info("embedded postgres ready at %s", handle.dsn())
    return handle


@contextmanager
def pgserver_session(data_dir: Path | None = None) -> Iterator[PgHandle]:
    """Context manager yielding a :class:`PgHandle`; stops postgres on exit.

    The server is stopped even on error, so scripts and tests can scope their
    own cluster lifecycle. Reuses a live instance without restarting it.
    """
    handle = ensure_pgserver(data_dir)
    try:
        yield handle
    finally:
        handle.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    handle = ensure_pgserver()
    try:
        with handle.get_conn() as conn:
            one = conn.execute("SELECT 1").fetchone()
        if one is None:
            msg = "SELECT 1 returned no rows"
            raise DbError(msg)
        print("SELECT 1 ->", one[0])
        print("dsn:", handle.dsn())
        print("data_dir:", handle.data_dir)
    finally:
        handle.stop()
