"""Tests for the embedded postgres bootstrap (idea_finder.db.bootstrap_pgserver)."""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import psycopg
import pytest

from idea_finder.db.bootstrap_pgserver import (
    DB_NAME,
    DbError,
    PgHandle,
    _validate_unix_dsn,
    default_data_dir,
    ensure_pgserver,
    pgserver_session,
)


@pytest.fixture(scope="module")
def handle(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgHandle]:
    """Start one real embedded postgres cluster for the whole module."""
    data_dir = tmp_path_factory.mktemp("pgdata") / "pg"
    pg = ensure_pgserver(data_dir)
    yield pg
    pg.stop()


def test_select_one_over_unix_socket(handle: PgHandle) -> None:
    with handle.get_conn() as conn:
        row = conn.execute("SELECT 1").fetchone()
    assert row is not None
    assert row[0] == 1


def test_dsn_targets_unix_socket_only(handle: PgHandle) -> None:
    dsn = handle.dsn()
    assert "?host=" in dsn
    socket_dir = Path(dsn.split("?host=")[1])
    assert socket_dir.is_absolute()
    assert "host=localhost" not in dsn
    assert "port=" not in dsn
    assert ":5432/" not in dsn


def test_socket_file_exists(handle: PgHandle) -> None:
    assert handle.socket_path().exists()


@dataclass
class _FakeInfo:
    host: str


@dataclass
class _FakeTcpConnection:
    """Stand-in for a psycopg connection that reports a TCP host."""

    info: _FakeInfo = field(default_factory=lambda: _FakeInfo(host="127.0.0.1"))
    closed: bool = False

    def close(self) -> None:
        self.closed = True


def _assert_unix_socket_host(host: str) -> None:
    if not host.startswith("/"):
        msg = f"connection did not use the unix socket: host={host!r}"
        raise DbError(msg)


def test_connection_with_tcp_host_is_rejected() -> None:
    fake = _FakeTcpConnection()
    host = fake.info.host
    with pytest.raises(DbError, match="unix socket"):
        _assert_unix_socket_host(host)
    assert not fake.closed


def test_connection_with_socket_host_is_accepted() -> None:
    _assert_unix_socket_host("/tmp/sock-dir")
    _assert_unix_socket_host("/work/projects/idea-finder/data/pg")


def test_repeated_ensure_is_idempotent(handle: PgHandle) -> None:
    again = ensure_pgserver(handle.data_dir)
    assert again.data_dir == handle.data_dir
    assert again.dsn() == handle.dsn()
    # No extra cluster directory was spawned next to the existing one.
    siblings = sorted(p.name for p in handle.data_dir.parent.iterdir())
    assert siblings == ["pg"]


def test_database_persisted_across_reuse(handle: PgHandle) -> None:
    with handle.get_conn() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS bootstrap_probe (id int)")
        conn.commit()
    with ensure_pgserver(handle.data_dir).get_conn() as conn:
        conn.execute("INSERT INTO bootstrap_probe (id) VALUES (1)")
        conn.commit()
    with handle.get_conn() as conn:
        count = conn.execute("SELECT count(*) FROM bootstrap_probe").fetchone()
    assert count is not None
    assert count[0] == 1


def test_kill_nine_recovery(tmp_path_factory: pytest.TempPathFactory) -> None:
    """A hard-killed postmaster must be recoverable on the next ensure call."""
    data_dir = tmp_path_factory.mktemp("pgkill") / "pg"
    pg = ensure_pgserver(data_dir)
    with pg.get_conn() as conn:
        conn.execute("CREATE TABLE crash_probe (id int)")
        conn.commit()

    postmaster_pid = int((data_dir / "postmaster.pid").read_text().splitlines()[0])
    os.kill(postmaster_pid, signal.SIGKILL)
    time.sleep(0.5)

    revived = ensure_pgserver(data_dir)
    assert revived.socket_path().exists()
    with revived.get_conn() as conn:
        # WAL replay restores data committed before the crash.
        conn.execute("INSERT INTO crash_probe (id) VALUES (42)")
        conn.commit()
        row = conn.execute("SELECT id FROM crash_probe").fetchone()
    assert row is not None
    assert row[0] == 42
    revived.stop()


def test_pgserver_session_stops_on_exit(tmp_path_factory: pytest.TempPathFactory) -> None:
    data_dir = tmp_path_factory.mktemp("pgsession") / "pg"
    with pgserver_session(data_dir) as pg:
        assert isinstance(pg, PgHandle)
        with pg.get_conn() as conn:
            assert conn.execute("SELECT 1").fetchone() == (1,)
        socket_dir = Path(pg.dsn().split("?host=")[1])
    # The context manager must have stopped the server: no live socket left.
    assert not (socket_dir / ".s.PGSQL.5432").exists()
    assert not (data_dir / "postmaster.pid").exists()


def test_default_data_dir_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PGDATA_DIR", str(tmp_path / "custom"))
    assert default_data_dir() == (tmp_path / "custom").resolve()
    monkeypatch.delenv("PGDATA_DIR")
    assert default_data_dir().name == "pg"


@pytest.mark.parametrize(
    ("bad_dsn", "fragment"),
    [
        ("postgresql://postgres:@/x?host=localhost", "not TCP"),
        ("postgresql://postgres:@/x?host=127.0.0.1", "not TCP"),
        ("postgresql://postgres:pw@localhost:5432/x", "unix-socket host"),
        ("postgresql://postgres:@/x?host=/tmp/sock&port=5433", "TCP port"),
        ("postgresql://postgres:@/x", "unix-socket host"),
    ],
)
def test_validate_unix_dsn_rejects_tcp(bad_dsn: str, fragment: str) -> None:
    with pytest.raises(DbError, match=fragment):
        _validate_unix_dsn(bad_dsn)


def test_database_name_is_idea_finder(handle: PgHandle) -> None:
    with handle.get_conn() as conn:
        current = conn.execute("SELECT current_database()").fetchone()
    assert current is not None
    assert current[0] == DB_NAME


def test_psycopg_connects_only_via_dsn_socket_dir(handle: PgHandle) -> None:
    """A connection made without the socket dir in the DSN must fail."""
    with pytest.raises(psycopg.OperationalError):
        psycopg.connect("postgresql://postgres:@/idea_finder", connect_timeout=2)
