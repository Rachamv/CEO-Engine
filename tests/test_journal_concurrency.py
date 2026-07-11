"""
Tests for journal.py's SQLite connection handling fix:
  - WAL mode + a busy_timeout are set on every connection, so a second
    writer waits instead of immediately raising "database is locked"
  - every method now goes through a context manager that closes the
    connection even if the query raises, instead of a bare conn.close()
    that would never run on an exception
  - concurrent writers (dashboard HTTP thread + live engine thread, in
    production) can both write without one failing
"""

import os
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone

import pytest

from ceo_engine_mt5.journal import Journal


@pytest.fixture
def db_path():
    path = tempfile.mktemp(suffix=".db")
    yield path
    if os.path.exists(path):
        os.unlink(path)
    for suffix in ("-wal", "-shm"):
        extra = path + suffix
        if os.path.exists(extra):
            os.unlink(extra)


@pytest.fixture
def journal(db_path):
    return Journal(db_path)


def _open_trade(journal, ticket, direction="long", entry=2350, sl=2345):
    journal.log_trade_open(
        ticket=ticket, symbol="XAUUSD", tf="M15", direction=direction,
        entry=entry, sl=sl, tp1=entry + 2, tp2=entry + 9, tp3=entry + 16,
        lots=0.1, quality=70, model="LQ",
        bar_time=datetime.now(timezone.utc), session="london",
    )


# ─────────────────────────────────────────────────────────────────────────────
# WAL mode + busy_timeout are actually applied
# ─────────────────────────────────────────────────────────────────────────────

class TestConnectionPragmas:
    def test_connect_enables_wal_mode(self, journal):
        conn = journal._connect()
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            conn.close()

    def test_connect_sets_a_nonzero_busy_timeout(self, journal):
        conn = journal._connect()
        try:
            timeout_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert timeout_ms > 0
        finally:
            conn.close()

    def test_connection_context_manager_closes_on_success(self, journal):
        with journal._connection() as conn:
            conn.execute("SELECT 1")
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")   # closed connections raise on use


# ─────────────────────────────────────────────────────────────────────────────
# Connections are closed even when the query inside raises
# ─────────────────────────────────────────────────────────────────────────────

class TestConnectionClosesOnException:
    def test_connection_closed_even_if_query_raises(self, journal):
        captured = {}
        with pytest.raises(sqlite3.OperationalError):
            with journal._connection() as conn:
                captured["conn"] = conn
                conn.execute("SELECT * FROM this_table_does_not_exist")
        # The connection object from the failed `with` block must already
        # be closed -- using it again raises ProgrammingError, not silently
        # succeed (which is what happened before: the bare conn.close()
        # at the end of each method never ran on an exception).
        with pytest.raises(sqlite3.ProgrammingError):
            captured["conn"].execute("SELECT 1")

    def test_log_signal_raising_midway_does_not_leak_the_connection(self, journal, monkeypatch):
        """A malformed call that raises inside log_signal (e.g. a bad
        bind parameter) must not leave the connection open."""
        opened = []
        real_connect = journal._connect

        def _tracking_connect():
            conn = real_connect()
            opened.append(conn)
            return conn

        monkeypatch.setattr(journal, "_connect", _tracking_connect)

        with pytest.raises(Exception):
            journal.log_signal(
                symbol="XAUUSD", tf="M15", direction="long", quality=70,
                model="LQ", bar_time=datetime.now(timezone.utc),
                entry=object(),   # not bindable -> sqlite3 raises
                sl=2345, tp1=2352, tp2=2359, tp3=2366,
            )
        assert len(opened) == 1
        with pytest.raises(sqlite3.ProgrammingError):
            opened[0].execute("SELECT 1")   # confirms it was actually closed


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent writers don't fail with "database is locked"
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrentWriters:
    def test_two_threads_writing_simultaneously_both_succeed(self, journal):
        """Simulates the dashboard HTTP thread and the live engine thread
        both logging trades around the same time. Before the WAL +
        busy_timeout fix, SQLite's default 0ms busy timeout meant the
        second writer could get 'database is locked' immediately instead
        of simply waiting its turn."""
        errors = []
        barrier = threading.Barrier(2)

        def _writer(start_ticket):
            try:
                barrier.wait(timeout=5)
                for i in range(15):
                    _open_trade(journal, ticket=start_ticket + i)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=_writer, args=(1000,))
        t2 = threading.Thread(target=_writer, args=(2000,))
        t1.start(); t2.start()
        t1.join(timeout=10); t2.join(timeout=10)

        assert errors == [], f"concurrent writers raised: {errors}"
        assert len(journal.open_trades()) == 30

    def test_reader_not_blocked_by_concurrent_writer(self, journal):
        """WAL mode's main benefit: a reader shouldn't be blocked by a
        writer holding the database open."""
        _open_trade(journal, ticket=1)
        errors = []

        def _slow_writer():
            try:
                conn = journal._connect()
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO daily_summary (trade_date) VALUES ('2024-01-01')")
                import time
                time.sleep(0.3)
                conn.commit()
                conn.close()
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=_slow_writer)
        t.start()
        import time
        time.sleep(0.05)   # let the writer grab its lock first
        # This read must not raise "database is locked" while the writer
        # above is mid-transaction.
        result = journal.open_trades()
        t.join(timeout=5)

        assert errors == []
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Existing behavior still intact after the refactor
# ─────────────────────────────────────────────────────────────────────────────

class TestExistingBehaviorPreserved:
    def test_full_trade_lifecycle_still_works(self, journal):
        _open_trade(journal, ticket=1)
        assert len(journal.open_trades()) == 1
        journal.log_trade_close(1, 2366, datetime.now(timezone.utc), "tp3", 160.0, True, True)
        assert journal.open_trades() == []
        stats = journal.performance_stats()
        assert stats["trades"] == 1
        assert stats["total_pnl"] == 160.0

    def test_stats_by_model_and_session_still_work(self, journal):
        _open_trade(journal, ticket=1)
        journal.log_trade_close(1, 2366, datetime.now(timezone.utc), "tp3", 160.0, True, True)
        assert journal.stats_by_model()[0]["model"] == "LQ"
        assert journal.stats_by_session()[0]["session"] == "london"

    def test_export_csv_still_works(self, journal, tmp_path):
        _open_trade(journal, ticket=1)
        journal.log_trade_close(1, 2366, datetime.now(timezone.utc), "tp3", 160.0, True, True)
        out_path = str(tmp_path / "export.csv")
        journal.export_csv(out_path)
        assert os.path.exists(out_path)
        content = open(out_path).read()
        assert "XAUUSD" in content
