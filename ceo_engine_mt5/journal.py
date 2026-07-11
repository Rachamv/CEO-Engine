"""
The CEO Protocol — Phase 4: Trade Journal
==============================================
Auto-logs every trade with full context. Tracks outcomes.
Feeds statistics back into the system for continuous improvement.

Storage
-------
    SQLite database (ceo_journal.db) — persistent, queryable
    CSV export on demand

Tables
------
    trades          — one row per trade (open + closed)
    signals         — every signal fired (traded or not)
    daily_summary   — aggregated daily stats

Usage
-----
    from .journal import Journal

    j = Journal("ceo_journal.db")

    # Log a signal (whether traded or not)
    j.log_signal(symbol, tf, direction, quality, model, bar_time,
                 entry, sl, tp1, tp2, tp3, traded=True, block_reason=None)

    # Log trade open
    j.log_trade_open(ticket, symbol, tf, direction, entry, sl,
                     tp1, tp2, tp3, lots, quality, model, bar_time)

    # Log trade close
    j.log_trade_close(ticket, close_price, close_time,
                      close_reason, pnl, tp1_hit, tp2_hit)

    # Get stats
    stats = j.performance_stats()
    by_model = j.stats_by_model()
    by_session = j.stats_by_session()
"""

import sqlite3
import csv
import os
import contextlib
import time
from datetime import datetime, timezone
from typing import Optional, List
import json


# ─────────────────────────────────────────────────────────────────────────────
# Journal
# ─────────────────────────────────────────────────────────────────────────────

class Journal:

    def __init__(self, db_path: str = "ceo_journal.db"):
        self.db_path = db_path
        self._init_db()

    # ── Database setup ────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        # The dashboard (HTTP request thread) and the live engine (its own
        # thread) can both hit this DB at the same time. SQLite's default
        # busy timeout is 0 -- a second writer gets "database is locked"
        # immediately instead of waiting. WAL mode also lets readers run
        # without blocking a concurrent writer, which the default rollback
        # journal mode does not.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _retry(self, func):
        max_attempts = 5
        delay = 0.1
        for attempt in range(max_attempts):
            try:
                return func()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_attempts - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 1.0)
                    continue
                raise

    @contextlib.contextmanager
    def _connection(self):
        """Yields a connection that's guaranteed to be closed even if the
        caller raises mid-query -- every method below used to open its own
        connection with a bare conn.close() at the end and no try/finally,
        so an exception between connect() and close() would leave the
        connection (and any lock it held) open until the interpreter
        eventually garbage-collects it."""
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._connection() as conn:
            c = conn.cursor()

            c.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    ticket          INTEGER PRIMARY KEY,
                    symbol          TEXT,
                    tf              TEXT,
                    direction       TEXT,
                    model           TEXT,
                    quality         REAL,
                    entry           REAL,
                    sl              REAL,
                    tp1             REAL,
                    tp2             REAL,
                    tp3             REAL,
                    lots            REAL,
                    open_time       TEXT,
                    bar_time        TEXT,
                    session         TEXT,
                    close_price     REAL,
                    close_time      TEXT,
                    close_reason    TEXT,
                    pnl             REAL,
                    tp1_hit         INTEGER DEFAULT 0,
                    tp2_hit         INTEGER DEFAULT 0,
                    status          TEXT DEFAULT 'open',
                    r_multiple      REAL,
                    tags            TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT,
                    tf              TEXT,
                    direction       TEXT,
                    model           TEXT,
                    quality         REAL,
                    bar_time        TEXT,
                    session         TEXT,
                    entry           REAL,
                    sl              REAL,
                    tp1             REAL,
                    tp2             REAL,
                    tp3             REAL,
                    traded          INTEGER DEFAULT 0,
                    block_reason    TEXT,
                    pat_name        TEXT,
                    ceo_valid       INTEGER DEFAULT 0,
                    bos_confirmed   INTEGER DEFAULT 0,
                    in_discount     INTEGER DEFAULT 0
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS daily_summary (
                    trade_date      TEXT PRIMARY KEY,
                    trades          INTEGER DEFAULT 0,
                    wins            INTEGER DEFAULT 0,
                    losses          INTEGER DEFAULT 0,
                    breakeven       INTEGER DEFAULT 0,
                    total_pnl       REAL DEFAULT 0,
                    total_r         REAL DEFAULT 0,
                    signals_fired   INTEGER DEFAULT 0
                )
            """)

            conn.commit()

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_signal(
        self,
        symbol:       str,
        tf:           str,
        direction:    str,
        quality:      float,
        model:        str,
        bar_time:     datetime,
        entry:        float,
        sl:           float,
        tp1:          float,
        tp2:          float,
        tp3:          float,
        traded:       bool   = False,
        block_reason: Optional[str] = None,
        session:      str    = "",
        pat_name:     str    = "",
        ceo_valid:    bool   = False,
        bos_confirmed: bool  = False,
        in_discount:  bool   = False,
    ) -> int:
        """Log every signal fired, whether traded or not."""
        def _insert():
            with self._connection() as conn:
                c = conn.cursor()
                c.execute("""
                    INSERT INTO signals
                    (symbol, tf, direction, model, quality, bar_time, session,
                     entry, sl, tp1, tp2, tp3, traded, block_reason,
                     pat_name, ceo_valid, bos_confirmed, in_discount)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    symbol, tf, direction, model, quality,
                    _ts(bar_time), session, entry, sl, tp1, tp2, tp3,
                    int(traded), block_reason, pat_name,
                    int(ceo_valid), int(bos_confirmed), int(in_discount),
                ))
                row_id = c.lastrowid
                conn.commit()
            return row_id

        row_id = self._retry(_insert)

        # Update daily signal count
        self._update_daily(bar_time, signals_delta=1)
        return row_id

    def log_trade_open(
        self,
        ticket:    int,
        symbol:    str,
        tf:        str,
        direction: str,
        entry:     float,
        sl:        float,
        tp1:       float,
        tp2:       float,
        tp3:       float,
        lots:      float,
        quality:   float,
        model:     str,
        bar_time:  datetime,
        session:   str  = "",
        tags:      Optional[List[str]] = None,
    ):
        """Log a trade at open."""
        def _insert():
            with self._connection() as conn:
                c   = conn.cursor()
                now = _ts(datetime.now(timezone.utc))
                c.execute("""
                    INSERT OR REPLACE INTO trades
                    (ticket, symbol, tf, direction, model, quality,
                     entry, sl, tp1, tp2, tp3, lots, open_time, bar_time,
                     session, status, tags)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',?)
                """, (
                    ticket, symbol, tf, direction, model, quality,
                    entry, sl, tp1, tp2, tp3, lots, now, _ts(bar_time),
                    session, json.dumps(tags or []),
                ))
                conn.commit()
        self._retry(_insert)

    def log_trade_close(
        self,
        ticket:       int,
        close_price:  float,
        close_time:   datetime,
        close_reason: str,
        pnl:          float,
        tp1_hit:      bool = False,
        tp2_hit:      bool = False,
    ):
        """Update a trade record at close. Computes R-multiple."""
        def _update_trade():
            with self._connection() as conn:
                c = conn.cursor()

                # Get entry/sl to compute R
                row = c.execute(
                    "SELECT entry, sl FROM trades WHERE ticket=?", (ticket,)
                ).fetchone()

                r_multiple = None
                if row:
                    entry = row["entry"]
                    sl    = row["sl"]
                    sl_dist = abs(entry - sl)
                    if sl_dist > 0:
                        r_multiple = round(pnl / (sl_dist * 100), 3)  # approximate

                c.execute("""
                    UPDATE trades SET
                        close_price  = ?,
                        close_time   = ?,
                        close_reason = ?,
                        pnl          = ?,
                        tp1_hit      = ?,
                        tp2_hit      = ?,
                        status       = 'closed',
                        r_multiple   = ?
                    WHERE ticket = ?
                """, (
                    close_price, _ts(close_time), close_reason,
                    pnl, int(tp1_hit), int(tp2_hit), r_multiple, ticket,
                ))
                conn.commit()

        self._retry(_update_trade)

        # Update daily summary
        self._update_daily(close_time, pnl=pnl)

    def _update_daily(
        self,
        dt:             datetime,
        pnl:            Optional[float] = None,
        signals_delta:  int = 0,
    ):
        def _update():
            ds = dt.date().isoformat() if hasattr(dt, "date") else str(dt)[:10]
            with self._connection() as conn:
                c = conn.cursor()

                existing = c.execute(
                    "SELECT * FROM daily_summary WHERE trade_date=?", (ds,)
                ).fetchone()

                if not existing:
                    c.execute("""
                        INSERT INTO daily_summary (trade_date) VALUES (?)
                    """, (ds,))

                if signals_delta:
                    c.execute("""
                        UPDATE daily_summary SET signals_fired = signals_fired + ?
                        WHERE trade_date = ?
                    """, (signals_delta, ds))

                if pnl is not None:
                    win  = 1 if pnl > 0 else 0
                    loss = 1 if pnl < 0 else 0
                    be   = 1 if pnl == 0 else 0
                    c.execute("""
                        UPDATE daily_summary SET
                            trades    = trades + 1,
                            wins      = wins + ?,
                            losses    = losses + ?,
                            breakeven = breakeven + ?,
                            total_pnl = total_pnl + ?
                        WHERE trade_date = ?
                    """, (win, loss, be, pnl, ds))

                conn.commit()
        self._retry(_update)

    # ── Statistics ────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_pct(numerator: float, denominator: float, ndigits: int = 1) -> float:
        """0-safe percentage: numerator/denominator * 100, or 0 if denominator is 0."""
        return round(numerator / denominator * 100, ndigits) if denominator else 0.0

    @staticmethod
    def _safe_avg(values: list, ndigits: int = 2) -> float:
        """0-safe average, or 0 if the list is empty."""
        return round(sum(values) / len(values), ndigits) if values else 0.0

    @staticmethod
    def _compute_performance_stats(rows) -> dict:
        """Pure computation over already-fetched closed-trade rows."""
        if not rows:
            return {"trades": 0}

        pnls   = [r["pnl"]        for r in rows if r["pnl"] is not None]
        rs     = [r["r_multiple"] for r in rows if r["r_multiple"] is not None]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        tp3s   = [r for r in rows if r["close_reason"] == "tp3"]
        sls    = [r for r in rows if r["close_reason"] == "sl"]
        n      = len(rows)

        return {
            "trades":           n,
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate":         Journal._safe_pct(len(wins), n),
            "total_pnl":        round(sum(pnls), 2),
            "avg_win":          Journal._safe_avg(wins),
            "avg_loss":         Journal._safe_avg(losses),
            "profit_factor":    round(-sum(wins) / sum(losses), 2) if sum(losses) else 0.0,
            "avg_r":            Journal._safe_avg(rs, ndigits=3),
            "tp3_rate":         Journal._safe_pct(len(tp3s), n),
            "sl_rate":          Journal._safe_pct(len(sls), n),
            "tp1_hit_rate":     Journal._safe_pct(sum(r["tp1_hit"] for r in rows), n),
            "tp2_hit_rate":     Journal._safe_pct(sum(r["tp2_hit"] for r in rows), n),
        }

    def performance_stats(self) -> dict:
        """Overall performance statistics."""
        with self._connection() as conn:
            c = conn.cursor()
            rows = c.execute("""
                SELECT pnl, r_multiple, close_reason, tp1_hit, tp2_hit,
                       direction, session, model
                FROM trades WHERE status='closed'
            """).fetchall()

        return self._compute_performance_stats(rows)

    def stats_by_model(self) -> List[dict]:
        """Performance breakdown per model."""
        with self._connection() as conn:
            rows = conn.execute("""
                SELECT model,
                       COUNT(*) as trades,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(pnl) as total_pnl,
                       AVG(r_multiple) as avg_r
                FROM trades WHERE status='closed'
                GROUP BY model ORDER BY avg_r DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def stats_by_session(self) -> List[dict]:
        """Performance breakdown per session."""
        with self._connection() as conn:
            rows = conn.execute("""
                SELECT session,
                       COUNT(*) as trades,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(pnl) as total_pnl,
                       AVG(r_multiple) as avg_r
                FROM trades WHERE status='closed'
                GROUP BY session ORDER BY avg_r DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def stats_by_symbol(self) -> List[dict]:
        """Performance breakdown per symbol."""
        with self._connection() as conn:
            rows = conn.execute("""
                SELECT symbol,
                       COUNT(*) as trades,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(pnl) as total_pnl,
                       AVG(r_multiple) as avg_r
                FROM trades WHERE status='closed'
                GROUP BY symbol ORDER BY total_pnl DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def open_trades(self) -> List[dict]:
        """All currently open trades."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='open' ORDER BY open_time DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_trades(self, limit: int = 20) -> List[dict]:
        """Most recent closed trades."""
        with self._connection() as conn:
            rows = conn.execute("""
                SELECT * FROM trades WHERE status='closed'
                ORDER BY close_time DESC LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def daily_summary(self, days: int = 30) -> List[dict]:
        """Daily summary for the last N days."""
        with self._connection() as conn:
            rows = conn.execute("""
                SELECT * FROM daily_summary
                ORDER BY trade_date DESC LIMIT ?
            """, (days,)).fetchall()
        return [dict(r) for r in rows]

    def export_csv(self, path: str = "ceo_trades_export.csv") -> str:
        """Export all closed trades to CSV."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY open_time DESC"
            ).fetchall()

        if not rows:
            print("  No trades to export.")
            return path

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows([dict(r) for r in rows])

        print(f"  ✅  Exported {len(rows)} trades to {path}")
        return path

    def print_summary(self):
        """Print a formatted performance summary."""
        s = self.performance_stats()
        if s.get("trades", 0) == 0:
            print("  No closed trades yet.")
            return

        print("\n" + "═" * 50)
        print("  CEO ENGINE — PERFORMANCE SUMMARY")
        print("═" * 50)
        print(f"  Trades        : {s['trades']}")
        print(f"  Win Rate      : {s['win_rate']:.1f}%  ({s['wins']}W / {s['losses']}L)")
        print(f"  Total P&L     : ${s['total_pnl']:+.2f}")
        print(f"  Avg R         : {s['avg_r']:+.3f}R")
        print(f"  Profit Factor : {s['profit_factor']:.2f}")
        print(f"  TP3 Rate      : {s['tp3_rate']:.1f}%")
        print(f"  SL Rate       : {s['sl_rate']:.1f}%")
        print(f"  TP1 Hit Rate  : {s['tp1_hit_rate']:.1f}%")
        print(f"  TP2 Hit Rate  : {s['tp2_hit_rate']:.1f}%")

        by_model = self.stats_by_model()
        if by_model:
            print(f"\n  ── By Model ──")
            for r in by_model:
                wr = round(r["wins"] / r["trades"] * 100, 1) if r["trades"] else 0
                print(f"  {r['model']:<30} "
                      f"T={r['trades']:>3} "
                      f"WR={wr:.0f}% "
                      f"R={r['avg_r']:+.3f}" if r['avg_r'] else "")

        print("═" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts(dt: datetime) -> str:
    if dt is None:
        return ""
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
