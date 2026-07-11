"""
Tests for journal.py — SQLite trade/signal logging and performance stats.
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from ceo_engine_mt5.journal import Journal


@pytest.fixture
def journal():
    db = tempfile.mktemp(suffix=".db")
    j = Journal(db)
    yield j
    if os.path.exists(db):
        os.unlink(db)


def _open_trade(journal, ticket, direction="long", entry=2350, sl=2345):
    journal.log_trade_open(
        ticket=ticket, symbol="XAUUSD", tf="M15", direction=direction,
        entry=entry, sl=sl, tp1=entry + 2, tp2=entry + 9, tp3=entry + 16,
        lots=0.1, quality=70, model="LQ",
        bar_time=datetime.now(timezone.utc), session="london",
    )


class TestTradeLifecycle:
    def test_open_trade_appears_in_open_trades(self, journal):
        _open_trade(journal, ticket=1)
        open_trades = journal.open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]["ticket"] == 1

    def test_closed_trade_no_longer_open(self, journal):
        _open_trade(journal, ticket=1)
        journal.log_trade_close(1, 2366, datetime.now(timezone.utc), "tp3", 160.0, True, True)
        assert journal.open_trades() == []

    def test_recent_trades_returns_newest_first(self, journal):
        for ticket in [1, 2, 3]:
            _open_trade(journal, ticket=ticket)
            journal.log_trade_close(ticket, 2360, datetime.now(timezone.utc),
                                     "tp1", 20.0, True, False)
        recent = journal.recent_trades(limit=10)
        assert [r["ticket"] for r in recent] == [3, 2, 1]


class TestPerformanceStats:
    """
    Exact-value checks on the stats math (refactored in v2.2.0 to use
    _safe_pct/_safe_avg helpers — these pin the behavior those helpers
    must preserve).
    """

    def test_zero_trades_returns_minimal_dict(self, journal):
        assert journal.performance_stats() == {"trades": 0}

    def test_known_outcomes_produce_exact_stats(self, journal):
        _open_trade(journal, ticket=1, direction="long")
        _open_trade(journal, ticket=2, direction="short")
        journal.log_trade_close(1, 2366, datetime.now(timezone.utc), "tp3", 160.0, True, True)
        journal.log_trade_close(2, 2355, datetime.now(timezone.utc), "sl", -50.0, False, False)

        stats = journal.performance_stats()
        assert stats["trades"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == 50.0
        assert stats["total_pnl"] == 110.0
        assert stats["avg_win"] == 160.0
        assert stats["avg_loss"] == -50.0
        assert stats["profit_factor"] == pytest.approx(3.2)
        assert stats["tp3_rate"] == 50.0
        assert stats["sl_rate"] == 50.0
        assert stats["tp1_hit_rate"] == 50.0
        assert stats["tp2_hit_rate"] == 50.0

    def test_all_wins_no_losses_profit_factor_is_zero_not_crash(self, journal):
        """profit_factor divides by sum(losses) — must not divide by zero
        when there are no losses at all."""
        _open_trade(journal, ticket=1)
        journal.log_trade_close(1, 2366, datetime.now(timezone.utc), "tp3", 100.0, True, True)
        stats = journal.performance_stats()
        assert stats["profit_factor"] == 0.0
        assert stats["avg_loss"] == 0.0

    def test_all_losses_no_wins_avg_win_is_zero_not_crash(self, journal):
        _open_trade(journal, ticket=1)
        journal.log_trade_close(1, 2345, datetime.now(timezone.utc), "sl", -50.0, False, False)
        stats = journal.performance_stats()
        assert stats["avg_win"] == 0.0
        assert stats["profit_factor"] == 0.0
