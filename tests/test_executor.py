"""
Tests for executor.py — TradeRecord dataclass and the TP/SL lifecycle
handlers extracted during the v2.2.0 complexity pass (_check_trade went
from cyclomatic complexity 41 to 8 by splitting into these 5 methods).
"""

from datetime import datetime, timezone

import pytest

from ceo_engine_mt5.executor import TradeRecord, Executor


def _long_trade(**overrides) -> TradeRecord:
    kwargs = dict(
        ticket=1, symbol="XAUUSD", tf="M15", direction="long",
        entry=2350.0, sl=2345.0, tp1=2352.0, tp2=2359.0, tp3=2366.0,
        lots=0.1, atr=5.0, quality=72.5, model="LQ",
        open_time=datetime.now(timezone.utc), bar_time=datetime.now(timezone.utc),
        magic=20250101,
    )
    kwargs.update(overrides)
    return TradeRecord(**kwargs)


class TestTradeRecordDataclass:
    """Regression tests for the @dataclass conversion (v2.2.0) — the manual
    __init__ used to set these defaults by hand; __post_init__ must do the
    same thing now."""

    def test_lots_remaining_defaults_to_lots(self):
        t = _long_trade(lots=0.25)
        assert t.lots_remaining == 0.25

    def test_current_price_defaults_to_entry(self):
        t = _long_trade(entry=2350.0)
        assert t.current_price == 2350.0

    def test_last_update_defaults_to_open_time(self):
        t = _long_trade()
        assert t.last_update == t.open_time

    def test_starts_open_with_no_tp_hits(self):
        t = _long_trade()
        assert t.status == "open"
        assert t.tp1_hit is False and t.tp2_hit is False
        assert t.close_price is None and t.pnl is None

    def test_to_dict_round_trips_key_fields(self):
        t = _long_trade()
        d = t.to_dict()
        assert d["ticket"] == 1
        assert d["symbol"] == "XAUUSD"
        assert d["close_time"] is None


@pytest.fixture
def executor():
    """Executor in simulation mode (no real MT5 connection), with a
    permissive FundedAccountGuard since guard is a required constructor
    param by design (every executor must have drawdown protection).
    journal_file=None -- otherwise this fixture silently writes real CSV
    rows to "ceo_trades.csv" in whatever directory pytest is invoked
    from, on every test that uses it."""
    from ceo_engine_mt5.funded_account_guard import FundedAccountGuard
    guard = FundedAccountGuard(account_size=10_000.0)
    return Executor(connection=None, risk_engine=None, guard=guard,
                     simulation=True, journal_file=None)


class TestComputeTpSlHits:
    """_compute_tp_sl_hits — pure boolean check, both directions."""

    def test_long_all_targets_hit(self, executor):
        t = _long_trade()
        hits = executor._compute_tp_sl_hits(t, price=2370.0, is_long=True)
        assert hits == (True, True, True, False)

    def test_long_sl_hit(self, executor):
        t = _long_trade()
        hits = executor._compute_tp_sl_hits(t, price=2340.0, is_long=True)
        assert hits == (False, False, False, True)

    def test_short_mirrors_long(self, executor):
        t = _long_trade(direction="short", entry=2350, sl=2355,
                        tp1=2348, tp2=2341, tp3=2334)
        hits = executor._compute_tp_sl_hits(t, price=2330.0, is_long=False)
        assert hits == (True, True, True, False)


class TestUpdateFloatingState:
    """_update_floating_state — floating R / P&L tracking."""

    def test_floating_r_positive_when_price_favorable(self, executor):
        t = _long_trade()  # entry=2350, sl=2345 -> risk=5
        executor._update_floating_state(t, price=2360.0, is_long=True)
        assert t.floating_r == pytest.approx(2.0)  # (2360-2350)/5

    def test_floating_r_negative_when_price_unfavorable(self, executor):
        t = _long_trade()
        executor._update_floating_state(t, price=2347.0, is_long=True)
        assert t.floating_r == pytest.approx(-0.6)  # (2347-2350)/5

    def test_current_price_updates(self, executor):
        t = _long_trade()
        executor._update_floating_state(t, price=2355.0, is_long=True)
        assert t.current_price == 2355.0

    def test_place_trade_skips_when_mt5_connection_is_disconnected(self, monkeypatch):
        class _FakeConn:
            def is_connected(self):
                return False
        class _FakeRiskEngine:
            def evaluate(self, **kwargs):
                return 0.1, {"sizing": {"risk_amount": 25.0}}
        class _FakeGuard:
            def pre_trade_check(self, **kwargs):
                return True, ""
        executor = Executor(
            connection=_FakeConn(),
            risk_engine=_FakeRiskEngine(),
            guard=_FakeGuard(),
            simulation=False,
            journal_file=None,
        )
        ticket = executor.place_trade(
            symbol="XAUUSD", tf="M15", direction="long",
            entry=2350.0, sl=2345.0, tp1=2352.0, tp2=2359.0, tp3=2366.0,
            quality=72.5, model="LQ", bar_time=datetime.now(timezone.utc), atr=5.0,
        )
        assert ticket is None


class TestHandleTp1Hit:
    """_handle_tp1_hit — partial close + optional move-to-breakeven."""

    def test_marks_tp1_hit(self, executor):
        t = _long_trade()
        executor._handle_tp1_hit(t, verbose=False)
        assert t.tp1_hit is True

    def test_moves_sl_to_breakeven_when_configured(self, executor):
        executor.move_sl_at_tp1 = True
        t = _long_trade()
        executor._handle_tp1_hit(t, verbose=False)
        assert t.sl == t.entry
        assert t.sl_moved is True

    def test_does_not_move_sl_when_disabled(self, executor):
        executor.move_sl_at_tp1 = False
        t = _long_trade()
        original_sl = t.sl
        executor._handle_tp1_hit(t, verbose=False)
        assert t.sl == original_sl
        assert t.sl_moved is False

    def test_does_not_move_sl_twice(self, executor):
        executor.move_sl_at_tp1 = True
        t = _long_trade()
        executor._handle_tp1_hit(t, verbose=False)
        t.entry = 9999  # if moved again, sl would become this
        executor._handle_tp1_hit(t, verbose=False)
        assert t.sl != 9999


class TestHandleTp2Hit:
    """_handle_tp2_hit — partial close + optional trail-to-TP1."""

    def test_marks_tp2_hit(self, executor):
        t = _long_trade()
        executor._handle_tp2_hit(t, verbose=False)
        assert t.tp2_hit is True

    def test_trails_sl_to_tp1_when_configured(self, executor):
        executor.trail_sl_at_tp2 = True
        t = _long_trade()
        executor._handle_tp2_hit(t, verbose=False)
        assert t.sl == t.tp1


class TestHandleFullClose:
    """_handle_full_close — TP3/SL terminal close, records outcome and tells the guard."""

    def test_closed_trade_moves_out_of_open_trades(self, executor):
        t = _long_trade()
        executor._open_trades[t.ticket] = t
        executor._handle_full_close(t, price=2366.0, reason="tp3", verbose=False)
        assert t.ticket not in executor._open_trades
        assert t in executor._closed_trades

    def test_status_and_close_fields_set(self, executor):
        t = _long_trade()
        executor._open_trades[t.ticket] = t
        event = executor._handle_full_close(t, price=2345.0, reason="sl", verbose=False)
        assert t.status == "closed"
        assert t.close_reason == "sl"
        assert t.close_price == 2345.0
        assert event["reason"] == "sl"
        assert event["ticket"] == t.ticket

    def test_pnl_estimated_in_simulation_mode(self, executor):
        t = _long_trade()
        executor._open_trades[t.ticket] = t
        event = executor._handle_full_close(t, price=2366.0, reason="tp3", verbose=False)
        # Long trade closing above entry should show a positive estimated P&L
        assert event["pnl"] > 0


class TestCheckTradeOrchestration:
    """_check_trade — the thin orchestrator wired up correctly end-to-end."""

    def test_full_lifecycle_tp1_then_tp2_then_tp3(self, executor):
        t = _long_trade()
        executor._open_trades[t.ticket] = t

        result1 = executor._check_trade(t, price=2353.0, bid=2353.0, ask=2353.2, verbose=False)
        assert result1 is None and t.tp1_hit is True

        result2 = executor._check_trade(t, price=2360.0, bid=2360.0, ask=2360.2, verbose=False)
        assert result2 is None and t.tp2_hit is True

        result3 = executor._check_trade(t, price=2367.0, bid=2367.0, ask=2367.2, verbose=False)
        assert result3 is not None
        assert result3["reason"] == "tp3"

    def test_sl_hit_closes_immediately_without_any_tp(self, executor):
        t = _long_trade()
        executor._open_trades[t.ticket] = t
        result = executor._check_trade(t, price=2344.0, bid=2344.0, ask=2344.2, verbose=False)
        assert result is not None
        assert result["reason"] == "sl"

    def test_no_exit_returns_none_and_trade_stays_open(self, executor):
        t = _long_trade()
        executor._open_trades[t.ticket] = t
        result = executor._check_trade(t, price=2351.0, bid=2351.0, ask=2351.2, verbose=False)
        assert result is None
        assert t.ticket in executor._open_trades
