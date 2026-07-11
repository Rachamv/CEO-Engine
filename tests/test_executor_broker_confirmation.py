"""
Tests for the broker-confirmation gating in executor.py's live (non-
simulation) trade-management path -- _mt5_partial_close / _mt5_modify_sl /
_mt5_close now return True/False based on the broker's retcode, and every
caller (_handle_tp1_hit, _handle_tp2_hit, _handle_full_close,
_update_trailing_sl, manual_close, manual_modify_sl, close_all) only
commits the corresponding in-memory state change when the broker actually
confirmed it.

Why this needs its own file: MetaTrader5 only installs on Windows, so
`Executor.__init__` forces `self.simulation = True` on every other
platform regardless of what's passed in -- meaning none of these "live
mode" code paths have ever been exercised by the existing test suite
(that's exactly how the original missing-retcode-check bug went
unnoticed). These tests inject a fake `mt5` module directly into the
executor module's namespace and force MT5_AVAILABLE=True so the real
branches actually run.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import ceo_engine_mt5.executor as executor_module
from ceo_engine_mt5.executor import TradeRecord, Executor
from ceo_engine_mt5.funded_account_guard import FundedAccountGuard


# ─────────────────────────────────────────────────────────────────────────────
# Fake MT5 module
# ─────────────────────────────────────────────────────────────────────────────

class _FakeOrderResult:
    def __init__(self, retcode, comment="", order=555):
        self.retcode = retcode
        self.comment = comment
        self.order = order


class _FakePosition:
    def __init__(self, volume, type_=0):
        self.volume = volume
        self.type = type_   # 0 = long/buy position, 1 = short/sell position


class _FakeTick:
    def __init__(self, bid=2350.0, ask=2350.2):
        self.bid = bid
        self.ask = ask
        self.time = 0


class FakeMT5:
    """Minimal stand-in for the MetaTrader5 module surface executor.py uses."""
    TRADE_ACTION_DEAL  = "DEAL"
    TRADE_ACTION_SLTP  = "SLTP"
    ORDER_TYPE_BUY     = 0
    ORDER_TYPE_SELL    = 1
    TRADE_RETCODE_DONE = 10009
    ORDER_TIME_GTC     = 0
    ORDER_FILLING_IOC  = 0

    def __init__(self):
        self.positions: dict = {}          # ticket -> _FakePosition
        self.tick = _FakeTick()
        self.next_retcode = self.TRADE_RETCODE_DONE
        self.next_comment = "Done"
        self.sent_requests: list = []

    def positions_get(self, ticket=None):
        pos = self.positions.get(ticket)
        return [pos] if pos else []

    def symbol_info_tick(self, symbol):
        return self.tick

    def order_send(self, request):
        self.sent_requests.append(request)
        return _FakeOrderResult(self.next_retcode, self.next_comment)

    def history_deals_get(self, position=None):
        return []   # forces _get_real_pnl() -> None -> falls back to _estimate_pnl


REJECTED = 10004   # arbitrary non-DONE retcode ("requote", illustratively)


@pytest.fixture
def fake_mt5(monkeypatch):
    fake = FakeMT5()
    monkeypatch.setattr(executor_module, "mt5", fake, raising=False)
    monkeypatch.setattr(executor_module, "MT5_AVAILABLE", True)
    return fake


@pytest.fixture
def live_executor(fake_mt5):
    """Executor forced into live (non-simulation) mode against the fake
    MT5 module -- journal_file=None keeps tests from touching disk."""
    guard = FundedAccountGuard(account_size=10_000.0)
    return Executor(connection=None, risk_engine=None, guard=guard,
                     simulation=False, journal_file=None)


def _open_long(executor, fake_mt5, ticket=1, lots=0.3, volume_open=0.3):
    trade = TradeRecord(
        ticket=ticket, symbol="XAUUSD", tf="M15", direction="long",
        entry=2350.0, sl=2345.0, tp1=2352.0, tp2=2359.0, tp3=2366.0,
        lots=lots, atr=5.0, quality=72.5, model="LQ",
        open_time=datetime.now(timezone.utc), bar_time=datetime.now(timezone.utc),
        magic=20250101,
    )
    executor._open_trades[ticket] = trade
    fake_mt5.positions[ticket] = _FakePosition(volume=volume_open, type_=0)
    return trade


# ─────────────────────────────────────────────────────────────────────────────
# Low-level broker call methods
# ─────────────────────────────────────────────────────────────────────────────

class TestMt5CallsReturnConfirmation:
    def test_modify_sl_returns_true_on_done(self, live_executor, fake_mt5):
        fake_mt5.next_retcode = fake_mt5.TRADE_RETCODE_DONE
        assert live_executor._mt5_modify_sl(1, 2345.0, 2366.0) is True

    def test_modify_sl_returns_false_on_rejection(self, live_executor, fake_mt5):
        fake_mt5.next_retcode = REJECTED
        assert live_executor._mt5_modify_sl(1, 2345.0, 2366.0) is False

    def test_close_returns_true_on_done(self, live_executor, fake_mt5):
        _open_long(live_executor, fake_mt5, ticket=1)
        fake_mt5.next_retcode = fake_mt5.TRADE_RETCODE_DONE
        assert live_executor._mt5_close(1, 0.3, "XAUUSD") is True

    def test_close_returns_false_on_rejection(self, live_executor, fake_mt5):
        _open_long(live_executor, fake_mt5, ticket=1)
        fake_mt5.next_retcode = REJECTED
        assert live_executor._mt5_close(1, 0.3, "XAUUSD") is False

    def test_close_returns_true_when_position_already_gone(self, live_executor, fake_mt5):
        # No position registered in fake_mt5.positions -> already closed
        # on the broker side (e.g. stopped out). Must not be treated as
        # a failure, or a legitimately-gone position blocks finalization
        # forever.
        assert live_executor._mt5_close(999, 0.3, "XAUUSD") is True

    def test_close_caps_requested_volume_to_what_is_actually_open(self, live_executor, fake_mt5):
        # lots_remaining could drift above the real open volume through
        # float rounding; the request must never ask for more than exists.
        _open_long(live_executor, fake_mt5, ticket=1, volume_open=0.05)
        live_executor._mt5_close(1, 0.30, "XAUUSD")
        sent = fake_mt5.sent_requests[-1]
        assert sent["volume"] == 0.05

    def test_partial_close_returns_true_on_done(self, live_executor, fake_mt5):
        _open_long(live_executor, fake_mt5, ticket=1)
        assert live_executor._mt5_partial_close(1, 0.1, "XAUUSD") is True

    def test_partial_close_returns_false_when_position_not_found(self, live_executor, fake_mt5):
        # Unlike full close, a missing position on a *partial* close
        # attempt is treated as failure/unknown, not silently as success --
        # the caller shouldn't assume a specific partial fill happened.
        assert live_executor._mt5_partial_close(999, 0.1, "XAUUSD") is False


# ─────────────────────────────────────────────────────────────────────────────
# _handle_tp1_hit / _handle_tp2_hit — confirmation gating
# ─────────────────────────────────────────────────────────────────────────────

class TestTp1HitGating:
    def test_confirmed_partial_close_commits_state(self, live_executor, fake_mt5):
        trade = _open_long(live_executor, fake_mt5, lots=0.3)
        fake_mt5.next_retcode = fake_mt5.TRADE_RETCODE_DONE
        live_executor._handle_tp1_hit(trade, verbose=False)
        assert trade.tp1_hit is True
        assert trade.lots_remaining == pytest.approx(0.3 - round(0.3 * live_executor.tp1_lots_pct, 5))
        assert trade.sl_moved is True
        assert trade.sl == trade.entry

    def test_rejected_partial_close_leaves_state_untouched(self, live_executor, fake_mt5):
        trade = _open_long(live_executor, fake_mt5, lots=0.3)
        original_lots_remaining = trade.lots_remaining
        original_sl = trade.sl
        fake_mt5.next_retcode = REJECTED
        live_executor._handle_tp1_hit(trade, verbose=False)
        assert trade.tp1_hit is False
        assert trade.lots_remaining == original_lots_remaining
        assert trade.sl_moved is False
        assert trade.sl == original_sl

    def test_rejected_then_confirmed_retry_eventually_commits(self, live_executor, fake_mt5):
        trade = _open_long(live_executor, fake_mt5, lots=0.3)
        fake_mt5.next_retcode = REJECTED
        live_executor._handle_tp1_hit(trade, verbose=False)
        assert trade.tp1_hit is False   # first attempt: broker said no

        fake_mt5.next_retcode = fake_mt5.TRADE_RETCODE_DONE
        live_executor._handle_tp1_hit(trade, verbose=False)
        assert trade.tp1_hit is True    # retry succeeds

    def test_sl_to_be_not_attempted_when_partial_close_rejected(self, live_executor, fake_mt5):
        trade = _open_long(live_executor, fake_mt5, lots=0.3)
        fake_mt5.next_retcode = REJECTED
        live_executor._handle_tp1_hit(trade, verbose=False)
        # No SL-modify request should have been sent at all -- the method
        # returns before reaching that step on a rejected partial close.
        assert all(r["action"] != fake_mt5.TRADE_ACTION_SLTP for r in fake_mt5.sent_requests)


class TestTp2HitGating:
    def test_confirmed_partial_close_commits_state(self, live_executor, fake_mt5):
        trade = _open_long(live_executor, fake_mt5, lots=0.3)
        fake_mt5.next_retcode = fake_mt5.TRADE_RETCODE_DONE
        live_executor._handle_tp2_hit(trade, verbose=False)
        assert trade.tp2_hit is True
        assert trade.sl == trade.tp1

    def test_rejected_partial_close_leaves_state_untouched(self, live_executor, fake_mt5):
        trade = _open_long(live_executor, fake_mt5, lots=0.3)
        original_sl = trade.sl
        fake_mt5.next_retcode = REJECTED
        live_executor._handle_tp2_hit(trade, verbose=False)
        assert trade.tp2_hit is False
        assert trade.sl == original_sl


# ─────────────────────────────────────────────────────────────────────────────
# _handle_full_close — confirmation gating (the core of the fix)
# ─────────────────────────────────────────────────────────────────────────────

class TestFullCloseGating:
    def test_confirmed_close_finalizes_and_removes_from_open_trades(self, live_executor, fake_mt5):
        trade = _open_long(live_executor, fake_mt5, ticket=1)
        fake_mt5.next_retcode = fake_mt5.TRADE_RETCODE_DONE
        event = live_executor._handle_full_close(trade, price=2366.0, reason="tp3", verbose=False)
        assert event is not None
        assert event["ticket"] == 1
        assert 1 not in live_executor._open_trades
        assert trade.status == "closed"

    def test_rejected_close_keeps_trade_open_and_returns_none(self, live_executor, fake_mt5):
        trade = _open_long(live_executor, fake_mt5, ticket=1)
        fake_mt5.next_retcode = REJECTED
        event = live_executor._handle_full_close(trade, price=2366.0, reason="tp3", verbose=False)
        # This is the core regression this fix targets: previously the
        # trade was deleted from _open_trades and marked "closed" even
        # though the broker rejected the close request.
        assert event is None
        assert 1 in live_executor._open_trades
        assert live_executor._open_trades[1].status == "open"

    def test_guard_not_credited_with_pnl_on_rejected_close(self, live_executor, fake_mt5):
        trade = _open_long(live_executor, fake_mt5, ticket=1)
        recorded = []
        live_executor.guard.record_closed_trade = lambda pnl, **kw: recorded.append(pnl)
        fake_mt5.next_retcode = REJECTED
        live_executor._handle_full_close(trade, price=2366.0, reason="tp3", verbose=False)
        assert recorded == []   # guard's daily-loss/drawdown tracking must not see a phantom close

    def test_retry_after_rejection_eventually_confirms(self, live_executor, fake_mt5):
        trade = _open_long(live_executor, fake_mt5, ticket=1)
        fake_mt5.next_retcode = REJECTED
        assert live_executor._handle_full_close(trade, price=2366.0, reason="tp3", verbose=False) is None
        assert 1 in live_executor._open_trades

        fake_mt5.next_retcode = fake_mt5.TRADE_RETCODE_DONE
        event = live_executor._handle_full_close(trade, price=2366.0, reason="tp3", verbose=False)
        assert event is not None
        assert 1 not in live_executor._open_trades

    def test_close_uses_lots_remaining_not_original_lots(self, live_executor, fake_mt5):
        # After a confirmed TP1 partial close, lots_remaining should be
        # less than the original lot size, and the final close request
        # must ask for lots_remaining, not the full original size.
        trade = _open_long(live_executor, fake_mt5, ticket=1, lots=0.3, volume_open=0.3)
        fake_mt5.next_retcode = fake_mt5.TRADE_RETCODE_DONE
        live_executor._handle_tp1_hit(trade, verbose=False)
        assert trade.lots_remaining < 0.3

        # Now the broker's open volume reflects the partial close.
        fake_mt5.positions[1] = _FakePosition(volume=trade.lots_remaining, type_=0)
        fake_mt5.sent_requests.clear()
        live_executor._handle_full_close(trade, price=2366.0, reason="tp3", verbose=False)
        close_request = fake_mt5.sent_requests[-1]
        assert close_request["volume"] == pytest.approx(trade.lots_remaining)


# ─────────────────────────────────────────────────────────────────────────────
# _update_trailing_sl — confirmation gating
# ─────────────────────────────────────────────────────────────────────────────

class TestTrailingSlGating:
    def _trailing_trade(self, executor, fake_mt5, ticket=1):
        trade = _open_long(executor, fake_mt5, ticket=ticket)
        trade.trail_active = True
        trade.trail_atr_mult = 1.0
        trade.trail_step_pct = 0.1
        trade.atr = 5.0
        trade.trail_sl_high = trade.entry
        return trade

    def test_confirmed_trail_move_updates_sl(self, live_executor, fake_mt5):
        trade = self._trailing_trade(live_executor, fake_mt5)
        fake_mt5.next_retcode = fake_mt5.TRADE_RETCODE_DONE
        live_executor._update_trailing_sl(trade, price=2365.0)   # far above entry
        assert trade.sl > 2345.0   # moved up from the original SL

    def test_rejected_trail_move_leaves_sl_untouched(self, live_executor, fake_mt5):
        trade = self._trailing_trade(live_executor, fake_mt5)
        original_sl = trade.sl
        fake_mt5.next_retcode = REJECTED
        live_executor._update_trailing_sl(trade, price=2365.0)
        assert trade.sl == original_sl


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard-facing manual operations
# ─────────────────────────────────────────────────────────────────────────────

class TestManualCloseGating:
    def test_confirmed_manual_close_succeeds(self, live_executor, fake_mt5):
        _open_long(live_executor, fake_mt5, ticket=1)
        fake_mt5.next_retcode = fake_mt5.TRADE_RETCODE_DONE
        result = live_executor.manual_close(1)
        assert result["ok"] is True
        assert 1 not in live_executor._open_trades

    def test_rejected_manual_close_reports_error_and_keeps_trade_open(self, live_executor, fake_mt5):
        _open_long(live_executor, fake_mt5, ticket=1)
        fake_mt5.next_retcode = REJECTED
        result = live_executor.manual_close(1)
        assert result["ok"] is False
        assert 1 in live_executor._open_trades


class TestManualModifySlGating:
    def test_confirmed_modify_commits_new_sl(self, live_executor, fake_mt5):
        trade = _open_long(live_executor, fake_mt5, ticket=1)
        trade.current_price = 2360.0
        fake_mt5.next_retcode = fake_mt5.TRADE_RETCODE_DONE
        result = live_executor.manual_modify_sl(1, 2350.0)
        assert result["ok"] is True
        assert trade.sl == 2350.0

    def test_rejected_modify_leaves_sl_unchanged_and_reports_error(self, live_executor, fake_mt5):
        trade = _open_long(live_executor, fake_mt5, ticket=1)
        trade.current_price = 2360.0
        original_sl = trade.sl
        fake_mt5.next_retcode = REJECTED
        result = live_executor.manual_modify_sl(1, 2350.0)
        assert result["ok"] is False
        assert trade.sl == original_sl


class TestCloseAllGating:
    def test_confirmed_close_all_clears_open_trades(self, live_executor, fake_mt5):
        _open_long(live_executor, fake_mt5, ticket=1)
        _open_long(live_executor, fake_mt5, ticket=2)
        fake_mt5.next_retcode = fake_mt5.TRADE_RETCODE_DONE
        live_executor.close_all(reason="emergency")
        assert live_executor._open_trades == {}

    def test_rejected_close_leaves_that_trade_in_open_trades(self, live_executor, fake_mt5):
        _open_long(live_executor, fake_mt5, ticket=1)
        _open_long(live_executor, fake_mt5, ticket=2)
        fake_mt5.next_retcode = REJECTED
        live_executor.close_all(reason="emergency")
        # Neither ticket confirmed -> both stay open, not silently dropped.
        assert set(live_executor._open_trades.keys()) == {1, 2}
