"""
Tests for funded_account_guard.py — prop-firm rule enforcement. This
directly protects real capital on a funded account, so the daily-loss
and drawdown math gets exact-value assertions, not just "it ran".

Key behavior this suite documents: the daily-loss gate is driven by the
guard's *own* tracked P&L (via record_closed_trade()), not by reading
account["balance"] vs account["equity"] directly — that pair only feeds
the drawdown-from-peak check. A test that wants to trigger the daily-loss
gate must actually record a loss first.
"""

import os
import tempfile

import pytest

from ceo_engine_mt5.funded_account_guard import FundedAccountGuard


@pytest.fixture
def guard():
    log_file = tempfile.mktemp(suffix=".json")
    g = FundedAccountGuard(
        account_size=10_000.0,
        daily_loss_limit_pct=5.0,
        max_drawdown_pct=10.0,
        consistency_pct=0.0,
        buffer_pct=0.5,
        log_file=log_file,
    )
    yield g
    if os.path.exists(log_file):
        os.unlink(log_file)


def _account(equity=10_000.0):
    return {"balance": equity, "equity": equity}


# ─────────────────────────────────────────────────────────────────────────────
# Daily loss limit
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyLossLimit:
    def test_allows_trade_with_no_recorded_loss_yet(self, guard):
        ok, reason = guard.pre_trade_check(
            account=_account(), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is True

    def test_blocks_trade_when_daily_loss_limit_breached(self, guard):
        # limit=$500, buffer=$50 → blocks at >= $450 lost today
        guard.record_closed_trade(pnl=-600.0, verbose=False)
        ok, reason = guard.pre_trade_check(
            account=_account(9_400.0), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is False
        assert "loss" in reason.lower()

    def test_buffer_blocks_before_the_hard_limit(self):
        """buffer_pct=1.0 → blocks at (limit - $100), not at the full $500."""
        log_file = tempfile.mktemp(suffix=".json")
        g = FundedAccountGuard(account_size=10_000.0, daily_loss_limit_pct=5.0,
                                buffer_pct=1.0, log_file=log_file)
        # limit=$500, buffer=$100 → blocked at $400; -450 crosses it while
        # still being under the *hard* $500 limit — the buffer is working.
        g.record_closed_trade(pnl=-450.0, verbose=False)
        ok, reason = g.pre_trade_check(
            account=_account(9_550.0), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is False
        if os.path.exists(log_file):
            os.unlink(log_file)

    def test_just_under_buffer_threshold_is_still_allowed(self, guard):
        """$440 loss with a $450 threshold — trade should still be allowed."""
        guard.record_closed_trade(pnl=-440.0, verbose=False)
        ok, _ = guard.pre_trade_check(
            account=_account(9_560.0), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is True

    def test_halt_persists_after_daily_loss_trigger(self, guard):
        """Once the daily-loss gate fires and triggers a halt, subsequent
        calls must also be blocked — even if called with a clean account."""
        guard.record_closed_trade(pnl=-600.0, verbose=False)
        guard.pre_trade_check(
            account=_account(9_400.0), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        ok, reason = guard.pre_trade_check(
            account=_account(10_000.0), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is False
        assert "halt" in reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Drawdown limit
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawdownLimit:
    def test_blocks_when_trailing_drawdown_exceeds_max_without_tripping_daily_loss(self):
        """Isolate the drawdown gate: high daily-loss limit so that gate
        never fires; tight drawdown limit so this one does."""
        log_file = tempfile.mktemp(suffix=".json")
        g = FundedAccountGuard(account_size=10_000.0, daily_loss_limit_pct=90.0,
                                max_drawdown_pct=5.0, buffer_pct=0.0, log_file=log_file)
        # Establish a peak of $10,500 via the first call (side-effect of
        # _current_drawdown updating peak on every pre_trade_check call).
        g.pre_trade_check(account=_account(10_500.0), open_trades=0,
                           has_sl=True, symbol="XAUUSD", direction="long", verbose=False)
        assert g._peak_equity == 10_500.0

        # $700 drawdown from $10,500 peak = 6.67% → exceeds the 5% limit
        ok, reason = g.pre_trade_check(
            account=_account(9_800.0), open_trades=0, has_sl=True,
            symbol="XAUUSD", direction="long", verbose=False,
        )
        assert ok is False
        assert "drawdown" in reason.lower()
        if os.path.exists(log_file):
            os.unlink(log_file)

    def test_allows_when_drawdown_within_limit(self):
        log_file = tempfile.mktemp(suffix=".json")
        g = FundedAccountGuard(account_size=10_000.0, daily_loss_limit_pct=90.0,
                                max_drawdown_pct=10.0, buffer_pct=0.0, log_file=log_file)
        # $200 drawdown on a $10,000 account = 2% — well within the 10% limit
        ok, _ = g.pre_trade_check(
            account=_account(9_800.0), open_trades=0, has_sl=True,
            symbol="XAUUSD", direction="long", verbose=False,
        )
        assert ok is True
        if os.path.exists(log_file):
            os.unlink(log_file)


# ─────────────────────────────────────────────────────────────────────────────
# SL requirement
# ─────────────────────────────────────────────────────────────────────────────

class TestRequireStopLoss:
    def test_blocks_trade_without_stop_loss(self, guard):
        ok, reason = guard.pre_trade_check(
            account=_account(), open_trades=0, has_sl=False, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is False
        assert "stop" in reason.lower() or "sl" in reason.lower()

    def test_allows_when_sl_present(self, guard):
        ok, _ = guard.pre_trade_check(
            account=_account(), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is True


# ─────────────────────────────────────────────────────────────────────────────
# Max open trades
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxOpenTrades:
    def test_blocks_when_at_max_open_trades(self):
        log_file = tempfile.mktemp(suffix=".json")
        g = FundedAccountGuard(account_size=10_000.0, max_open_trades=2, log_file=log_file)
        ok, reason = g.pre_trade_check(
            account=_account(), open_trades=2, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is False
        assert "open" in reason.lower()
        if os.path.exists(log_file):
            os.unlink(log_file)

    def test_allows_when_one_below_max(self):
        log_file = tempfile.mktemp(suffix=".json")
        g = FundedAccountGuard(account_size=10_000.0, max_open_trades=2, log_file=log_file)
        ok, _ = g.pre_trade_check(
            account=_account(), open_trades=1, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is True
        if os.path.exists(log_file):
            os.unlink(log_file)

    def test_max_open_trades_zero_means_unlimited(self, guard):
        """0 = unlimited — should never block on open-trade count."""
        assert guard.max_open_trades == 0
        ok, _ = guard.pre_trade_check(
            account=_account(), open_trades=999, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is True


# ─────────────────────────────────────────────────────────────────────────────
# Daily trade count
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxDailyTrades:
    def test_blocks_at_daily_trade_limit(self):
        log_file = tempfile.mktemp(suffix=".json")
        g = FundedAccountGuard(account_size=10_000.0, max_daily_trades=3, log_file=log_file)
        for _ in range(3):
            g.record_closed_trade(pnl=10.0, verbose=False)
        ok, reason = g.pre_trade_check(
            account=_account(), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is False
        assert "daily" in reason.lower()
        if os.path.exists(log_file):
            os.unlink(log_file)

    def test_allows_below_daily_trade_limit(self):
        log_file = tempfile.mktemp(suffix=".json")
        g = FundedAccountGuard(account_size=10_000.0, max_daily_trades=3, log_file=log_file)
        for _ in range(2):
            g.record_closed_trade(pnl=10.0, verbose=False)
        ok, _ = g.pre_trade_check(
            account=_account(), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is True
        if os.path.exists(log_file):
            os.unlink(log_file)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation exposure
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrelationExposure:
    def test_does_not_crash_on_correlated_symbol_check(self, guard):
        """EURUSD and GBPUSD are both USD-negative. The check must not crash."""
        open_trades_detail = [{"symbol": "EURUSD", "direction": "long"}]
        ok, reason = guard.pre_trade_check(
            account=_account(), open_trades=1, has_sl=True, symbol="GBPUSD",
            direction="long", open_trade_details=open_trades_detail, verbose=False,
        )
        assert isinstance(ok, bool)
        assert isinstance(reason, str)

    def test_blocks_correlated_same_direction(self, guard):
        """GBPUSD long while EURUSD long is open → both USD-negative, same
        direction → effective double exposure → must be blocked."""
        open_trades_detail = [{"symbol": "EURUSD", "direction": "long"}]
        ok, reason = guard.pre_trade_check(
            account=_account(), open_trades=1, has_sl=True, symbol="GBPUSD",
            direction="long", open_trade_details=open_trades_detail, verbose=False,
        )
        assert ok is False
        assert "correlation" in reason.lower()

    def test_allows_correlated_opposite_direction(self, guard):
        """EURUSD long + GBPUSD short is a hedge, not double-risk — allowed."""
        open_trades_detail = [{"symbol": "EURUSD", "direction": "long"}]
        ok, _ = guard.pre_trade_check(
            account=_account(), open_trades=1, has_sl=True, symbol="GBPUSD",
            direction="short", open_trade_details=open_trades_detail, verbose=False,
        )
        assert ok is True

    def test_allows_uncorrelated_symbols_same_direction(self, guard):
        """XAUUSD and EURUSD have no correlation entry — two longs are fine."""
        open_trades_detail = [{"symbol": "EURUSD", "direction": "long"}]
        ok, _ = guard.pre_trade_check(
            account=_account(), open_trades=1, has_sl=True, symbol="XAUUSD",
            direction="long", open_trade_details=open_trades_detail, verbose=False,
        )
        assert ok is True

    def test_crypto_correlation_blocked(self, guard):
        """BTCUSD long + ETHUSD long → known crypto pair → blocked."""
        open_trades_detail = [{"symbol": "BTCUSD", "direction": "long"}]
        ok, reason = guard.pre_trade_check(
            account=_account(), open_trades=1, has_sl=True, symbol="ETHUSD",
            direction="long", open_trade_details=open_trades_detail, verbose=False,
        )
        assert ok is False
        assert "correlation" in reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Hold time (pre_close_check)
# ─────────────────────────────────────────────────────────────────────────────

class TestHoldTime:
    def test_blocks_close_before_minimum_hold_time(self):
        from datetime import datetime, timedelta, timezone
        log_file = tempfile.mktemp(suffix=".json")
        g = FundedAccountGuard(account_size=10_000.0, min_hold_minutes=5, log_file=log_file)
        open_time = datetime.now(timezone.utc) - timedelta(minutes=2)
        ok, reason = g.pre_close_check(trade_open_time=open_time)
        assert ok is False
        assert "hold" in reason.lower()
        if os.path.exists(log_file):
            os.unlink(log_file)

    def test_allows_close_after_minimum_hold_time(self):
        from datetime import datetime, timedelta, timezone
        log_file = tempfile.mktemp(suffix=".json")
        g = FundedAccountGuard(account_size=10_000.0, min_hold_minutes=5, log_file=log_file)
        open_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        ok, _ = g.pre_close_check(trade_open_time=open_time)
        assert ok is True
        if os.path.exists(log_file):
            os.unlink(log_file)

    def test_hold_time_zero_always_allows(self, guard):
        """min_hold_minutes=0 means no requirement — any close time is fine."""
        from datetime import datetime, timezone
        assert guard.min_hold_minutes == 0
        ok, _ = guard.pre_close_check(
            trade_open_time=datetime.now(timezone.utc)
        )
        assert ok is True


# ─────────────────────────────────────────────────────────────────────────────
# Consistency score (post-trade warning, not a hard block)
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistencyScore:
    def _guard_with_consistency(self, pct=30.0):
        log_file = tempfile.mktemp(suffix=".json")
        return FundedAccountGuard(
            account_size=10_000.0,
            daily_loss_limit_pct=90.0,  # ensure daily-loss gate doesn't interfere
            consistency_pct=pct,
            log_file=log_file,
        ), log_file

    def test_consistency_breach_recorded_without_blocking_trade(self):
        """Consistency is a *warning*, not a pre-trade block. A single day
        that accounts for more than the consistency % of total profits
        triggers a console warning but must NOT return ok=False."""
        g, lf = self._guard_with_consistency(pct=30.0)
        # Day 1: profit $100 → total_profit = $100
        from datetime import date, timedelta
        g.record_closed_trade(pnl=100.0, trade_date=date(2024, 1, 1), verbose=False)
        # Day 2: profit $200 → today_profit is 66% of total ($300) > 30% limit
        g.record_closed_trade(pnl=200.0, trade_date=date(2024, 1, 2), verbose=False)
        # The gate should still pass — consistency is advisory, not blocking
        ok, _ = g.pre_trade_check(
            account=_account(), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is True
        if os.path.exists(lf):
            os.unlink(lf)

    def test_consistency_zero_disables_check(self):
        """consistency_pct=0 means the check is disabled — no warning ever."""
        g, lf = self._guard_with_consistency(pct=0.0)
        g.record_closed_trade(pnl=10_000.0, verbose=False)   # absurdly large
        ok, _ = g.pre_trade_check(
            account=_account(), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is True
        if os.path.exists(lf):
            os.unlink(lf)


# ─────────────────────────────────────────────────────────────────────────────
# Emergency halt
# ─────────────────────────────────────────────────────────────────────────────

class TestHaltAndReset:
    def test_trigger_halt_blocks_all_subsequent_trades(self, guard):
        guard._trigger_halt("manual test halt")
        ok, reason = guard.pre_trade_check(
            account=_account(), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is False

    def test_reset_halt_without_confirm_raises(self, guard):
        guard._trigger_halt("manual test halt")
        with pytest.raises(ValueError):
            guard.reset_halt(confirm=False)
        # still halted after the failed reset
        ok, _ = guard.pre_trade_check(
            account=_account(), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is False

    def test_reset_halt_with_confirm_actually_resets(self, guard):
        guard._trigger_halt("manual test halt")
        guard.reset_halt(confirm=True)
        ok, _ = guard.pre_trade_check(
            account=_account(), open_trades=0, has_sl=True, symbol="XAUUSD",
            direction="long", verbose=False,
        )
        assert ok is True


# ─────────────────────────────────────────────────────────────────────────────
# Record closed trade
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordClosedTrade:
    def test_today_pnl_accumulates_across_trades(self, guard):
        guard.record_closed_trade(pnl=100.0, verbose=False)
        guard.record_closed_trade(pnl=-30.0, verbose=False)
        assert guard._today_pnl() == pytest.approx(70.0)

    def test_today_trades_counts_each_call(self, guard):
        guard.record_closed_trade(pnl=10.0, verbose=False)
        guard.record_closed_trade(pnl=-5.0, verbose=False)
        guard.record_closed_trade(pnl=20.0, verbose=False)
        assert guard._today_trades() == 3

    def test_trading_days_set_grows_with_each_new_date(self, guard):
        from datetime import date
        guard.record_closed_trade(pnl=10.0, trade_date=date(2024, 1, 1), verbose=False)
        guard.record_closed_trade(pnl=20.0, trade_date=date(2024, 1, 2), verbose=False)
        guard.record_closed_trade(pnl=30.0, trade_date=date(2024, 1, 2), verbose=False)
        assert len(guard._trading_days_set) == 2   # two distinct dates

    def test_total_profit_only_sums_profitable_days(self, guard):
        from datetime import date
        guard.record_closed_trade(pnl=200.0, trade_date=date(2024, 1, 1), verbose=False)
        guard.record_closed_trade(pnl=-50.0, trade_date=date(2024, 1, 2), verbose=False)
        # Only day 1 ($200) counts toward total_profit — losing days excluded
        assert guard._total_profit() == pytest.approx(200.0)
