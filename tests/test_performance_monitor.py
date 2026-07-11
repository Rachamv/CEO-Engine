"""
Tests for performance_monitor.py — the live-vs-backtest feedback loop.
update() was refactored from cyclomatic complexity 34 to 15 in v2.2.0 by
splitting into 5 helper methods; these scenarios are the same ones used
to verify that refactor changed nothing, now locked in as a real suite.
"""

import pytest

from ceo_engine_mt5.performance_monitor import PerformanceMonitor


class FakeJournal:
    def __init__(self, trades):
        self._trades = trades

    def recent_trades(self, limit):
        return self._trades[:limit]


class FakeRiskEngine:
    def __init__(self):
        self.risk_pct = 1.0
        self.min_quality = 50.0


def _pm(trades, **kwargs):
    re = FakeRiskEngine()
    pm = PerformanceMonitor(
        journal=FakeJournal(trades), risk_engine=re,
        baseline_risk_pct=1.0, rolling_window=10,
        loss_streak_trigger=3, win_rate_floor=35.0,
        **kwargs,
    )
    return pm, re


class TestLossStreakReduction:
    def test_three_consecutive_losses_reduces_risk(self):
        trades = ([{"pnl": -10, "r_multiple": -1.0}] * 3 +
                  [{"pnl": 50, "r_multiple": 2.0}] * 7)
        pm, re = _pm(trades)
        state = pm.update(verbose=False)
        assert state.risk_reduced is True
        assert re.risk_pct < 1.0
        assert "consecutive losses" in state.reduction_reason

    def test_two_losses_do_not_trigger_the_three_loss_gate(self):
        trades = ([{"pnl": -10, "r_multiple": -1.0}] * 2 +
                  [{"pnl": 50, "r_multiple": 2.0}] * 8)
        pm, re = _pm(trades)
        state = pm.update(verbose=False)
        assert state.risk_reduced is False
        assert re.risk_pct == 1.0


class TestWinRateFloor:
    def test_win_rate_below_floor_reduces_risk(self):
        # newest-first; starts with a win so the loss-streak gate (checked
        # first, by priority) doesn't also fire — isolates the win-rate-floor
        # trigger specifically. Still 7 losses / 3 wins = 30% WR, below the 35% floor.
        trades = ([{"pnl": 50, "r_multiple": 2.0}] +
                  [{"pnl": -10, "r_multiple": -1.0}] * 7 +
                  [{"pnl": 50, "r_multiple": 2.0}] * 2)
        pm, re = _pm(trades)
        state = pm.update(verbose=False)
        assert state.risk_reduced is True
        assert state.consecutive_losses == 0  # confirms the loss-streak gate didn't fire
        assert "win rate" in state.reduction_reason


class TestHealthyBaseline:
    def test_all_wins_stays_at_baseline_not_reduced(self):
        trades = [{"pnl": 50, "r_multiple": 2.0}] * 10
        pm, re = _pm(trades)
        state = pm.update(verbose=False)
        assert state.risk_reduced is False
        assert re.risk_pct == 1.0
        assert state.reduction_reason == ""

    def test_not_enough_trades_leaves_baseline_untouched(self):
        trades = [{"pnl": 50, "r_multiple": 2.0}] * 2  # below min_trades_for_eval
        pm, re = _pm(trades)
        state = pm.update(verbose=False)
        assert re.risk_pct == 1.0


class TestRecovery:
    def test_partial_recovery_steps_back_toward_baseline(self):
        """3 recent consecutive wins, but overall WR (80%) and avg_r (+0.6)
        are healthy — the ONLY active trigger should be recovery."""
        trades = (
            [{"pnl": 50, "r_multiple": 1.0}] * 3 +
            [{"pnl": -10, "r_multiple": -1.0}] +
            [{"pnl": 50, "r_multiple": 1.0}] * 3 +
            [{"pnl": -10, "r_multiple": -1.0}] +
            [{"pnl": 50, "r_multiple": 1.0}] * 2
        )
        pm, re = _pm(trades, recovery_wins_required=5)
        pm._state.risk_reduced = True
        pm._state.current_risk_pct = 0.5
        pm._state.current_min_quality = 60.0

        state = pm.update(verbose=False)
        assert state.risk_reduced is True
        assert 0.5 < re.risk_pct < 1.0
        assert "recovering" in state.reduction_reason

    def test_full_recovery_after_enough_consecutive_wins(self):
        trades = [{"pnl": 50, "r_multiple": 1.0}] * 10
        pm, re = _pm(trades, recovery_wins_required=5)
        pm._state.risk_reduced = True
        pm._state.current_risk_pct = 0.5
        pm._state.current_min_quality = 60.0

        state = pm.update(verbose=False)
        assert state.risk_reduced is False
        assert re.risk_pct == 1.0
        assert state.reduction_reason == ""

    def test_recovery_does_not_override_an_active_loss_streak(self):
        """If a NEW loss streak triggers while a previous reduction was
        already in effect, the loss-streak reason should win, not recovery."""
        trades = [{"pnl": -10, "r_multiple": -1.0}] * 3 + [{"pnl": 50, "r_multiple": 1.0}] * 7
        pm, re = _pm(trades)
        pm._state.risk_reduced = True
        pm._state.current_risk_pct = 0.5
        pm._state.current_min_quality = 60.0

        state = pm.update(verbose=False)
        assert "consecutive losses" in state.reduction_reason


class TestDivergenceCheck:
    def test_flags_when_live_wr_far_below_backtest_wr(self):
        trades = [{"pnl": -10, "r_multiple": -0.5}] * 7 + [{"pnl": 50, "r_multiple": 1.0}] * 3
        pm, re = _pm(trades)
        pm.register_backtest_winrate("XAUUSD", "M15", 70.0)
        state = pm.update(symbol="XAUUSD", tf="M15", verbose=False)
        assert state.divergence_flag is True
        assert state.backtest_win_rate == 70.0

    def test_no_divergence_flag_without_a_registered_backtest_winrate(self):
        trades = [{"pnl": 50, "r_multiple": 1.0}] * 10
        pm, re = _pm(trades)
        state = pm.update(symbol="UNKNOWN", tf="M15", verbose=False)
        assert state.divergence_flag is False
        assert state.backtest_win_rate is None
