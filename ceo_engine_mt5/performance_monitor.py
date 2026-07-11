"""
The CEO Protocol — Performance Feedback Loop
==================================================
Reads the trade journal, computes rolling performance over the
last N trades, and dynamically adjusts RiskEngine.risk_pct and
RiskEngine.min_quality based on how the system is actually performing
live — not just what the backtest predicted.

Core idea
---------
A backtest tells you what *should* happen on average. Live trading
tells you what *is* happening right now. When the two diverge —
say the model backtested at 55% WR but the last 10 live trades are
30% WR — that's a signal to reduce risk before a real drawdown
compounds, not after.

This module is read-only with respect to trade history (it never
writes to the journal) and only ever ADJUSTS sizing/quality
thresholds — it never places or closes trades itself.

Behaviour
---------
    Losing streak detection
        N consecutive losses → risk_pct drops by a configurable factor
        Recovers automatically once a win breaks the streak

    Rolling win-rate / expectancy tracking
        Last N trades (default 10) win rate and avg R computed
        If win rate drops below a floor → risk reduced
        If expectancy turns negative → risk reduced further

    Backtest vs live divergence flag
        Compares live rolling win rate to the backtest win rate
        the model was registered with. Large negative divergence
        triggers a flag for manual review (does NOT auto-halt —
        that's FundedAccountGuard's job).

    Recovery
        After a configurable number of consecutive wins, or once
        rolling expectancy returns positive, risk_pct restores
        toward the baseline gradually (not instantly).

This module does NOT:
    - Place or close trades
    - Override FundedAccountGuard (which remains the hard safety net)
    - Permanently change the registered model

Usage
-----
    from .performance_monitor import PerformanceMonitor

    monitor = PerformanceMonitor(
        journal             = journal,
        risk_engine         = risk_engine,
        baseline_risk_pct   = 1.0,
        rolling_window      = 10,
        loss_streak_trigger = 3,
        loss_streak_factor  = 0.5,
    )

    # Call once per bar close, or on a timer (e.g. every hour)
    monitor.update(symbol="XAUUSD", tf="M15", verbose=True)

    # Check current state
    state = monitor.status()
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional



# ─────────────────────────────────────────────────────────────────────────────
# State snapshot
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerfState:
    """Current performance feedback state."""
    rolling_trades:       int     = 0
    rolling_win_rate:     float   = 0.0
    rolling_avg_r:        float   = 0.0
    rolling_expectancy:   float   = 0.0
    consecutive_losses:   int     = 0
    consecutive_wins:     int     = 0
    current_risk_pct:     float   = 0.0
    baseline_risk_pct:    float   = 0.0
    current_min_quality:  float   = 0.0
    baseline_min_quality: float   = 0.0
    risk_reduced:         bool    = False
    reduction_reason:     str     = ""
    backtest_win_rate:    Optional[float] = None
    divergence_flag:      bool    = False
    divergence_detail:    str     = ""
    last_update:          Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"\n{'─'*55}",
            f"  PERFORMANCE FEEDBACK STATUS",
            f"{'─'*55}",
            f"  Rolling window    : last {self.rolling_trades} trades",
            f"  Rolling Win Rate  : {self.rolling_win_rate:.1f}%",
            f"  Rolling Avg R     : {self.rolling_avg_r:+.3f}R",
            f"  Rolling Expectancy: {self.rolling_expectancy:+.3f}",
            f"  Consecutive losses: {self.consecutive_losses}",
            f"  Consecutive wins  : {self.consecutive_wins}",
            f"  ─────────────────────────────────────",
            f"  Risk pct (current): {self.current_risk_pct:.2f}%  "
            f"(baseline {self.baseline_risk_pct:.2f}%)",
            f"  Min quality (cur) : {self.current_min_quality:.0f}  "
            f"(baseline {self.baseline_min_quality:.0f})",
        ]
        if self.risk_reduced:
            lines.append(f"  ⚠️  REDUCED: {self.reduction_reason}")
        if self.divergence_flag:
            lines.append(f"  🚩  DIVERGENCE: {self.divergence_detail}")
        lines.append(f"{'─'*55}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PerformanceMonitor
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceMonitor:
    """
    Reads the journal and adjusts RiskEngine sizing/quality thresholds
    based on rolling live performance.

    Parameters
    ----------
    journal               : Journal instance (read-only access)
    risk_engine            : RiskEngine instance to adjust
    baseline_risk_pct      : the "normal" risk_pct to recover toward
                             (default: whatever risk_engine.risk_pct is at init)
    baseline_min_quality   : the "normal" min_quality to recover toward
    rolling_window         : number of recent trades to evaluate (default 10)
    min_trades_for_eval    : don't adjust until at least this many trades
                             exist in the rolling window (default 5)
    loss_streak_trigger    : consecutive losses that triggers a risk cut (default 3)
    loss_streak_factor     : multiply risk_pct by this factor on trigger (default 0.5)
    win_rate_floor         : if rolling win rate drops below this %, reduce risk (default 35.0)
    win_rate_floor_factor  : risk reduction factor when win rate floor breached (default 0.6)
    negative_expectancy_factor : risk reduction factor when rolling expectancy < 0 (default 0.5)
    recovery_wins_required : consecutive wins needed to fully restore baseline (default 3)
    recovery_step          : fraction of the gap to baseline restored per recovery win (default 0.34)
    divergence_threshold   : percentage-point drop vs backtest WR that triggers a flag (default 20.0)
    quality_increase_on_reduction : raise min_quality by this many points when risk is cut (default 10.0)
    """

    def __init__(
        self,
        journal,
        risk_engine,
        baseline_risk_pct:          Optional[float] = None,
        baseline_min_quality:       Optional[float] = None,
        rolling_window:             int   = 10,
        min_trades_for_eval:        int   = 5,
        loss_streak_trigger:        int   = 3,
        loss_streak_factor:         float = 0.5,
        win_rate_floor:             float = 35.0,
        win_rate_floor_factor:      float = 0.6,
        negative_expectancy_factor: float = 0.5,
        recovery_wins_required:     int   = 3,
        recovery_step:              float = 0.34,
        divergence_threshold:       float = 20.0,
        quality_increase_on_reduction: float = 10.0,
    ):
        self.journal     = journal
        self.risk_engine = risk_engine

        self.baseline_risk_pct = (
            baseline_risk_pct if baseline_risk_pct is not None
            else risk_engine.risk_pct
        )
        self.baseline_min_quality = (
            baseline_min_quality if baseline_min_quality is not None
            else getattr(risk_engine, "min_quality", 50.0)
        )

        self.rolling_window             = rolling_window
        self.min_trades_for_eval        = min_trades_for_eval
        self.loss_streak_trigger        = loss_streak_trigger
        self.loss_streak_factor         = loss_streak_factor
        self.win_rate_floor             = win_rate_floor
        self.win_rate_floor_factor      = win_rate_floor_factor
        self.negative_expectancy_factor = negative_expectancy_factor
        self.recovery_wins_required     = recovery_wins_required
        self.recovery_step              = recovery_step
        self.divergence_threshold       = divergence_threshold
        self.quality_increase_on_reduction = quality_increase_on_reduction

        self._state = PerfState(
            current_risk_pct     = self.baseline_risk_pct,
            baseline_risk_pct    = self.baseline_risk_pct,
            current_min_quality  = self.baseline_min_quality,
            baseline_min_quality = self.baseline_min_quality,
        )
        self._consecutive_wins_since_reduction = 0
        self._backtest_win_rates: Dict[str, float] = {}   # "SYMBOL_TF" -> WR

    # ── Backtest reference registration ──────────────────────────────────────

    def register_backtest_winrate(self, symbol: str, tf: str, win_rate: float):
        """
        Call once after walk-forward / backtest model selection to give
        the monitor a reference point for divergence detection.
        """
        key = f"{symbol}_{tf}".upper()
        self._backtest_win_rates[key] = win_rate

    # ── Core update ───────────────────────────────────────────────────────────

    @staticmethod
    def _compute_streaks(trades: list) -> tuple:
        """
        Consecutive loss/win streak from the most recent trades (newest
        first). Stops counting at the first trade that breaks the streak
        or has no recorded P&L.
        """
        consecutive_losses = 0
        consecutive_wins = 0
        for t in trades:
            pnl = t.get("pnl")
            if pnl is None:
                break
            if pnl < 0:
                if consecutive_wins > 0:
                    break
                consecutive_losses += 1
            elif pnl > 0:
                if consecutive_losses > 0:
                    break
                consecutive_wins += 1
            else:
                break
        return consecutive_losses, consecutive_wins

    def _determine_risk_adjustment(self, consecutive_losses: int, win_rate: float,
                                    avg_r: float) -> tuple:
        """
        Decides whether to reduce risk/raise the quality bar based on the
        three independent triggers (loss streak, win-rate floor, negative
        expectancy), in that priority order. Returns
        (new_risk_pct, new_min_quality, reduced, reason).
        """
        new_risk_pct    = self.baseline_risk_pct
        new_min_quality = self.baseline_min_quality
        reduced = False
        reason  = ""

        if consecutive_losses >= self.loss_streak_trigger:
            new_risk_pct    = self.baseline_risk_pct * self.loss_streak_factor
            new_min_quality = self.baseline_min_quality + self.quality_increase_on_reduction
            reduced = True
            reason  = (f"{consecutive_losses} consecutive losses "
                      f"(trigger: {self.loss_streak_trigger})")

        elif win_rate < self.win_rate_floor:
            candidate = self.baseline_risk_pct * self.win_rate_floor_factor
            if candidate < new_risk_pct:
                new_risk_pct = candidate
                new_min_quality = self.baseline_min_quality + self.quality_increase_on_reduction / 2
                reduced = True
                reason = (f"rolling win rate {win_rate:.1f}% below floor "
                         f"{self.win_rate_floor:.1f}%")

        elif avg_r < 0:
            candidate = self.baseline_risk_pct * self.negative_expectancy_factor
            if candidate < new_risk_pct:
                new_risk_pct = candidate
                new_min_quality = self.baseline_min_quality + self.quality_increase_on_reduction / 2
                reduced = True
                reason = f"rolling expectancy negative ({avg_r:+.3f}R avg)"

        return new_risk_pct, new_min_quality, reduced, reason

    def _apply_recovery(self, reduced: bool, consecutive_wins: int,
                         new_risk_pct: float, new_min_quality: float) -> tuple:
        """
        If a previous reduction is in effect and we're now winning again,
        steps risk back toward baseline (fully, if recovery_wins_required
        consecutive wins have been reached; partially otherwise). Returns
        the possibly-adjusted (new_risk_pct, new_min_quality, reduced, reason).
        """
        was_reduced = self._state.risk_reduced
        reason = ""
        if was_reduced and not reduced and consecutive_wins > 0:
            self._consecutive_wins_since_reduction = consecutive_wins
            if consecutive_wins >= self.recovery_wins_required:
                new_risk_pct    = self.baseline_risk_pct
                new_min_quality = self.baseline_min_quality
            else:
                gap_risk = self.baseline_risk_pct - self._state.current_risk_pct
                gap_q    = self.baseline_min_quality - self._state.current_min_quality
                new_risk_pct    = self._state.current_risk_pct + gap_risk * self.recovery_step
                new_min_quality = self._state.current_min_quality + gap_q * self.recovery_step
                reduced = True
                reason  = (f"recovering — {consecutive_wins}/"
                          f"{self.recovery_wins_required} wins since reduction")
        elif not reduced:
            self._consecutive_wins_since_reduction = 0
        return new_risk_pct, new_min_quality, reduced, reason

    def _check_divergence(self, symbol: Optional[str], tf: Optional[str],
                           win_rate: float) -> tuple:
        """
        Compares live rolling win rate against the registered backtest
        win rate for this symbol/tf, if one was registered. Returns
        (divergence_flag, divergence_detail, backtest_win_rate).
        """
        if not (symbol and tf):
            return False, "", None
        key = f"{symbol}_{tf}".upper()
        backtest_wr = self._backtest_win_rates.get(key)
        if backtest_wr is None:
            return False, "", None
        drop = backtest_wr - win_rate
        if drop < self.divergence_threshold:
            return False, "", backtest_wr
        detail = (
            f"Live WR {win_rate:.1f}% is {drop:.1f}pp below "
            f"backtest WR {backtest_wr:.1f}% for {symbol} {tf} "
            f"— model may be underperforming live, review recommended"
        )
        return True, detail, backtest_wr

    def update(
        self,
        symbol:  Optional[str] = None,
        tf:      Optional[str] = None,
        verbose: bool = True,
    ) -> PerfState:
        """
        Recompute rolling stats from the journal and adjust risk_engine.
        Call this periodically (e.g. once per bar close, or on a timer).

        Delegates to the five helper methods above — this method just
        sequences them: rolling stats → streaks → risk adjustment →
        recovery → divergence check → apply to RiskEngine → update state.
        """
        trades = self.journal.recent_trades(limit=self.rolling_window)

        if len(trades) < self.min_trades_for_eval:
            # Not enough data yet — leave at baseline
            self._state.rolling_trades = len(trades)
            self._state.last_update = datetime.now(timezone.utc).isoformat()
            return self._state

        pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
        rs   = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]

        wins     = [p for p in pnls if p > 0]
        win_rate = (len(wins) / len(pnls) * 100) if pnls else 0.0
        avg_r    = (sum(rs) / len(rs)) if rs else 0.0

        consecutive_losses, consecutive_wins = self._compute_streaks(trades)

        new_risk_pct, new_min_quality, reduced, reason = self._determine_risk_adjustment(
            consecutive_losses, win_rate, avg_r
        )
        new_risk_pct, new_min_quality, reduced, recovery_reason = self._apply_recovery(
            reduced, consecutive_wins, new_risk_pct, new_min_quality
        )
        reason = recovery_reason or reason

        divergence_flag, divergence_detail, backtest_wr = self._check_divergence(
            symbol, tf, win_rate
        )

        # ── Apply to RiskEngine ──────────────────────────────────────────────
        risk_changed = abs(new_risk_pct - self.risk_engine.risk_pct) > 1e-6
        self.risk_engine.risk_pct    = round(new_risk_pct, 4)
        self.risk_engine.min_quality = round(new_min_quality, 2)

        # ── Update state ─────────────────────────────────────────────────────
        self._state = PerfState(
            rolling_trades       = len(trades),
            rolling_win_rate     = round(win_rate, 1),
            rolling_avg_r        = round(avg_r, 3),
            rolling_expectancy   = round(avg_r, 3),
            consecutive_losses   = consecutive_losses,
            consecutive_wins     = consecutive_wins,
            current_risk_pct     = round(new_risk_pct, 4),
            baseline_risk_pct    = self.baseline_risk_pct,
            current_min_quality  = round(new_min_quality, 2),
            baseline_min_quality = self.baseline_min_quality,
            risk_reduced         = reduced,
            reduction_reason     = reason,
            backtest_win_rate    = backtest_wr,
            divergence_flag      = divergence_flag,
            divergence_detail    = divergence_detail,
            last_update          = datetime.now(timezone.utc).isoformat(),
        )

        if verbose and (reduced or risk_changed or divergence_flag):
            print(self._state.summary())

        return self._state

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Returns the current state as a dict (for dashboard / logging)."""
        s = self._state
        return {
            "rolling_trades":        s.rolling_trades,
            "rolling_win_rate":      s.rolling_win_rate,
            "rolling_avg_r":         s.rolling_avg_r,
            "consecutive_losses":    s.consecutive_losses,
            "consecutive_wins":      s.consecutive_wins,
            "current_risk_pct":      s.current_risk_pct,
            "baseline_risk_pct":     s.baseline_risk_pct,
            "current_min_quality":   s.current_min_quality,
            "baseline_min_quality":  s.baseline_min_quality,
            "risk_reduced":          s.risk_reduced,
            "reduction_reason":      s.reduction_reason,
            "divergence_flag":       s.divergence_flag,
            "divergence_detail":     s.divergence_detail,
            "last_update":           s.last_update,
        }

    def reset(self):
        """Reset to baseline — e.g. after manual review or new trading day."""
        self.risk_engine.risk_pct    = self.baseline_risk_pct
        self.risk_engine.min_quality = self.baseline_min_quality
        self._consecutive_wins_since_reduction = 0
        self._state = PerfState(
            current_risk_pct     = self.baseline_risk_pct,
            baseline_risk_pct    = self.baseline_risk_pct,
            current_min_quality  = self.baseline_min_quality,
            baseline_min_quality = self.baseline_min_quality,
        )
        print("  ✅  PerformanceMonitor reset to baseline.")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
