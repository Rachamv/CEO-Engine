"""
The CEO Protocol — Phase 2: Risk Engine
============================================
Handles position sizing, model selection from backtest results,
spread validation, and session-aware trade filtering.

Modules
-------
    ModelSelector       — picks best model per symbol/timeframe from backtest
    PositionSizer       — calculates lot size from account balance + SL distance
    SessionFilter       — London/NY/Asian session awareness + spread check
    RiskEngine          — combines all three into one pre-trade gate

Usage
-----
    from .risk_engine import RiskEngine

    engine = RiskEngine(
        risk_pct        = 1.0,    # 1% per trade
        max_spread_ratio= 0.20,   # spread must be < 20% of SL
        sessions        = ["london", "new_york"],
    )

    # After running backtest, register results
    engine.register_backtest(results_table, symbol="XAUUSD", tf="M15")

    # At signal time — returns lot size or 0 (blocked)
    lot = engine.evaluate(
        symbol   = "XAUUSD",
        tf       = "M15",
        direction= "long",
        entry    = 2345.00,
        sl       = 2340.00,
        account  = account_info_dict,   # from mt5_connect
        sym_info = symbol_info_dict,    # from mt5_connect
        bar_time = datetime.now(timezone.utc),
    )
"""

import math
from datetime import datetime, time as dtime, timezone
from typing import Optional, Dict, List, Tuple

import pandas as pd

from .ceo_logging import get_logger
from .session_filter import SESSION_WINDOWS as SESSIONS, TRADE_ANYTIME

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Session definitions (UTC) — imported from session_filter.py above so the
# live order gate and the backtest session mask can never drift apart again.
# ─────────────────────────────────────────────────────────────────────────────

# Minimum trade quality thresholds per session
SESSION_MIN_QUALITY = {
    "asian":      70,   # higher bar during thin session
    "pre_london": 55,   # setup window, thinner liquidity than full London
    "london":     50,
    "new_york":   50,
    "overlap":    45,   # most liquid — accept lower threshold
    "post_ny":    70,   # thin, widening spreads — same bar as Asian
}


# ─────────────────────────────────────────────────────────────────────────────
# Model Selector
# ─────────────────────────────────────────────────────────────────────────────

class ModelSelector:
    """
    Selects the best performing model per symbol+timeframe
    based on backtest expectancy, optionally gated by walk-forward
    consistency to avoid deploying curve-fit models.

    Expectancy = Win Rate × Avg Win R − Loss Rate × 1.0
    Only deploys a model if it has >= min_trades and expectancy > 0.

    If min_consistency > 0, register_with_walkforward() additionally
    requires the model to have been profitable in at least that
    fraction of walk-forward windows (e.g. 0.6 = profitable in 60%+
    of windows). A model can look great on the full-period backtest
    while only winning in 1 of 4 windows — that's curve-fit risk this
    gate is designed to catch.
    """

    def __init__(
        self,
        min_trades:      int   = 30,
        min_expectancy:  float = 0.0,
        min_consistency: float = 0.0,   # 0 = no walk-forward gate
    ):
        self.min_trades      = min_trades
        self.min_expectancy  = min_expectancy
        self.min_consistency = min_consistency
        self._registry: Dict[str, dict] = {}   # key = "SYMBOL_TF"

    def _key(self, symbol: str, tf: str) -> str:
        return f"{symbol.upper()}_{tf.upper()}"

    def register(
        self,
        results: pd.DataFrame,
        symbol:  str,
        tf:      str,
    ) -> Optional[str]:
        """
        Register backtest results for a symbol/TF pair.
        results must be the DataFrame from backtest.results_table().

        Returns the selected model name, or None if no valid model found.
        """
        key = self._key(symbol, tf)

        valid = results[
            (results["Trades"] >= self.min_trades) &
            results["Net R"].notna() &
            results["Win Rate"].notna() &
            results["Avg R"].notna()
        ].copy()

        if valid.empty:
            logger.warning(
                "ModelSelector: no valid model for %s %s (need >= %s trades)",
                symbol, tf, self.min_trades)
            self._registry[key] = {"model": None, "expectancy": None}
            return None

        # Avg R from results_table = r_result.mean() = average R across ALL trades
        # (wins and losses combined). This IS the per-trade expectancy directly.
        # The previous WR × AvgWin − (1−WR) × 1.0 formula was double-counting
        # losses because Avg R already reflects losing trades.
        valid["expectancy"] = valid["Avg R"]

        best = valid[valid["expectancy"] > self.min_expectancy]
        if best.empty:
            logger.warning("ModelSelector: no model with positive expectancy for %s %s",
                           symbol, tf)
            self._registry[key] = {"model": None, "expectancy": None}
            return None

        best_row = best.loc[best["expectancy"].idxmax()]
        model_name = best_row.name

        self._registry[key] = {
            "model":       model_name,
            "expectancy":  float(best_row["expectancy"]),
            "win_rate":    float(best_row["Win Rate"]),
            "avg_r":       float(best_row["Avg R"]),
            "net_r":       float(best_row["Net R"]),
            "trades":      int(best_row["Trades"]),
            "max_dd_r":    float(best_row.get("Max DD R", 0.0)),
            "registered":  datetime.now(timezone.utc).isoformat(),
        }

        logger.info("ModelSelector: %s %s → '%s' (E=%+0.3f, WR=%0.1f%%, Trades=%s)",
                    symbol, tf, model_name, best_row['expectancy'],
                    best_row['Win Rate'], int(best_row['Trades']))

        return model_name

    def register_with_walkforward(
        self,
        results:    pd.DataFrame,    # full-period results_table()
        wf_summary: pd.DataFrame,    # walkforward._build_summary() output
        symbol:     str,
        tf:         str,
    ) -> Optional[str]:
        """
        Like register(), but additionally requires the candidate model
        to be consistent across walk-forward windows before deployment.

        wf_summary is the DataFrame from walk_forward(df, ...)["summary"],
        which has 'Positive Windows' and 'Valid Windows' per model.

        A model that passes the expectancy gate on the full-period
        backtest but was only profitable in a minority of walk-forward
        windows is rejected — it's the classic curve-fit signature
        (great on average, fragile in practice).
        """
        key = self._key(symbol, tf)

        valid = results[
            (results["Trades"] >= self.min_trades) &
            results["Net R"].notna() &
            results["Win Rate"].notna() &
            results["Avg R"].notna()
        ].copy()

        if valid.empty:
            logger.warning(
                "ModelSelector: no valid model for %s %s (need >= %s trades)",
                symbol, tf, self.min_trades)
            self._registry[key] = {"model": None, "expectancy": None}
            return None

        valid["expectancy"] = valid["Avg R"]

        # Attach walk-forward consistency to each candidate
        consistency = {}
        for model_name in valid.index:
            if model_name in wf_summary.index:
                row = wf_summary.loc[model_name]
                vw  = row.get("Valid Windows", 0)
                pw  = row.get("Positive Windows", 0)
                consistency[model_name] = (pw / vw) if vw > 0 else 0.0
            else:
                consistency[model_name] = 0.0

        valid["consistency"] = valid.index.map(consistency)

        # Apply both gates: expectancy AND consistency
        passing = valid[
            (valid["expectancy"] > self.min_expectancy) &
            (valid["consistency"] >= self.min_consistency)
        ]

        if passing.empty:
            # Report the best near-miss for diagnostics
            best_candidate = valid.loc[valid["expectancy"].idxmax()] if not valid.empty else None
            if best_candidate is not None:
                logger.warning(
                    "ModelSelector: no model passed both gates for %s %s. "
                    "Best candidate '%s' had E=%+0.3f, consistency=%0.0f%% "
                    "(required: E>%s, consistency>=%0.0f%%)",
                    symbol, tf, best_candidate.name,
                    best_candidate['expectancy'], best_candidate['consistency'] * 100,
                    self.min_expectancy, self.min_consistency * 100,
                )
            else:
                logger.warning("ModelSelector: no candidates for %s %s", symbol, tf)
            self._registry[key] = {"model": None, "expectancy": None}
            return None

        # Among passing models, prefer the most consistent; break ties by expectancy
        best_row = passing.sort_values(
            ["consistency", "expectancy"], ascending=False
        ).iloc[0]
        model_name = best_row.name

        self._registry[key] = {
            "model":       model_name,
            "expectancy":  float(best_row["expectancy"]),
            "win_rate":    float(best_row["Win Rate"]),
            "avg_r":       float(best_row["Avg R"]),
            "net_r":       float(best_row["Net R"]),
            "trades":      int(best_row["Trades"]),
            "max_dd_r":    float(best_row.get("Max DD R", 0.0)),
            "consistency": float(best_row["consistency"]),
            "registered":  datetime.now(timezone.utc).isoformat(),
        }

        logger.info("ModelSelector: %s %s → '%s' (E=%+0.3f, WR=%0.1f%%, Trades=%s, Consistency=%0.0f%%)",
                    symbol, tf, model_name, best_row['expectancy'],
                    best_row['Win Rate'], int(best_row['Trades']),
                    best_row['consistency'] * 100)

        return model_name

    def get_model(self, symbol: str, tf: str) -> Optional[str]:
        """Returns the selected model name for a symbol/TF, or None."""
        return self._registry.get(self._key(symbol, tf), {}).get("model")

    def get_stats(self, symbol: str, tf: str) -> dict:
        """Returns full backtest stats for the selected model."""
        return self._registry.get(self._key(symbol, tf), {})

    def is_deployable(self, symbol: str, tf: str) -> bool:
        """True if a valid model has been registered and selected."""
        return self.get_model(symbol, tf) is not None

    def summary(self) -> pd.DataFrame:
        """Returns a summary DataFrame of all registered models."""
        rows = []
        for key, stats in self._registry.items():
            symbol, tf = key.split("_", 1)
            rows.append({
                "symbol":     symbol,
                "tf":         tf,
                "model":      stats.get("model"),
                "expectancy": stats.get("expectancy"),
                "win_rate":   stats.get("win_rate"),
                "net_r":      stats.get("net_r"),
                "trades":     stats.get("trades"),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Position Sizer
# ─────────────────────────────────────────────────────────────────────────────

class PositionSizer:
    """
    Calculates MT5 lot size from:
        - Account balance
        - Risk percentage per trade
        - SL distance in price
        - Symbol tick size and tick value

    Enforces min/max lot size and lot step from broker symbol info.
    """

    def __init__(
        self,
        risk_pct:     float = 1.0,     # % of balance to risk per trade
        min_sl_pips:  float = 5.0,     # minimum SL in pips (blocks tiny SLs)
        max_sl_pips:  float = 500.0,   # maximum SL in pips (blocks huge SLs)
    ):
        self.risk_pct    = risk_pct
        self.min_sl_pips = min_sl_pips
        self.max_sl_pips = max_sl_pips

    def calculate(
        self,
        balance:   float,
        entry:     float,
        sl:        float,
        sym_info:  dict,     # from MT5Connection.symbol_info()
        direction: str,      # "long" or "short"
    ) -> Tuple[float, dict]:
        """
        Returns (lot_size, breakdown_dict).
        lot_size is 0.0 if trade should be blocked.
        """
        sym_info["digits"]
        tick_size   = sym_info["tick_size"]
        tick_value  = sym_info["tick_value"]
        vol_min     = sym_info["volume_min"]
        vol_max     = sym_info["volume_max"]
        vol_step    = sym_info["volume_step"]

        sl_distance = abs(entry - sl)
        if sl_distance <= 0:
            return 0.0, {"error": "SL distance is zero"}

        # Convert SL distance to pips
        point = sym_info.get("point", tick_size)
        sl_pips = sl_distance / point

        if sl_pips < self.min_sl_pips:
            return 0.0, {
                "error": f"SL too small: {sl_pips:.1f} pips < min {self.min_sl_pips}",
                "sl_pips": sl_pips,
            }

        if sl_pips > self.max_sl_pips:
            return 0.0, {
                "error": f"SL too large: {sl_pips:.1f} pips > max {self.max_sl_pips}",
                "sl_pips": sl_pips,
            }

        # Risk amount in account currency
        risk_amount = balance * (self.risk_pct / 100.0)

        # Value per pip per lot
        pip_value_per_lot = tick_value / tick_size * point

        if pip_value_per_lot <= 0:
            return 0.0, {"error": "Cannot compute pip value — check symbol info"}

        # Raw lot size
        raw_lots = risk_amount / (sl_pips * pip_value_per_lot)

        # Round to broker lot step
        lots = math.floor(raw_lots / vol_step) * vol_step
        lots = round(lots, 8)   # floating point cleanup

        # Enforce broker limits
        if lots < vol_min:
            return 0.0, {
                "error": f"Calculated lot {lots:.5f} < broker minimum {vol_min}",
                "raw_lots": raw_lots,
            }

        lots = min(lots, vol_max)

        breakdown = {
            "balance":          balance,
            "risk_pct":         self.risk_pct,
            "risk_amount":      risk_amount,
            "sl_distance":      sl_distance,
            "sl_pips":          sl_pips,
            "pip_value_per_lot": pip_value_per_lot,
            "raw_lots":         raw_lots,
            "lots":             lots,
            "vol_min":          vol_min,
            "vol_max":          vol_max,
        }

        return lots, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Session Filter
# ─────────────────────────────────────────────────────────────────────────────

class SessionFilter:
    """
    Validates that a trade fires during an allowed session,
    and that the spread is within acceptable bounds relative to SL size.
    """

    def __init__(
        self,
        allowed_sessions:    List[str]  = (TRADE_ANYTIME,),
        max_spread_ratio:    float      = 0.20,    # spread / SL < this
        max_spread_pips_abs: float      = 30.0,    # hard cap in pips
        block_friday_close:  bool       = True,    # block after 20:00 UTC Friday
        block_sunday_open:   bool       = True,    # block before 05:00 UTC Sunday
    ):
        self.allowed_sessions    = [s.lower() for s in allowed_sessions]
        self.trade_anytime       = TRADE_ANYTIME in self.allowed_sessions
        self.max_spread_ratio    = max_spread_ratio
        self.max_spread_pips_abs = max_spread_pips_abs
        self.block_friday_close  = block_friday_close
        self.block_sunday_open   = block_sunday_open

    def active_sessions(self, bar_time: datetime) -> List[str]:
        """Returns list of session names active at bar_time (UTC)."""
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)
        t = bar_time.time()
        active = []
        for name, (start, end) in SESSIONS.items():
            if start <= t < end:
                active.append(name)
        return active

    def is_session_allowed(self, bar_time: datetime) -> Tuple[bool, str]:
        """Returns (allowed, reason)."""
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        weekday = bar_time.weekday()   # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        t       = bar_time.time()

        # Block Friday close
        if self.block_friday_close and weekday == 4 and t >= dtime(20, 0):
            return False, "Friday close — market illiquid"

        # Block Sunday open
        if self.block_sunday_open and weekday == 6 and t < dtime(5, 0):
            return False, "Sunday open — wide spreads"

        # Saturday — always closed
        if weekday == 5:
            return False, "Saturday — market closed"

        # Check allowed sessions — skipped entirely in trade-anytime mode
        if self.trade_anytime:
            return True, "OK (trade-anytime — session names not restricted)"

        active = self.active_sessions(bar_time)
        allowed = any(s in self.allowed_sessions for s in active)

        if not allowed:
            active_str = ", ".join(active) if active else "no active session"
            allowed_str = ", ".join(self.allowed_sessions)
            return False, f"Outside allowed sessions ({active_str}; allowed: {allowed_str})"

        return True, "OK"

    def is_spread_acceptable(
        self,
        spread_pips:  float,
        sl_pips:      float,
        sym_info:     dict,
    ) -> Tuple[bool, str]:
        """Returns (acceptable, reason)."""
        # Hard cap
        if spread_pips > self.max_spread_pips_abs:
            return False, (f"Spread {spread_pips:.1f} pips exceeds hard cap "
                           f"{self.max_spread_pips_abs:.1f} pips")

        # Ratio check
        if sl_pips > 0:
            ratio = spread_pips / sl_pips
            if ratio > self.max_spread_ratio:
                return False, (f"Spread/SL ratio {ratio:.2%} exceeds max "
                               f"{self.max_spread_ratio:.2%} "
                               f"(spread={spread_pips:.1f}, SL={sl_pips:.1f} pips)")

        return True, "OK"

    def min_quality_for_session(self, bar_time: datetime) -> int:
        """Returns the minimum quality score required for the current session."""
        active = self.active_sessions(bar_time)
        if self.trade_anytime:
            # Trade-anytime mode: still apply the per-session quality bar
            # (Asian/post-NY require a cleaner signal), just don't restrict
            # *which* sessions are tradeable.
            if not active:
                return 60   # outside all named windows (rare) — sane default
            return min(SESSION_MIN_QUALITY.get(s, 60) for s in active)
        if not active:
            return 100   # outside sessions = require perfect score (blocks everything)
        qualities = [SESSION_MIN_QUALITY.get(s, 60) for s in active
                     if s in self.allowed_sessions]
        return min(qualities) if qualities else 60


# ─────────────────────────────────────────────────────────────────────────────
# Risk Engine (master gate)
# ─────────────────────────────────────────────────────────────────────────────

class RiskEngine:
    """
    Combines ModelSelector + PositionSizer + SessionFilter
    into a single pre-trade evaluation gate.

    Call evaluate() before every trade. It returns lot size (0 = blocked).

    Parameters
    ----------
    risk_pct            : % of account balance to risk per trade (default 1.0)
    max_spread_ratio    : max spread as fraction of SL (default 0.20)
    max_spread_pips_abs : hard spread cap in pips (default 30)
    sessions            : allowed session names (default ["all"] — trade
                          anytime the market is open; pass specific names
                          like ["london","new_york"] to restrict)
    min_quality         : global minimum signal quality (default 50)
    min_trades          : min backtest trades to deploy a model (default 30)
    min_expectancy      : min expectancy to deploy a model (default 0.05)
    min_consistency     : min fraction of walk-forward windows a model must
                          be profitable in, when using register_walkforward()
                          (default 0.0 = no walk-forward gate)
    min_sl_pips         : minimum SL in pips (default 5)
    max_sl_pips         : maximum SL in pips (default 500)
    """

    def __init__(
        self,
        risk_pct:             float      = 1.0,
        max_spread_ratio:     float      = 0.20,
        max_spread_pips_abs:  float      = 30.0,
        sessions:             List[str]  = (TRADE_ANYTIME,),
        min_quality:          float      = 50.0,
        min_trades:           int        = 30,
        min_expectancy:       float      = 0.05,
        min_consistency:      float      = 0.0,
        min_sl_pips:          float      = 5.0,
        max_sl_pips:          float      = 500.0,
    ):
        self.min_quality = min_quality

        self.model_selector = ModelSelector(
            min_trades      = min_trades,
            min_expectancy  = min_expectancy,
            min_consistency = min_consistency,
        )
        self.position_sizer = PositionSizer(
            risk_pct    = risk_pct,
            min_sl_pips = min_sl_pips,
            max_sl_pips = max_sl_pips,
        )
        self.session_filter = SessionFilter(
            allowed_sessions    = list(sessions),
            max_spread_ratio    = max_spread_ratio,
            max_spread_pips_abs = max_spread_pips_abs,
        )

    # ── Backtest registration ─────────────────────────────────────────────────

    def register_backtest(
        self,
        results: pd.DataFrame,
        symbol:  str,
        tf:      str,
    ) -> Optional[str]:
        """Register backtest results. Returns selected model name or None."""
        return self.model_selector.register(results, symbol, tf)

    def register_walkforward(
        self,
        results:    pd.DataFrame,
        wf_summary: pd.DataFrame,
        symbol:     str,
        tf:         str,
    ) -> Optional[str]:
        """
        Register using walk-forward validated results. Requires the
        model to pass both the expectancy gate (full-period backtest)
        and the consistency gate (min_consistency across wf windows).

        Usage:
            wf = walk_forward(df, n_windows=4)
            tbl = results_table(run_backtest(df))
            engine.register_walkforward(tbl, wf["summary"], "XAUUSD", "M15")
        """
        return self.model_selector.register_with_walkforward(
            results, wf_summary, symbol, tf
        )

    # ── Main evaluation gate ──────────────────────────────────────────────────

    def evaluate(
        self,
        symbol:      str,
        tf:          str,
        direction:   str,       # "long" or "short"
        entry:       float,
        sl:          float,
        quality:     float,
        account:     dict,      # from MT5Connection.account_info()
        sym_info:    dict,      # from MT5Connection.symbol_info()
        bar_time:    datetime,
        spread_pips: Optional[float] = None,
        verbose:     bool = True,
    ) -> Tuple[float, dict]:
        """
        Full pre-trade gate. Returns (lot_size, report).
        lot_size = 0.0 means the trade is BLOCKED.

        report contains all gate outcomes for logging.
        """
        report = {
            "symbol":    symbol,
            "tf":        tf,
            "direction": direction,
            "entry":     entry,
            "sl":        sl,
            "quality":   quality,
            "bar_time":  bar_time.isoformat() if hasattr(bar_time, "isoformat") else str(bar_time),
            "gates":     {},
            "lot_size":  0.0,
            "blocked_by": None,
        }

        def _block(gate: str, reason: str):
            report["gates"][gate] = {"passed": False, "reason": reason}
            report["blocked_by"] = gate
            if verbose:
                logger.warning("[%s] %s", gate, reason)
            return 0.0, report

        def _pass(gate: str, note: str = "OK"):
            report["gates"][gate] = {"passed": True, "reason": note}

        # ── Gate 1: Model deployable ──────────────────────────────────────────
        if not self.model_selector.is_deployable(symbol, tf):
            return _block("model_selector",
                          f"No valid model registered for {symbol} {tf}. "
                          f"Run register_backtest() first.")
        _pass("model_selector", self.model_selector.get_model(symbol, tf))

        # ── Gate 2: Signal quality ────────────────────────────────────────────
        session_min_q = self.session_filter.min_quality_for_session(bar_time)
        eff_min_q     = max(self.min_quality, session_min_q)
        if quality < eff_min_q:
            return _block("quality",
                          f"Signal quality {quality:.1f} < required {eff_min_q:.1f}")
        _pass("quality", f"{quality:.1f} >= {eff_min_q:.1f}")

        # ── Gate 3: Session filter ────────────────────────────────────────────
        sess_ok, sess_reason = self.session_filter.is_session_allowed(bar_time)
        if not sess_ok:
            return _block("session", sess_reason)
        _pass("session", sess_reason)

        # ── Gate 4: Spread check ──────────────────────────────────────────────
        if spread_pips is not None:
            point   = sym_info.get("point", sym_info["tick_size"])
            digits  = sym_info.get("digits", 5)
            # MT5 info.spread is always in raw points (integer).
            # Normalise to true pips so the hard cap (max_spread_pips_abs)
            # is symbol-agnostic:
            #   5-digit forex (EURUSD, digits=5): 1 pip = 10 points
            #   XAUUSD / most indices (digits=2): 1 pip = 1 point
            #   3-digit JPY pairs (digits=3):     1 pip = 100 points
            if digits >= 3:
                pip_size = 10 ** -(digits - 1)   # e.g. digits=5 -> 0.0001
            else:
                pip_size = point                  # XAUUSD, indices: pip == point
            points_per_pip  = pip_size / point    # e.g. 0.0001/0.00001 = 10
            spread_in_pips  = spread_pips / points_per_pip
            sl_pips         = abs(entry - sl) / pip_size
            spread_ok, spread_reason = self.session_filter.is_spread_acceptable(
                spread_in_pips, sl_pips, sym_info
            )
            if not spread_ok:
                return _block("spread", spread_reason)
            _pass("spread", spread_reason)
        else:
            _pass("spread", "spread not provided — skipped")

        # ── Gate 5: Position sizing ───────────────────────────────────────────
        balance  = account["balance"]
        lots, sizing = self.position_sizer.calculate(
            balance   = balance,
            entry     = entry,
            sl        = sl,
            sym_info  = sym_info,
            direction = direction,
        )

        if lots <= 0:
            return _block("position_sizer", sizing.get("error", "Zero lot size"))
        _pass("position_sizer", f"{lots:.5f} lots (risk={sizing['risk_amount']:.2f} {account.get('currency', '?')})")

        report["lot_size"]  = lots
        report["sizing"]    = sizing
        report["model"]     = self.model_selector.get_model(symbol, tf)
        report["model_stats"] = self.model_selector.get_stats(symbol, tf)

        if verbose:
            logger.info("Trade approved: %0.5f lots | Risk: %0.2f %s | SL: %0.1f pips",
                        lots, sizing['risk_amount'], account.get('currency', '?'),
                        sizing['sl_pips'])

        return lots, report

    def model_summary(self) -> pd.DataFrame:
        """Returns registered model summary."""
        return self.model_selector.summary()

    @property
    def risk_pct(self) -> float:
        return self.position_sizer.risk_pct

    @risk_pct.setter
    def risk_pct(self, value: float):
        self.position_sizer.risk_pct = value


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (no MT5 required)
# ─────────────────────────────────────────────────────────────────────────────
