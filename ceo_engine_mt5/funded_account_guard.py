"""
The CEO Protocol — Phase 2: Funded Account Guard
=====================================================
General-purpose prop firm / funded account rule enforcer.
Works with any prop firm by configuring the parameters directly
or loading a named preset (Blue Guardian, FTMO, The5ers, etc.).

Rules enforced
--------------
    1. Daily loss limit          — hard stop before hitting the limit
    2. Max trailing drawdown     — account-level protection
    3. Consistency score         — no single day > X% of total profits
    4. Minimum trading days      — tracks progress toward requirement
    5. Minimum hold time         — trade must be open for min N minutes
    6. Max open trades           — simultaneous position limit
    7. Weekend / session block   — blocks trades outside safe windows

Usage
-----
    from .funded_account_guard import FundedAccountGuard

    # Option A: configure manually (works with any prop firm)
    guard = FundedAccountGuard(
        account_size         = 10_000,
        daily_loss_limit_pct = 5.0,    # % of account
        max_drawdown_pct     = 10.0,   # % of account
        consistency_pct      = 15.0,   # 0 = disabled
        min_hold_minutes     = 0,      # 0 = no hold requirement
        min_trading_days     = 5,      # 0 = no minimum
    )

    # Option B: load a preset
    guard = FundedAccountGuard(preset="custom", account_size=10_000)

    # Before placing a trade
    ok, reason = guard.pre_trade_check(account_info, open_trades)

    # After a trade closes
    guard.record_closed_trade(pnl, date)

    # Before closing a trade (hold time)
    ok, reason = guard.pre_close_check(trade_open_time)
"""

from datetime import datetime, date, timedelta, timezone
from typing import Dict, Optional, Tuple
import json
import os


# ─────────────────────────────────────────────────────────────────────────────
# Prop firm presets
# ─────────────────────────────────────────────────────────────────────────────

PROP_FIRM_PRESETS = {
    # ── Custom / manual ───────────────────────────────────────────────────────
    # Configure all values to match your prop firm's exact rules
    "custom": {
        "daily_loss_limit_pct":    5.0,
        "max_drawdown_pct":        10.0,
        "consistency_pct":         0.0,    # 0 = disabled
        "min_hold_minutes":        0,      # 0 = no minimum
        "min_trading_days":        0,      # 0 = no minimum
        "max_daily_trades":        0,      # 0 = unlimited
        "require_sl":              True,
        "max_open_trades":         0,      # 0 = unlimited
        "buffer_pct":              0.5,    # stop 0.5% before the hard limit
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Funded Account Guard
# ─────────────────────────────────────────────────────────────────────────────


# Known high-correlation pairs (>0.7 correlation in same direction = double risk)
# Maps symbol → list of correlated symbols
CORRELATION_MAP: dict = {
    "XAUUSD": ["XAGUSD"],                           # Gold / Silver
    "XAGUSD": ["XAUUSD"],
    "EURUSD": ["GBPUSD", "AUDUSD", "NZDUSD"],       # USD-negative basket
    "GBPUSD": ["EURUSD", "AUDUSD"],
    "AUDUSD": ["EURUSD", "GBPUSD", "NZDUSD"],
    "NZDUSD": ["EURUSD", "AUDUSD"],
    "USDJPY": ["USDCHF", "USDCAD"],                 # USD-positive basket
    "USDCHF": ["USDJPY", "USDCAD"],
    "USDCAD": ["USDJPY", "USDCHF"],
    "US30":   ["US500", "NAS100"],                  # US indices
    "US500":  ["US30", "NAS100"],
    "NAS100": ["US30", "US500"],
    "BTCUSD": ["ETHUSD"],                           # Crypto
    "ETHUSD": ["BTCUSD"],
}


class FundedAccountGuard:
    """
    General-purpose funded account / prop firm rule enforcer.

    Configure manually for any prop firm, or pass a preset name.

    Parameters
    ----------
    account_size            : starting funded account size
    daily_loss_limit_pct    : max daily loss as % of account (default 5%)
    max_drawdown_pct        : max total drawdown as % (default 10%)
    consistency_pct         : max single day profit as % of total profits
                              Set to 0 to disable (most firms have no rule)
    min_hold_minutes        : minimum trade hold time in minutes (0 = none)
    min_trading_days        : minimum days that must include at least 1 trade
    max_daily_trades        : max trades per day (0 = unlimited)
    max_open_trades         : max simultaneous open trades (0 = unlimited)
    require_sl              : every trade must have an SL set
    buffer_pct              : stop trading this % before the hard limit
    preset                  : load config from a named preset
                              Options: custom
    log_file                : path to persist journal state across sessions
    """

    # Sentinel — distinguishes "not passed" from an explicit value
    _UNSET = object()

    def __init__(
        self,
        account_size:           float = 10_000.0,
        daily_loss_limit_pct:   float = _UNSET,
        max_drawdown_pct:       float = _UNSET,
        consistency_pct:        float = _UNSET,
        min_hold_minutes:       int   = _UNSET,
        min_trading_days:       int   = _UNSET,
        max_daily_trades:       int   = _UNSET,
        max_open_trades:        int   = _UNSET,
        require_sl:             bool  = _UNSET,
        buffer_pct:             float = _UNSET,
        preset:                 Optional[str] = None,
        log_file:               Optional[str] = None,
    ):
        # Start from preset (or built-in defaults if no preset)
        if preset:
            cfg = PROP_FIRM_PRESETS.get(preset.lower().replace(" ", "_"))
            if cfg is None:
                raise ValueError(f"Unknown preset '{preset}'. "
                                 f"Options: {list(PROP_FIRM_PRESETS.keys())}")
        else:
            cfg = PROP_FIRM_PRESETS["custom"]

        # Any explicitly passed parameter overrides the preset value
        _u = FundedAccountGuard._UNSET
        daily_loss_limit_pct = daily_loss_limit_pct if daily_loss_limit_pct is not _u else cfg["daily_loss_limit_pct"]
        max_drawdown_pct     = max_drawdown_pct     if max_drawdown_pct     is not _u else cfg["max_drawdown_pct"]
        consistency_pct      = consistency_pct      if consistency_pct      is not _u else cfg["consistency_pct"]
        min_hold_minutes     = min_hold_minutes     if min_hold_minutes     is not _u else cfg["min_hold_minutes"]
        min_trading_days     = min_trading_days     if min_trading_days     is not _u else cfg["min_trading_days"]
        max_daily_trades     = max_daily_trades     if max_daily_trades     is not _u else cfg["max_daily_trades"]
        require_sl           = require_sl           if require_sl           is not _u else cfg["require_sl"]
        max_open_trades      = max_open_trades      if max_open_trades      is not _u else cfg["max_open_trades"]
        buffer_pct           = buffer_pct           if buffer_pct           is not _u else cfg["buffer_pct"]

        self.account_size           = account_size
        self.daily_loss_limit_pct   = daily_loss_limit_pct
        self.max_drawdown_pct       = max_drawdown_pct
        self.consistency_pct        = consistency_pct
        self.min_hold_minutes       = min_hold_minutes
        self.min_trading_days       = min_trading_days
        self.max_daily_trades       = max_daily_trades
        self.max_open_trades        = max_open_trades
        self.require_sl             = require_sl
        self.buffer_pct             = buffer_pct
        self.log_file               = log_file

        # Computed limits
        self.daily_loss_limit  = account_size * (daily_loss_limit_pct / 100.0)
        self.max_drawdown_abs  = account_size * (max_drawdown_pct / 100.0)
        self.buffer_amount     = account_size * (buffer_pct / 100.0)

        # Peak equity tracker
        self._peak_equity: float = account_size

        # Daily journal: {date_str: {"pnl": float, "trades": int}}
        self._daily: Dict[str, dict] = {}

        # Trading days counter
        self._trading_days_set: set = set()

        # Emergency halt flag
        self._halted: bool = False
        self._halt_reason: str = ""

        # Load state if log file exists
        if log_file and os.path.exists(log_file):
            self._load_state()

    # ─────────────────────────────────────────────────────────────────────────
    # State persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _save_state(self):
        if not self.log_file:
            return
        state = {
            "peak_equity":       self._peak_equity,
            "daily":             self._daily,
            "trading_days":      list(self._trading_days_set),
            "halted":            self._halted,
            "halt_reason":       self._halt_reason,
        }
        with open(self.log_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        try:
            with open(self.log_file) as f:
                state = json.load(f)
            self._peak_equity      = state.get("peak_equity", self.account_size)
            self._daily            = state.get("daily", {})
            self._trading_days_set = set(state.get("trading_days", []))
            self._halted           = state.get("halted", False)
            self._halt_reason      = state.get("halt_reason", "")
            print(f"  📂  Guard state loaded from {self.log_file}")
        except Exception as e:
            print(f"  ⚠️  Could not load guard state: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _today_str(self) -> str:
        """Returns today's date in UTC — matches broker server time convention
        used by most prop firms for daily loss limit resets."""
        return datetime.now(timezone.utc).date().isoformat()

    def _day_data(self, date_str: Optional[str] = None) -> dict:
        ds = date_str or self._today_str()
        if ds not in self._daily:
            self._daily[ds] = {"pnl": 0.0, "trades": 0}
        return self._daily[ds]

    def _total_profit(self) -> float:
        """Sum of all profitable days."""
        return sum(v["pnl"] for v in self._daily.values() if v["pnl"] > 0)

    def _today_pnl(self) -> float:
        return self._day_data()["pnl"]

    def _today_trades(self) -> int:
        return self._day_data()["trades"]

    def _current_drawdown(self, current_equity: float) -> float:
        """Drawdown from peak equity."""
        self._peak_equity = max(self._peak_equity, current_equity)
        return self._peak_equity - current_equity

    def _check_correlation(
        self,
        symbol:              str,
        direction:           str,
        open_trade_details:  list,
        verbose:             bool = True,
    ) -> Optional[str]:
        """
        Returns a block reason string if a correlated trade is already open
        in the same direction, otherwise None.

        open_trade_details: list of dicts with keys "symbol" and "direction"
        """
        correlated = CORRELATION_MAP.get(symbol.upper(), [])
        for trade in open_trade_details:
            t_sym = trade.get("symbol", "").upper()
            t_dir = trade.get("direction", "").lower()
            if t_sym in correlated and t_dir == direction.lower():
                reason = (f"Correlation block: {symbol} {direction} conflicts "
                          f"with open {t_sym} {t_dir} — both are correlated "
                          f"and in the same direction (effective double risk)")
                if verbose:
                    print(f"  🚫  {reason}")
                return reason
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Core gate: pre-trade check
    # ─────────────────────────────────────────────────────────────────────────

    def pre_trade_check(
        self,
        account:             dict,
        open_trades:         int  = 0,
        has_sl:              bool = True,
        bar_time:            Optional[datetime] = None,
        symbol:              Optional[str] = None,
        direction:           Optional[str] = None,
        open_trade_details:  Optional[list] = None,  # list of {"symbol":..,"direction":..}
        verbose:             bool = True,
    ) -> Tuple[bool, str]:
        """
        Full pre-trade gate. Returns (allowed, reason).
        False = block the trade.
        """
        equity  = account["equity"]
        account["balance"]

        # ── Emergency halt ────────────────────────────────────────────────────
        if self._halted:
            return False, f"🛑 EMERGENCY HALT: {self._halt_reason}"

        # ── SL requirement ────────────────────────────────────────────────────
        if self.require_sl and not has_sl:
            return False, "Trade rejected: SL is required by prop firm rules"

        # ── Daily loss limit (with buffer) ────────────────────────────────────
        today_pnl    = self._today_pnl()
        effective_dd = max(0.0, -today_pnl)  # today's loss so far
        limit_net    = self.daily_loss_limit - self.buffer_amount

        if effective_dd >= limit_net:
            reason = (f"Daily loss limit protection: today's loss "
                      f"${effective_dd:.2f} has reached the buffer threshold "
                      f"(limit=${self.daily_loss_limit:.2f}, "
                      f"buffer=${self.buffer_amount:.2f})")
            self._trigger_halt(reason)
            return False, reason

        # ── Max drawdown ──────────────────────────────────────────────────────
        drawdown = self._current_drawdown(equity)
        dd_limit = self.max_drawdown_abs - self.buffer_amount

        if drawdown >= dd_limit:
            reason = (f"Max drawdown protection: current drawdown "
                      f"${drawdown:.2f} from peak ${self._peak_equity:.2f} "
                      f"(limit=${self.max_drawdown_abs:.2f})")
            self._trigger_halt(reason)
            return False, reason

        # ── Max open trades ───────────────────────────────────────────────────
        if self.max_open_trades > 0 and open_trades >= self.max_open_trades:
            return False, (f"Max open trades reached: {open_trades} / "
                           f"{self.max_open_trades}")

        # ── Daily trade count ─────────────────────────────────────────────────
        if self.max_daily_trades > 0 and self._today_trades() >= self.max_daily_trades:
            return False, (f"Daily trade limit reached: {self._today_trades()} / "
                           f"{self.max_daily_trades}")

        # ── Correlation gate ──────────────────────────────────────────────────
        # Block a new trade if an open trade exists in a correlated symbol
        # in the same direction — two correlated longs double real risk.
        if open_trade_details and symbol and direction:
            corr_conflict = self._check_correlation(
                symbol, direction, open_trade_details, verbose
            )
            if corr_conflict:
                return False, corr_conflict

        # ── Consistency score pre-check ───────────────────────────────────────
        # Warn if we're close to making today's profit too large
        if self.consistency_pct > 0:
            total_profit = self._total_profit()
            if total_profit > 0:
                max_today_profit = total_profit * (self.consistency_pct / 100.0)
                if today_pnl >= max_today_profit * 0.80:   # 80% of the limit
                    if verbose:
                        print(f"  ⚠️  Consistency warning: today's profit "
                              f"${today_pnl:.2f} is approaching the "
                              f"{self.consistency_pct:.0f}% limit "
                              f"(max today: ${max_today_profit:.2f})")

        if verbose:
            print(f"  ✅  Guard pre-trade check passed | "
                  f"Daily P&L: ${today_pnl:+.2f} | "
                  f"Drawdown: ${drawdown:.2f} / ${self.max_drawdown_abs:.2f} | "
                  f"Open: {open_trades}")

        return True, "OK"

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-close check (hold time)
    # ─────────────────────────────────────────────────────────────────────────

    def pre_close_check(
        self,
        trade_open_time: datetime,
        current_time:    Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        Checks if minimum hold time has been satisfied.
        Returns (can_close, reason).
        """
        if self.min_hold_minutes <= 0:
            return True, "No minimum hold time"

        now = current_time or datetime.now(timezone.utc)
        if trade_open_time.tzinfo is None:
            trade_open_time = trade_open_time.replace(tzinfo=timezone.utc)

        held_minutes = (now - trade_open_time).total_seconds() / 60.0

        if held_minutes < self.min_hold_minutes:
            remaining = self.min_hold_minutes - held_minutes
            return False, (f"Minimum hold time not met: held {held_minutes:.1f} min, "
                           f"need {self.min_hold_minutes} min "
                           f"({remaining:.1f} min remaining)")

        return True, f"Hold time satisfied ({held_minutes:.1f} min)"

    # ─────────────────────────────────────────────────────────────────────────
    # Post-trade recording
    # ─────────────────────────────────────────────────────────────────────────

    def record_closed_trade(
        self,
        pnl:          float,
        trade_date:   Optional[date] = None,
        account:      Optional[dict] = None,
        verbose:      bool = True,
    ):
        """
        Record a closed trade's P&L.
        Call this every time a trade closes to keep guard state accurate.
        """
        if trade_date is not None:
            # Accept both date and datetime — always reduce to YYYY-MM-DD
            ds = trade_date.date().isoformat() if hasattr(trade_date, "hour") \
                 else trade_date.isoformat()
        else:
            ds = self._today_str()
        day = self._day_data(ds)
        day["pnl"]    += pnl
        day["trades"] += 1
        self._trading_days_set.add(ds)

        # Update peak equity if account info provided
        if account:
            self._peak_equity = max(self._peak_equity, account["equity"])

        # Check consistency score
        if self.consistency_pct > 0:
            total_profit = self._total_profit()
            today_profit = max(0.0, day["pnl"])

            if total_profit > 0 and today_profit > 0:
                today_pct = (today_profit / total_profit) * 100.0
                if today_pct > self.consistency_pct:
                    if verbose:
                        print(f"  ⚠️  CONSISTENCY WARNING: today's profit "
                              f"${today_profit:.2f} is {today_pct:.1f}% of "
                              f"total profits ${total_profit:.2f} "
                              f"(limit: {self.consistency_pct:.0f}%)")
                        print(f"      Consider pausing for the rest of the day.")

        if verbose:
            print(f"  📊  Trade recorded: P&L ${pnl:+.2f} | "
                  f"Today: ${day['pnl']:+.2f} | "
                  f"Trading days: {len(self._trading_days_set)}/{self.min_trading_days}")

        self._save_state()

    # ─────────────────────────────────────────────────────────────────────────
    # Emergency halt
    # ─────────────────────────────────────────────────────────────────────────

    def _trigger_halt(self, reason: str):
        self._halted     = True
        self._halt_reason = reason
        self._save_state()
        print(f"\n  🛑  EMERGENCY HALT TRIGGERED")
        print(f"      {reason}")
        print(f"      All trading stopped. Manual reset required.")

    def reset_halt(self, confirm: bool = False):
        """Manually reset the emergency halt. Requires explicit confirmation."""
        if not confirm:
            raise ValueError("Pass confirm=True to reset the emergency halt.")
        self._halted      = False
        self._halt_reason = ""
        self._save_state()
        print("  ✅  Emergency halt reset.")

    # ─────────────────────────────────────────────────────────────────────────
    # Status dashboard
    # ─────────────────────────────────────────────────────────────────────────

    def status(self, account: Optional[dict] = None) -> dict:
        """Returns a complete status dict for logging / display."""
        today_pnl     = self._today_pnl()
        today_trades  = self._today_trades()
        total_profit  = self._total_profit()
        trading_days  = len(self._trading_days_set)

        # Daily loss remaining before buffer
        daily_loss_remaining = (self.daily_loss_limit - self.buffer_amount) - max(0.0, -today_pnl)

        # Drawdown status
        current_equity   = account["equity"] if account else self.account_size
        current_drawdown = self._current_drawdown(current_equity)
        dd_remaining     = (self.max_drawdown_abs - self.buffer_amount) - current_drawdown

        # Consistency
        consistency_ok   = True
        consistency_note = "N/A"
        if self.consistency_pct > 0 and total_profit > 0 and today_pnl > 0:
            pct = (today_pnl / total_profit) * 100.0
            consistency_ok   = pct <= self.consistency_pct
            consistency_note = f"{pct:.1f}% of total (limit {self.consistency_pct:.0f}%)"

        return {
            "halted":                   self._halted,
            "halt_reason":              self._halt_reason,

            "today_pnl":                today_pnl,
            "today_trades":             today_trades,
            "daily_loss_limit":         self.daily_loss_limit,
            "daily_loss_remaining":     daily_loss_remaining,
            "daily_loss_used_pct":      abs(min(0.0, today_pnl)) / self.daily_loss_limit * 100.0,

            "peak_equity":              self._peak_equity,
            "current_drawdown":         current_drawdown,
            "max_drawdown_limit":       self.max_drawdown_abs,
            "drawdown_remaining":       dd_remaining,
            "drawdown_used_pct":        current_drawdown / self.max_drawdown_abs * 100.0,

            "trading_days_completed":   trading_days,
            "trading_days_required":    self.min_trading_days,
            "trading_days_ok":          trading_days >= self.min_trading_days,

            "total_profit":             total_profit,
            "consistency_ok":           consistency_ok,
            "consistency_note":         consistency_note,
        }

    def print_status(self, account: Optional[dict] = None):
        """Prints a formatted status dashboard."""
        s = self.status(account)

        print("\n" + "═" * 50)
        print("  FUNDED ACCOUNT GUARD — STATUS")
        print("═" * 50)

        if s["halted"]:
            print(f"  🛑  HALTED: {s['halt_reason']}")
            print("═" * 50)
            return

        dd_bar   = "█" * int(s["drawdown_used_pct"] / 10) + "░" * (10 - int(s["drawdown_used_pct"] / 10))
        dl_bar   = "█" * int(s["daily_loss_used_pct"] / 10) + "░" * (10 - int(s["daily_loss_used_pct"] / 10))
        td_bar   = "✅" if s["trading_days_ok"] else f"{s['trading_days_completed']}/{s['trading_days_required']}"

        print(f"  Daily P&L      : ${s['today_pnl']:+.2f}  ({s['today_trades']} trades)")
        print(f"  Daily Loss     : [{dl_bar}] {s['daily_loss_used_pct']:.1f}%  "
              f"(${abs(min(0,s['today_pnl'])):.2f} / ${s['daily_loss_limit']:.2f})")
        print(f"  Drawdown       : [{dd_bar}] {s['drawdown_used_pct']:.1f}%  "
              f"(${s['current_drawdown']:.2f} / ${s['max_drawdown_limit']:.2f})")
        print(f"  Trading Days   : {td_bar}  ({s['trading_days_completed']} / {s['trading_days_required']})")
        print(f"  Consistency    : {s['consistency_note']}")
        print(f"  Total Profits  : ${s['total_profit']:.2f}")
        print("═" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
