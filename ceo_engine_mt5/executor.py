"""
The CEO Protocol — Phase 2: Executor
=========================================
MT5 order placement, partial TP management, SL trailing,
and automated trade lifecycle — all governed by the Risk Engine
and Funded Account Guard.

Architecture
------------
    Signal fires (bar close)
          ↓
    RiskEngine.evaluate()      → lot size or blocked
          ↓
    FundedAccountGuard.pre_trade_check()  → allowed or blocked
          ↓
    Executor.place_trade()     → sends order to MT5
          ↓
    Executor.manage_open()     → runs every bar, handles TP1/TP2/SL trail
          ↓
    FundedAccountGuard.record_closed_trade()

Usage
-----
    from .executor import Executor

    executor = Executor(
        connection   = mt5_connection,    # MT5Connection instance
        risk_engine  = risk_engine,       # RiskEngine instance
        guard        = funded_guard,      # FundedAccountGuard instance
        magic_number = 20250101,          # unique ID for CEO engine trades
    )

    # On bar close signal
    trade_id = executor.place_trade(
        symbol    = "XAUUSD",
        tf        = "M15",
        direction = "long",
        entry     = 2345.00,
        sl        = 2338.00,
        tp1       = 2352.00,
        tp2       = 2359.00,
        tp3       = 2366.00,
        quality   = 72.5,
        model     = "LQ + Displacement",
        bar_time  = datetime.now(timezone.utc),
    )

    # On every new bar
    executor.manage_open_trades()
"""

import os
import sys
import csv
import math
from datetime import datetime, timezone
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from ceo_engine_mt5.ceo_logging import get_logger, log_to_dashboard
logger = get_logger(__name__)

# MT5 availability guard
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = sys.platform == "win32"
except ImportError:
    MT5_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Represents a single open or closed trade managed by the executor."""
    ticket:     int
    symbol:     str
    tf:         str
    direction:  str       # "long" or "short"
    entry:      float
    sl:         float
    tp1:        float
    tp2:        float
    tp3:        float
    lots:       float
    atr:        float
    quality:    float
    model:      str
    open_time:  datetime
    bar_time:   datetime
    magic:      int

    lots_remaining: float = field(default=None)
    tp1_hit:    bool = False
    tp2_hit:    bool = False
    sl_moved:   bool = False
    status:     str = "open"
    # ── Trailing SL configuration (per-trade, set after open) ────────────────
    trail_active:   bool  = False   # is trailing SL enabled for this trade?
    trail_atr_mult: float = 1.5     # trail distance = ATR * this multiplier
    trail_step_pct: float = 0.3     # only move SL if it improves by >= this % of ATR
    trail_sl_high:  float = field(default=None)  # highest/lowest price seen (tracks trail)
    close_price: Optional[float] = None
    close_time:  Optional[datetime] = None
    close_reason: Optional[str] = None
    pnl:         Optional[float] = None
    # Live floating state — updated every manage_open_trades() call
    current_price: float = field(default=None)
    floating_pnl:  float = 0.0
    floating_r:    float = 0.0
    last_update:   Optional[datetime] = field(default=None)

    def __post_init__(self):
        if self.lots_remaining is None:
            self.lots_remaining = self.lots
        if self.current_price is None:
            self.current_price = self.entry
        if self.last_update is None:
            self.last_update = self.open_time
        if self.trail_sl_high is None:
            # For longs, track the highest price seen; for shorts, the lowest
            self.trail_sl_high = self.entry

    def to_dict(self) -> dict:
        return {
            "ticket":       self.ticket,
            "symbol":       self.symbol,
            "tf":           self.tf,
            "direction":    self.direction,
            "entry":        self.entry,
            "sl":           self.sl,
            "tp1":          self.tp1,
            "tp2":          self.tp2,
            "tp3":          self.tp3,
            "lots":         self.lots,
            "quality":      self.quality,
            "model":        self.model,
            "open_time":    self.open_time.isoformat(),
            "bar_time":     self.bar_time.isoformat(),
            "tp1_hit":      self.tp1_hit,
            "tp2_hit":      self.tp2_hit,
            "sl_moved":     self.sl_moved,
            "status":       self.status,
            "close_price":  self.close_price,
            "close_time":   self.close_time.isoformat() if self.close_time else None,
            "close_reason": self.close_reason,
            "pnl":          self.pnl,
            "current_price":self.current_price,
            "floating_pnl": round(self.floating_pnl, 2),
            "floating_r":   round(self.floating_r, 3),
            "trail_active":   self.trail_active,
            "trail_atr_mult": self.trail_atr_mult,
            "trail_step_pct": self.trail_step_pct,
            "atr":            round(self.atr, 5) if self.atr else 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

class Executor:
    """
    MT5 order executor with full trade lifecycle management.

    On non-Windows or when MT5 is unavailable, runs in SIMULATION mode:
    all orders are logged but not actually sent to MT5.
    Simulation mode is useful for testing the full pipeline on Mac/Linux.
    """

    def __init__(
        self,
        connection,           # MT5Connection instance (or None for sim mode)
        risk_engine,          # RiskEngine instance
        guard,                # FundedAccountGuard instance
        magic_number:  int    = 20250101,
        deviation:     int    = 20,        # max price deviation in points
        tp1_lots_pct:  float  = 0.333,     # close 1/3 at TP1
        tp2_lots_pct:  float  = 0.333,     # close 1/3 at TP2
        move_sl_at_tp1: bool  = True,      # move SL to BE after TP1
        trail_sl_at_tp2: bool = True,      # trail SL to TP1 level after TP2
        journal_file:  Optional[str] = "ceo_trades.csv",
        simulation:    bool   = False,
        alerts_obj             = None,   # optional AlertSystem, for trail-SL notifications
    ):
        self.conn          = connection
        self.risk_engine   = risk_engine
        self.guard         = guard
        self.magic         = magic_number
        self.alerts_obj    = alerts_obj
        self.deviation     = deviation
        self.tp1_lots_pct  = tp1_lots_pct
        self.tp2_lots_pct  = tp2_lots_pct
        self.move_sl_at_tp1 = move_sl_at_tp1
        self.trail_sl_at_tp2 = trail_sl_at_tp2
        self.journal_file  = journal_file

        # Force simulation if MT5 not available
        self.simulation = simulation or not MT5_AVAILABLE
        if self.simulation and not simulation:
            msg = "ℹ️   MT5 not available — Executor running in SIMULATION mode"
            logger.info(msg)
            try:
                log_to_dashboard(msg, level="info")
            except Exception:
                pass

        # Active trades tracked in memory
        self._open_trades: Dict[int, TradeRecord] = {}
        self._closed_trades: List[TradeRecord]    = []
        self._next_sim_ticket = 100000   # fake tickets in sim mode

        # Ensure journal file has headers
        if journal_file:
            self._init_journal()

    # ─────────────────────────────────────────────────────────────────────────
    # Journal
    # ─────────────────────────────────────────────────────────────────────────

    def _init_journal(self):
        if not os.path.exists(self.journal_file):
            headers = [
                "ticket","symbol","tf","direction","model","quality",
                "entry","sl","tp1","tp2","tp3","lots",
                "open_time","bar_time","tp1_hit","tp2_hit","sl_moved",
                "close_price","close_time","close_reason","pnl","status",
            ]
            with open(self.journal_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=headers).writeheader()

    def _log_trade(self, trade: TradeRecord):
        if not self.journal_file:
            return
        row = trade.to_dict()
        headers = list(row.keys())
        with open(self.journal_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writerow(row)

    def is_connected(self) -> bool:
        if self.simulation or not self.conn:
            return False
        try:
            return self.conn.is_connected()
        except Exception as e:
            logger.warning("Executor connection health check failed: %s", e)
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Place trade
    # ─────────────────────────────────────────────────────────────────────────

    def place_trade(
        self,
        symbol:    str,
        tf:        str,
        direction: str,       # "long" or "short"
        entry:     float,     # reference entry (next bar open estimate)
        sl:        float,
        tp1:       float,
        tp2:       float,
        tp3:       float,
        quality:   float,
        model:     str,
        bar_time:  datetime,
        atr:       float      = 0.0,
        verbose:   bool       = True,
    ) -> Optional[int]:
        """
        Full trade placement pipeline.
        Returns ticket number on success, None if blocked or failed.
        """
        if verbose:
            print(f"\n  📡  Signal: {symbol} {tf} {'▲ LONG' if direction == 'long' else '▼ SHORT'} "
                  f"| Quality: {quality:.1f} | {model}")

        # ── Get live account and symbol info ──────────────────────────────────
        if self.simulation:
            account  = self._sim_account()
            sym_info = self._sim_sym_info(symbol)
        else:
            if not self.conn or not self.conn.is_connected():
                logger.warning("MT5 disconnected while placing trade for %s — skipping order", symbol)
                return None
            account  = self.conn.account_info()
            sym_info = self.conn.symbol_info(symbol)

        # ── Risk engine gate ──────────────────────────────────────────────────
        lots, risk_report = self.risk_engine.evaluate(
            symbol      = symbol,
            tf          = tf,
            direction   = direction,
            entry       = entry,
            sl          = sl,
            quality     = quality,
            account     = account,
            sym_info    = sym_info,
            bar_time    = bar_time,
            spread_pips = float(sym_info.get("spread", 0)),
            verbose     = verbose,
        )

        if lots <= 0:
            return None

        # ── Funded account guard ──────────────────────────────────────────────
        open_count = len(self._open_trades)
        guard_ok, guard_reason = self.guard.pre_trade_check(
            account     = account,
            open_trades = open_count,
            has_sl      = True,
            bar_time    = bar_time,
            verbose     = verbose,
        )

        if not guard_ok:
            if verbose:
                print(f"  🚫  Guard blocked: {guard_reason}")
            return None

        # ── Compute actual TP lots (partial position sizing) ──────────────────
        tp1_lots = round(math.floor(lots * self.tp1_lots_pct / sym_info["volume_step"]) * sym_info["volume_step"], 8)
        tp2_lots = tp1_lots
        tp3_lots = round(lots - tp1_lots - tp2_lots, 8)
        tp3_lots = max(tp3_lots, sym_info["volume_min"])

        # ── Send order ────────────────────────────────────────────────────────
        if self.simulation:
            ticket = self._sim_place(symbol, direction, lots, entry, sl, tp3)
        else:
            ticket = self._mt5_place(symbol, direction, lots, sl, tp3, sym_info)

        if ticket is None:
            return None

        # ── Record trade ──────────────────────────────────────────────────────
        now = datetime.now(timezone.utc)
        trade = TradeRecord(
            ticket    = ticket,
            symbol    = symbol,
            tf        = tf,
            direction = direction,
            entry     = entry,
            sl        = sl,
            tp1       = tp1,
            tp2       = tp2,
            tp3       = tp3,
            lots      = lots,
            atr       = atr,
            quality   = quality,
            model     = model,
            open_time = now,
            bar_time  = bar_time,
            magic     = self.magic,
        )
        self._open_trades[ticket] = trade

        mode_str = "SIM" if self.simulation else "LIVE"
        if verbose:
            print(f"  ✅  [{mode_str}] Trade opened: #{ticket} | "
                  f"{direction.upper()} {lots:.5f} lots @ ~{entry:.5f} | "
                  f"SL={sl:.5f} TP3={tp3:.5f}")

        return ticket

    # ─────────────────────────────────────────────────────────────────────────
    # Manage open trades (call on every bar close)
    # ─────────────────────────────────────────────────────────────────────────

    def manage_open_trades(self, verbose: bool = False) -> List[dict]:
        """
        Check all open trades against current prices.
        Handles TP1/TP2 partial closes and SL moves.
        Returns list of close events this bar.
        """
        if not self._open_trades:
            return []

        events = []

        for ticket, trade in list(self._open_trades.items()):
            # Get current price
            if self.simulation:
                # In sim mode, caller must update prices manually
                # or pass price in manage_open_trades_with_prices()
                continue

            tick = self._get_tick(trade.symbol)
            if tick is None:
                continue

            bid = tick["bid"]
            ask = tick["ask"]
            price = bid if trade.direction == "long" else ask

            event = self._check_trade(trade, price, bid, ask)
            if event:
                events.append(event)

        return events

    def get_open_trades(self) -> List[dict]:
        """All open trades as dicts with live floating P&L. Call after manage_open_trades()."""
        return [t.to_dict() for t in self._open_trades.values()]

    def manage_open_trades_with_prices(
        self,
        prices: Dict[str, float],   # {symbol: current_mid_price}
        verbose: bool = False,
    ) -> List[dict]:
        """
        Simulation-mode version of manage_open_trades.
        prices: dict mapping symbol → current price.
        """
        events = []
        for ticket, trade in list(self._open_trades.items()):
            price = prices.get(trade.symbol)
            if price is None:
                continue
            event = self._check_trade(trade, price, price, price, verbose=verbose)
            if event:
                events.append(event)
        return events

    def _update_floating_state(self, trade: TradeRecord, price: float, is_long: bool) -> None:
        """Updates current_price/floating_r/floating_pnl for an open trade."""
        from datetime import datetime as _dt, timezone as _tz
        trade.current_price = price
        trade.last_update   = _dt.now(_tz.utc)
        sl_dist = abs(trade.entry - trade.sl) if trade.sl != trade.entry else 1.0
        price_diff = (price - trade.entry) if is_long else (trade.entry - price)
        trade.floating_r = round(price_diff / sl_dist, 4) if sl_dist > 0 else 0.0
        si = self._sim_sym_info(trade.symbol)
        ts, tv = si.get("tick_size", 0.01), si.get("tick_value", 1.0)
        lots_rem = getattr(trade, "lots_remaining", trade.lots)
        trade.floating_pnl = round(
            (price_diff / ts * tv * lots_rem) if ts > 0
            else price_diff * lots_rem * 100, 2)

    @staticmethod
    def _compute_tp_sl_hits(trade: TradeRecord, price: float, is_long: bool):
        """Returns (hit_tp1, hit_tp2, hit_tp3, hit_sl) booleans for the current price."""
        hit_tp1 = (is_long and price >= trade.tp1) or (not is_long and price <= trade.tp1)
        hit_tp2 = (is_long and price >= trade.tp2) or (not is_long and price <= trade.tp2)
        hit_tp3 = (is_long and price >= trade.tp3) or (not is_long and price <= trade.tp3)
        hit_sl  = (is_long and price <= trade.sl)  or (not is_long and price >= trade.sl)
        return hit_tp1, hit_tp2, hit_tp3, hit_sl

    def _handle_tp1_hit(self, trade: TradeRecord, verbose: bool) -> None:
        """
        Partial close at TP1, then move SL to breakeven if configured.
        In live mode, tp1_hit / lots_remaining / the SL-to-BE move are only
        committed to the in-memory trade record once the broker confirms
        each action -- a rejected order now retries on the next tick
        instead of being silently assumed to have gone through.
        """
        tp1_lots = round(trade.lots * self.tp1_lots_pct, 5)
        if self.simulation:
            trade.tp1_hit = True
            trade.lots_remaining = round(max(trade.lots_remaining - tp1_lots, 0.0), 8)
            if verbose:
                print(f"  🎯  TP1 hit: #{trade.ticket} {trade.symbol} "
                      f"| Close {tp1_lots:.3f} lots @ {trade.tp1:.5f}")
        else:
            if not self._mt5_partial_close(trade.ticket, tp1_lots, trade.symbol):
                return   # unconfirmed -- retry next tick, don't touch SL yet
            trade.tp1_hit = True
            trade.lots_remaining = round(max(trade.lots_remaining - tp1_lots, 0.0), 8)
            if verbose:
                print(f"  🎯  TP1 hit: #{trade.ticket} {trade.symbol} "
                      f"| Close {tp1_lots:.3f} lots @ {trade.tp1:.5f}")

        if self.move_sl_at_tp1 and not trade.sl_moved:
            if self.simulation:
                trade.sl = trade.entry
                trade.sl_moved = True
                if verbose:
                    print(f"  📍  SL moved to BE: #{trade.ticket} @ {trade.entry:.5f}")
            elif self._mt5_modify_sl(trade.ticket, trade.entry, trade.tp3):
                trade.sl = trade.entry
                trade.sl_moved = True
                if verbose:
                    print(f"  📍  SL moved to BE: #{trade.ticket} @ {trade.entry:.5f}")
            # else: leave sl_moved False -- retried on the next tick

    def _handle_tp2_hit(self, trade: TradeRecord, verbose: bool) -> None:
        """
        Partial close at TP2, then trail SL to the TP1 level if configured.
        Same broker-confirmation gating as _handle_tp1_hit.
        """
        tp2_lots = round(trade.lots * self.tp2_lots_pct, 5)
        if self.simulation:
            trade.tp2_hit = True
            trade.lots_remaining = round(max(trade.lots_remaining - tp2_lots, 0.0), 8)
            if verbose:
                print(f"  🎯  TP2 hit: #{trade.ticket} {trade.symbol} "
                      f"| Close {tp2_lots:.3f} lots @ {trade.tp2:.5f}")
        else:
            if not self._mt5_partial_close(trade.ticket, tp2_lots, trade.symbol):
                return
            trade.tp2_hit = True
            trade.lots_remaining = round(max(trade.lots_remaining - tp2_lots, 0.0), 8)
            if verbose:
                print(f"  🎯  TP2 hit: #{trade.ticket} {trade.symbol} "
                      f"| Close {tp2_lots:.3f} lots @ {trade.tp2:.5f}")

        if self.trail_sl_at_tp2:
            if self.simulation:
                trade.sl = trade.tp1
                if verbose:
                    print(f"  📍  SL trailed to TP1: #{trade.ticket} @ {trade.tp1:.5f}")
            elif self._mt5_modify_sl(trade.ticket, trade.tp1, trade.tp3):
                trade.sl = trade.tp1
                if verbose:
                    print(f"  📍  SL trailed to TP1: #{trade.ticket} @ {trade.tp1:.5f}")

    def _handle_full_close(self, trade: TradeRecord, price: float, reason: str,
                            verbose: bool) -> Optional[dict]:
        """
        Closes the remaining position (TP3 or SL hit) and records the
        outcome. Returns None instead of finalizing if a live close is
        rejected by the broker -- the trade stays in _open_trades and
        _check_trade() will retry the close on the next tick (hit_tp3/
        hit_sl stay true while price remains past that level), rather
        than the bot losing track of a position that's still actually open.
        """
        if not self.simulation:
            if not self._mt5_close(trade.ticket, trade.lots_remaining, trade.symbol):
                logger.warning("Full close rejected by broker for #%s %s -- "
                                "will retry next tick", trade.ticket, trade.symbol)
                return None

        close_price = price
        pnl = self._get_real_pnl(trade.ticket)
        if pnl is None:
            pnl = self._estimate_pnl(trade, close_price)
        trade.close_price  = close_price
        trade.close_time   = datetime.now(timezone.utc)
        trade.close_reason = reason
        trade.pnl          = pnl
        trade.status       = "closed"

        self.guard.record_closed_trade(pnl=pnl, verbose=verbose)

        del self._open_trades[trade.ticket]
        self._closed_trades.append(trade)
        self._log_trade(trade)

        icon = "✅" if pnl >= 0 else "❌"
        print(f"  {icon}  Trade closed: #{trade.ticket} {trade.symbol} "
              f"| {reason.upper()} @ {close_price:.5f} | P&L: ${pnl:+.2f} "
              f"| TP1={'✓' if trade.tp1_hit else '✗'} "
              f"TP2={'✓' if trade.tp2_hit else '✗'}")

        return {"ticket": trade.ticket, "reason": reason, "pnl": pnl, "trade": trade}

    def _check_trade(
        self,
        trade:   TradeRecord,
        price:   float,
        bid:     float,
        ask:     float,
        verbose: bool = False,
    ) -> Optional[dict]:
        """
        Check one trade against current price. Returns a close event dict
        if the trade fully closed this call (TP3 or SL hit), else None.

        Orchestrates the TP1/TP2/TP3/SL logic via the five helper methods
        above — this method itself just sequences them, so the branching
        for each (partial close, SL-to-BE, SL trail, full close) lives in
        one place each instead of all four interleaved in a single block.
        """
        is_long = trade.direction == "long"

        self._update_floating_state(trade, price, is_long)
        # Run trailing SL update before checking hits — SL may have moved
        self._update_trailing_sl(trade, price)
        hit_tp1, hit_tp2, hit_tp3, hit_sl = self._compute_tp_sl_hits(trade, price, is_long)

        if hit_tp1 and not trade.tp1_hit:
            self._handle_tp1_hit(trade, verbose)

        if hit_tp2 and not trade.tp2_hit:
            self._handle_tp2_hit(trade, verbose)

        if hit_tp3 or hit_sl:
            reason = "tp3" if hit_tp3 else "sl"
            return self._handle_full_close(trade, price, reason, verbose)

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # MT5 order operations
    # ─────────────────────────────────────────────────────────────────────────

    def _mt5_place(
        self,
        symbol:    str,
        direction: str,
        lots:      float,
        sl:        float,
        tp:        float,
        sym_info:  dict,
    ) -> Optional[int]:
        """Send a market order to MT5. Returns ticket or None."""
        if not MT5_AVAILABLE:
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"  ❌  Cannot get tick for {symbol}")
            return None

        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
        price      = tick.ask if direction == "long" else tick.bid

        request = {
            "action":     mt5.TRADE_ACTION_DEAL,
            "symbol":     symbol,
            "volume":     lots,
            "type":       order_type,
            "price":      price,
            "sl":         sl,
            "tp":         tp,
            "deviation":  self.deviation,
            "magic":      self.magic,
            "comment":    "CEO_Engine",
            "type_time":  mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            desc = result.comment if result else "No response"
            print(f"  ❌  MT5 order failed: retcode={code} — {desc}")
            return None

        return result.order

    def _mt5_partial_close(self, ticket: int, lots: float, symbol: str) -> bool:
        """
        Close part of an open position. Returns True only if the broker
        confirmed the fill -- callers must not mark tp1_hit/tp2_hit or
        decrement lots_remaining unless this returns True, or a rejected
        order (requote, invalid volume, trade-context-busy, etc.) leaves
        the bot's bookkeeping out of sync with what's actually open.
        """
        if not MT5_AVAILABLE:
            return False
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning("MT5 partial close skipped for #%s: position not found "
                            "(already closed?)", ticket)
            return False
        pos       = position[0]
        direction = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick      = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error("MT5 partial close failed for #%s: no tick for %s", ticket, symbol)
            return False
        price     = tick.bid if direction == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action":     mt5.TRADE_ACTION_DEAL,
            "position":   ticket,
            "symbol":     symbol,
            "volume":     min(lots, pos.volume),   # never request more than is actually open
            "type":       direction,
            "price":      price,
            "deviation":  self.deviation,
            "magic":      self.magic,
            "comment":    "CEO_TP_PARTIAL",
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            desc = result.comment if result else "No response"
            logger.error("MT5 partial close failed for #%s: retcode=%s — %s", ticket, code, desc)
            print(f"  ❌  MT5 partial close failed: #{ticket} retcode={code} — {desc}")
            return False
        return True

    def _mt5_modify_sl(self, ticket: int, new_sl: float, tp: float) -> bool:
        """
        Modify the SL of an open position. Returns True only if the
        broker confirmed the modification -- callers must not treat the
        in-memory trade.sl as authoritative when this returns False (a
        rejected modify -- invalid stops level, requote, market closed --
        would otherwise leave the bot believing a stop moved when the
        real position on the broker still has the old one).
        """
        if not MT5_AVAILABLE:
            return False
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       new_sl,
            "tp":       tp,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            desc = result.comment if result else "No response"
            logger.error("MT5 SL modify failed for #%s: retcode=%s — %s", ticket, code, desc)
            print(f"  ❌  MT5 SL modify failed: #{ticket} retcode={code} — {desc}")
            return False
        return True

    def _mt5_close(self, ticket: int, lots: float, symbol: str) -> bool:
        """
        Close remaining position. Returns True only if the broker
        confirmed the close (or the position was already gone on the
        broker side, e.g. stopped out or closed manually) -- callers must
        not mark a trade closed / drop it from tracking unless this
        returns True, or a rejected close leaves a real open position
        that the bot has stopped watching entirely.
        """
        if not MT5_AVAILABLE:
            return False
        position = mt5.positions_get(ticket=ticket)
        if not position:
            # Nothing left to close -- already closed on the broker side.
            return True
        pos       = position[0]
        direction = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick      = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error("MT5 close failed for #%s: no tick for %s", ticket, symbol)
            return False
        price     = tick.bid if direction == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol":   symbol,
            "volume":   min(lots, pos.volume),   # never request more than is actually open
            "type":     direction,
            "price":    price,
            "deviation": self.deviation,
            "magic":    self.magic,
            "comment":  "CEO_CLOSE",
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            desc = result.comment if result else "No response"
            logger.error("MT5 close failed for #%s: retcode=%s — %s", ticket, code, desc)
            print(f"  ❌  MT5 close failed: #{ticket} retcode={code} — {desc}")
            return False
        return True

    def _get_tick(self, symbol: str) -> Optional[dict]:
        if not MT5_AVAILABLE:
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _sim_place(self, symbol, direction, lots, entry, sl, tp) -> int:
        ticket = self._next_sim_ticket
        self._next_sim_ticket += 1
        print(f"  🔵  [SIM] Order #{ticket}: {direction.upper()} {lots:.3f} lots "
              f"{symbol} | entry≈{entry:.5f} SL={sl:.5f} TP={tp:.5f}")
        return ticket

    def _sim_account(self) -> dict:
        return {
            "balance":     10_000.0,
            "equity":      10_000.0,
            "currency":    "USD",
            "leverage":    100,
            "profit":      0.0,
            "free_margin": 10_000.0,
        }

    def _sim_sym_info(self, symbol: str) -> dict:
        """Default sim symbol info — approximates XAUUSD."""
        return {
            "symbol":          symbol,
            "digits":          2,
            "tick_size":       0.01,
            "tick_value":      1.0,
            "point":           0.01,
            "volume_min":      0.01,
            "volume_max":      500.0,
            "volume_step":     0.01,
            "spread":          30,
            "currency_profit": "USD",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # P&L calculation — real from MT5 history (live) or estimated (sim)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_real_pnl(self, ticket: int) -> Optional[float]:
        """
        Pull actual realised P&L from MT5 deal history for a closed ticket.
        Returns None if not available (falls back to estimate).
        The `* 100` rough multiplier used previously was instrument-specific
        and wrong for anything other than XAUUSD at ~$1/pip/0.01lot.
        """
        if not MT5_AVAILABLE:
            return None
        try:
            deals = mt5.history_deals_get(position=ticket)
            if deals is None or len(deals) == 0:
                return None
            # Sum profit across all deal legs (partial closes + final close)
            return round(sum(d.profit for d in deals), 2)
        except Exception as e:
            logger.warning("MT5 history_deals_get() failed for ticket %s, "
                            "falling back to estimated P&L: %s", ticket, e)
            return None

    def _estimate_pnl(self, trade: TradeRecord, close_price: float) -> float:
        """
        Fallback P&L estimate used in simulation mode only.
        In live mode, _get_real_pnl() is called first and takes priority.

        Uses tick_value from sym_info if available, otherwise falls back
        to a price-points approximation.
        """
        direction_mult = 1 if trade.direction == "long" else -1

        # Partial TP profits already banked
        partial_r = 0.0
        if trade.tp1_hit:
            partial_r += abs(trade.tp1 - trade.entry) * self.tp1_lots_pct
        if trade.tp2_hit:
            partial_r += abs(trade.tp2 - trade.entry) * self.tp2_lots_pct

        remaining_pct = 1.0 \
            - (self.tp1_lots_pct if trade.tp1_hit else 0.0) \
            - (self.tp2_lots_pct if trade.tp2_hit else 0.0)
        remaining_pnl = (close_price - trade.entry) * direction_mult * remaining_pct

        price_pnl = partial_r + remaining_pnl

        # Try to convert price points → account currency using tick_value
        sym_info = self._sim_sym_info(trade.symbol)   # safe fallback
        tick_size  = sym_info.get("tick_size",  0.01)
        tick_value = sym_info.get("tick_value", 1.0)
        if tick_size > 0:
            pnl = price_pnl / tick_size * tick_value * trade.lots
        else:
            pnl = price_pnl * trade.lots * 100  # original rough estimate

        return round(pnl, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # Trailing SL — continuous ATR-based trail
    # ─────────────────────────────────────────────────────────────────────────

    def set_trailing_sl(self, ticket: int, atr: float,
                        atr_mult: float = 1.5,
                        step_pct: float = 0.3) -> bool:
        """
        Enable ATR-based trailing SL for an open trade.

        Parameters
        ----------
        ticket    : trade ticket number
        atr       : current ATR value for the symbol/tf (from live bar data)
        atr_mult  : trail distance = atr * atr_mult  (default 1.5 ATR)
        step_pct  : minimum improvement before moving SL, as a fraction of
                    atr (default 0.3 = only move when SL improves by 0.3 ATR).
                    Prevents constant tiny SL adjustments causing slippage.

        The trail runs inside _update_trailing_sl() called on every
        manage_open_trades() tick. It only ever moves SL in the favourable
        direction — never widens it.
        """
        trade = self._open_trades.get(ticket)
        if trade is None:
            logger.warning("set_trailing_sl: ticket %s not found", ticket)
            return False
        trade.trail_active   = True
        trade.trail_atr_mult = atr_mult
        trade.trail_step_pct = step_pct
        trade.atr            = atr
        # Reset the watermark to current price so trail starts from here
        trade.trail_sl_high  = trade.current_price or trade.entry
        logger.info("Trailing SL enabled: #%s %s %s | %.1f ATR | step %.1f%%",
                    ticket, trade.symbol, trade.direction, atr_mult, step_pct * 100)
        return True

    def _update_trailing_sl(self, trade: TradeRecord, price: float) -> None:
        """
        Called inside _check_trade() on every price update.
        Moves the SL in the direction of the trade if price has advanced
        by at least step_pct of ATR beyond the previous watermark.
        Never widens the SL.
        """
        if not trade.trail_active or trade.atr <= 0:
            return

        atr          = trade.atr
        trail_dist   = atr * trade.trail_atr_mult
        step_min     = atr * trade.trail_step_pct
        is_long      = trade.direction == "long"

        if is_long:
            # Update watermark — only track highest price seen
            if price > (trade.trail_sl_high or trade.entry):
                trade.trail_sl_high = price
            # Proposed new SL = watermark - trail_dist
            new_sl = trade.trail_sl_high - trail_dist
            # Only move SL if it improves by >= step_min and stays above current SL
            if new_sl > trade.sl + step_min:
                old_sl = trade.sl
                rounded_sl = round(new_sl, 5)
                confirmed = True if self.simulation else self._mt5_modify_sl(trade.ticket, rounded_sl, trade.tp3)
                if confirmed:
                    trade.sl = rounded_sl
                    logger.info("Trail SL moved UP: #%s %s %.5f → %.5f (price %.5f)",
                                trade.ticket, trade.symbol, old_sl, trade.sl, price)
                    if self.alerts_obj:
                        try:
                            self.alerts_obj.trail_moved(trade.ticket, trade.symbol, old_sl, trade.sl, price)
                        except Exception:
                            pass
                else:
                    logger.warning("Trail SL move rejected by broker for #%s %s -- "
                                    "will retry next tick", trade.ticket, trade.symbol)
        else:
            # Short: track lowest price seen
            if price < (trade.trail_sl_high or trade.entry):
                trade.trail_sl_high = price
            # Proposed new SL = watermark + trail_dist
            new_sl = trade.trail_sl_high + trail_dist
            # Only move SL if it improves (lower than current) by >= step_min
            if new_sl < trade.sl - step_min:
                old_sl = trade.sl
                rounded_sl = round(new_sl, 5)
                confirmed = True if self.simulation else self._mt5_modify_sl(trade.ticket, rounded_sl, trade.tp3)
                if confirmed:
                    trade.sl = rounded_sl
                    logger.info("Trail SL moved DOWN: #%s %s %.5f → %.5f (price %.5f)",
                                trade.ticket, trade.symbol, old_sl, trade.sl, price)
                    if self.alerts_obj:
                        try:
                            self.alerts_obj.trail_moved(trade.ticket, trade.symbol, old_sl, trade.sl, price)
                        except Exception:
                            pass
                else:
                    logger.warning("Trail SL move rejected by broker for #%s %s -- "
                                    "will retry next tick", trade.ticket, trade.symbol)

    # ─────────────────────────────────────────────────────────────────────────
    # Manual trade operations (called from dashboard UI)
    # ─────────────────────────────────────────────────────────────────────────

    def manual_close(self, ticket: int, reason: str = "manual") -> dict:
        """
        Close a single trade by ticket, called from the dashboard Close button.
        Returns {"ok": True, "pnl": float} or {"ok": False, "error": str}.
        """
        trade = self._open_trades.get(ticket)
        if trade is None:
            return {"ok": False, "error": f"Ticket {ticket} not found in open trades"}
        try:
            if self.simulation:
                price = trade.current_price or trade.entry
            else:
                tick = self._get_tick(trade.symbol)
                if tick is None:
                    return {"ok": False, "error": f"Cannot get tick for {trade.symbol}"}
                price = tick["bid"] if trade.direction == "long" else tick["ask"]
                if not self._mt5_close(ticket, trade.lots_remaining, trade.symbol):
                    return {"ok": False, "error": "Broker rejected the close order — "
                                                   "position is still open. Check server logs."}
            pnl = self._get_real_pnl(ticket) or self._estimate_pnl(trade, price)
            trade.close_price  = price
            trade.close_time   = datetime.now(timezone.utc)
            trade.close_reason = reason
            trade.pnl          = pnl
            trade.status       = "closed"
            self.guard.record_closed_trade(pnl=pnl)
            del self._open_trades[ticket]
            self._closed_trades.append(trade)
            self._log_trade(trade)
            logger.info("Manual close: #%s %s @ %.5f P&L=%.2f", ticket, trade.symbol, price, pnl)
            return {"ok": True, "ticket": ticket, "pnl": pnl, "price": price}
        except Exception as e:
            logger.exception("manual_close failed for ticket %s", ticket)
            return {"ok": False, "error": str(e)}

    def manual_modify_sl(self, ticket: int, new_sl: float) -> dict:
        """
        Manually change the SL of an open trade from the dashboard.
        Returns {"ok": True} or {"ok": False, "error": str}.
        Validates that new SL doesn't cross current price.
        """
        trade = self._open_trades.get(ticket)
        if trade is None:
            return {"ok": False, "error": f"Ticket {ticket} not found"}
        price = trade.current_price or trade.entry
        is_long = trade.direction == "long"
        if is_long and new_sl >= price:
            return {"ok": False, "error": f"SL {new_sl} must be below current price {price:.5f}"}
        if not is_long and new_sl <= price:
            return {"ok": False, "error": f"SL {new_sl} must be above current price {price:.5f}"}
        old_sl   = trade.sl
        if not self.simulation:
            if not self._mt5_modify_sl(ticket, new_sl, trade.tp3):
                return {"ok": False, "error": "Broker rejected the SL modification. Check server logs."}
        trade.sl = new_sl
        logger.info("Manual SL modify: #%s %s %.5f → %.5f", ticket, trade.symbol, old_sl, new_sl)
        return {"ok": True, "ticket": ticket, "old_sl": old_sl, "new_sl": new_sl}

    def close_all(self, reason: str = "manual"):
        """
        Emergency close all open trades. A trade whose close is rejected
        by the broker stays in _open_trades (not silently dropped) so
        it's still visible and manageable rather than orphaned.
        """
        print(f"\n  ⚠️   Closing all {len(self._open_trades)} open trades: {reason}")
        for ticket, trade in list(self._open_trades.items()):
            if not self.simulation:
                if not self._mt5_close(ticket, trade.lots_remaining, trade.symbol):
                    logger.error("close_all: broker rejected close for #%s %s -- left open",
                                 ticket, trade.symbol)
                    continue
            trade.status       = "closed"
            trade.close_reason = reason
            trade.close_time   = datetime.now(timezone.utc)
            self._log_trade(trade)
            del self._open_trades[ticket]
            self._closed_trades.append(trade)

    def open_count(self) -> int:
        return len(self._open_trades)

    def trade_summary(self) -> dict:
        closed = self._closed_trades
        if not closed:
            return {"trades": 0}
        pnls   = [t.pnl for t in closed if t.pnl is not None]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        return {
            "trades":      len(closed),
            "wins":        len(wins),
            "losses":      len(losses),
            "win_rate":    len(wins) / len(closed) * 100 if closed else 0,
            "total_pnl":   sum(pnls),
            "avg_win":     sum(wins) / len(wins) if wins else 0,
            "avg_loss":    sum(losses) / len(losses) if losses else 0,
            "open_trades": len(self._open_trades),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (simulation mode)
# ─────────────────────────────────────────────────────────────────────────────
