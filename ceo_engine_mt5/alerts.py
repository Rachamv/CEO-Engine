"""
The CEO Protocol — Phase 4: Alert System
=============================================
Sends rich Telegram alerts on signal, trade open, TP hits,
SL hit, and daily summary. Attaches chart PNG to signal alerts.

Setup
-----
1. Create a Telegram bot via @BotFather → get BOT_TOKEN
2. Send a message to your bot, then get CHAT_ID:
   https://api.telegram.org/bot<TOKEN>/getUpdates
3. Set environment variables:
       export CEO_TELEGRAM_TOKEN="your_token"
       export CEO_TELEGRAM_CHAT_ID="your_chat_id"
   Or pass them directly to AlertSystem().

Usage
-----
    from .alerts import AlertSystem

    alerts = AlertSystem(
        token   = "your_bot_token",
        chat_id = "your_chat_id",
    )

    # On signal
    alerts.signal(symbol, tf, direction, entry, sl, tp1, tp2, tp3,
                  quality, model, session, chart_path=None)

    # On trade events
    alerts.trade_opened(ticket, symbol, direction, lots, entry, sl, tp3)
    alerts.tp_hit(ticket, symbol, direction, tp_num, price, pnl_so_far)
    alerts.sl_hit(ticket, symbol, direction, price, pnl)
    alerts.daily_summary(stats, guard_status)
"""

import os
import time
import requests
from datetime import datetime, timezone
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Alert System
# ─────────────────────────────────────────────────────────────────────────────

class AlertSystem:
    """
    Telegram alert dispatcher for the CEO Engine.

    All methods fail silently — alerts should never crash the trading system.
    Set verbose=True to see send confirmations in the console.
    """

    TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
    _MIN_INTERVAL = 1.1   # seconds between messages (Telegram allows ~1/sec per chat)

    def __init__(
        self,
        token:       Optional[str] = None,
        chat_id:     Optional[str] = None,
        verbose:     bool = True,
        dry_run:     bool = False,
        # Per-event toggles — set any to False to suppress that alert type
        on_signal:   bool = True,
        on_trade_open: bool = True,
        on_tp_hit:   bool = True,
        on_sl_hit:   bool = True,
        on_daily_summary: bool = True,
        on_system:   bool = True,
        on_guard_halt: bool = True,
        on_trail_moved: bool = False,   # off by default — would be very noisy
    ):
        self.token   = token   or os.environ.get("CEO_TELEGRAM_TOKEN",   "")
        self.chat_id = chat_id or os.environ.get("CEO_TELEGRAM_CHAT_ID", "")
        self.verbose = verbose
        self.dry_run = dry_run
        self._last_sent: float = 0.0

        # Per-event toggles
        self.on_signal        = on_signal
        self.on_trade_open    = on_trade_open
        self.on_tp_hit        = on_tp_hit
        self.on_sl_hit        = on_sl_hit
        self.on_daily_summary = on_daily_summary
        self.on_system        = on_system
        self.on_guard_halt    = on_guard_halt
        self.on_trail_moved   = on_trail_moved

        if not self.token or not self.chat_id:
            print("  ⚠️   AlertSystem: no Telegram token/chat_id — running in dry_run mode")
            self.dry_run = True

    @classmethod
    def from_config(cls, cfg: dict) -> "AlertSystem":
        """
        Build an AlertSystem from a config dict (e.g. from ceo_engine_config.json).
        Handles missing keys gracefully so old configs stay compatible.
        """
        alert_cfg = cfg.get("alerts", {})
        return cls(
            token       = cfg.get("telegram_token") or "",
            chat_id     = cfg.get("telegram_chat")  or "",
            verbose     = cfg.get("verbose_alerts", True),
            on_signal        = alert_cfg.get("on_signal",        True),
            on_trade_open    = alert_cfg.get("on_trade_open",    True),
            on_tp_hit        = alert_cfg.get("on_tp_hit",        True),
            on_sl_hit        = alert_cfg.get("on_sl_hit",        True),
            on_daily_summary = alert_cfg.get("on_daily_summary", True),
            on_system        = alert_cfg.get("on_system",        True),
            on_guard_halt    = alert_cfg.get("on_guard_halt",    True),
            on_trail_moved   = alert_cfg.get("on_trail_moved",   False),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public alert methods
    # ─────────────────────────────────────────────────────────────────────────

    def signal(
        self,
        symbol:     str,
        tf:         str,
        direction:  str,
        entry:      float,
        sl:         float,
        tp1:        float,
        tp2:        float,
        tp3:        float,
        quality:    float,
        model:      str,
        session:    str       = "",
        pat_name:   str       = "",
        ceo_valid:  bool      = False,
        bos:        bool      = False,
        in_discount: bool     = False,
        chart_path: Optional[str] = None,
    ):
        """Send a signal alert, optionally with chart image attached."""
        if not self.on_signal:
            return
        arrow   = "🟢 ▲ LONG" if direction == "long" else "🔴 ▼ SHORT"
        rr1     = abs(tp1 - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        rr3     = abs(tp3 - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0

        # CEO sequence checkmarks
        checks  = []
        if bos:        checks.append("✅ BOS confirmed")
        if in_discount: checks.append("✅ Discount zone")
        if ceo_valid:  checks.append("✅ CEO sequence valid")
        if pat_name:   checks.append(f"📐 Pattern: {pat_name}")

        checks_str = "\n".join(checks) if checks else "⚡ Base sweep signal"

        msg = (
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 *CEO ENGINE SIGNAL*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{arrow}  `{symbol}`  `{tf}`\n\n"
            f"📊 *Levels*\n"
            f"  Entry  : `{entry:.5f}`\n"
            f"  SL     : `{sl:.5f}`\n"
            f"  TP1    : `{tp1:.5f}` _(1:{rr1:.1f}R)_\n"
            f"  TP2    : `{tp2:.5f}`\n"
            f"  TP3    : `{tp3:.5f}` _(1:{rr3:.1f}R)_\n\n"
            f"🔬 *Quality*: `{quality:.1f}` | Model: _{model}_\n"
            f"🕐 *Session*: {session.title()}\n\n"
            f"{checks_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_"
        )

        if chart_path and os.path.exists(chart_path):
            self._send_photo(chart_path, caption=msg)
        else:
            self._send_message(msg)

    def trade_opened(
        self,
        ticket:    int,
        symbol:    str,
        direction: str,
        lots:      float,
        entry:     float,
        sl:        float,
        tp3:       float,
    ):
        if not self.on_trade_open:
            return
        arrow = "🟢 LONG" if direction == "long" else "🔴 SHORT"
        msg   = (
            f"✅ *TRADE OPENED*\n"
            f"#{ticket}  {arrow}  `{symbol}`\n"
            f"Lots: `{lots:.3f}` @ `{entry:.5f}`\n"
            f"SL: `{sl:.5f}`  TP: `{tp3:.5f}`"
        )
        self._send_message(msg)

    def tp_hit(
        self,
        ticket:     int,
        symbol:     str,
        direction:  str,
        tp_num:     int,
        price:      float,
        pnl_so_far: float,
    ):
        if not self.on_tp_hit:
            return
        msg = (
            f"🎯 *TP{tp_num} HIT*  #{ticket}\n"
            f"`{symbol}`  {'▲' if direction == 'long' else '▼'} @ `{price:.5f}`\n"
            f"P&L so far: `${pnl_so_far:+.2f}`\n"
            f"{'SL moved to breakeven ✅' if tp_num == 1 else 'SL trailed to TP1 ✅'}"
        )
        self._send_message(msg)

    def sl_hit(
        self,
        ticket:    int,
        symbol:    str,
        direction: str,
        price:     float,
        pnl:       float,
    ):
        if not self.on_sl_hit:
            return
        msg = (
            f"❌ *SL HIT*  #{ticket}\n"
            f"`{symbol}`  {'▲' if direction == 'long' else '▼'} @ `{price:.5f}`\n"
            f"P&L: `${pnl:+.2f}`"
        )
        self._send_message(msg)

    def trade_closed(
        self,
        ticket:       int,
        symbol:       str,
        direction:    str,
        close_reason: str,
        close_price:  float,
        pnl:          float,
        tp1_hit:      bool,
        tp2_hit:      bool,
    ):
        icon  = "✅" if pnl >= 0 else "❌"
        tps   = f"TP1{'✓' if tp1_hit else '✗'} TP2{'✓' if tp2_hit else '✗'}"
        msg   = (
            f"{icon} *TRADE CLOSED*  #{ticket}\n"
            f"`{symbol}`  {'▲' if direction == 'long' else '▼'} "
            f"@ `{close_price:.5f}`\n"
            f"Reason: _{close_reason.upper()}_  |  {tps}\n"
            f"P&L: `${pnl:+.2f}`"
        )
        self._send_message(msg)

    def daily_summary(
        self,
        stats:        dict,
        guard_status: Optional[dict] = None,
    ):
        """Send end-of-day summary."""
        if not self.on_daily_summary:
            return
        trades    = stats.get("trades", 0)
        wins      = stats.get("wins", 0)
        losses    = stats.get("losses", 0)
        total_pnl = stats.get("total_pnl", 0.0)
        win_rate  = stats.get("win_rate", 0.0)
        avg_r     = stats.get("avg_r", 0.0)

        guard_str = ""
        if guard_status:
            dd_pct = guard_status.get("drawdown_used_pct", 0)
            dl_pct = guard_status.get("daily_loss_used_pct", 0)
            td     = guard_status.get("trading_days_completed", 0)
            td_req = guard_status.get("trading_days_required", 0)
            guard_str = (
                f"\n\n🛡 *Account Guard*\n"
                f"  Daily Loss Used : `{dl_pct:.1f}%`\n"
                f"  Drawdown Used   : `{dd_pct:.1f}%`\n"
                f"  Trading Days    : `{td}/{td_req}`"
            )

        pnl_icon = "📈" if total_pnl >= 0 else "📉"
        msg = (
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{pnl_icon} *DAILY SUMMARY*\n"
            f"_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}_\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Trades  : `{trades}` ({wins}W / {losses}L)\n"
            f"Win Rate: `{win_rate:.1f}%`\n"
            f"Avg R   : `{avg_r:+.3f}R`\n"
            f"P&L     : `${total_pnl:+.2f}`"
            f"{guard_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )
        self._send_message(msg)

    def guard_halt(self, reason: str, daily_loss_pct: float = 0, dd_pct: float = 0):
        """Send a halt alert — always sent regardless of on_guard_halt (safety critical)."""
        msg = (
            f"🛑 *ENGINE HALTED*\n"
            f"Reason: _{reason}_\n"
            f"Daily loss used: `{daily_loss_pct:.1f}%`\n"
            f"Drawdown used: `{dd_pct:.1f}%`\n"
            f"_All trading suspended. Restart manually._"
        )
        self._send_message(msg)   # always fires regardless of toggle

    def trail_moved(self, ticket: int, symbol: str, old_sl: float, new_sl: float, price: float):
        """Send a trailing SL move notification (off by default — noisy)."""
        if not self.on_trail_moved:
            return
        direction = "UP" if new_sl > old_sl else "DOWN"
        msg = (
            f"⬆ *TRAIL SL {direction}*  #{ticket}\n"
            f"`{symbol}` @ `{price:.5f}`\n"
            f"SL: `{old_sl:.5f}` → `{new_sl:.5f}`"
        )
        self._send_message(msg)

    def system_alert(self, message: str, level: str = "info"):
        """Send a system-level alert (errors, halts, restarts)."""
        if not self.on_system:
            return
        icons = {"info": "ℹ️", "warning": "⚠️", "error": "🚨", "halt": "🛑"}
        icon  = icons.get(level, "ℹ️")
        msg   = f"{icon} *CEO ENGINE*\n{message}"
        self._send_message(msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Transport
    # ─────────────────────────────────────────────────────────────────────────

    def _send_message(self, text: str, retries: int = 2):
        if self.dry_run:
            print(f"\n  [DRY RUN ALERT]\n{text}\n")
            return True

        # Rate limit — enforce minimum interval between sends
        now = time.time()
        gap = now - self._last_sent
        if gap < self._MIN_INTERVAL:
            time.sleep(self._MIN_INTERVAL - gap)

        url     = self.TELEGRAM_API.format(token=self.token, method="sendMessage")
        payload = {
            "chat_id":    self.chat_id,
            "text":       text,
            "parse_mode": "Markdown",
        }

        for attempt in range(retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=10)
                if r.status_code == 200:
                    self._last_sent = time.time()
                    if self.verbose:
                        print(f"  📨  Alert sent (message)")
                    return True
                elif r.status_code == 429:
                    # Telegram rate limit — back off
                    retry_after = r.json().get("parameters", {}).get("retry_after", 5)
                    if self.verbose:
                        print(f"  ⚠️   Telegram rate limit — waiting {retry_after}s")
                    time.sleep(retry_after)
                else:
                    if self.verbose:
                        print(f"  ⚠️   Alert failed: {r.status_code} {r.text[:80]}")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️   Alert error (attempt {attempt+1}): {e}")
                if attempt < retries:
                    time.sleep(2)

        return False

    def _send_photo(self, image_path: str, caption: str = "", retries: int = 2):
        if self.dry_run:
            print(f"\n  [DRY RUN PHOTO ALERT]\n  Image: {image_path}\n{caption}\n")
            return True

        url = self.TELEGRAM_API.format(token=self.token, method="sendPhoto")

        for attempt in range(retries + 1):
            try:
                with open(image_path, "rb") as img:
                    r = requests.post(
                        url,
                        data={
                            "chat_id":    self.chat_id,
                            "caption":    caption[:1024],  # Telegram caption limit
                            "parse_mode": "Markdown",
                        },
                        files={"photo": img},
                        timeout=20,
                    )
                if r.status_code == 200:
                    if self.verbose:
                        print(f"  📨  Alert sent (photo + caption)")
                    return True
                else:
                    # Fall back to text-only
                    if self.verbose:
                        print(f"  ⚠️   Photo send failed ({r.status_code}), sending text")
                    return self._send_message(caption)
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️   Photo alert error (attempt {attempt+1}): {e}")
                if attempt < retries:
                    time.sleep(2)

        return self._send_message(caption)

    def test_connection(self) -> bool:
        """Send a test message to verify the connection works."""
        return self._send_message(
            "✅ *CEO Engine connected*\nAlert system is live."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (dry run)
# ─────────────────────────────────────────────────────────────────────────────
