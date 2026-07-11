"""
The CEO Protocol — Phase 4: Web Dashboard  v3.6
====================================================
Layout: MT5-faithful wireframe design.
Chart:  TradingView Lightweight Charts (native candlestick feel).
P&L:    1-second floating tick — only the numbers update, not the whole row.

Run:
    python run_dashboard_preview.py   # preview with mock data (project root)
"""

import os
import sys
import json as _json_auth
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict

from flask import Flask, jsonify, render_template_string, request, Response
from ceo_engine_mt5.ceo_logging import get_logger

logger = get_logger(__name__)


# ── State store ───────────────────────────────────────────────────────────────

class DashboardState:
    def __init__(self):
        self._lock       = threading.Lock()
        self.signals:    List[dict]        = []
        self.trades:     List[dict]        = []
        self.account:    dict              = {}
        self.guard:      dict              = {}
        self.stats:      dict              = {}
        self.log:        List[dict]        = []
        self.candles:    Dict[str, list]   = {}   # key = "SYMBOL_TF" → list of OHLCV dicts
        self.structure:  Dict[str, dict]   = {}   # key = "SYMBOL_TF" → CEO structure overlay payload
        self.market_ticks: Dict[str, dict] = {}   # symbol → {bid, ask}
        self.backtest_results: Dict[str, dict] = {}  # "SYMBOL_TF" → rows
        self._executor = None   # set by set_executor() after engine starts
        self._conn     = None   # MT5Connection reference for manual trades
        self.charts:     Dict[str, dict]   = {}   # legacy HTML chart cache
        self.journal_path: Optional[str]   = None
        self.last_update = datetime.now(timezone.utc).isoformat()

    def update_signal(self, signal: dict):
        with self._lock:
            self.signals = [signal] + [s for s in self.signals
                if not (s.get("symbol") == signal.get("symbol") and s.get("tf") == signal.get("tf"))]
            self.signals = self.signals[:50]
            self.last_update = datetime.now(timezone.utc).isoformat()

    def update_trades(self, trades: List[dict]):
        with self._lock:
            self.trades = trades
            self.last_update = datetime.now(timezone.utc).isoformat()

    def update_account(self, account: dict):
        with self._lock:
            self.account = account

    def update_guard(self, guard: dict):
        with self._lock:
            self.guard = guard

    def update_stats(self, stats: dict):
        with self._lock:
            self.stats = stats

    def add_log(self, message: str, level: str = "info"):
        with self._lock:
            self.log = [{"time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                         "message": message, "level": level}] + self.log[:19]

    def update_candles(self, symbol: str, tf: str, bars: list):
        """
        Push OHLCV bars for Lightweight Charts rendering.
        bars: list of dicts with keys: time (unix ts), open, high, low, close, volume
        """
        with self._lock:
            self.candles[f"{symbol}_{tf}"] = bars[-500:]   # keep last 500 bars

    def get_candles(self, symbol: str, tf: str) -> list:
        with self._lock:
            return self.candles.get(f"{symbol}_{tf}", [])

    def update_structure(self, symbol: str, tf: str, structure: dict):
        """Push CEO structure overlay (order block/FVG/QM zones, Fib 50%,
        BOS + CEO-valid markers) for the live LightweightCharts view."""
        with self._lock:
            self.structure[f"{symbol}_{tf}"] = structure

    def get_structure(self, symbol: str, tf: str) -> dict:
        with self._lock:
            return self.structure.get(f"{symbol}_{tf}",
                                       {"zoneLines": [], "priceLines": [], "markers": []})

    def set_executor(self, executor, conn=None):
        """Called by run_live() once the executor and MT5 connection are ready."""
        with self._lock:
            self._executor = executor
            self._conn     = conn

    def get_executor(self):
        with self._lock:
            return self._executor

    def update_market_tick(self, symbol: str, bid: float, ask: float):
        """Push live bid/ask for Market Watch. Called every 2s."""
        with self._lock:
            self.market_ticks[symbol] = {"symbol": symbol, "bid": bid, "ask": ask,
                "time": datetime.now(timezone.utc).isoformat()}

    def update_backtest(self, symbol: str, tf: str, results: list):
        """Store backtest rows for dashboard Backtest tab."""
        with self._lock:
            self.backtest_results[f"{symbol}_{tf}"] = results

    # Legacy HTML chart cache — kept for backward compat with chart_lwc.py callers
    def update_chart(self, symbol: str, tf: str, html: str):
        with self._lock:
            self.charts[f"{symbol}_{tf}"] = {"html": html,
                "cached_at": datetime.now(timezone.utc).isoformat()}

    def get_chart(self, symbol: str, tf: str) -> str:
        with self._lock:
            return self.charts.get(f"{symbol}_{tf}", {}).get("html", "")

    def snapshot(self) -> dict:
        with self._lock:
            return {"signals": self.signals[:20], "trades": self.trades,
                    "account": self.account, "guard": self.guard,
                    "stats": self.stats, "log": self.log,
                    "market_ticks": self.market_ticks, "backtest_results": self.backtest_results, "last_update": self.last_update}

    def trades_snapshot(self) -> list:
        """Lightweight snapshot for 1-second P&L tick — floating state only."""
        with self._lock:
            return [
                {"ticket": t.get("ticket"),
                 "floating_pnl": t.get("floating_pnl"),
                 "floating_r":   t.get("floating_r"),
                 "current_price":t.get("current_price")}
                for t in self.trades
            ]


state = DashboardState()
app   = Flask(__name__)

# ── Dashboard access control ────────────────────────────────────────────────
# Every route below can read MT5 credentials, place trades, and close
# positions. Binding to 0.0.0.0 without auth means anyone on the same
# network (or the internet, if port-forwarded) can hit those endpoints.
# A single shared password, checked via HTTP Basic Auth, closes that gap
# without requiring any changes to the frontend JS (browsers cache and
# resend Basic Auth credentials automatically once entered).
import secrets as _secrets

_AUTH_CONFIG_PATH = "ceo_dashboard_auth.json"


def _load_or_create_dashboard_password() -> str:
    """Use CEO_DASHBOARD_PASSWORD if set, else reuse/create a saved one."""
    env_pw = os.environ.get("CEO_DASHBOARD_PASSWORD")
    if env_pw:
        return env_pw
    if os.path.exists(_AUTH_CONFIG_PATH):
        try:
            with open(_AUTH_CONFIG_PATH) as f:
                saved = _json_auth.load(f)
            if saved.get("password"):
                return saved["password"]
        except Exception:
            pass
    pw = _secrets.token_urlsafe(18)
    try:
        with open(_AUTH_CONFIG_PATH, "w") as f:
            _json_auth.dump({"password": pw}, f, indent=2)
        os.chmod(_AUTH_CONFIG_PATH, 0o600)
    except Exception:
        pass
    return pw


_DASHBOARD_USER = "admin"
_DASHBOARD_PASSWORD = _load_or_create_dashboard_password()


# ── Rate limiting ────────────────────────────────────────────────────────────
# Auth alone doesn't stop repeated password guesses or a flood of trade
# requests from a single source. This is a small in-memory sliding-window
# limiter -- no extra dependency, good enough for a single-process
# dashboard. Login attempts are throttled per-IP regardless of outcome;
# trade-mutating endpoints get a looser per-IP cap on top of that.
import time as _time
import threading as _threading_rl
from collections import defaultdict as _defaultdict

_rate_lock    = _threading_rl.Lock()
_rate_buckets: dict = _defaultdict(list)   # key -> [timestamps]

LOGIN_MAX_ATTEMPTS  = 8     # per window, per source IP
LOGIN_WINDOW_SECS   = 60
TRADE_MAX_REQUESTS  = 20    # per window, per source IP
TRADE_WINDOW_SECS   = 60


def _rate_limited(key: str, max_calls: int, window_secs: int) -> bool:
    """Returns True if `key` has exceeded max_calls within window_secs,
    recording this call regardless. Thread-safe, O(window size) per call."""
    now = _time.time()
    with _rate_lock:
        bucket = _rate_buckets[key]
        cutoff = now - window_secs
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        if len(bucket) >= max_calls:
            return True
        bucket.append(now)
        return False


def _rate_check(key: str, max_calls: int, window_secs: int) -> bool:
    """Like _rate_limited but read-only -- does not record this call.
    Used for the login gate, where only *failed* attempts should count
    against the limit (recorded separately via _rate_record), so normal
    polling by an already-authenticated client is never throttled."""
    now = _time.time()
    with _rate_lock:
        bucket = _rate_buckets[key]
        cutoff = now - window_secs
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        return len(bucket) >= max_calls


def _rate_record(key: str) -> None:
    with _rate_lock:
        _rate_buckets[key].append(_time.time())


def _client_ip() -> str:
    """
    Client IP used to key the rate limiter. Deliberately does NOT trust
    X-Forwarded-For by default -- that header is attacker-controlled
    unless a real reverse proxy sits in front and overwrites it, which
    this dashboard normally doesn't have (it's typically reached
    directly, not through nginx/etc). Trusting it unconditionally would
    let anyone defeat both the login and trade rate limiters just by
    sending a different X-Forwarded-For value on every request.
    Set CEO_TRUST_PROXY=1 only if you've actually put this dashboard
    behind a proxy that sets/overwrites this header itself.
    """
    if os.environ.get("CEO_TRUST_PROXY") == "1":
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


@app.before_request
def _require_dashboard_auth():
    ip = _client_ip()
    login_key = f"login:{ip}"
    if _rate_check(login_key, LOGIN_MAX_ATTEMPTS, LOGIN_WINDOW_SECS):
        return Response(
            "Too many failed login attempts. Try again shortly.", 429,
            {"Retry-After": str(LOGIN_WINDOW_SECS)})
    auth = request.authorization
    if not auth or auth.username != _DASHBOARD_USER or auth.password != _DASHBOARD_PASSWORD:
        _rate_record(login_key)
        return Response(
            "Authentication required.", 401,
            {"WWW-Authenticate": 'Basic realm="The CEO Protocol"'})
app.config["JSON_SORT_KEYS"] = False


def _rate_limit_trade(fn):
    """Decorator for trade-mutating endpoints: caps requests per source IP
    so a stuck client (or an attacker who has somehow gotten past auth)
    can't hammer order placement/closing."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ip = _client_ip()
        if _rate_limited(f"trade:{ip}", TRADE_MAX_REQUESTS, TRADE_WINDOW_SECS):
            return jsonify({"ok": False, "error": "Too many requests, slow down."}), 429
        return fn(*args, **kwargs)
    return wrapper


def _safe_error_response(exc: Exception, context: str, status: int = 500, include_ok: bool = True):
    """Logs the full exception server-side and returns a generic message to
    the client -- the raw exception text can include file paths, internal
    object state, or other details that shouldn't leave the server."""
    logger.exception("%s failed", context)
    msg = f"{context} failed. Check server logs for details."
    payload = {"ok": False, "error": msg} if include_ok else {"error": msg}
    return jsonify(payload), status


# Every route below that places, closes, or modifies a live trade shares
# this lock so two concurrent requests (e.g. a double-click, or the
# dashboard auto-refresh racing a manual click) can't interleave against
# the same executor/connection state.
_trade_op_lock = threading.Lock()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/state")
def api_state():
    return jsonify(state.snapshot())

@app.route("/api/pnl")
def api_pnl():
    """1-second lightweight endpoint — floating P&L only, no signal/account data."""
    return jsonify(state.trades_snapshot())

@app.route("/api/candles/<symbol>/<tf>")
def api_candles(symbol, tf):
    """OHLCV bars for Lightweight Charts. Returns [] when no data yet."""
    bars = state.get_candles(symbol, tf)
    return jsonify(bars)

@app.route("/api/structure/<symbol>/<tf>")
def api_structure(symbol, tf):
    """CEO structure overlay (order block/FVG/QM zones, Fib 50%, BOS +
    CEO-valid markers) for the live chart. Empty shape when no data yet."""
    return jsonify(state.get_structure(symbol, tf))

@app.route("/api/chart/<symbol>/<tf>")
def api_chart(symbol, tf):
    """Legacy HTML chart endpoint — used as fallback if candles not available."""
    html = state.get_chart(symbol, tf)
    return html if html else ""

@app.route("/api/log", methods=["POST"])
def api_log():
    data = request.get_json(silent=True) or {}
    state.add_log(data.get("message", ""), data.get("level", "info"))
    return jsonify({"ok": True})

@app.route("/api/history")
def api_history():
    if state.journal_path and os.path.exists(state.journal_path):
        try:
            from .journal import Journal
            j = Journal(state.journal_path)
            return jsonify({"recent_trades": j.recent_trades(50),
                            "daily_summary": j.daily_summary(30),
                            "stats": j.performance_stats(),
                            "by_model": j.stats_by_model(),
                            "by_session": j.stats_by_session()})
        except Exception as e:
            return _safe_error_response(e, "api_history", include_ok=False)
    return jsonify({"recent_trades": [], "daily_summary": [], "stats": {}})

@app.route("/api/market_ticks")
def api_market_ticks():
    """Live bid/ask for all symbols -- polled every 2s by Market Watch."""
    return jsonify(state.market_ticks)

@app.route("/api/backtest")
def api_backtest():
    """Backtest results for all registered symbols."""
    return jsonify(state.backtest_results)

# -- Engine process control ---------------------------------------------------
import subprocess as _subprocess, json as _json, threading as _threading2
_engine_proc = None
_engine_lock = _threading2.Lock()
_config_path = "ceo_engine_config.json"

@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    """Read or write the engine config JSON."""
    import os
    if request.method == "POST":
        cfg = request.get_json(silent=True) or {}
        with open(_config_path, "w") as f:
            _json.dump(cfg, f, indent=2)
        # This file can hold the MT5 password and Telegram bot token in
        # plaintext -- lock it down to the owner, same as the auth file.
        try:
            os.chmod(_config_path, 0o600)
        except Exception:
            pass
        return jsonify({"ok": True})
    if os.path.exists(_config_path):
        with open(_config_path) as f:
            return jsonify(_json.load(f))
    return jsonify({"symbols": ["XAUUSD"], "tf": "H1", "risk_pct": 1.0,
                    "account_size": 10000, "daily_loss_pct": 5.0,
                    "max_dd_pct": 10.0, "auto_trade": False,
                    "dashboard_port": 5000, "journal": "ceo_journal.db",
                    "mt5_login": None, "mt5_password": None, "mt5_server": None})

@app.route("/api/start", methods=["POST"])
def api_start():
    """Start the engine subprocess using the saved config."""
    global _engine_proc
    import sys, os
    with _engine_lock:
        if _engine_proc and _engine_proc.poll() is None:
            return jsonify({"ok": False, "error": "Engine already running"})
        if not os.path.exists(_config_path):
            return jsonify({"ok": False, "error": "No config saved. Complete the setup wizard first."})
        with open(_config_path) as f:
            cfg = _json.load(f)
        cmd = [sys.executable, "run.py"]
        for s in cfg.get("symbols", ["XAUUSD"]):
            cmd += ["--symbol", s]
        cmd += ["--tf", cfg.get("tf", "H1"), "--source", "mt5", "--live"]
        if cfg.get("mt5_login"):      cmd += ["--mt5-login",      str(cfg["mt5_login"])]
        if cfg.get("mt5_password"):   cmd += ["--mt5-password",   cfg["mt5_password"]]
        if cfg.get("mt5_server"):     cmd += ["--mt5-server",     cfg["mt5_server"]]
        if cfg.get("auto_trade"):     cmd += ["--auto-trade"]
        if cfg.get("risk_pct"):       cmd += ["--risk-pct",       str(cfg["risk_pct"])]
        if cfg.get("account_size"):   cmd += ["--account-size",   str(cfg["account_size"])]
        if cfg.get("daily_loss_pct"): cmd += ["--daily-loss-pct", str(cfg["daily_loss_pct"])]
        if cfg.get("max_dd_pct"):     cmd += ["--max-dd-pct",     str(cfg["max_dd_pct"])]
        cmd += ["--journal", cfg.get("journal", "ceo_journal.db")]
        cmd += ["--dashboard-port", str(cfg.get("dashboard_port", 5000))]
        if cfg.get("telegram_token"): cmd += ["--telegram-token", cfg["telegram_token"]]
        if cfg.get("telegram_chat"):  cmd += ["--telegram-chat",  cfg["telegram_chat"]]
        if cfg.get("min_quality"):    cmd += ["--min-quality",    str(cfg["min_quality"])]
        if cfg.get("confluence_mode") and cfg["confluence_mode"] != "sweep":
            cmd += ["--confluence-mode", cfg["confluence_mode"]]
        if cfg.get("min_consistency"): cmd += ["--min-consistency", str(cfg["min_consistency"])]
        if cfg.get("sessions"):       cmd += ["--sessions"]        + cfg["sessions"]
        # News filter
        if cfg.get("no_news_filter"): cmd += ["--no-news-filter"]
        if cfg.get("news_pre_mins"):  cmd += ["--news-pre-mins",   str(cfg["news_pre_mins"])]
        if cfg.get("news_post_mins"): cmd += ["--news-post-mins",  str(cfg["news_post_mins"])]
        if cfg.get("news_medium"):    cmd += ["--news-medium"]
        if cfg.get("fcsapi_key"):     cmd += ["--fcsapi-key",      cfg["fcsapi_key"]]
        # MTF stack
        if cfg.get("mtf_tfs"):        cmd += ["--mtf"] + cfg["mtf_tfs"]
        if cfg.get("mtf_mode"):       cmd += ["--mtf-mode",        cfg["mtf_mode"]]
        if cfg.get("mtf_min_tfs"):    cmd += ["--mtf-min-tfs",     str(cfg["mtf_min_tfs"])]
        if cfg.get("mtf_min_score"):  cmd += ["--mtf-min-score",   str(cfg["mtf_min_score"])]
        # Performance feedback
        if cfg.get("perf_feedback"):  cmd += ["--perf-feedback"]
        if cfg.get("perf_window"):    cmd += ["--perf-window",     str(cfg["perf_window"])]
        if cfg.get("perf_loss_streak"):cmd+= ["--perf-loss-streak",str(cfg["perf_loss_streak"])]
        if cfg.get("perf_wr_floor"):  cmd += ["--perf-wr-floor",   str(cfg["perf_wr_floor"])]
        # Walk-forward
        if cfg.get("walkforward"):    cmd += ["--walkforward"]
        if cfg.get("wf_windows"):     cmd += ["--wf-windows",      str(cfg["wf_windows"])]
        # Sound
        if cfg.get("sound"):          cmd += ["--sound"]
        try:
            _engine_proc = _subprocess.Popen(
                cmd, stdout=_subprocess.PIPE,
                stderr=_subprocess.STDOUT, text=True, bufsize=1)
            state.add_log(f"Engine started (PID {_engine_proc.pid})", level="info")
            return jsonify({"ok": True, "pid": _engine_proc.pid})
        except Exception as e:
            return _safe_error_response(e, "api_start")

@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop the engine subprocess."""
    global _engine_proc
    with _engine_lock:
        if _engine_proc and _engine_proc.poll() is None:
            _engine_proc.terminate()
            state.add_log("Engine stopped by user", level="warn")
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "Engine is not running"})

@app.route("/api/engine_status")
def api_engine_status():
    running = _engine_proc is not None and _engine_proc.poll() is None
    return jsonify({"running": running,
                    "pid": _engine_proc.pid if running else None})

@app.route("/api/test_telegram", methods=["POST"])
def api_test_telegram():
    """Test Telegram connection with provided token + chat_id."""
    data = request.get_json(silent=True) or {}
    token   = data.get("token","").strip()
    chat_id = data.get("chat_id","").strip()
    if not token or not chat_id:
        return jsonify({"ok": False, "error": "token and chat_id required"})
    try:
        import requests as _req
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r   = _req.post(url, json={
            "chat_id": chat_id,
            "text": "✅ *CEO Engine* — Telegram connection test successful!",
            "parse_mode": "Markdown"
        }, timeout=10)
        if r.status_code == 200:
            return jsonify({"ok": True, "message": "Test message sent successfully!"})
        err = r.json().get("description", f"HTTP {r.status_code}")
        return jsonify({"ok": False, "error": err})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/find_chat_id", methods=["POST"])
def api_find_chat_id():
    """
    Reads recent messages sent to the user's bot via getUpdates and returns
    the chat ID of the most recent sender. This replaces the manual
    "open this URL and read the JSON" step for non-technical users.
    Each user calls this with their own bot token — the chat ID returned
    is unique to whoever messaged that specific bot.
    """
    data  = request.get_json(silent=True) or {}
    token = data.get("token", "").strip()
    if not token:
        return jsonify({"ok": False, "error": "Enter your Bot Token first"})
    try:
        import requests as _req
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        r = _req.get(url, timeout=10)
        if r.status_code != 200:
            err = r.json().get("description", f"HTTP {r.status_code}")
            return jsonify({"ok": False, "error": f"Telegram rejected the token: {err}"})
        updates = r.json().get("result", [])
        if not updates:
            return jsonify({"ok": False,
                "error": "No messages found yet. Open Telegram, search for your bot, "
                         "and send it any message (e.g. 'hi'), then click Find my Chat ID again."})
        # Use the most recent message — handles both direct messages and group messages
        last = updates[-1]
        chat = (last.get("message") or last.get("channel_post") or {}).get("chat", {})
        chat_id = chat.get("id")
        chat_title = chat.get("title") or chat.get("username") or chat.get("first_name") or "your chat"
        if chat_id is None:
            return jsonify({"ok": False, "error": "Could not read chat ID from the last message."})
        return jsonify({"ok": True, "chat_id": chat_id, "chat_name": chat_title})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/mt5_detect")
def api_mt5_detect():
    """
    Attempt a no-credentials MT5 connection and return the active account.
    Returns {ok: true, account: {...}} when MT5 is open and logged in.
    Returns {ok: false, error: "..."} when MT5 is not running or not on Windows.
    Safe to call repeatedly -- disconnects immediately after reading.
    """
    import sys
    if sys.platform != "win32":
        return jsonify({"ok": False, "error": "MT5 is only available on Windows."})
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return jsonify({"ok": False, "error": "MetaTrader5 package not installed."})
    try:
        if not mt5.initialize():
            err = mt5.last_error()
            mt5.shutdown()
            return jsonify({"ok": False,
                            "error": f"MT5 not running or not logged in. ({err})"})
        acct = mt5.account_info()
        term = mt5.terminal_info()
        # Read all available symbols for the wizard chip list
        all_syms = [s.name for s in (mt5.symbols_get() or [])]
        # Common tradeable groups
        common = ["XAUUSD","GBPUSD","EURUSD","USDJPY","USDCAD","AUDUSD",
                  "NZDUSD","USDCHF","BTCUSD","ETHUSD","XAGUSD",
                  "GBPJPY","EURJPY","EURGBP","AUDJPY"]
        broker_syms = [s for s in common if s in all_syms]
        mt5.shutdown()
        return jsonify({
            "ok": True,
            "account": {
                "login":       acct.login,
                "server":      acct.server,
                "name":        acct.name,
                "currency":    acct.currency,
                "balance":     round(acct.balance, 2),
                "equity":      round(acct.equity, 2),
                "leverage":    acct.leverage,
                "account_type": "demo" if "demo" in acct.server.lower() else "live",
            },
            "terminal": {
                "name":    term.name,
                "build":   term.build,
                "path":    term.path,
            },
            "broker_symbols": broker_syms,
            "all_symbols_count": len(all_syms),
        })
    except Exception as e:
        try: mt5.shutdown()
        except Exception: pass
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/trade", methods=["POST"])
@_rate_limit_trade
def api_trade():
    """
    Place a manual trade from a signal card.
    Body: {symbol, direction, entry, sl, tp1, tp2, tp3, quality, model, tf, lots (optional)}
    Returns {ok, ticket} or {ok: false, error}
    """
    data = request.get_json(silent=True) or {}
    ex = state.get_executor()
    if ex is None:
        return jsonify({"ok": False,
            "error": "Engine not running or auto-trade not enabled. "
                     "Start the engine with auto-trade turned on first."}), 400
    sym = data.get("symbol"); direction = data.get("direction")
    if not sym or not direction:
        return jsonify({"ok": False, "error": "symbol and direction required"}), 400
    with _trade_op_lock:
        try:
            # Get current account info for lot sizing
            conn = state._conn
            account_info = conn.account_info() if conn else None
            sym_info     = conn.symbol_info(sym) if conn else None
            if account_info is None or sym_info is None:
                return jsonify({"ok": False,
                    "error": "Cannot read account/symbol info from MT5. Is MT5 connected?"}), 400
            result = ex.place_trade(
                symbol      = sym,
                tf          = data.get("tf", "H1"),
                direction   = direction,
                entry       = float(data["entry"]),
                sl          = float(data["sl"]),
                tp1         = float(data.get("tp1") or data["entry"]),
                tp2         = float(data.get("tp2") or data["entry"]),
                tp3         = float(data.get("tp3") or data["entry"]),
                quality     = int(data.get("quality", 70)),
                model       = data.get("model", "manual"),
                bar_time    = None,
                atr         = float(data.get("atr", 0)),
                account_info= account_info,
                sym_info    = sym_info,
            )
            if result and result.get("ticket"):
                state.add_log(
                    f"Manual trade placed: #{result['ticket']} {sym} "
                    f"{direction.upper()} @ {data['entry']}", level="info")
                return jsonify({"ok": True, "ticket": result["ticket"],
                                "lots": result.get("lots"), "message": "Trade placed in MT5"})
            return jsonify({"ok": False, "error": result.get("error", "place_trade returned no ticket")})
        except Exception as e:
            return _safe_error_response(e, "api_trade")


@app.route("/api/close", methods=["POST"])
@_rate_limit_trade
def api_close():
    """
    Close an open trade from the dashboard Close button.
    Body: {ticket: int, reason: str (optional)}
    Returns {ok, pnl, price} or {ok: false, error}
    """
    data = request.get_json(silent=True) or {}
    ex = state.get_executor()
    if ex is None:
        return jsonify({"ok": False, "error": "Engine not running"}), 400
    ticket = data.get("ticket")
    if not ticket:
        return jsonify({"ok": False, "error": "ticket required"}), 400
    with _trade_op_lock:
        try:
            result = ex.manual_close(int(ticket), reason=data.get("reason", "manual"))
        except Exception as e:
            return _safe_error_response(e, "api_close")
    if result["ok"]:
        state.add_log(
            f"Trade closed: #{ticket} P&L ${result.get('pnl', 0):.2f}", level="info")
    return jsonify(result)


@app.route("/api/modify_sl", methods=["POST"])
@_rate_limit_trade
def api_modify_sl():
    """
    Manually change the SL of an open trade.
    Body: {ticket: int, new_sl: float}
    Returns {ok, old_sl, new_sl} or {ok: false, error}
    """
    data = request.get_json(silent=True) or {}
    ex = state.get_executor()
    if ex is None:
        return jsonify({"ok": False, "error": "Engine not running"}), 400
    ticket = data.get("ticket"); new_sl = data.get("new_sl")
    if not ticket or new_sl is None:
        return jsonify({"ok": False, "error": "ticket and new_sl required"}), 400
    with _trade_op_lock:
        try:
            result = ex.manual_modify_sl(int(ticket), float(new_sl))
        except Exception as e:
            return _safe_error_response(e, "api_modify_sl")
    if result["ok"]:
        state.add_log(
            f"SL modified: #{ticket} → {new_sl}", level="info")
    return jsonify(result)


@app.route("/api/set_trail", methods=["POST"])
@_rate_limit_trade
def api_set_trail():
    """
    Enable ATR-based trailing SL for an open trade.
    Body: {ticket: int, atr: float, atr_mult: float (default 1.5), step_pct: float (default 0.3)}
    Returns {ok} or {ok: false, error}
    """
    data = request.get_json(silent=True) or {}
    ex = state.get_executor()
    if ex is None:
        return jsonify({"ok": False, "error": "Engine not running"}), 400
    ticket = data.get("ticket"); atr = data.get("atr")
    if not ticket or not atr:
        return jsonify({"ok": False, "error": "ticket and atr required"}), 400
    with _trade_op_lock:
        try:
            ok = ex.set_trailing_sl(
                ticket   = int(ticket),
                atr      = float(atr),
                atr_mult = float(data.get("atr_mult", 1.5)),
                step_pct = float(data.get("step_pct", 0.3)),
            )
        except Exception as e:
            return _safe_error_response(e, "api_set_trail")
    if ok:
        state.add_log(
            f"Trailing SL enabled: #{ticket} | "
            f"{data.get('atr_mult',1.5)}x ATR | "
            f"step {float(data.get('step_pct',0.3))*100:.0f}%", level="info")
    return jsonify({"ok": ok,
        "error": None if ok else f"Ticket {ticket} not found in open trades"})


@app.route("/api/exec", methods=["POST"])
def api_exec():
    data = request.get_json(silent=True) or {}
    cmd  = (data.get("cmd") or "").strip()
    if not cmd:
        return jsonify({"ok": False, "error": "no command"}), 400
    try:
        if cmd == "tail_log":
            requested = (data.get("path") or "ceo_engine.log").strip()
            # The client only ever needs to tail the engine's own log files
            # in the working directory. Without this, "path" was passed
            # straight to open() -- a client could send an absolute path
            # or "../../../etc/passwd" and read any file the process can
            # see. basename() discards any directory component (whether
            # relative "../" traversal or an absolute path), so only a
            # bare filename inside cwd can ever be opened here.
            safe_name = os.path.basename(requested)
            if not safe_name or safe_name in (".", ".."):
                return jsonify({"ok": False, "error": "Invalid log path"}), 400
            log_file = os.path.join(os.getcwd(), safe_name)
            if not os.path.exists(log_file):
                return jsonify({"ok": True, "output": f"Log file not found: {safe_name}"})
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[-200:]
            return jsonify({"ok": True, "output": "".join(lines)})
        if cmd == "journal_head":
            n = int(data.get("n", 30))
            if not state.journal_path or not os.path.exists(state.journal_path):
                return jsonify({"ok": True, "output": "No journal connected."})
            try:
                from .journal import Journal
                j = Journal(state.journal_path)
                return jsonify({"ok": True, "output": j.recent_trades(n)})
            except Exception as e:
                return _safe_error_response(e, "api_exec(journal_head)")
        if cmd == "list_signals":
            return jsonify({"ok": True, "output": state.signals})
        return jsonify({"ok": False, "error": f"Unknown command: {cmd}"}), 400
    except Exception as e:
        return _safe_error_response(e, "api_exec")


# ── Public API ────────────────────────────────────────────────────────────────

def update_signal(signal: dict):        state.update_signal(signal)
def update_account(account: dict):      state.update_account(account)
def update_guard(guard: dict):          state.update_guard(guard)
def update_stats(stats: dict):          state.update_stats(stats)
def update_trades(trades: List[dict]):  state.update_trades(trades)
def add_log(message: str, level: str = "info"): state.add_log(message, level)

def update_candles(symbol: str, tf: str, bars: list):
    """
    Push OHLCV bars to the dashboard's Lightweight Charts renderer.

    Call this once per bar close (same cadence as update_chart).
    bars format: [{"time": unix_ts_int, "open": f, "high": f, "low": f, "close": f, "volume": f}, ...]

    Helper to convert a DataFrame row to a bar dict:
        def df_to_bars(df):
            return [{"time": int(ts.timestamp()), "open": r.open, "high": r.high,
                     "low": r.low, "close": r.close, "volume": float(r.get("volume", 0))}
                    for ts, r in df.iterrows()]
    """
    state.update_candles(symbol, tf, bars)

def update_structure(symbol: str, tf: str, structure: dict):
    """
    Push a CEO structure overlay to the dashboard's live chart.

    structure format (see chart_lwc.build_structure_payload()):
        {"zoneLines": [...], "priceLines": [...], "markers": [...]}
    Call this alongside update_candles() -- same cadence, same symbol/tf key.
    """
    state.update_structure(symbol, tf, structure)

def update_chart(symbol: str, tf: str, html: str):
    """Legacy HTML chart — still accepted, stored as fallback."""
    state.update_chart(symbol, tf, html)

def set_executor(executor, conn=None):
    """Called by run_live() once executor and MT5 connection are ready."""
    state.set_executor(executor, conn)

def update_market_tick(symbol: str, bid: float, ask: float):
    """Push live bid/ask tick for Market Watch. Called from trade management tick."""
    state.update_market_tick(symbol, bid, ask)

def update_backtest(symbol: str, tf: str, results: list):
    """Push backtest results rows for the Backtest tab.
    results: list of dicts with keys matching results_table() columns.
    """
    state.update_backtest(symbol, tf, results)

def set_journal(path: str):
    """Connect dashboard to a Journal SQLite database."""
    state.journal_path = path

def start_dashboard(host: Optional[str] = None, port: int = 5000, debug: bool = False):
    """
    Starts the dashboard Flask app in a background thread.

    host: defaults to 127.0.0.1 (localhost-only) unless explicitly overridden
    -- either by passing host= directly, or by setting CEO_DASHBOARD_HOST
    (e.g. "0.0.0.0" to expose on the network). Binding wide open by default
    meant anyone on the same network, or the internet if the port is
    forwarded, could reach a server that can read MT5 credentials and place
    trades; auth + rate limiting reduce but don't eliminate that exposure,
    so the safer default is localhost-only with an explicit opt-in.
    """
    if host is None:
        host = os.environ.get("CEO_DASHBOARD_HOST", "127.0.0.1")
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=debug, use_reloader=False),
        daemon=True)
    t.start()
    msg = f"🌐  Dashboard running at http://localhost:{port}"
    logger.info(msg)
    if host not in ("127.0.0.1", "localhost"):
        logger.warning(f"⚠️  Dashboard bound to {host} — reachable beyond this machine. "
                        f"Make sure that's intentional.")
    logger.info(f"🔐  Dashboard login configured — username: {_DASHBOARD_USER} "
                f"(password saved to {_AUTH_CONFIG_PATH}, permissions 0600; "
                f"not written to this log file)")
    print(f"  🔐  Login — username: {_DASHBOARD_USER}  password: {_DASHBOARD_PASSWORD}")
    print(f"      (password saved to {_AUTH_CONFIG_PATH}; "
          f"override anytime by setting CEO_DASHBOARD_PASSWORD)")
    try:
        state.add_log(msg, level="info")
    except Exception:
        pass
    # Start a lightweight MT5 poller so the dashboard can show account
    # and market-tick data even while the main engine isn't running.
    def _mt5_poller(interval: int = 5):
        import sys
        if sys.platform != "win32":
            return
        try:
            import MetaTrader5 as mt5
        except Exception:
            return
        import time as _local_time
        common = [
            "XAUUSD","GBPUSD","EURUSD","USDJPY","USDCAD","AUDUSD",
            "NZDUSD","USDCHF","BTCUSD","ETHUSD","XAGUSD",
            "GBPJPY","EURJPY","EURGBP","AUDJPY"
        ]
        while True:
            try:
                if not mt5.initialize():
                    try:
                        mt5.shutdown()
                    except Exception:
                        pass
                    _local_time.sleep(interval)
                    continue
                acct = mt5.account_info()
                if acct:
                    try:
                        state.update_account({
                            "login": acct.login,
                            "server": acct.server,
                            "name": acct.name,
                            "currency": acct.currency,
                            "balance": round(acct.balance, 2),
                            "equity": round(acct.equity, 2),
                            "leverage": acct.leverage,
                            "account_type": "demo" if "demo" in acct.server.lower() else "live",
                        })
                    except Exception:
                        pass
                # Update some common symbol ticks if available
                try:
                    all_syms = [s.name for s in (mt5.symbols_get() or [])]
                    broker_syms = [s for s in common if s in all_syms]
                    for sym in broker_syms:
                        try:
                            tk = mt5.symbol_info_tick(sym)
                            if tk:
                                state.update_market_tick(sym, float(tk.bid), float(tk.ask))
                        except Exception:
                            continue
                except Exception:
                    pass
                try:
                    mt5.shutdown()
                except Exception:
                    pass
            except Exception:
                try:
                    mt5.shutdown()
                except Exception:
                    pass
            _local_time.sleep(interval)

    try:
        poll_thread = threading.Thread(target=_mt5_poller, daemon=True)
        poll_thread.start()
    except Exception:
        pass

    return t


# ── HTML ─────────────────────────────────────────────────────────────────────

def _dashboard_template_path() -> Path:
    """
    Resolves the dashboard's HTML/CSS/JS template file, both when running
    from source and when frozen into the PyInstaller onedir bundle built
    by ceo_engine.spec (which bundles this same "ceo_engine_mt5/templates"
    folder via its `datas` list, mirroring the source layout so both
    branches below use an identical relative path from their own base).
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)          # PyInstaller bundle root
    else:
        base = Path(__file__).resolve().parent.parent   # project root (source checkout)
    return base / "ceo_engine_mt5" / "templates" / "dashboard.html"


def _load_dashboard_template() -> str:
    path = _dashboard_template_path()
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dashboard template not found at {path}. If running from a "
            f"PyInstaller build, check that ceo_engine.spec's `datas` list "
            f"includes ('ceo_engine_mt5/templates', 'ceo_engine_mt5/templates'). "
            f"If running from source, check that "
            f"ceo_engine_mt5/templates/dashboard.html exists."
        ) from None


DASHBOARD_HTML = _load_dashboard_template()
